"""SQLite persistence: :class:`SQLiteMemoryStore` and :class:`SQLiteRecollectionQueue`.

Both live in ONE database file so separate processes (a stdio MCP server, the HTTP ingestion
server, a Claude Code hook) share memories, trigger history and the push queue.

Layout (``PRAGMA user_version`` = :data:`SCHEMA_VERSION`)::

    schema_version(version PK, applied_at)
    meta(key PK, value)                       -- flags, e.g. vec_dirty
    memories(rowid PK AUTOINCREMENT, id UNIQUE, content, media_ref, created_at, ts, ts_us,
             geohash, salience, tags_json, metadata_json, observations_json)
    channels(id PK AUTOINCREMENT, name UNIQUE, dim, model_id)
    vectors(memory_rowid, channel, embedding BLOB float32)   -- canonical copy
    vec_ch<channels.id>  -- sqlite-vec vec0(embedding float[dim]), rowid = memories.rowid
    raw_blobs(memory_id, channel, data)        -- only for observations that kept ``data``
    triggers(memory_id, at, at_us)
    queue(seq PK AUTOINCREMENT, rec_id, memory_id, payload_json, enqueued_at)
                                               -- payload: memory id + scores, no content

Schema history: 1 = initial; 2 = slim queue payloads (codec v2: the memory id and score
explanation only; drained recollections are rehydrated from ``memories``). Opening a v1 file
migrates its queue rows in place.

The ``vectors`` table is the source of truth; ``vec0`` tables are a KNN index over it. When
the sqlite-vec extension cannot be loaded, ``nearest`` falls back to numpy brute force over
``vectors`` (a warning is logged once) and writes set ``meta.vec_dirty``. A connection that
*can* load the extension checks the flag before each KNN query and uses brute force while it
is set (so mixed processes never miss memories); it rebuilds the index when it opens and on
its next write.

Concurrency: one connection per :class:`SQLiteDatabase` (``check_same_thread=False``)
guarded by a lock; WAL mode and ``busy_timeout`` across connections/processes; every write
is a ``BEGIN IMMEDIATE`` transaction, which makes ``drain`` and ``claim_trigger`` atomic
across processes.
"""

# Every interpolated SQL fragment is generated internally (integer vec table ids, "?" placeholder
# lists, module constants); all user data is bound as parameters.
# ruff: noqa: S608

from __future__ import annotations

import json
import logging
import os
import sqlite3
import stat
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from madeleine.core.types import (
    ChannelName,
    ChannelSpec,
    Memory,
    MemoryNotFoundError,
    Observation,
    Recollection,
    RecollectionListener,
    SensoryContext,
    SensoryFingerprint,
    StoreError,
    Unsubscribe,
    Vector,
)
from madeleine.store.codec import (
    QueuedRecollection,
    observation_from_dict,
    queued_from_dict,
    vector_from_bytes,
    vector_to_bytes,
)
from madeleine.store.memory import _check_limit, _notify, l2_nearest

__all__ = [
    "SCHEMA_VERSION",
    "SQLiteDatabase",
    "SQLiteMemoryStore",
    "SQLiteRecollectionQueue",
    "open_sqlite",
]

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 2
VEC_MAX_K = 4096
"""sqlite-vec's KNN ``k`` limit; larger requests use the brute-force path."""
_CHUNK = 256
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_fallback_warned = False
_warn_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS memories (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    media_ref TEXT,
    created_at TEXT NOT NULL,
    ts TEXT NOT NULL,
    ts_us INTEGER NOT NULL,
    geohash TEXT,
    salience REAL NOT NULL,
    tags_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    observations_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS channels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    dim INTEGER NOT NULL,
    model_id TEXT
);
CREATE TABLE IF NOT EXISTS vectors (
    memory_rowid INTEGER NOT NULL,
    channel TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (channel, memory_rowid)
);
CREATE INDEX IF NOT EXISTS vectors_by_memory ON vectors(memory_rowid);
CREATE TABLE IF NOT EXISTS raw_blobs (
    memory_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (memory_id, channel)
);
CREATE TABLE IF NOT EXISTS triggers (
    memory_id TEXT NOT NULL,
    at TEXT NOT NULL,
    at_us INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS triggers_by_memory ON triggers(memory_id, at_us);
CREATE TABLE IF NOT EXISTS queue (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_id TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    enqueued_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS queue_by_memory ON queue(memory_id);
"""


def _micros(dt: datetime) -> int:
    return (dt - _EPOCH) // timedelta(microseconds=1)


def _is_file_path(path: str) -> bool:
    return path not in ("", ":memory:") and not path.startswith("file:")


def _restrict_permissions(path: str) -> None:
    """Make ``path`` owner-only (``0o600``), best effort (R16).

    A missing file is created with that mode before SQLite opens it; an existing file that
    is group/other accessible is tightened. SQLite creates the ``-wal`` / ``-shm`` siblings
    with the main file's mode; they are tightened too if they already exist. No-op where
    POSIX modes do not apply (Windows) or when the file is not ours to chmod.
    """
    if os.name == "nt" or not _is_file_path(path):
        return
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    except OSError as exc:
        _log.debug("cannot pre-create %s owner-only: %s", path, exc)
    else:
        os.close(fd)
    for name in (path, f"{path}-wal", f"{path}-shm"):
        try:
            mode = stat.S_IMODE(os.stat(name).st_mode)
            if mode & 0o077:
                os.chmod(name, mode & 0o700)
        except FileNotFoundError:
            continue
        except OSError as exc:
            _log.debug("cannot restrict permissions of %s: %s", name, exc)


def _load_sqlite_vec(conn: sqlite3.Connection) -> str | None:
    """Try to load sqlite-vec into ``conn``; return ``None`` on success or the reason."""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        conn.execute("SELECT vec_version()").fetchone()
    except (ImportError, AttributeError, OSError, sqlite3.Error) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


class SQLiteDatabase:
    """One SQLite connection (+ lock) shared by a store and a queue.

    * ``path`` - file path or ``":memory:"`` (then store and queue MUST share this object).
    * ``use_sqlite_vec`` - ``None`` (default): use the extension when it loads, otherwise
      fall back to numpy with a one-time warning; ``False``: force the fallback; ``True``:
      require it (:class:`StoreError` if unavailable).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        use_sqlite_vec: bool | None = None,
        busy_timeout: timedelta = timedelta(seconds=30),
    ) -> None:
        self.path = os.fspath(path)
        self._lock = threading.RLock()
        self._refs = 0
        self._closed = False
        _restrict_permissions(self.path)
        try:
            self._conn = sqlite3.connect(
                self.path,
                timeout=busy_timeout.total_seconds(),
                check_same_thread=False,
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open SQLite database {self.path!r}: {exc}") from exc
        try:
            conn = self._conn
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout.total_seconds() * 1000)}")
            # Freed pages are zeroed so forgotten content and raw bytes are not recoverable
            # from the file (R7). Per connection: every process must set it.
            conn.execute("PRAGMA secure_delete = ON")
            if self.path != ":memory:" and not self.path.startswith("file::memory:"):
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                _restrict_permissions(self.path)
            self.vec_enabled = False
            if use_sqlite_vec is not False:
                reason = _load_sqlite_vec(conn)
                if reason is None:
                    self.vec_enabled = True
                elif use_sqlite_vec:
                    raise StoreError(f"sqlite-vec extension unavailable: {reason}")
                else:
                    _warn_fallback(reason)
            self._init_schema()
        except BaseException:
            self._conn.close()
            raise

    # -- connection plumbing -------------------------------------------------------------
    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """Locked access to the connection (autocommit; for reads and single statements)."""
        with self._lock:
            if self._closed:
                raise StoreError("database is closed")
            try:
                yield self._conn
            except sqlite3.Error as exc:
                raise StoreError(f"SQLite error: {exc}") from exc

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """A ``BEGIN IMMEDIATE`` transaction (committed on success, rolled back on error)."""
        with self.read() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def acquire(self) -> SQLiteDatabase:
        with self._lock:
            self._refs += 1
        return self

    def release(self) -> None:
        """Drop one reference; the connection closes when the last owner releases it."""
        with self._lock:
            self._refs -= 1
            if self._refs <= 0 and not self._closed:
                self._closed = True
                self._conn.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def checkpoint(self) -> None:
        """Passive WAL checkpoint: copy committed (securely deleted) pages into the main file.

        Best effort: never blocks writers, and a busy WAL is simply checkpointed later (the
        last connection to close checkpoints and removes the WAL anyway).
        """
        with self._lock:
            if self._closed:
                return
            try:
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            except sqlite3.Error as exc:
                _log.debug("passive WAL checkpoint failed: %s", exc)

    # -- schema --------------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self.write() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise StoreError(
                    f"database schema version {version} is newer than supported "
                    f"({SCHEMA_VERSION}); upgrade madeleine"
                )
            for stmt in _SCHEMA.split(";"):
                if stmt.strip():
                    conn.execute(stmt)
            if 1 <= version < 2:
                _migrate_queue_to_v2(conn)
            if version < SCHEMA_VERSION:
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, datetime.now(UTC).isoformat()),
                )
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            if self.vec_enabled:
                self._sync_vec_tables(conn)

    def _sync_vec_tables(self, conn: sqlite3.Connection) -> None:
        rebuild = _vec_dirty(conn)
        for ch_id, name, dim in conn.execute("SELECT id, name, dim FROM channels").fetchall():
            table = _vec_table(ch_id)
            if rebuild and _table_exists(conn, table):
                conn.execute(f"DROP TABLE {table}")
            _ensure_vec_table(conn, ch_id, name, dim)
        if rebuild:
            _log.info("rebuilt sqlite-vec index after fallback-mode writes")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('vec_dirty', '0')")

    def vec_index_current(self) -> bool:
        """Whether KNN may use the sqlite-vec index (R14).

        ``False`` when the extension is not loaded or another connection wrote without it
        (``meta.vec_dirty``): ``nearest`` then uses exact brute force, so a long-running
        process with the extension never misses vectors written by a fallback-mode process.
        Checked before every KNN query (one primary-key lookup); reads never rebuild.
        """
        if not self.vec_enabled:
            return False
        with self.read() as conn:
            return not _vec_dirty(conn)

    def repair_vec_index(self, conn: sqlite3.Connection) -> None:
        """Inside a write transaction: rebuild the index if fallback-mode writes made it
        stale (so the next write of a sqlite-vec connection restores fast KNN)."""
        if self.vec_enabled and _vec_dirty(conn):
            self._sync_vec_tables(conn)

    def mark_vec_dirty(self, conn: sqlite3.Connection) -> None:
        """Called inside write transactions made without the extension."""
        if not self.vec_enabled:
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('vec_dirty', '1')")


def _warn_fallback(reason: str) -> None:
    global _fallback_warned
    with _warn_lock:
        if _fallback_warned:
            return
        _fallback_warned = True
    _log.warning(
        "sqlite-vec could not be loaded (%s); using numpy brute-force nearest neighbours", reason
    )


def _migrate_queue_to_v2(conn: sqlite3.Connection) -> None:
    """Schema 1 -> 2: rewrite queue payloads without the memory (content, embeddings and
    observations). Rows that cannot be decoded or whose memory is gone are dropped; with
    ``secure_delete`` the old payload bytes are zeroed."""
    kept = dropped = 0
    rows = conn.execute("SELECT seq, payload_json FROM queue ORDER BY seq").fetchall()
    for seq, payload in rows:
        try:
            queued = queued_from_dict(json.loads(payload))
        except (ValueError, KeyError, TypeError):
            queued = None
        exists = queued is not None and (
            conn.execute("SELECT 1 FROM memories WHERE id = ?", (queued.memory_id,)).fetchone()
            is not None
        )
        if queued is None or not exists:
            conn.execute("DELETE FROM queue WHERE seq = ?", (seq,))
            dropped += 1
            continue
        conn.execute(
            "UPDATE queue SET payload_json = ?, memory_id = ? WHERE seq = ?",
            (_queued_payload(queued), queued.memory_id, seq),
        )
        kept += 1
    if rows:
        _log.info("migrated %d queued recollection(s) to slim payloads (%d dropped)", kept, dropped)


def _vec_dirty(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM meta WHERE key = 'vec_dirty'").fetchone()
    return row is not None and row[0] == "1"


def _vec_table(channel_id: int) -> str:
    return f"vec_ch{int(channel_id)}"


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (name,)).fetchone()
    return row is not None


def _ensure_vec_table(conn: sqlite3.Connection, channel_id: int, name: str, dim: int) -> str:
    """Create (and backfill from ``vectors``) the vec0 index of a channel if missing."""
    table = _vec_table(channel_id)
    if not _table_exists(conn, table):
        conn.execute(f"CREATE VIRTUAL TABLE {table} USING vec0(embedding float[{int(dim)}])")
        conn.execute(
            f"INSERT INTO {table}(rowid, embedding) "
            "SELECT memory_rowid, embedding FROM vectors WHERE channel = ?",
            (name,),
        )
    return table


def open_sqlite(
    path: str | os.PathLike[str], *, use_sqlite_vec: bool | None = None
) -> tuple[SQLiteMemoryStore, SQLiteRecollectionQueue]:
    """Open a store and a queue sharing one connection to ``path`` (``":memory:"`` works)."""
    db = SQLiteDatabase(path, use_sqlite_vec=use_sqlite_vec)
    return SQLiteMemoryStore(db), SQLiteRecollectionQueue(db)


def _db_for(
    target: str | os.PathLike[str] | SQLiteDatabase, use_sqlite_vec: bool | None
) -> SQLiteDatabase:
    if isinstance(target, SQLiteDatabase):
        return target.acquire()
    return SQLiteDatabase(target, use_sqlite_vec=use_sqlite_vec).acquire()


# --------------------------------------------------------------------------------------
# Memory store
# --------------------------------------------------------------------------------------

_MEM_COLS = (
    "rowid, id, content, media_ref, created_at, ts, geohash, salience, "
    "tags_json, metadata_json, observations_json"
)


def _hydrate(
    conn: sqlite3.Connection, rows: Sequence[Sequence[Any]], *, with_raw: bool = True
) -> list[Memory]:
    """Build memories from ``_MEM_COLS`` rows (+ their vectors, and raw bytes if asked)."""
    if not rows:
        return []
    rowids = [int(r[0]) for r in rows]
    ids = [str(r[1]) for r in rows]
    marks = ",".join("?" * len(rows))
    vectors: dict[int, dict[ChannelName, Vector]] = {}
    for mrow, channel, blob in conn.execute(
        f"SELECT memory_rowid, channel, embedding FROM vectors WHERE memory_rowid IN ({marks})",
        rowids,
    ):
        vectors.setdefault(int(mrow), {})[str(channel)] = vector_from_bytes(blob)
    raws: dict[str, dict[ChannelName, bytes]] = {}
    if with_raw:
        for mid, channel, data in conn.execute(
            f"SELECT memory_id, channel, data FROM raw_blobs WHERE memory_id IN ({marks})", ids
        ):
            raws.setdefault(str(mid), {})[str(channel)] = bytes(data)
    out: list[Memory] = []
    for r in rows:
        rowid, mid = int(r[0]), str(r[1])
        observations: dict[ChannelName, Observation] = {}
        for c, d in json.loads(r[10]).items():
            obs = observation_from_dict(d)
            raw = raws.get(mid, {}).get(c)
            if raw is not None:
                obs = Observation(
                    channel=obs.channel,
                    text=obs.text,
                    data=raw,
                    mime_type=obs.mime_type,
                    uri=obs.uri,
                    numbers=obs.numbers,
                    timestamp=obs.timestamp,
                )
            observations[c] = obs
        ctx = SensoryContext(
            timestamp=datetime.fromisoformat(r[5]),
            geohash=r[6],
            tags=frozenset(json.loads(r[8])),
            salience=float(r[7]),
        )
        out.append(
            Memory(
                id=mid,
                content=str(r[2]),
                fingerprint=SensoryFingerprint(
                    embeddings=vectors.get(rowid, {}), context=ctx, observations=observations
                ),
                created_at=datetime.fromisoformat(r[4]),
                media_ref=r[3],
                metadata=json.loads(r[9]),
            )
        )
    return out


def _memories_by_id(
    conn: sqlite3.Connection, memory_ids: Sequence[str], *, with_raw: bool = True
) -> dict[str, Memory]:
    wanted = list(dict.fromkeys(memory_ids))
    by_id: dict[str, Memory] = {}
    for i in range(0, len(wanted), _CHUNK):
        chunk = wanted[i : i + _CHUNK]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT {_MEM_COLS} FROM memories WHERE id IN ({marks})", chunk
        ).fetchall()
        for m in _hydrate(conn, rows, with_raw=with_raw):
            by_id[m.id] = m
    return by_id


class SQLiteMemoryStore:
    """:class:`~madeleine.core.types.MemoryStore` backed by SQLite (+ sqlite-vec)."""

    def __init__(
        self,
        target: str | os.PathLike[str] | SQLiteDatabase,
        *,
        use_sqlite_vec: bool | None = None,
    ) -> None:
        self._db = _db_for(target, use_sqlite_vec)
        self._closed = False

    @property
    def database(self) -> SQLiteDatabase:
        return self._db

    @property
    def uses_sqlite_vec(self) -> bool:
        """Whether ``nearest`` runs on sqlite-vec (False: numpy brute-force fallback)."""
        return self._db.vec_enabled

    # -- channels ------------------------------------------------------------------------
    def register_channel(self, spec: ChannelSpec) -> None:
        with self._db.write() as conn:
            self._db.repair_vec_index(conn)
            row = conn.execute(
                "SELECT id, dim, model_id FROM channels WHERE name = ?", (spec.name,)
            ).fetchone()
            if row is None:
                cur = conn.execute(
                    "INSERT INTO channels(name, dim, model_id) VALUES (?, ?, ?)",
                    (spec.name, spec.dim, spec.model_id),
                )
                ch_id = int(cur.lastrowid or 0)
                if self._db.vec_enabled:
                    _ensure_vec_table(conn, ch_id, spec.name, spec.dim)
                else:
                    self._db.mark_vec_dirty(conn)
                return
            ch_id, dim, model_id = int(row[0]), int(row[1]), row[2]
            if (dim, model_id) == (spec.dim, spec.model_id):
                if self._db.vec_enabled:
                    _ensure_vec_table(conn, ch_id, spec.name, spec.dim)
                return
            held = conn.execute(
                "SELECT COUNT(*) FROM vectors WHERE channel = ?", (spec.name,)
            ).fetchone()[0]
            if held:
                raise StoreError(_model_change_message(spec, int(held), dim, model_id))
            conn.execute(
                "UPDATE channels SET dim = ?, model_id = ? WHERE id = ?",
                (spec.dim, spec.model_id, ch_id),
            )
            if self._db.vec_enabled:
                if dim != spec.dim and _table_exists(conn, _vec_table(ch_id)):
                    conn.execute(f"DROP TABLE {_vec_table(ch_id)}")
                _ensure_vec_table(conn, ch_id, spec.name, spec.dim)
            else:
                self._db.mark_vec_dirty(conn)

    def check_channel(self, spec: ChannelSpec) -> bool:
        """Read-only counterpart of :meth:`register_channel` (for recall paths, R30).

        ``True`` if ``spec`` is registered as is; ``False`` if the channel is unknown or holds
        no vectors under a different ``dim``/``model_id`` (nothing to match: treat as
        empty). Raises :class:`StoreError` exactly where :meth:`register_channel` would.
        Never writes.
        """
        with self._db.read() as conn:
            row = conn.execute(
                "SELECT dim, model_id FROM channels WHERE name = ?", (spec.name,)
            ).fetchone()
            if row is None:
                return False
            dim, model_id = int(row[0]), row[1]
            if (dim, model_id) == (spec.dim, spec.model_id):
                return True
            held = int(
                conn.execute(
                    "SELECT COUNT(*) FROM vectors WHERE channel = ?", (spec.name,)
                ).fetchone()[0]
            )
        if held:
            raise StoreError(_model_change_message(spec, held, dim, model_id))
        return False

    def _channels(self, conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
        rows = conn.execute("SELECT name, id, dim FROM channels").fetchall()
        return {str(n): (int(i), int(d)) for n, i, d in rows}

    # -- memories ------------------------------------------------------------------------
    def add(self, memory: Memory) -> None:
        fp = memory.fingerprint
        ctx = fp.context
        with self._db.write() as conn:
            self._db.repair_vec_index(conn)
            channels = self._channels(conn)
            for name, vec in fp.embeddings.items():
                if name not in channels:
                    raise ValueError(f"channel {name!r} is not registered")
                if len(vec) != channels[name][1]:
                    raise ValueError(
                        f"vector for {name!r} has length {len(vec)}, not {channels[name][1]}"
                    )
            if conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory.id,)).fetchone():
                raise ValueError(f"memory {memory.id!r} already exists")
            observations = {
                c: o.to_dict(include_raw=False) for c, o in sorted(fp.observations.items())
            }
            cur = conn.execute(
                "INSERT INTO memories(id, content, media_ref, created_at, ts, ts_us, geohash, "
                "salience, tags_json, metadata_json, observations_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    memory.content,
                    memory.media_ref,
                    memory.created_at.isoformat(),
                    ctx.timestamp.isoformat(),
                    _micros(ctx.timestamp),
                    ctx.geohash,
                    ctx.salience,
                    json.dumps(sorted(ctx.tags)),
                    json.dumps(dict(memory.metadata), sort_keys=True),
                    json.dumps(observations, sort_keys=True),
                ),
            )
            rowid = int(cur.lastrowid or 0)
            for name, vec in fp.embeddings.items():
                blob = vector_to_bytes(vec)
                conn.execute(
                    "INSERT INTO vectors(memory_rowid, channel, embedding) VALUES (?, ?, ?)",
                    (rowid, name, blob),
                )
                if self._db.vec_enabled:
                    ch_id, dim = channels[name]
                    table = _ensure_vec_table(conn, ch_id, name, dim)
                    conn.execute(
                        f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)", (rowid, blob)
                    )
            for name, obs in fp.observations.items():
                if obs.data is not None:
                    conn.execute(
                        "INSERT INTO raw_blobs(memory_id, channel, data) VALUES (?, ?, ?)",
                        (memory.id, name, obs.data),
                    )
            self._db.mark_vec_dirty(conn)

    def update(self, memory: Memory) -> None:
        """Replace a stored memory with an edited version of itself, in one transaction.

        The id and the set of channels must stay the same. Changed vectors are rewritten
        (and re-indexed); raw blobs are never touched. Raises :class:`MemoryNotFoundError` or
        ``ValueError``."""
        fp = memory.fingerprint
        ctx = fp.context
        with self._db.write() as conn:
            self._db.repair_vec_index(conn)
            row = conn.execute("SELECT rowid FROM memories WHERE id = ?", (memory.id,)).fetchone()
            if row is None:
                raise MemoryNotFoundError(memory.id)
            rowid = int(row[0])
            stored = {
                str(name): bytes(blob)
                for name, blob in conn.execute(
                    "SELECT channel, embedding FROM vectors WHERE memory_rowid = ?", (rowid,)
                ).fetchall()
            }
            if set(stored) != set(fp.embeddings):
                raise ValueError("update cannot add or remove channels")
            channels = self._channels(conn)
            for name, vec in fp.embeddings.items():
                if len(vec) != channels[name][1]:
                    raise ValueError(f"vector for {name!r} has the wrong length")
            observations = {
                c: o.to_dict(include_raw=False) for c, o in sorted(fp.observations.items())
            }
            conn.execute(
                "UPDATE memories SET content = ?, media_ref = ?, ts = ?, ts_us = ?, geohash = ?, "
                "salience = ?, tags_json = ?, metadata_json = ?, observations_json = ? "
                "WHERE rowid = ?",
                (
                    memory.content,
                    memory.media_ref,
                    ctx.timestamp.isoformat(),
                    _micros(ctx.timestamp),
                    ctx.geohash,
                    ctx.salience,
                    json.dumps(sorted(ctx.tags)),
                    json.dumps(dict(memory.metadata), sort_keys=True),
                    json.dumps(observations, sort_keys=True),
                    rowid,
                ),
            )
            changed = False
            for name, vec in fp.embeddings.items():
                blob = vector_to_bytes(vec)
                if blob == stored[name]:
                    continue
                changed = True
                conn.execute(
                    "UPDATE vectors SET embedding = ? WHERE memory_rowid = ? AND channel = ?",
                    (blob, rowid, name),
                )
                if self._db.vec_enabled:
                    ch_id, dim = channels[name]
                    table = _ensure_vec_table(conn, ch_id, name, dim)
                    conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
                    conn.execute(
                        f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)", (rowid, blob)
                    )
            if changed:
                self._db.mark_vec_dirty(conn)

    def get(self, memory_id: str) -> Memory:
        with self._db.read() as conn:
            rows = conn.execute(
                f"SELECT {_MEM_COLS} FROM memories WHERE id = ?", (memory_id,)
            ).fetchall()
            found = _hydrate(conn, rows)
        if not found:
            raise MemoryNotFoundError(memory_id)
        return found[0]

    def get_many(self, memory_ids: Sequence[str]) -> list[Memory]:
        with self._db.read() as conn:
            by_id = _memories_by_id(conn, memory_ids)
        return [by_id[i] for i in memory_ids if i in by_id]

    def delete(self, memory_id: str) -> bool:
        """Delete the memory, its vectors, raw data, triggers and pending queue entries in one
        transaction, then checkpoint the WAL (freed pages are zeroed: ``secure_delete``)."""
        with self._db.write() as conn:
            self._db.repair_vec_index(conn)
            row = conn.execute("SELECT rowid FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                conn.execute("DELETE FROM triggers WHERE memory_id = ?", (memory_id,))
                return False
            rowid = int(row[0])
            if self._db.vec_enabled:
                channels = self._channels(conn)
                for (name,) in conn.execute(
                    "SELECT channel FROM vectors WHERE memory_rowid = ?", (rowid,)
                ).fetchall():
                    if name in channels:
                        table = _vec_table(channels[name][0])
                        if _table_exists(conn, table):
                            conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
            conn.execute("DELETE FROM vectors WHERE memory_rowid = ?", (rowid,))
            conn.execute("DELETE FROM raw_blobs WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM triggers WHERE memory_id = ?", (memory_id,))
            # The queue table lives in the same file: its pending recollections of this memory
            # go in the same transaction, whichever connection the queue uses (R15).
            conn.execute("DELETE FROM queue WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE rowid = ?", (rowid,))
            self._db.mark_vec_dirty(conn)
        self._db.checkpoint()
        return True

    def iter_memories(self, *, tags: frozenset[str] | None = None) -> Iterator[Memory]:
        """All memories in insertion order (fetched in chunks; the lock is not held between)."""
        last = 0
        while True:
            with self._db.read() as conn:
                rows = conn.execute(
                    f"SELECT {_MEM_COLS} FROM memories WHERE rowid > ? ORDER BY rowid LIMIT ?",
                    (last, _CHUNK),
                ).fetchall()
                batch = _hydrate(conn, rows)
            if not rows:
                return
            last = int(rows[-1][0])
            for m in batch:
                if not tags or tags <= m.context.tags:
                    yield m

    def count(self) -> int:
        with self._db.read() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    # -- nearest -------------------------------------------------------------------------
    def nearest(self, channel: ChannelName, vector: Vector, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        use_index = k <= VEC_MAX_K and self._db.vec_index_current()
        with self._db.read() as conn:
            info = self._channels(conn).get(channel)
            if info is None:
                return []
            ch_id, dim = info
            if len(vector) != dim:
                raise ValueError(
                    f"query vector for {channel!r} has length {len(vector)}, not {dim}"
                )
            table = _vec_table(ch_id)
            if use_index and _table_exists(conn, table):
                hits = conn.execute(
                    f"SELECT rowid, distance FROM {table} WHERE embedding MATCH ? AND k = ? "
                    "ORDER BY distance",
                    (vector_to_bytes(vector), k),
                ).fetchall()
                if not hits:
                    return []
                marks = ",".join("?" * len(hits))
                id_of = dict(
                    conn.execute(
                        f"SELECT rowid, id FROM memories WHERE rowid IN ({marks})",
                        [h[0] for h in hits],
                    ).fetchall()
                )
                return [(str(id_of[r]), float(d)) for r, d in hits if r in id_of]
            rows = conn.execute(
                "SELECT m.id, v.embedding FROM vectors v JOIN memories m "
                "ON m.rowid = v.memory_rowid WHERE v.channel = ?",
                (channel,),
            ).fetchall()
        return _brute_force(rows, vector, k, dim)

    def nearest_tagged(
        self, channel: ChannelName, vector: Vector, k: int, tags: frozenset[str]
    ) -> list[tuple[str, float]]:
        """Like :meth:`nearest`, restricted to memories having *all* ``tags``.

        The tag filter runs in SQL (``json_each`` over ``tags_json``) and the distances are
        exact brute force over that subset, so a tagged memory cannot be starved by closer
        untagged ones (R9). An extension of the
        :class:`~madeleine.core.types.MemoryStore` protocol used by the engine.
        """
        if not tags:
            return self.nearest(channel, vector, k)
        if k <= 0:
            return []
        wanted = sorted(tags)
        marks = ",".join("?" * len(wanted))
        with self._db.read() as conn:
            info = self._channels(conn).get(channel)
            if info is None:
                return []
            dim = info[1]
            if len(vector) != dim:
                raise ValueError(
                    f"query vector for {channel!r} has length {len(vector)}, not {dim}"
                )
            rows = conn.execute(
                "SELECT m.id, v.embedding FROM vectors v JOIN memories m "
                "ON m.rowid = v.memory_rowid WHERE v.channel = ? AND ("
                f"SELECT COUNT(DISTINCT t.value) FROM json_each(m.tags_json) t "
                f"WHERE t.value IN ({marks})) = ?",
                (channel, *wanted, len(wanted)),
            ).fetchall()
        return _brute_force(rows, vector, k, dim)

    # -- triggers ------------------------------------------------------------------------
    def record_trigger(self, memory_id: str, at: datetime) -> None:
        with self._db.write() as conn:
            _insert_trigger(conn, memory_id, at)

    def last_triggered(self, memory_id: str) -> datetime | None:
        with self._db.read() as conn:
            return _last_trigger(conn, memory_id)

    def claim_trigger(self, memory_id: str, at: datetime, refractory_period: timedelta) -> bool:
        """Atomically (across processes) record a trigger unless the memory is refractory.

        Refractory means *any* recorded trigger ``t`` with ``abs(at - t) < refractory_period``
        (not only the latest one), so out-of-order and future-dated cues cannot re-open the
        window (R13). Returns ``False`` without recording anything when the memory does not
        exist (any more): a ``sense`` racing a ``forget`` cannot leave an orphan trigger or
        enqueue the forgotten memory (R15).
        """
        with self._db.write() as conn:
            if conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone() is None:
                return False
            if _fired_within(conn, memory_id, at, refractory_period):
                return False
            _insert_trigger(conn, memory_id, at)
            return True

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._db.release()


def _model_change_message(spec: ChannelSpec, held: int, dim: int, model_id: object) -> str:
    return (
        f"channel {spec.name!r} holds {held} vectors with dim={dim}, "
        f"model_id={model_id!r}; refusing dim={spec.dim}, model_id={spec.model_id!r} "
        "(vectors from different models are not comparable; use a new database "
        "or re-embed)"
    )


def _brute_force(
    rows: Sequence[Sequence[Any]], vector: Vector, k: int, dim: int
) -> list[tuple[str, float]]:
    if not rows:
        return []
    ids = [str(r[0]) for r in rows]
    matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype="<f4").reshape(len(rows), dim)
    return l2_nearest(ids, matrix, vector, k)


def _insert_trigger(conn: sqlite3.Connection, memory_id: str, at: datetime) -> None:
    conn.execute(
        "INSERT INTO triggers(memory_id, at, at_us) VALUES (?, ?, ?)",
        (memory_id, at.isoformat(), _micros(at)),
    )


def _fired_within(
    conn: sqlite3.Connection, memory_id: str, at: datetime, period: timedelta
) -> bool:
    """Whether a trigger exists strictly within ``period`` of ``at`` (range on the index)."""
    if period <= timedelta(0):
        return False
    at_us, span = _micros(at), period // timedelta(microseconds=1)
    row = conn.execute(
        "SELECT 1 FROM triggers WHERE memory_id = ? AND at_us > ? AND at_us < ? LIMIT 1",
        (memory_id, at_us - span, at_us + span),
    ).fetchone()
    return row is not None


def _last_trigger(conn: sqlite3.Connection, memory_id: str) -> datetime | None:
    row = conn.execute(
        "SELECT at FROM triggers WHERE memory_id = ? ORDER BY at_us DESC LIMIT 1", (memory_id,)
    ).fetchone()
    return datetime.fromisoformat(row[0]) if row else None


# --------------------------------------------------------------------------------------
# Recollection queue
# --------------------------------------------------------------------------------------


class SQLiteRecollectionQueue:
    """Persistent FIFO shared by every process that opens the same file.

    ``drain`` runs in a ``BEGIN IMMEDIATE`` transaction, so each recollection is delivered to
    exactly one drainer across threads, connections and processes.

    Payloads are :func:`~madeleine.store.codec.queued_to_dict` JSON: the memory id, the score
    explanation and the recollection fields, never the memory's content, embeddings,
    observations or raw bytes (R20). ``peek``/``drain`` rehydrate the memory from the
    ``memories`` table of the same file (without raw bytes); a recollection whose memory no
    longer exists is dropped, and ``put`` ignores one whose memory is not in the file. So
    the queue must share its file with the store (``open_sqlite``, ``create_engine``).
    """

    def __init__(
        self,
        target: str | os.PathLike[str] | SQLiteDatabase,
        *,
        use_sqlite_vec: bool | None = None,
    ) -> None:
        self._db = _db_for(target, use_sqlite_vec)
        self._listeners: list[RecollectionListener] = []
        self._listeners_lock = threading.Lock()
        self._closed = False

    @property
    def database(self) -> SQLiteDatabase:
        return self._db

    def put(self, recollection: Recollection) -> None:
        """Append unless the memory is not (any more) in the database (then nothing is
        queued and listeners are not called: a racing ``forget`` won)."""
        payload = _queued_payload(QueuedRecollection.from_recollection(recollection))
        memory_id = recollection.memory.id
        with self._db.read() as conn:
            cur = conn.execute(
                "INSERT INTO queue(rec_id, memory_id, payload_json, enqueued_at) "
                "SELECT ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM memories WHERE id = ?)",
                (recollection.id, memory_id, payload, datetime.now(UTC).isoformat(), memory_id),
            )
        if cur.rowcount != 1:
            _log.debug("not queueing %s: memory %s does not exist", recollection.id, memory_id)
            return
        with self._listeners_lock:
            listeners = list(self._listeners)
        _notify(listeners, recollection)

    def peek(self, limit: int | None = None) -> list[Recollection]:
        n = _check_limit(limit)
        with self._db.read() as conn:
            rows = conn.execute(
                "SELECT seq, payload_json FROM queue ORDER BY seq LIMIT ?",
                (-1 if n is None else n,),
            ).fetchall()
            return _decode(conn, rows)

    def drain(self, limit: int | None = None) -> list[Recollection]:
        n = _check_limit(limit)
        if n == 0:
            return []
        with self._db.write() as conn:
            rows = conn.execute(
                "SELECT seq, payload_json FROM queue ORDER BY seq LIMIT ?",
                (-1 if n is None else n,),
            ).fetchall()
            if rows:
                conn.execute("DELETE FROM queue WHERE seq <= ?", (rows[-1][0],))
            return _decode(conn, rows)

    def pending_count(self) -> int:
        with self._db.read() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM queue").fetchone()[0])

    def discard_memory(self, memory_id: str) -> int:
        with self._db.write() as conn:
            return int(conn.execute("DELETE FROM queue WHERE memory_id = ?", (memory_id,)).rowcount)

    def subscribe(self, listener: RecollectionListener) -> Unsubscribe:
        with self._listeners_lock:
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            with self._listeners_lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return _unsubscribe

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._db.release()


def _queued_payload(queued: QueuedRecollection) -> str:
    """Re-encode a decoded entry in the current (slim) codec."""
    return json.dumps(queued.to_dict(), sort_keys=True)


def _decode(conn: sqlite3.Connection, rows: Sequence[Sequence[Any]]) -> list[Recollection]:
    """Decode queue rows (codec v1 or v2) and rehydrate their memories from the store."""
    queued: list[tuple[int, QueuedRecollection]] = []
    for seq, payload in rows:
        try:
            data: Mapping[str, Any] = json.loads(payload)
            queued.append((int(seq), queued_from_dict(data)))
        except (ValueError, KeyError, TypeError):
            _log.exception("dropping undecodable queue entry seq=%s", seq)
    memories = _memories_by_id(conn, [q.memory_id for _, q in queued], with_raw=False)
    out: list[Recollection] = []
    for seq, q in queued:
        memory = memories.get(q.memory_id)
        if memory is None:
            _log.debug("dropping queue entry seq=%s: memory %s no longer exists", seq, q.memory_id)
            continue
        out.append(q.with_memory(memory))
    return out
