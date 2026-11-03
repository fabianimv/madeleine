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

The ``vectors`` table is the source of truth; ``vec0`` tables are a KNN index over it. When
the sqlite-vec extension cannot be loaded, ``nearest`` falls back to numpy brute force over
``vectors`` (a warning is logged once) and writes set ``meta.vec_dirty`` so that the next
connection that *can* load the extension rebuilds the index.

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
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from madeleine.core.fusion import is_refractory
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
    observation_from_dict,
    recollection_from_dict,
    recollection_to_dict,
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

SCHEMA_VERSION = 1
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
            if self.path != ":memory:" and not self.path.startswith("file::memory:"):
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
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
            if version < SCHEMA_VERSION:
                conn.execute(
                    "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, datetime.now(UTC).isoformat()),
                )
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            if self.vec_enabled:
                self._sync_vec_tables(conn)

    def _sync_vec_tables(self, conn: sqlite3.Connection) -> None:
        dirty = conn.execute("SELECT value FROM meta WHERE key = 'vec_dirty'").fetchone()
        rebuild = dirty is not None and dirty[0] == "1"
        for ch_id, name, dim in conn.execute("SELECT id, name, dim FROM channels").fetchall():
            table = _vec_table(ch_id)
            if rebuild and _table_exists(conn, table):
                conn.execute(f"DROP TABLE {table}")
            _ensure_vec_table(conn, ch_id, name, dim)
        if rebuild:
            _log.info("rebuilt sqlite-vec index after fallback-mode writes")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('vec_dirty', '0')")

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
                raise StoreError(
                    f"channel {spec.name!r} holds {held} vectors with dim={dim}, "
                    f"model_id={model_id!r}; refusing dim={spec.dim}, model_id={spec.model_id!r} "
                    "(vectors from different models are not comparable; use a new database "
                    "or re-embed)"
                )
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

    def _channels(self, conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
        rows = conn.execute("SELECT name, id, dim FROM channels").fetchall()
        return {str(n): (int(i), int(d)) for n, i, d in rows}

    # -- memories ------------------------------------------------------------------------
    def add(self, memory: Memory) -> None:
        fp = memory.fingerprint
        ctx = fp.context
        with self._db.write() as conn:
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

    def _hydrate(self, conn: sqlite3.Connection, rows: Sequence[Sequence[Any]]) -> list[Memory]:
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

    def get(self, memory_id: str) -> Memory:
        with self._db.read() as conn:
            rows = conn.execute(
                f"SELECT {_MEM_COLS} FROM memories WHERE id = ?", (memory_id,)
            ).fetchall()
            found = self._hydrate(conn, rows)
        if not found:
            raise MemoryNotFoundError(memory_id)
        return found[0]

    def get_many(self, memory_ids: Sequence[str]) -> list[Memory]:
        wanted = list(dict.fromkeys(memory_ids))
        by_id: dict[str, Memory] = {}
        with self._db.read() as conn:
            for i in range(0, len(wanted), _CHUNK):
                chunk = wanted[i : i + _CHUNK]
                marks = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT {_MEM_COLS} FROM memories WHERE id IN ({marks})", chunk
                ).fetchall()
                for m in self._hydrate(conn, rows):
                    by_id[m.id] = m
        return [by_id[i] for i in memory_ids if i in by_id]

    def delete(self, memory_id: str) -> bool:
        with self._db.write() as conn:
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
            conn.execute("DELETE FROM memories WHERE rowid = ?", (rowid,))
            self._db.mark_vec_dirty(conn)
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
                batch = self._hydrate(conn, rows)
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
            if self._db.vec_enabled and k <= VEC_MAX_K and _table_exists(conn, table):
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

    # -- triggers ------------------------------------------------------------------------
    def record_trigger(self, memory_id: str, at: datetime) -> None:
        with self._db.write() as conn:
            _insert_trigger(conn, memory_id, at)

    def last_triggered(self, memory_id: str) -> datetime | None:
        with self._db.read() as conn:
            return _last_trigger(conn, memory_id)

    def claim_trigger(self, memory_id: str, at: datetime, refractory_period: timedelta) -> bool:
        """Atomically (across processes) record a trigger unless the memory is refractory."""
        with self._db.write() as conn:
            if is_refractory(_last_trigger(conn, memory_id), at, refractory_period):
                return False
            _insert_trigger(conn, memory_id, at)
            return True

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._db.release()


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
    exactly one drainer across threads, connections and processes. Payloads are
    :func:`~madeleine.store.codec.recollection_to_dict` JSON (never raw media bytes).
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
        payload = json.dumps(recollection_to_dict(recollection), sort_keys=True)
        with self._db.read() as conn:
            conn.execute(
                "INSERT INTO queue(rec_id, memory_id, payload_json, enqueued_at) "
                "VALUES (?, ?, ?, ?)",
                (recollection.id, recollection.memory.id, payload, datetime.now(UTC).isoformat()),
            )
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
        return _decode(rows)

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
        return _decode(rows)

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


def _decode(rows: Sequence[Sequence[Any]]) -> list[Recollection]:
    out: list[Recollection] = []
    for seq, payload in rows:
        try:
            data: Mapping[str, Any] = json.loads(payload)
            out.append(recollection_from_dict(data))
        except (ValueError, KeyError, TypeError):
            _log.exception("dropping undecodable queue entry seq=%s", seq)
    return out
