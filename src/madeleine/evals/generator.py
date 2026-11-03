"""Deterministic synthetic "life" for benchmarking cue-triggered recall.

Everything here is driven by a single seeded :class:`random.Random` instance created per
call to :func:`generate` (no global random state), so the same arguments always produce
byte-identical datasets.

The synthetic life:

* **Places** - a few cities (fixed UTC offsets, no DST), each with five place kinds (home,
  office, café, park, waterfront) a few kilometres apart. Memories at a place are jittered
  within ~60 m of its centre.
* **Routines** - each place kind has typical local hours and weekdays (office: 9-17 on
  weekdays, home: mornings and evenings...).
* **Topics** - ten topics with distinctive vocabularies (cooking, football, sea...) with a
  place affinity, plus three *negative* topics (astronomy, chess, plumbing) that never
  appear in memories and are used for negative cues.
* **Memories** - a content sentence (person + topic words + a concrete detail), a place, a
  local-offset timestamp spread over ~3 years before ``now``, tags, salience and optional
  tiny synthetic PNG / WAV bytes.

Cues live in the "present" (``now`` .. ``now + 28 days``) and carry ground truth: the set of
memory indices that are relevant and whether an involuntary trigger should happen. See
``docs/benchmarks.md`` for what each category tests.
"""

from __future__ import annotations

import io
import math
import random
import struct
import wave
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Final

from madeleine.core import geohash as gh
from madeleine.core.types import AUDIO, IMAGE, TEXT, RawSignal, SensoryContext

__all__ = [
    "ALL_CATEGORIES",
    "DEFAULT_CATEGORY_MIX",
    "DEFAULT_NOW",
    "NEGATIVE_CATEGORIES",
    "POSITIVE_CATEGORIES",
    "Place",
    "SyntheticCue",
    "SyntheticDataset",
    "SyntheticMemory",
    "circular_hour_diff",
    "generate",
    "local_hour",
]

DEFAULT_NOW: Final = datetime(2026, 6, 1, tzinfo=UTC)
"""Fixed "present" anchor: memories precede it, cues follow it. Never the wall clock."""

GEOHASH_PRECISION: Final = 9
HISTORY_DAYS: Final = 3 * 365
PRESENT_DAYS: Final = 28
OLD_MEMORY_MIN_AGE: Final = timedelta(days=2 * 365)
CONTEXT_HOUR_WINDOW: Final = 1.0
"""A memory is relevant to a ``context`` cue iff same place and |Δ hour-of-day| <= this."""

# -- categories --------------------------------------------------------------------------

CONTEXT: Final = "context"
TEXT_PARAPHRASE: Final = "text_paraphrase"
MULTI: Final = "multi"
EXACT_MEDIA: Final = "exact_media"
PERTURBED_MEDIA: Final = "perturbed_media"
OLD_MEMORY: Final = "old_memory"
DISTRACTOR: Final = "distractor"
TIME_ONLY: Final = "time_only"
HARD_NEGATIVE: Final = "hard_negative"

POSITIVE_CATEGORIES: Final[tuple[str, ...]] = (
    CONTEXT,
    TEXT_PARAPHRASE,
    MULTI,
    EXACT_MEDIA,
    PERTURBED_MEDIA,
    OLD_MEMORY,
)
NEGATIVE_CATEGORIES: Final[tuple[str, ...]] = (DISTRACTOR, TIME_ONLY, HARD_NEGATIVE)
ALL_CATEGORIES: Final[tuple[str, ...]] = POSITIVE_CATEGORIES + NEGATIVE_CATEGORIES

DEFAULT_CATEGORY_MIX: Final[Mapping[str, float]] = {
    CONTEXT: 0.14,
    TEXT_PARAPHRASE: 0.14,
    MULTI: 0.14,
    EXACT_MEDIA: 0.08,
    PERTURBED_MEDIA: 0.08,
    OLD_MEMORY: 0.10,
    DISTRACTOR: 0.12,
    TIME_ONLY: 0.08,
    HARD_NEGATIVE: 0.12,
}

# -- world description -------------------------------------------------------------------

# (slug, display name, lat, lon, utc offset hours, weight of time spent there)
_CITIES: Final = (
    ("lisbon", "Lisbon", 38.7223, -9.1393, 0, 0.55),
    ("santiago", "Santiago", -33.4489, -70.6693, -3, 0.30),
    ("tokyo", "Tokyo", 35.6762, 139.6503, 9, 0.15),
)

# Far-away points used only by distractor cues (no memory is within thousands of km).
_FAR_POINTS: Final = (
    (64.1466, -21.9426),  # Reykjavik
    (-1.2921, 36.8219),  # Nairobi
    (-33.8688, 151.2093),  # Sydney
    (61.2181, -149.9003),  # Anchorage
    (-54.8019, -68.3030),  # Ushuaia
)

_WEEKDAYS: Final = frozenset(range(5))
_WEEKEND: Final = frozenset({5, 6})
_ALL_DAYS: Final = frozenset(range(7))


@dataclass(frozen=True, slots=True)
class _Routine:
    label: str
    offset_km: tuple[float, float]  # (east, north) from the city centre
    hours: tuple[tuple[int, float], ...]  # (local hour, weight)
    days: frozenset[int]
    topics: tuple[str, ...]


_ROUTINES: Final = (
    _Routine(
        "home",
        (-2.0, 1.5),
        ((7, 2.0), (8, 1.0), (19, 1.0), (20, 2.0), (21, 2.0), (22, 1.0)),
        _ALL_DAYS,
        ("cooking", "family", "books", "music"),
    ),
    _Routine(
        "office",
        (1.5, 0.5),
        ((9, 2.0), (10, 2.0), (11, 1.0), (14, 1.0), (15, 2.0), (16, 1.0)),
        _WEEKDAYS,
        ("project", "books", "health"),
    ),
    _Routine(
        "café",
        (0.0, -1.0),
        ((8, 2.0), (9, 1.0), (16, 2.0), (17, 1.0)),
        _ALL_DAYS,
        ("coffee", "books", "family", "music"),
    ),
    _Routine(
        "park",
        (-1.5, -2.5),
        ((11, 1.0), (12, 2.0), (13, 1.0), (18, 1.0)),
        _ALL_DAYS,
        ("football", "gardening", "health", "family"),
    ),
    _Routine(
        "waterfront",
        (3.0, -2.0),
        ((10, 1.0), (13, 1.0), (14, 2.0), (15, 2.0), (17, 1.0)),
        _WEEKEND,
        ("sea", "health", "music"),
    ),
)


def _words(text: str) -> tuple[str, ...]:
    return tuple(text.split())


_TOPICS: Final[Mapping[str, tuple[str, ...]]] = {
    "cooking": _words(
        "saffron risotto oven basil garlic simmer dough recipe cinnamon skillet butter "
        "oregano broth knead apron ladle"
    ),
    "family": _words(
        "grandmother cousin birthday nephew wedding uncle aunt toddler lullaby christening "
        "siblings heirloom anniversary reunion cradle godfather"
    ),
    "project": _words(
        "deadline roadmap sprint invoice prototype budget spreadsheet quarterly client "
        "milestone deployment contract forecast audit backlog slides"
    ),
    "music": _words(
        "violin chorus melody vinyl rehearsal saxophone concert ukulele harmony drummer "
        "playlist tempo ballad orchestra metronome encore"
    ),
    "football": _words(
        "goalkeeper penalty stadium referee striker jersey halftime derby offside corner "
        "trophy midfielder whistle scarf kickoff league"
    ),
    "gardening": _words(
        "tulips compost seedlings hedge trowel lavender pruning greenhouse ferns mulch "
        "orchid sunflowers watering bulbs ivy hydrangea"
    ),
    "coffee": _words(
        "espresso cortado barista croissant foam roast cappuccino macchiato saucer grinder "
        "pastry latte beans mug terrace sugar"
    ),
    "sea": _words(
        "waves seagulls tide surfboard seashells lighthouse sand sunscreen pier sailboat "
        "driftwood dunes kelp snorkel harbour salt"
    ),
    "health": _words(
        "physio stretching marathon dentist vitamins yoga sprained ankle therapist pulse "
        "treadmill bandage clinic posture insomnia inhaler"
    ),
    "books": _words(
        "novel poetry library chapter manuscript bookmark paperback librarian sonnet "
        "footnote anthology author bookshop epilogue hardcover essay"
    ),
}

# Topics that never occur in memories: used for distractor / hard-negative cue text.
_NEGATIVE_TOPICS: Final[Mapping[str, tuple[str, ...]]] = {
    "astronomy": _words(
        "telescope nebula comet galaxy orbit asteroid eclipse constellation meteor "
        "planetarium quasar supernova"
    ),
    "chess": _words(
        "bishop checkmate gambit rook pawn stalemate castling endgame knight grandmaster "
        "zugzwang blitz"
    ),
    "plumbing": _words(
        "faucet pipe wrench leak boiler valve drain gasket sewer plunger radiator cistern"
    ),
}

_NAMES: Final = _words(
    "Ana Bruno Camila Diego Elena Felipe Greta Hugo Ines Joaquin Kenji Lucia Mateo Nadia "
    "Oscar Paula Quentin Rosa Sofia Tomas Ursula Victor Wendy Ximena Yuki Zoe Amaru Beatriz "
    "Cyril Dafne Emil Fatima Gael Hana Ivo Jana"
)
_ADJECTIVES: Final = _words(
    "red crooked rusty velvet chipped striped faded golden muddy tiny enormous squeaky "
    "crumpled polished frozen lopsided borrowed dusty turquoise wobbly"
)
_OBJECTS: Final = _words(
    "umbrella bicycle lantern suitcase kite teapot scooter postcard harmonica backpack "
    "notebook fountain bench tram parrot accordion hammock compass typewriter sombrero "
    "telegram carousel ferry pinecone thermos kaleidoscope windmill jukebox canoe gramophone"
)
_VERBS: Final = (
    "talked about",
    "laughed about",
    "argued about",
    "planned",
    "daydreamed about",
    "worried about",
    "celebrated",
)  # fmt: skip

# Vocabulary deliberately disjoint from memory templates for negative cue text.
_NEGATIVE_TEMPLATES: Final = (
    "someone online keeps posting about {w1}, {w2} and {w3}",
    "a podcast episode covering {w1} plus {w2}, then {w3}",
    "an advert for {w1} followed by {w2} with {w3}",
)


# -- public dataclasses -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Place:
    """One place of the synthetic life."""

    id: int
    city: str
    kind: str
    lat: float
    lon: float
    utc_offset_hours: int

    @property
    def label(self) -> str:
        return f"{self.kind}@{self.city}"


@dataclass(frozen=True, slots=True)
class SyntheticMemory:
    """A memory to insert with ``engine.remember(content, signals, context=context)``."""

    index: int
    content: str
    place: int
    """Index into :attr:`SyntheticDataset.places`."""
    lat: float
    lon: float
    timestamp: datetime
    """Local-offset (tz-aware) timestamp."""
    topic: str
    tags: frozenset[str]
    salience: float
    image: bytes | None = None
    audio: bytes | None = None

    @property
    def geohash(self) -> str:
        return gh.encode(self.lat, self.lon, GEOHASH_PRECISION)

    @property
    def context(self) -> SensoryContext:
        return SensoryContext(
            timestamp=self.timestamp,
            geohash=self.geohash,
            tags=self.tags,
            salience=self.salience,
        )

    @property
    def signals(self) -> tuple[RawSignal, ...]:
        """Media signals (text is derived from ``content`` by the engine)."""
        out: list[RawSignal] = []
        if self.image is not None:
            out.append(RawSignal(IMAGE, self.image, mime_type="image/png"))
        if self.audio is not None:
            out.append(RawSignal(AUDIO, self.audio, mime_type="audio/wav"))
        return tuple(out)


@dataclass(frozen=True, slots=True)
class SyntheticCue:
    """A labelled cue with known ground truth.

    Build it with ``engine.build_cue(cue.signals, context=cue.context,
    include_time=cue.include_time, include_geo=cue.include_geo, source="bench")``.
    """

    index: int
    category: str
    signals: tuple[RawSignal, ...]
    context: SensoryContext
    relevant: frozenset[int]
    """Indices into :attr:`SyntheticDataset.memories` that a good recall should return."""
    should_fire: bool
    """Whether an involuntary (push) trigger is desired for this cue."""
    include_time: bool = True
    include_geo: bool = True
    target: int | None = None
    """The memory the cue was derived from (positives only)."""
    description: str = ""


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    seed: int
    now: datetime
    places: tuple[Place, ...]
    memories: tuple[SyntheticMemory, ...]
    cues: tuple[SyntheticCue, ...]

    def category_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(ALL_CATEGORIES, 0)
        for cue in self.cues:
            counts[cue.category] = counts.get(cue.category, 0) + 1
        return counts


# -- helpers ------------------------------------------------------------------------------


def local_hour(ts: datetime) -> float:
    """Hour of day (fractional) in the timestamp's own UTC offset."""
    return ts.hour + ts.minute / 60 + ts.second / 3600


def circular_hour_diff(a: float, b: float) -> float:
    """Distance between two hours of day on the 24 h circle, in ``[0, 12]``."""
    d = abs(a - b) % 24
    return min(d, 24 - d)


def _offset(lat: float, lon: float, east_km: float, north_km: float) -> tuple[float, float]:
    dlat = north_km / 111.32
    dlon = east_km / (111.32 * max(0.01, math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def _jitter(
    rng: random.Random, lat: float, lon: float, lo_km: float, hi_km: float
) -> tuple[float, float]:
    r = rng.uniform(lo_km, hi_km)
    theta = rng.uniform(0, 2 * math.pi)
    return _offset(lat, lon, r * math.cos(theta), r * math.sin(theta))


def _weighted(rng: random.Random, items: Sequence[tuple[int, float]]) -> int:
    return rng.choices([i for i, _ in items], weights=[w for _, w in items], k=1)[0]


def _png(pixels: Sequence[int], size: int) -> bytes:
    """Encode an RGB ``size`` x ``size`` image as a minimal valid PNG (stdlib only)."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = bytearray()
    for y in range(size):
        rows.append(0)  # filter: none
        rows.extend(pixels[y * size * 3 : (y + 1) * size * 3])
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


_IMG_SIZE: Final = 16


def _image_pixels(rng: random.Random) -> list[int]:
    base = [rng.randrange(256) for _ in range(3)]
    fx, fy = rng.uniform(0.2, 1.2), rng.uniform(0.2, 1.2)
    px: list[int] = []
    for y in range(_IMG_SIZE):
        for x in range(_IMG_SIZE):
            for c in range(3):
                v = base[c] + 90 * math.sin(fx * x + c) * math.cos(fy * y - c)
                px.append(max(0, min(255, int(v))))
    return px


def _perturb_pixels(rng: random.Random, px: Sequence[int]) -> list[int]:
    """Small, visually negligible change: +-1..3 on ~10% of channel values."""
    out = list(px)
    for i in range(len(out)):
        if rng.random() < 0.1:
            out[i] = max(0, min(255, out[i] + rng.choice((-3, -2, -1, 1, 2, 3))))
    return out


_SAMPLE_RATE: Final = 8000
_AUDIO_SAMPLES: Final = 2000  # 0.25 s


def _audio_samples(rng: random.Random) -> list[int]:
    freqs = [rng.uniform(180, 1200) for _ in range(rng.randint(1, 3))]
    out: list[int] = []
    for n in range(_AUDIO_SAMPLES):
        t = n / _SAMPLE_RATE
        v = sum(math.sin(2 * math.pi * f * t) for f in freqs) / len(freqs)
        out.append(int(12000 * v))
    return out


def _perturb_samples(rng: random.Random, samples: Sequence[int]) -> list[int]:
    """Low-level noise (~-50 dB) on every sample."""
    return [max(-32768, min(32767, s + rng.randint(-40, 40))) for s in samples]


def _wav(samples: Sequence[int]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


# -- generation ---------------------------------------------------------------------------


@dataclass(slots=True)
class _MemoryDraft:
    """Mutable working copy (keeps the parts needed to build cues)."""

    memory: SyntheticMemory
    name: str
    words: tuple[str, str, str]
    adjective: str
    obj: str
    image_pixels: list[int] | None
    audio_samples: list[int] | None


def _build_places(rng: random.Random) -> tuple[Place, ...]:
    places: list[Place] = []
    for slug, _name, lat, lon, tz, _w in _CITIES:
        for r in _ROUTINES:
            east = r.offset_km[0] + rng.uniform(-0.2, 0.2)
            north = r.offset_km[1] + rng.uniform(-0.2, 0.2)
            plat, plon = _offset(lat, lon, east, north)
            places.append(Place(len(places), slug, r.label, plat, plon, tz))
    return tuple(places)


def _routine(place: Place) -> _Routine:
    for r in _ROUTINES:
        if r.label == place.kind:
            return r
    raise KeyError(place.kind)  # pragma: no cover - places are built from _ROUTINES


def _city_name(slug: str) -> str:
    for s, name, *_ in _CITIES:
        if s == slug:
            return name
    raise KeyError(slug)  # pragma: no cover


def _memory_time(rng: random.Random, place: Place, now: datetime) -> datetime:
    r = _routine(place)
    tz = timezone(timedelta(hours=place.utc_offset_hours))
    start = (now - timedelta(days=HISTORY_DAYS)).astimezone(tz).date()
    for _ in range(50):
        day = start + timedelta(days=rng.randrange(HISTORY_DAYS - 2))
        if day.weekday() in r.days:
            break
    hour = _weighted(rng, r.hours)
    return datetime(day.year, day.month, day.day, hour, rng.randrange(60), tzinfo=tz)


def _make_memory(
    rng: random.Random, index: int, places: Sequence[Place], now: datetime
) -> _MemoryDraft:
    city = rng.choices([c[0] for c in _CITIES], weights=[c[5] for c in _CITIES], k=1)[0]
    place = rng.choice([p for p in places if p.city == city])
    topic = rng.choice(_routine(place).topics)
    w1, w2, w3 = rng.sample(_TOPICS[topic], 3)
    name, adj, obj = rng.choice(_NAMES), rng.choice(_ADJECTIVES), rng.choice(_OBJECTS)
    verb = rng.choice(_VERBS)
    content = (
        f"{name} and I {verb} {w1} and {w2} at the {place.kind} in {_city_name(city)}; "
        f"I still remember the {w3} and the {adj} {obj}."
    )
    lat, lon = _jitter(rng, place.lat, place.lon, 0.0, 0.06)
    ts = _memory_time(rng, place, now)
    salience = round(min(0.98, max(0.02, rng.betavariate(2.0, 2.0))), 2)
    roll = rng.random()
    img_px = _image_pixels(rng) if roll < 0.18 else None
    aud = _audio_samples(rng) if 0.18 <= roll < 0.30 else None
    mem = SyntheticMemory(
        index=index,
        content=content,
        place=place.id,
        lat=lat,
        lon=lon,
        timestamp=ts,
        topic=topic,
        tags=frozenset({topic, place.kind, city}),
        salience=salience,
        image=_png(img_px, _IMG_SIZE) if img_px is not None else None,
        audio=_wav(aud) if aud is not None else None,
    )
    return _MemoryDraft(mem, name, (w1, w2, w3), adj, obj, img_px, aud)


def _ensure_media(rng: random.Random, drafts: list[_MemoryDraft]) -> None:
    """Guarantee at least one image and one audio memory (for media cue categories)."""
    from dataclasses import replace

    if len(drafts) >= 2 and not any(d.image_pixels for d in drafts):
        d = drafts[0]
        d.image_pixels = _image_pixels(rng)
        d.memory = replace(d.memory, image=_png(d.image_pixels, _IMG_SIZE))
    if len(drafts) >= 2 and not any(d.audio_samples for d in drafts):
        d = next(x for x in reversed(drafts) if x.image_pixels is None)
        d.audio_samples = _audio_samples(rng)
        d.memory = replace(d.memory, audio=_wav(d.audio_samples))


def _allocate(n: int, mix: Mapping[str, float]) -> list[str]:
    """Largest-remainder allocation of ``n`` cues over categories (fixed order)."""
    cats = [c for c in ALL_CATEGORIES if mix.get(c, 0.0) > 0]
    cats += sorted(c for c in mix if c not in ALL_CATEGORIES and mix[c] > 0)
    unknown = [c for c in cats if c not in ALL_CATEGORIES]
    if unknown:
        raise ValueError(f"unknown cue categories: {unknown}")
    if not cats or n <= 0:
        return []
    total = sum(mix[c] for c in cats)
    exact = [n * mix[c] / total for c in cats]
    counts = [int(x) for x in exact]
    order = sorted(range(len(cats)), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    # Every category gets at least one cue when there are enough cues to go round.
    if n >= len(cats):
        for i, cnt in enumerate(counts):
            if cnt == 0:
                j = max(range(len(cats)), key=lambda k: counts[k])
                counts[j] -= 1
                counts[i] += 1
    out: list[str] = []
    for c, k in zip(cats, counts, strict=True):
        out.extend([c] * k)
    return out


def _present_time(
    rng: random.Random, target: datetime, now: datetime, hour_jitter_min: int
) -> datetime:
    """A present-day local time on the target's weekday near the target's hour."""
    tz = target.tzinfo
    today = now.astimezone(tz).date()
    days = [
        today + timedelta(days=d)
        for d in range(1, PRESENT_DAYS + 1)
        if (today + timedelta(days=d)).weekday() == target.weekday()
    ]
    day = rng.choice(days)
    base = datetime(day.year, day.month, day.day, target.hour, target.minute, tzinfo=tz)
    return base + timedelta(minutes=rng.randint(-hour_jitter_min, hour_jitter_min))


def _random_present_time(rng: random.Random, now: datetime, utc_offset: int) -> datetime:
    tz = timezone(timedelta(hours=utc_offset))
    day = now.astimezone(tz).date() + timedelta(days=rng.randint(1, PRESENT_DAYS))
    return datetime(day.year, day.month, day.day, rng.randrange(24), rng.randrange(60), tzinfo=tz)


def _paraphrase(rng: random.Random, d: _MemoryDraft) -> str:
    """Share part of the target's vocabulary: name, two of three topic words, the object."""
    w = list(d.words)
    rng.shuffle(w)
    templates = (
        "something about {a} and {b} with {name} comes back, that {obj} too",
        "{name} mentioned {a}, then {b}; I keep picturing a {obj}",
        "thinking of {name}: {a}, {b} and some {obj}",
    )
    return rng.choice(templates).format(a=w[0], b=w[1], name=d.name, obj=d.obj)


def _negative_text(rng: random.Random) -> str:
    topic = rng.choice(sorted(_NEGATIVE_TOPICS))
    w1, w2, w3 = rng.sample(_NEGATIVE_TOPICS[topic], 3)
    return rng.choice(_NEGATIVE_TEMPLATES).format(w1=w1, w2=w2, w3=w3)


def _context_relevant(
    memories: Sequence[SyntheticMemory], place: int, cue_hour: float
) -> frozenset[int]:
    return frozenset(
        m.index
        for m in memories
        if m.place == place
        and circular_hour_diff(local_hour(m.timestamp), cue_hour) <= CONTEXT_HOUR_WINDOW
    )


def _make_cue(
    rng: random.Random,
    index: int,
    category: str,
    drafts: Sequence[_MemoryDraft],
    places: Sequence[Place],
    now: datetime,
) -> SyntheticCue:
    memories = [d.memory for d in drafts]

    if category == CONTEXT:
        d = rng.choice(drafts)
        m = d.memory
        place = places[m.place]
        ts = _present_time(rng, m.timestamp, now, 60)
        lat, lon = _jitter(rng, place.lat, place.lon, 0.05, 0.3)
        ctx = SensoryContext(timestamp=ts, geohash=gh.encode(lat, lon, GEOHASH_PRECISION))
        relevant = _context_relevant(memories, m.place, local_hour(ts))
        return SyntheticCue(
            index, category, (), ctx, relevant | {m.index}, True, True, True, m.index,
            f"back at {place.label} around {ts:%H:%M}, no text",
        )  # fmt: skip

    if category == TEXT_PARAPHRASE:
        d = rng.choice(drafts)
        ts = _random_present_time(rng, now, places[d.memory.place].utc_offset_hours)
        text = _paraphrase(rng, d)
        return SyntheticCue(
            index, category, (RawSignal(TEXT, text),), SensoryContext(timestamp=ts),
            frozenset({d.memory.index}), True, False, False, d.memory.index, text,
        )  # fmt: skip

    if category in (MULTI, OLD_MEMORY):
        pool = list(drafts)
        if category == OLD_MEMORY:
            old = [x for x in drafts if now - x.memory.timestamp >= OLD_MEMORY_MIN_AGE]
            pool = (
                old or sorted(drafts, key=lambda x: x.memory.timestamp)[: max(1, len(drafts) // 5)]
            )
        d = rng.choice(pool)
        m = d.memory
        place = places[m.place]
        ts = _present_time(rng, m.timestamp, now, 30)
        lat, lon = _jitter(rng, m.lat, m.lon, 0.02, 0.15)
        ctx = SensoryContext(timestamp=ts, geohash=gh.encode(lat, lon, GEOHASH_PRECISION))
        text = _paraphrase(rng, d)
        age_days = (now - m.timestamp).days
        return SyntheticCue(
            index, category, (RawSignal(TEXT, text),), ctx, frozenset({m.index}), True,
            True, True, m.index, f"{text} (at {place.label}, target {age_days} d old)",
        )  # fmt: skip

    if category in (EXACT_MEDIA, PERTURBED_MEDIA):
        with_media = [x for x in drafts if x.image_pixels is not None or x.audio_samples]
        d = rng.choice(with_media or list(drafts))
        m = d.memory
        ts = _random_present_time(rng, now, places[m.place].utc_offset_hours)
        perturb = category == PERTURBED_MEDIA
        signal: RawSignal
        if d.image_pixels is not None:
            px = _perturb_pixels(rng, d.image_pixels) if perturb else d.image_pixels
            signal = RawSignal(IMAGE, _png(px, _IMG_SIZE), mime_type="image/png")
        elif d.audio_samples is not None:
            s = _perturb_samples(rng, d.audio_samples) if perturb else d.audio_samples
            signal = RawSignal(AUDIO, _wav(s), mime_type="audio/wav")
        else:  # pragma: no cover - only when the dataset has < 2 memories
            signal = RawSignal(TEXT, m.content)
        return SyntheticCue(
            index, category, (signal,), SensoryContext(timestamp=ts),
            frozenset({m.index}), True, False, False, m.index,
            f"{'perturbed' if perturb else 'identical'} {signal.channel} of memory {m.index}",
        )  # fmt: skip

    if category == DISTRACTOR:
        lat, lon = _jitter(rng, *rng.choice(_FAR_POINTS), 0.0, 5.0)
        ts = _random_present_time(rng, now, rng.randint(-10, 12))
        ctx = SensoryContext(timestamp=ts, geohash=gh.encode(lat, lon, GEOHASH_PRECISION))
        text = _negative_text(rng)
        return SyntheticCue(
            index, category, (RawSignal(TEXT, text),), ctx, frozenset(), False,
            description=f"far away: {text}",
        )  # fmt: skip

    if category == TIME_ONLY:
        d = rng.choice(drafts)
        ts = _present_time(rng, d.memory.timestamp, now, 10)
        return SyntheticCue(
            index, category, (), SensoryContext(timestamp=ts), frozenset(), False,
            True, False, None, f"only the hour {ts:%H:%M} (matches memory {d.memory.index})",
        )  # fmt: skip

    if category == HARD_NEGATIVE:
        d = rng.choice(drafts)
        place = places[d.memory.place]
        here = [m for m in memories if m.place == place.id]
        # The hour farthest from every memory at this place (ties: earliest hour).
        hour = max(
            range(24),
            key=lambda h: (
                min(circular_hour_diff(h + 0.5, local_hour(m.timestamp)) for m in here),
                -h,
            ),
        )
        tz = timezone(timedelta(hours=place.utc_offset_hours))
        day = now.astimezone(tz).date() + timedelta(days=rng.randint(1, PRESENT_DAYS))
        ts = datetime(day.year, day.month, day.day, hour, rng.randrange(60), tzinfo=tz)
        lat, lon = _jitter(rng, place.lat, place.lon, 0.05, 0.2)
        ctx = SensoryContext(timestamp=ts, geohash=gh.encode(lat, lon, GEOHASH_PRECISION))
        text = _negative_text(rng)
        return SyntheticCue(
            index, category, (RawSignal(TEXT, text),), ctx, frozenset(), False,
            description=f"at {place.label} at {ts:%H:%M}: {text}",
        )  # fmt: skip

    raise ValueError(f"unknown cue category {category!r}")


def generate(
    *,
    memories: int = 500,
    cues: int = 300,
    seed: int = 0,
    now: datetime = DEFAULT_NOW,
    category_mix: Mapping[str, float] | None = None,
) -> SyntheticDataset:
    """Generate a deterministic synthetic dataset.

    ``category_mix`` maps cue categories to relative weights (default
    :data:`DEFAULT_CATEGORY_MIX`); categories with weight 0 are omitted. Same arguments ->
    identical dataset.
    """
    if memories < 1:
        raise ValueError("memories must be >= 1")
    if cues < 0:
        raise ValueError("cues must be >= 0")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    rng = random.Random(seed)  # noqa: S311 - reproducible benchmark data, not crypto
    places = _build_places(rng)
    drafts = [_make_memory(rng, i, places, now) for i in range(memories)]
    _ensure_media(rng, drafts)
    categories = _allocate(cues, DEFAULT_CATEGORY_MIX if category_mix is None else category_mix)
    rng.shuffle(categories)
    cue_list = [_make_cue(rng, i, c, drafts, places, now) for i, c in enumerate(categories)]
    return SyntheticDataset(
        seed=seed,
        now=now,
        places=places,
        memories=tuple(d.memory for d in drafts),
        cues=tuple(cue_list),
    )
