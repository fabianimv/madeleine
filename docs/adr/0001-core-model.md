# ADR 0001: Core model — memories, cues, channels, fused recall, involuntary push

- Status: Accepted
- Date: 2026-11-03
- Deciders: lead

## Context

Agent memory systems today are almost all *voluntary*: the agent decides to search, writes a
query, and gets results. Human memory has a second, older pathway: *involuntary* recall,
where a sensory cue (the madeleine dipped in tea, a song, a street corner at dusk) brings a
memory back without any intention to remember. Cognitive psychology frames this as
**encoding specificity** (Tulving & Thomson, 1973): retrieval is most effective when the
cues present at retrieval overlap with the context present at encoding.

madeleine gives agents that second pathway. To do so we need a small, stable vocabulary
that every component (storage, channels, MCP server, ingestion, evals) builds against.

## Decision

### Entities

| Entity | Meaning |
|---|---|
| **RawSignal** | Un-normalised reading on one channel, as it arrives (`(lat, lon)`, PNG bytes, a string). |
| **Observation** | Normalised reading on one channel. One generic shape (`text`, `data`, `mime_type`, `uri`, `numbers`, `timestamp`); each channel documents the fields it reads. |
| **Channel** | A pluggable sense = `ChannelSpec` + `Adapter` (RawSignal → Observation) + `Embedder` (Observation → Vector) + `SimilarityFn` (Vector × Vector → [0, 1]). |
| **SensoryContext** | Structured context: tz-aware `timestamp`, `geohash`, `tags`, `salience ∈ [0,1]`. |
| **SensoryFingerprint** | Per-channel embeddings + context (+ normalised observations, raw bytes stripped). |
| **Memory** | `content` (+ optional `media_ref`, `metadata`) + fingerprint captured at encoding time. |
| **Cue** | A bundle of observations (≤ 1 per channel) + context, from some `source`. |
| **ScoredMemory** | Memory + fused score + per-channel similarity explanation. |
| **Recollection** | A ScoredMemory delivered because of a cue, with `mode ∈ {pull, push}`. |

All are frozen dataclasses in `src/madeleine/core/types.py`. Vectors are `tuple[float, ...]`
(immutable, JSON-friendly, no numpy in the contract).

### Time and place are ordinary channels derived from context

`time` and `geo` are channels like any other, but their observations are usually *derived*
from `SensoryContext.timestamp` / `.geohash` by the engine (`build_cue`, `remember`), so
callers just say "now, here". Explicit `time`/`geo` signals override derived ones.
Hour-of-day/weekday are computed in the timestamp's **own** UTC offset: callers should send
local-offset timestamps (`2026-06-01T21:30:00+02:00`), otherwise "evening" means UTC evening.

### Recall fusion (normative)

For a cue *q* and memory *m*: `S` = channels present in both with weight `w_c > 0`,
`C` = channels in the cue with `w_c > 0`, `s_c ∈ [0, 1]` = channel similarity.

```
base            = Σ_{c∈S} w_c·s_c / Σ_{c∈S} w_c                 (0 if S = ∅)
coverage        = (Σ_{c∈S} w_c / Σ_{c∈C} w_c) ^ coverage_exponent
salience_factor = 1 + salience_weight · (m.salience − 0.5)
decay_factor    = decay_floor + (1 − decay_floor) · 0.5 ^ (age / half_life)
score           = clamp(base · coverage · salience_factor · decay_factor, 0, 1)
```

`age = max(0, q.timestamp − m.timestamp)`. Defaults live in `RecallConfig`.

* Only channels present in **both** cue and memory contribute to `base` (a memory without
  audio is not penalised for not sounding like the cue), but `coverage` penalises a memory
  that matches only a small part of a rich cue.
* Decay has a **floor** (default 0.7): old memories fade but never vanish — the Proust
  effect is about memories from decades ago.
* Salience is a bounded multiplicative boost (default ×0.8 … ×1.2).

### Two modes

* **PULL** (`recall`): rank, threshold (`pull_threshold`), top-k. Pure, no side effects.
* **PUSH** (`sense`): a memory fires iff `score ≥ trigger_threshold`, `|S| ≥ min_shared_channels`,
  at least one channel in `S` has `solo_trigger=True`, and it is not **refractory**
  (fired less than `refractory_period` ago, measured on the cue's timestamp). At most
  `max_push_per_cue` fire per cue. Firing records a trigger time and enqueues a
  `Recollection` in a persistent `RecollectionQueue`.

`solo_trigger=False` exists because some channels are only *context*: every memory encoded at
9 am matches every 9 am. Time (and, by default, biometrics) can strengthen a match but never
cause one alone.

Refractory is symmetric in time: a memory is refractory iff
`|cue.timestamp − last_triggered| < refractory_period`, so a late-delivered (out-of-order) cue
just before the last fire is suppressed too; `refractory_period = 0` disables it.

The **refractory period** is the habituation mechanism: walking past the same café all
afternoon reminds you of it once, not on every GPS ping.

### Delivery of push recollections

The queue is persisted in the same SQLite file as the memories so that separate processes
share it (an HTTP ingestion server receiving webhooks; a stdio MCP server spawned by the
client; a Claude Code hook). Delivery paths, from most to least universal:

1. `UserPromptSubmit` hook (Claude Code) drains the queue and injects recollections as
   additional context — works with any client behaviour.
2. MCP tool `pending_recollections` — the agent can poll.
3. MCP resource `recollections://pending` + `notifications/resources/updated` — for clients
   that honour subscriptions.

`drain` is atomic: each recollection is delivered once.

### Storage

`MemoryStore` protocol; reference implementation SQLite + `sqlite-vec` (one `vec0` table
per channel, L2 KNN as candidate generator, exact re-scoring with the channel's similarity),
with a numpy brute-force fallback when the extension cannot be loaded. Stores record each
channel's `dim` and `model_id` and refuse to mix vectors from different models.

### Privacy

Raw media bytes (`Observation.data`) are used for embedding and dropped. They are persisted
only with `store_raw=True` (per call, or `MadeleineConfig.store_raw`). Cues are never
persisted; recollections keep only the cue id, source and channel names.

## Consequences

* Every component can be built and tested in parallel against `core/types.py` using the
  offline `HashEmbedder`.
* The fusion formula is simple, explainable (`channel_scores` + factors are returned) and
  tunable; it is not learned. A learned fusion is a future ADR.
* Similarities from different channels are on different scales (e.g. CLIP image cosines
  cluster high). Channels are responsible for mapping to a meaningful `[0, 1]`; evals
  measure the result.
* Changing any entity, the formula, or a protocol requires a new ADR.

## Alternatives considered

* **Single concatenated embedding**: simpler ANN, but breaks "only channels present in both"
  and makes channel weights opaque. Rejected.
* **Pure push via MCP notifications**: most clients ignore resource notifications today, so
  the hook path is primary. Kept notifications as progressive enhancement.
* **ImageBind as a universal embedder**: one space for all senses, but CC-BY-NC licence is
  incompatible with Apache-2.0 distribution. Documented as an optional user-installed
  backend only; never a dependency.
