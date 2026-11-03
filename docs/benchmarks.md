# Benchmarks

madeleine ships an **offline, deterministic, synthetic benchmark** of cue-triggered recall.
Its job is to show, honestly and reproducibly, where involuntary recall works and where it
fails. It is not a claim about real-world memory quality: every memory and cue is
generated, and the default text embedder is lexical (see below).

```bash
madeleine bench                          # 500 memories, 300 cues, seed 0, markdown report
madeleine bench --sweep --json --output bench.json
python -m madeleine.evals --memories 200 --cues 120 --seed 3
```

```python
from madeleine.evals import DEFAULT_SWEEP, run_bench

report = run_bench(memories=500, cues=300, seed=0, thresholds=DEFAULT_SWEEP)
print(report.to_markdown())          # or report.to_dict() for JSON
```

`run_bench(*, memories, cues, seed, engine_factory=None, recall_config=None, thresholds=None,
habituation=True)` builds every engine through `engine_factory(recall_config) ->
MemoryEngine`. The default factory creates an in-memory engine
(`MadeleineConfig(db_path=":memory:", text_backend="hash")`) with the hash channel registry,
so nothing is downloaded and the default run takes seconds. Pass your own factory to
benchmark another configuration, e.g. a registry with real text/image/audio models.

## What is synthetic

`madeleine.evals.generator.generate(memories, cues, seed)` builds a synthetic "life" from a
single seeded `random.Random` (no global state). The same arguments always give the same
dataset; a different seed gives a different one.

* **Places**: three cities (Lisbon UTC+0, Santiago UTC-3, Tokyo UTC+9; fixed offsets, no
  DST), each with five place kinds (home, office, café, park, waterfront). Places are at
  least 1.5 km apart. A memory's location is jittered by up to 60 m around its place.
* **Routines**: each place kind has typical local hours and weekdays (office 9-16 on
  weekdays, home mornings and evenings, waterfront weekend afternoons...).
* **Topics**: ten topics with 16 distinctive words each (cooking, family, project, music,
  football, gardening, coffee, sea, health, books), each with a place affinity. Three more
  topics (astronomy, chess, plumbing) **never** appear in memories and supply negative cue
  text.
* **Memories**: `"{name} and I {verb} {w1} and {w2} at the {place} in {city}; I still
  remember the {w3} and the {adjective} {object}."` plus a local-offset timestamp spread
  over the 3 years before a fixed `now` (2026-06-01 UTC, never the wall clock), tags
  `{topic, place kind, city}`, a salience drawn from Beta(2, 2), and, for ~18% / ~12% of
  memories, a tiny valid synthetic PNG (16×16 RGB) or WAV (0.25 s, 8 kHz).
* **Cues** happen in the "present" (the 28 days after `now`) and carry ground truth: the
  set of relevant memory indices and whether an involuntary trigger is wanted
  (`should_fire`).

## Cue categories

| category | signals in the cue | relevant set | should fire | what it tests |
|---|---|---|---|---|
| `context` | geo 50-300 m from the place + time (same weekday, ±1 h), **no text** | every memory at that place with local hour within ±1 h | yes | encoding specificity: being back in the same place at the same time of day |
| `text_paraphrase` | text only, sharing the target's name, 2 of its 3 topic words and its object, in a different sentence | the target | yes | lexical text recall (PULL) and text-only triggering |
| `multi` | the paraphrase + geo 20-150 m from the target + time (same weekday, ±30 min) | the target | yes | fusion of several channels |
| `exact_media` | the target's image or audio, identical bytes | the target | yes | media channels |
| `perturbed_media` | the target's image/audio slightly changed (±1-3 on 10% of pixel values / −50 dB noise) | the target | yes | robustness of media embeddings: **expected to fail with the hash embedder** |
| `old_memory` | like `multi`, target at least 2 years old | the target | yes | the decay floor (old memories must still be reachable) |
| `distractor` | text from a never-used topic, a location thousands of km away, a random time | ∅ | no | nothing should fire on unrelated input |
| `time_only` | only the time channel, at the hour of some memory | ∅ | no | the `solo_trigger` rule: "same hour" alone must never fire |
| `hard_negative` | geo at a place with memories, at the hour farthest from all of them, text from a never-used topic | ∅ | no | place alone, at the wrong time and about something else, should not fire |

Notes on ground truth:

* Relevance is **strict**. For `multi`/`old_memory` only the target counts, even though
  other memories at the same place and hour are plausibly "related"; firing them lowers
  precision. For `context`, the relevant set ignores weekday and season (the `time` channel
  does not).
* Negatives have an empty relevant set, so recall@k is undefined (`n/a`) for them.

## Protocol

1. A fresh engine is created with `refractory_period = 0`, so every cue is measured
   independently of the others.
2. All memories are inserted with `engine.remember(content, signals, context=...)` (text is
   derived from the content, time and geo from the context).
3. Cues are processed in **chronological order** (time only moves forward). For each:
   `engine.build_cue(...)`, then PULL `engine.recall(RecallQuery(cue, k=10))`, then PUSH
   `engine.sense(cue)`, then the queue is drained so that it never grows.
4. **Habituation**: a second fresh engine with the configured refractory period (default
   6 h). One positive cue that fired its target in the main run (preferably `multi`) is
   repeated 5 times, 10 minutes apart. Expected: the target fires exactly once and no
   memory fires twice. The cue is then repeated whole weeks later, after the period ends,
   at the same local time. Expected: the target fires again. Reported as PASS/FAIL.
5. **Threshold sweep** (`--sweep`, thresholds 0.40-0.80). Under the ADR 0001 push rule, the
   memories that fire at threshold `t` are exactly those that fire at a lower threshold
   `t0 ≤ t` with `score ≥ t`. Eligibility does not depend on the threshold, and the top
   `max_push_per_cue` of a score-sorted prefix is a prefix of the top `max_push_per_cue`.
   The main run therefore uses `min(sweep + [configured threshold])`, and every
   threshold, the configured one included, is derived from that single pass. A test checks
   this against independent runs.

If the engine lacks a channel (e.g. no image support), memories are inserted without their
media and cues on that channel are skipped; both are counted under "Skipped".

## Metrics

Memories are referred to by dataset index. A metric that is undefined for its input is
reported as `n/a` (`null` in JSON), never as a misleading 0 or 1.

* **recall@k** (k = 1, 5, 10; PULL): capped recall,
  `|relevant ∩ top-k| / min(k, |relevant|)`, averaged over cues with a non-empty relevant
  set. Capping keeps it reachable when a `context` cue has 30 relevant memories.
* **Involuntary trigger precision** (PUSH): relevant fires / all fires, summed over cues
  (micro-average). Fires on negative cues count as irrelevant.
* **Trigger hit rate**: share of positive cues where at least one relevant memory fired.
* **False-trigger rate**: share of negative cues that fired anything.
* **Latency** p50/p95 (plus mean and max) in milliseconds for `remember`, `build_cue`,
  `recall` and `sense`, measured in-process with `time.perf_counter`. Habituation runs are
  not timed.

Everything is broken down per category and overall. Percentiles use linear interpolation
(numpy's default).

## Why the hash embedder is lexical, not semantic

The offline `HashEmbedder` turns text into a signed feature-hashing vector of word unigrams
and bigrams (stop words down-weighted). Two texts are similar only if they **share words**:
"car" and "automobile" are unrelated, while "the bank of the river" and "the bank raised
rates" are related. The generator's paraphrases deliberately reuse some of the target's
words, so the text categories measure *lexical* recall. With the `[text]` extra
(sentence-transformers), paraphrases with no word overlap would also be found. That
backend is not part of the default run.

For media, the hash embedder derives a pseudo-random vector from the **content hash**.
Identical bytes give similarity 1; changing a single byte gives a near-orthogonal vector.
So `exact_media` measures the plumbing, and `perturbed_media` is expected to fail until a
perceptual embedder (CLIP/CLAP via the `[image]`/`[audio]` extras) is plugged in.

## Known blind spots

* **Synthetic language**: a fixed template, English, ten topics. Real memories are messier,
  and real paraphrases share fewer words.
* **Fusion averages, it does not add**: `base` is a weighted mean over shared channels, so
  a weakly matching channel lowers a strong match. With the hash embedder a paraphrase
  scores about 0.3 on text, so a `multi` cue (text + place + time) can score *lower* than
  the same place and time with no text. Watch `multi` vs `context` in the results.
* **`max_push_per_cue` saturation**: `context` cues often have many relevant memories, so
  up to 3 fires per cue is normal there. Precision is lower for `multi`/`old_memory`, where
  only one memory is relevant.
* **No biometric cues**, no tags-filtered recall, no multi-person data, no DST or travel
  across time zones within a day.
* **Tiny media** (16×16 PNG, 0.25 s WAV). They are valid files, but real-model results on
  them say little about real photos and recordings.
* **Latency** is single-threaded and in-process on an in-memory database: no I/O, no MCP or
  HTTP overhead, no model inference unless a model-backed factory is supplied.
* **Ground truth is defined by the generator**, which shares assumptions with the engine
  (e.g. that place plus hour of day is a meaningful context). The benchmark checks the
  implementation against those assumptions; it does not validate the assumptions.

## Results

_To be filled in after integration (engine + channels merged) by running
`madeleine bench --sweep` on the default configuration; include the machine, the commit
and the text backend._
