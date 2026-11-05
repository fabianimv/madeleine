# ADR 0003: Reading and editing memories

- Status: Accepted
- Date: 2026-11-05
- Deciders: lead

## Context

v0.1 could create (`remember`), search (`recall`, `sense`) and delete (`forget`) memories,
but it could not read a single memory by id or correct one. Users (and agents) need to fix a
typo, re-tag a memory or change how much it matters, without forgetting it and losing the
moment it was encoded. Glama's coherence review also flagged the missing read/update
operations.

A memory has two kinds of data:

1. its **sensory fingerprint**: when and where it happened and what was sensed (text, image,
   sound, biometrics), captured at encoding time;
2. its **annotations**: the content the agent gets back, tags, salience, metadata and the
   media reference.

## Decision

* New engine methods `get_memory(id)` and `update_memory(id, *, content, tags, salience,
  metadata, media_ref)`, with a new `MemoryStore.update(memory)` that replaces a stored
  memory in place: same id, same set of channels, changed vectors re-indexed, raw blobs
  untouched.
* **Only annotations are editable.** The timestamp, geohash and observed signals record the
  moment of encoding; editing them would make involuntary recall lie about the past. To
  change them, `forget` and `remember` again.
* `tags` and `metadata` are **replaced**, not merged; `None` leaves a field unchanged.
* If `content` changes **and** the memory's `text` observation was derived from the old
  content (the default `embed_content=True` path), the text observation and its embedding
  follow the new content, so text matching stays consistent with what the agent reads. A text
  that was explicitly *sensed* (for example a song title heard at the time) is part of the
  fingerprint and is kept.
* `update_memory` is serialised with `sense` and `forget` (same engine lock), so a memory is
  never edited halfway through a trigger.
* Surfaces: MCP tools `get_memory` and `update_memory`; HTTP `GET` and `PATCH
  /memories/{id}`; CLI `madeleine show` and `madeleine edit`.

## Consequences

* The contract (`core/types.py`) grows by three methods. Third-party stores must implement
  `update`.
* Trigger history and refractory state are kept across edits: editing a memory does not make
  it fire again sooner.
* Editing can change which cues match a memory (through its text) but never its time or
  place.
