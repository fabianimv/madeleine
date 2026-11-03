# Concepts

## The Proust effect and encoding specificity

In *Swann's Way* (1913), Marcel Proust dips a madeleine in lime-blossom tea, and the taste
brings back, whole and unbidden, the Sunday mornings of his childhood in Combray. He had not
been trying to remember anything. Psychologists call this **involuntary autobiographical
memory**. It is common, it is usually triggered by a sensory or contextual cue, and it often
brings back memories that deliberate search would not find. The mechanism behind it is
**encoding specificity** (Tulving & Thomson, 1973): a memory is stored together with the
context present when it was formed, and a cue retrieves it to the degree that it overlaps
with that encoding context. Divers who learned a word list underwater recall it better
underwater (Godden & Baddeley, 1975). The place, the hour, the smell and the song are part
of the memory, not decoration around it.

madeleine applies this to agents. Each memory keeps its **sensory fingerprint**, and each
incoming **cue** is compared channel by channel against those fingerprints. Memories whose
encoding context overlaps strongly enough surface on their own.

## Vocabulary

| Term | Meaning |
|---|---|
| **Channel** | A sense: adapter (raw signal → observation) + embedder (→ vector) + similarity (→ [0, 1]). |
| **Observation** | One normalised reading on one channel (a text, a geohash, a timestamp, image bytes...). |
| **Sensory fingerprint** | The per-channel embeddings and context (time, place, tags, salience) captured when a memory was encoded. |
| **Cue** | A bundle of observations arriving now, from the agent, a webhook, a phone or a sensor. |
| **Recall (pull)** | The agent asks; results are ranked by fused score. |
| **Recollection (push)** | A memory that surfaced involuntarily because a cue matched it, queued for the agent. |
| **Salience** | How much a memory matters (0..1); salient memories surface more easily. |
| **Decay** | Older memories need stronger cues, but a floor keeps decades-old memories reachable. |
| **Refractory period** | After firing, a memory stays quiet for a while (habituation). |
| **Solo trigger** | Whether a channel can cause a recollection on its own. Pure context such as the hour cannot. |

## Fusion in one paragraph

Only channels present in **both** cue and memory are compared. A memory without audio is not
penalised for not sounding like the cue. Their similarities are averaged by weight (`base`).
The result is multiplied by how much of the cue those shared channels cover (`coverage`), by a
bounded salience boost, and by an age decay with a floor. A memory fires involuntarily when
that score clears a threshold, at least one shared channel is allowed to trigger on its own,
and the memory is not in its refractory period. The exact formula is normative and lives in
[ADR 0001](adr/0001-core-model.md). Every result reports its per-channel similarities, so
you can always see *why* something surfaced.

## What madeleine is not

It is not a vector database, and it does not decide what is worth remembering: your agent
or your integrations call `remember`. It does not fetch anything from the internet or read
your files unless you allow it. Its similarity functions are hand-designed and calibrated, not
learned.
