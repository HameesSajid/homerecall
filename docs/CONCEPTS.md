# Concepts

The ideas behind HomeRecall, the alternatives that were considered, and why each choice
was made. For the code itself see [CODE_GUIDE.md](CODE_GUIDE.md).

## 1. The problem: finding a moment, not describing a scene

LiveScene answers "what is in front of the camera now". HomeRecall answers "did this
happen, and when". In the research literature this is *video moment retrieval* (find the
segment that matches a sentence) combined with *video question answering*.

Two things make it different from live description:

- The question is not known when the footage is processed. Whatever is stored has to
  serve questions nobody has thought of yet.
- The answer has to come with evidence: a clip and a time someone can check.

## 2. Index once, ask many times

**Chosen:** process each recording once into a compact index (moments, captions, tags,
key frames), and answer questions from the index.

**Alternative:** run a video-language model over all the footage for every question.
That is the most accurate thing possible and completely impractical: a week of clips is
thousands of model calls per question.

**Alternative:** store only captions and search the text. Cheap, but anything the caption
did not mention is lost for good. A small model's caption of a person at a fridge will
rarely mention eggs.

The design here stores captions *and* key frames, so that a question can go back to the
pixels for the few moments that matter.

## 3. Moments: what gets indexed

A recording is cut into windows of at most 8 seconds where a person is present, with 2
seconds of overlap between neighbours.

- *Why person-based:* the camera records on person detection, and nearly every question
  is about what someone did. Frames with nobody in them are skipped, which also keeps
  the index small.
- *Why 8 seconds:* long enough to hold one action (walk to the fridge, open it, take
  something), short enough that a few key frames cover it and one caption can describe
  it. A 60-second visit becomes about ten moments, each separately findable.
- *Why overlap:* an action that straddles a cut would otherwise be split in two halves
  that each look like nothing. With overlap it is whole in at least one window.
- *Gaps of up to 2 seconds are bridged,* so someone bending behind a counter is one
  visit, and a single frame with a detection is ignored as noise.
- *Safety net:* if nobody is detected anywhere in a recording, the whole recording is
  still indexed, flagged as "nobody detected". The camera thought it saw someone; a
  detector miss must not make the clip unsearchable.

## 4. The activity crop

Key frames are cropped from the original full-resolution frame to the union of the
person boxes in the moment, grown by 35 % on each side.

A language model sees a frame at 512 to 900 pixels on its longest side. A wide-angle
camera frame shrunk to that size leaves a hand and what it holds only a few pixels
across. Cropping first spends those pixels on the person and what they can reach. The
margin is there because the thing being reached for (a shelf, a door handle) is outside
the person's own box. A crop that would cover most of the frame becomes the whole frame.

**Alternative:** send whole frames at high resolution. Costs several times the visual
tokens for mostly empty floor. **Alternative:** crop to hands only. Better for small
objects but needs a hand detector or pose model; it is on the roadmap.

## 5. Several frames in time order

"Took out" is a change over time: the hand is empty, then it is not. One frame cannot
show it. Each moment keeps 4 to 8 key frames spread evenly through it, and the language
model receives them in order and is told so.

## 6. Retrieve, then verify

This is the core of the design, and it is the same principle as LiveScene's
propose-verify-describe: do not trust one model's first statement.

**Retrieve** is tuned for *recall*. It may return wrong candidates; it must not lose the
right one. Three kinds of evidence, each cheap:

| Signal | What it catches | What it misses |
|--------|-----------------|----------------|
| Keywords (BM25 over captions and tags) | Moments the caption described in the question's words | Anything the caption left out or phrased differently |
| Detector lookup | The question's nouns visible in the key frames, even if no caption mentioned them | Actions; objects too small for the detector |
| Embedding similarity (SigLIP 2) | Meaning rather than wording; scenes that "look like" the question | Fine detail; it is a whole-image match |

**Verify** is tuned for *precision*. The top candidates are shown to the language model
again with the actual question, and it must say yes.

Why this beats either stage alone: captioning is open-ended ("say something about
this"), so a small detail is easily omitted. Verification is a pointed question about
that one detail, with the frames in front of the model. A model that would never write
"eggs" in a caption can still answer "is the person taking eggs out of the fridge?".

### Why retrieval does not delete unverified objects

LiveScene deletes any sentence naming an object the detector cannot confirm, because a
live description must not state false things. HomeRecall keeps the whole caption and
only records the detector's confidence. In search, a missed moment can never be
recovered, while a wrong candidate is caught at the verify stage. The direction of
caution is reversed because the cost of each error is reversed.

### Rank fusion

The three signals have incomparable scales (a BM25 score, a detector confidence, a
cosine similarity), so they are combined by *reciprocal rank fusion*: each signal
contributes 1 / (60 + rank) for each moment it scored. Only the order within each signal
matters. Moments with no evidence at all queue up behind, newest first.

**Alternative:** a learned weighting of the scores. Needs labelled data that does not
exist yet.

## 7. Confidence from the model's probability of "yes"

Instead of generating text and reading "yes" or "no", the model is run one step and the
probability it gives to a yes-token versus a no-token is read directly:
confidence = P(yes) / (P(yes) + P(no)).

- It is one forward pass, with no text generation, so it is fast.
- It is graded. Candidates can be ranked, and the threshold can be moved to trade missed
  moments against false matches. Text alone would only say yes or no.

It is not a calibrated probability. A confidence of 0.9 does not mean right nine times
in ten; it only orders candidates. Small models also lean towards yes (LiveScene's POPE
benchmark measured this for the same 500M model), which is a reason to raise the
threshold on the laptop profile.

## 8. Open-vocabulary detection at question time

A fixed-class detector knows 80 things. YOLOE takes words as prompts, so the question's
own nouns can be looked for in the stored key frames, and nobody had to predict at
indexing time that eggs would matter. The result is cached per word, so a word is only
looked for once per moment. Known home objects (a list of about forty) are tagged at
indexing time and cost nothing at question time.

When checking for a question word, the home vocabulary is given as competing prompts.
The detector assigns each region to the closest prompt it has, so asking about "egg"
alone would let any small round thing pass as one.

## 9. Trustworthy time

A clip is only evidence if its date is right. Three sources, in order:

1. the file name (camera apps normally name exports by date and time),
2. the creation time in the MP4 container's movie header,
3. the file's modification time, minus the clip's length.

The source is stored with every recording, and a result that rests on the third is
printed as approximate. A camera with an unset clock writes the year 1904 into the MP4
header; that is detected and ignored.

Time phrases in the question are parsed by rules and removed before the question goes to
the language model, which cannot know what "yesterday" means from frames.

**Alternative:** read the clock the camera draws on the picture. Most reliable of all,
but needs text recognition; it is on the roadmap.

## 10. Two runtimes, one index

The laptop has 6 GB of GPU memory, which fits a 500M-parameter language model. That
model is the accuracy ceiling locally. Colab fits a 3B or 7B model. Because the index
stores paths relative to itself, it can be built on Colab and queried on the laptop, so
the expensive, accuracy-critical work (captioning every moment) is done once by the
large model.

## 11. What is deliberately not done

- **Face recognition / identifying people.** Out of scope, and a privacy problem.
- **Direct camera access.** Kept out until the folder workflow is proven; it also means
  no camera password is ever needed.
- **Cloud APIs.** Footage of the inside of a home stays on machines you control.
