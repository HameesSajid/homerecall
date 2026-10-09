# Code guide

A file-by-file, function-by-function explanation of HomeRecall. Read
[CONCEPTS.md](CONCEPTS.md) first for the ideas; this document is about the code.

## The three paths

**Recording** (`recall.py watch <camera>`):

1. `cameras.resolve` looks the camera's address up in `cameras.json`.
2. `recorder.FrameSource` reads the stream on its own thread and stamps each frame with
   the time it arrived.
3. `recorder.watch` asks the detector for people twice a second and passes every frame,
   with the sighting if there was one, to `recorder.ClipRecorder`.
4. `ClipRecorder` starts, writes and closes clips, named by their start time.
5. Each closed clip goes to `recorder.IndexWorker`, which runs the indexing steps below
   on a background thread.

**Indexing** (`recall.py index <folder>`, or the worker above), per recording:

1. `indexer.find_clips` lists the videos; `Store.is_indexed` skips the ones already done.
2. `indexer.scan` reads the clip once and asks the detector for people twice a second.
3. `timestamps.clip_start` works out when the recording began.
4. `segments.make_windows` groups the frames into moments and computes each one's crop.
5. `indexer.save_keyframes` reads the clip a second time and writes the key frames.
6. `indexer.describe` captions each moment; `indexer.tag` records detector confidences.
7. `Store.add_clip`, `add_window`, `add_frame` write it all to SQLite.

**Asking** (`recall.py ask "..."`):

1. `text.parse_when` takes a time range out of the question.
2. `Store.windows` returns the moments in that range.
3. `Searcher.search` scores them three ways (`text.BM25`, `_detector_scores`,
   `_embedding_scores`) and combines the rankings with `text.fuse`.
4. The top candidates go to `vlm.yes_probability` with the question.
5. `search.merge` joins overlapping matches; `report.save` writes the contact sheet and
   the cut clip; `report.format_result` prints the answer.

---

## `recall.py` — command line

- **`build_parser()`** — the `camera`, `watch`, `index`, `ask`, `log` and `stats`
  commands and their options.
- **`say(...)`** — `print` that flushes, so lines appear at once in a long-running window.
- **`cmd_camera`** — `add` asks for the IP address and username with `input()` and the
  password with `getpass` (not shown, not in the shell history), builds the address and
  saves it; `list` prints the cameras with the login hidden; `remove` deletes one;
  `check` connects, reads one frame, and saves it as `results/camera_<name>.jpg`.
- **`cmd_watch`** — resolves the camera (or `--source`), loads the detector, starts the
  `IndexWorker` unless `--no-index`, and runs `recorder.watch` until Ctrl+C. On exit it
  lets the clip being indexed finish and says how to index any that are left.
- **`cmd_index`**, **`cmd_ask`**, **`cmd_log`** — one function per command. `cmd_ask`
  with no question loops on `input()`, so models load once for many questions. `ask`
  hands the model loaders to the searcher, so a search that needs no detector lookup
  never loads the detector.
- **`main()`** — `camera`, `log` and `stats` load no model. For the others it resolves
  the profile and, for a profile with `allow_download=False`, calls
  `paths.forbid_downloads()` unless `--allow-download` was given.

## `homerecall/models.py` — loading models

**`Models(cfg, log)`** builds the detector, language model and embedder the first time
each is asked for (`detector()`, `vlm()`, `embedder()`) and keeps them.

## `homerecall/cameras.py` — camera addresses

- **`CONFIG`** — `cameras.json` in the project folder: `{name: address}`.
- **`mask(source)`** — the address with `user:password@` replaced by `***@`. Everything
  that prints an address goes through this.
- **`rtsp_url(host, username, password, stream, port)`** — builds the address,
  percent-escaping the username and password so any characters work.
- **`load`**, **`save`** — read and write the file.
- **`resolve(name)`** — the named camera; with no name, the only camera in the file, or
  the `HOMERECALL_CAMERA_URL` environment variable. Errors say what to run next and
  never include an address.

## `homerecall/recorder.py` — recording from a stream

- **`FrameSource(source)`** — a thread that reads frames into `frames`, a queue of
  `(time, image)`.
  - `_open()` — opens a stream with FFmpeg over TCP with 10-second open and read
    timeouts; a webcam number or a file is opened normally.
  - `_put(item)` — for a live source, drops the oldest waiting frame when the queue is
    full (counted in `dropped`); for a file, waits for room instead.
  - `run()` — connect, read until the stream fails, reconnect with a growing delay (2 s
    up to 60 s). A file is read once and given the times it would have had playing live.
  - `connected`, `ended` — events the caller waits on. `stop()` ends the thread.
- **`open_writer(path, fps, size)`** — an MPEG-4 `VideoWriter`.
- **`ClipRecorder(folder, fps, pre_s, post_s, max_s, min_hits, min_free_gb, on_close)`**
  - `feed(t, frame, person)` — one frame; `person` is `True`/`False` if the detector
    looked at it, else `None`. While idle it keeps the last `pre_s` seconds as JPEGs and
    starts a clip after `min_hits` sightings in a row. While recording it writes the
    frame, closes the clip `post_s` after the last sighting, and rolls into a new clip
    at `max_s`.
  - `idle(now)` — closes an open clip when no frame has arrived for `post_s`.
  - `_disk_ok(t)` — refuses to start a clip when free space is below the limit, saying
    so at most every ten minutes.
  - `_begin(frames)` — names the clip after its first frame's time, opens the writer in
    `.incoming`, writes the pre-roll.
  - `_write(t, frame)` — writes as many copies of the frame as the clock says are owed
    (none if it is early), so clip time equals real time.
  - `close()` — releases the writer, moves the file into place, calls `on_close`.
- **`IndexWorker(index_dir, cfg, captions)`** — a thread with its own `Store` and
  `Models`. `add(path)` queues a clip; `run()` indexes them one at a time and reports,
  rather than raises, a clip that fails; `finish(drain)` stops after the current clip
  (or after all of them) and returns how many were left.
- **`watch(source, folder, detector, on_clip, ...)`** — the recording loop: wait for the
  connection, then for each frame decide whether it is time for a person check, feed the
  recorder, and report dropped frames every 500. Ctrl+C stops the source and closes the
  open clip.

## `homerecall/config.py` — profiles

- **`Config`** — one frozen dataclass per profile: detector weights and input size,
  language model spec, image side, key frames per moment, caption length, whether to ask
  for caption and objects separately (`two_step`, for small models), the embedder, the
  sampling rate, window length and overlap, key frame size, how many candidates are
  verified, the tag threshold, and whether downloads are allowed.
- **`PROFILES`** — `laptop`, `colab-t4`, `colab`.
- **`pick(name, **overrides)`** — returns a profile; `auto` chooses by GPU memory.
  Overrides that are `None` are ignored, so command-line flags pass straight in.

## `homerecall/paths.py` — locations and the download guard

- **`REPO`, `CACHE`, `WEIGHTS`, `INDEX`, `RESULTS`** — `CACHE` is `HOMERECALL_CACHE` if
  set; otherwise this project's `.cache`; unless that does not exist and a `livescene`
  folder with a `.cache` sits next to this project, in which case LiveScene's downloads
  are reused. `HF_HOME` is pointed at `CACHE/hf`, which is why this module is imported
  before the model libraries. `INDEX` can be moved with `HOMERECALL_INDEX`.
- **`forbid_downloads()`**, **`downloads_forbidden()`** — set and read Hugging Face's
  offline switch, so a model that is not on disk raises an error instead of downloading.
- **`in_weights_dir()`** — context manager that runs a block with the working directory
  set to `WEIGHTS`, because Ultralytics resolves weight files relative to it.

## `homerecall/timestamps.py` — when a recording started

- **`from_filename(path)`** — the first valid date and time in the file name. One regular
  expression accepts `20261006_074210`, `2026-10-06 07-42-10`, `2026.10.06_07.42.10`,
  `20261006T074210` and similar; impossible values (month 13) are skipped.
- **`_atoms(f, start, end)`** — walks the boxes of an MP4 file.
- **`from_metadata(path)`** — the creation time from the `moov/mvhd` box (seconds since
  1904, UTC), converted to local time. Dates before 2005 mean the clock was never set.
- **`clip_start(path, duration)`** — tries the two above, then the file's modification
  time minus the duration. Returns the time and which source it used.

## `homerecall/segments.py` — moments

Pure Python, no model.

- **`Sample`** — one examined frame: frame number, time, person boxes (0 to 1).
- **`Window`** — one moment: start, end, its samples, whether a person was found, the
  most people in one frame, the crop.
- **`_groups(samples, gap_s, min_hits)`** — runs of frames with a person, bridging gaps
  up to `gap_s`, dropping runs shorter than `min_hits` frames.
- **`_split(t0, t1, max_len, overlap)`** — equal pieces no longer than `max_len` that
  overlap by `overlap`.
- **`activity_crop(boxes, margin, min_side, full_above)`** — union of the boxes, grown
  by `margin`, at least `min_side` of the frame on each side, shifted to stay inside the
  frame, and replaced by the whole frame when it covers `full_above` of it.
- **`make_windows(samples, duration, ...)`** — the moments of one recording, with the
  whole-recording fallback when nobody was detected.
- **`pick_keyframes(window, k)`** — up to `k` samples spread evenly, preferring frames
  that contain a person.

## `homerecall/vocab.py` — word lists

`SUBJECTS` (who makes a moment worth indexing: `person`) and `HOME_VOCAB` (about forty
household objects tagged in every moment). Separate from the detector so that search and
the tests can import them without loading a model library.

## `homerecall/detector.py` — open-vocabulary detection

Adapted from LiveScene's `perception.py`.

- **`OpenVocabDetector`**
  - `__init__` — loads YOLOE from `WEIGHTS`; raises a clear error if the file is missing
    and downloads are forbidden.
  - `embed(names)` — text embeddings for words, cached per word.
  - `set_vocab(vocab)` — installs a vocabulary; a no-op when it is unchanged.
  - `detect(frame, conf)` — one forward pass.
  - `boxes(frame, vocab, conf)` — `(label, confidence, box in 0..1)` per detection; used
    to find people while scanning.
  - `check(frames, names, conf, competitors)` — best confidence per name over several
    frames; used for tagging and for question words.

## `homerecall/vlm.py` — language models

Adapted from LiveScene's `vlm.py`, with one addition.

- **`HFVLM`** — a Hugging Face image-text-to-text model.
  - `generate(frames, prompt, max_new_tokens)` — text generation over several frames.
  - `_answer_tokens()` — finds the token ids for "Yes", "yes", " Yes", "No" and so on.
  - `yes_probability(frames, question)` — one forward pass; returns
    P(yes) / (P(yes) + P(no)) for the first word of the answer.
- **`ServerVLM`** — a model behind an OpenAI-compatible endpoint; its `yes_probability`
  falls back to reading the generated word.
- **`parse_yes_no(reply)`** — 1.0, 0.0 or 0.5 from text.
- **`load_vlm(spec)`** — a model id, optionally with `@fp16` or `@4bit`, or `server:<url>`.

## `homerecall/embed.py` — embeddings

- **`Embedder`** — SigLIP 2. `images(frames)` returns one unit vector per frame;
  `text(sentence)` returns one for a question (lower-cased, padded to 64 tokens, as the
  model was trained). Used only by profiles that name an embedder; not yet run.

## `homerecall/text.py` — everything about words

Pure Python, no model.

- **Prompts** — `CAPTION_PROMPT`, `CAPTION_PROMPT_NOBODY`, `LIST_PROMPT` (small models,
  two calls), `PROPOSE_PROMPT` (capable models, one call), `VERIFY_PROMPT`.
- **`singular`**, **`clean_object`**, **`parse_list`**, **`parse_proposal`**,
  **`complete_sentences`** — turning a model's reply into a caption and object names.
- **`stem(word)`** — irregular forms from a table, then `-ing`, `-ed` and plurals, then
  home synonyms (`fridge` to `refrigerator`, `grab` to `take`).
- **`tokens(text)`** — content words, stemmed, stop words removed.
- **`query_objects(question)`** — the question's words worth giving to the detector:
  not stop words, not known action words.
- **`BM25`** — keyword scoring. `scores(query)` returns one score per document.
- **`fuse(rankings, k)`** — reciprocal rank fusion of several `{id: score}` dicts.
- **`parse_when(question, now)`** — returns `(start, end, rest)`. Understands ISO dates,
  "5 October" / "October 5th", today, yesterday, tonight, last night, weekdays, "last 3
  days", "this week", and a part of day attached to a day. `rest` is the question with
  the time words removed.

## `homerecall/store.py` — the index

- **`Store(root)`** — opens `root/index.db` and creates the tables.
- Writing: `is_indexed`, `remove_clip` (rows and key frame files), `add_clip`,
  `add_window`, `add_frame`, `set_probes`, `commit`.
- Paths: `rel`, `abs`, `frame_dir`. Key frame paths are stored relative to the index
  folder, so the index can be moved or built on another machine.
- Reading: `windows(start, end)` (moments overlapping a range, with their recording's
  fields), `frames(window_id)`, `embeddings(window_ids)`, `probes(terms)`, `stats()`.

| Table | One row per | Main columns |
|-------|-------------|--------------|
| `clips` | recording | path, size, modified time, start, start source, duration, profile |
| `windows` | moment | recording, t0, t1, absolute start and end, crop, caption, objects (JSON) |
| `frames` | key frame | moment, time, JPEG path, context JPEG path, embedding |
| `probes` | (moment, question word) | detector confidence |

## `homerecall/indexer.py` — the indexing pass

- **`find_clips(folder)`** — every video under a folder, skipping folders whose name
  starts with a dot (the recorder's `.incoming`).
- **`scan(path, detector, sample_fps)`** — pass 1. Uses `grab()` for skipped frames so
  only examined frames are decoded to images.
- **`save_keyframes(path, windows, k, side, folder)`** — pass 2. Writes each key frame
  as it is decoded, so memory does not grow with clip length. Also writes one context
  image per moment: the whole frame with the crop outlined.
- **`describe(frames, has_subject, vlm, cfg)`** — caption and proposed objects, in one
  or two model calls depending on the profile.
- **`tag(frames, proposed, detector, threshold)`** — detector confidence for the home
  vocabulary and the proposed objects. Returns `tags`, `proposed`, `confirmed`.
- **`index_clip(...)`** — all of the above for one recording.
- **`index_folder(...)`** — every new or changed recording in a folder.

## `homerecall/search.py` — answering

- **`as_question`**, **`_doc`**, **`similarity`**, **`parse_day`** — small helpers.
- **`merge(matches, gap)`** — joins matching moments of one recording that touch or
  overlap; the joined result keeps the frames and caption of its most confident part.
- **`Searcher`**
  - `_frames(window_id)` — a moment's key frames from disk.
  - `_detector_scores(windows, terms, probe)` — summed confidence for the question's
    nouns: from the stored tags for home vocabulary words, from the `probes` cache for
    others, running the detector for anything not yet cached.
  - `_embedding_scores(windows, question)` — best similarity per moment.
  - `search(question, start, end, verify, threshold, probe, now)` — the whole flow.
    Returns the parsed range, counts, the signals that fired, `matches` and `candidates`.

## `homerecall/report.py` — output

- **`describe_time(m)`** — date and time of a result, marked approximate when the time
  came from the file date.
- **`contact_sheet(store, window, path)`** — key frames side by side, each stamped with
  its own date and time.
- **`cut_clip(src, t0, t1, path)`** — the matching piece with 2 seconds either side,
  scaled to at most 1280 wide.
- **`save(result, store, folder)`** — sheet, cut clip and `result.json` per question.
- **`result_folder`**, **`format_result`** — the folder name and the printed answer. The
  answer always states how many moments were searched, which signals ranked them, and
  how many were checked.

## `tools/eval_search.py` — accuracy on labelled questions

- **`evaluate(searcher, labels, threshold)`** — runs each question with every moment
  verified; records the yes/no, whether it was right, and the retrieval rank of the
  first correct moment.
- **`summarise(rows)`** — verdict accuracy, found-when-present, correct-no-when-absent,
  retrieval recall at 1, 5 and 10.

## Tests

`tests/test_core.py`, 25 tests, no GPU: file name and MP4 timestamps (including an unset
clock), cutting moments (gaps, noise, splitting with overlap, the nobody-detected
fallback), the crop, key frame choice, stemming and synonyms, reply parsing, BM25, rank
fusion, time phrases (and words like "may" that must not be read as dates), the store
(round trip, time filter, removal), and search end to end with a stand-in language model
(verifies, merges overlapping matches, says no, respects the time range, labels
unverified results).

`tests/test_recorder.py`, 10 tests, no camera or GPU: a clip includes the pre-roll and
stops after the person leaves; one sighting does not start a clip; clip length follows
the clock when frames are lost; long activity rolls into a new clip; a break in the
stream closes the clip; stopping closes an open clip and names never collide; a full
disk stops recording; a clip written by the real writer reads back through
`FrameSource` with the right times and is found by `find_clips` while `.incoming` is
not; addresses are escaped and masked; the camera file and its error messages.

```
python tests/test_core.py
python tests/test_recorder.py
```
