# HomeRecall

Ask a folder of home security recordings a question in plain language and get back the
clip, with its date and time.

```
> did anyone take eggs out of the fridge yesterday

Searched 41 moments in 12 recordings from Wed 07 Oct 00:00 until Thu 08 Oct 00:00;
ranked by keywords, detector; checked 24 with the language model.

YES: 1 matching moment.
 1. Wed 07 Oct 2026, 07:42:16 to 07:42:30   confidence 0.91
    recording: D:\tapo\20261007_074210.mp4  at 0:06 to 0:20
    caption:   A person opens the refrigerator and takes out a carton of eggs.
    clip:      results\20261008-141334_did-anyone-take-eggs\1.mp4
    frames:    results\20261008-141334_did-anyone-take-eggs\1.jpg
```

(The block above shows the output format. The numbers measured so far are under
[Measured so far](#measured-so-far).)

It is built for home cameras such as a Tapo camera, and gets its footage in either of
two ways: it can watch the camera's stream itself and save a clip whenever a person
appears, or it can read recordings exported from the camera's app into a folder.
Nothing leaves your machine: there is no cloud service and no API key.

HomeRecall reuses the perception stack of
[LiveScene](https://github.com/HameesSajid/livescene) (YOLOE open-vocabulary detection, a
vision-language model, and the rule that one model has to confirm what another claims)
and applies it to a different problem: not "what is in front of the camera now" but
"when did this happen".

## How it works

**Indexing** happens once per recording, ahead of time.

| Step | What happens |
|------|--------------|
| Scan | The detector looks for people twice a second |
| Cut | Frames with a person are grouped into *moments* of at most 8 seconds, overlapping by 2 |
| Keep | A few key frames per moment are saved, cropped from the full-resolution frame to the area around the person |
| Describe | The language model captions the moment from its key frames, in time order |
| Tag | The detector looks for household objects and for every object the caption named |
| Embed | (Colab profiles) each key frame gets a SigLIP 2 image-text embedding |

Each moment is stored with its absolute date and time, which comes from the recording's
file name, or the time inside the MP4 file, or as a last resort the file's date. The
index remembers which, and a result based on the file's date is marked approximate.

**Asking** has two stages.

1. *Retrieve.* Every moment in the time range is ranked by cheap evidence: the question's
   words in captions and tags, the question's nouns found in the key frames by the
   detector (a word nobody anticipated, like "egg", is looked for now and cached), and
   embedding similarity when the index has embeddings. The rankings are fused.
2. *Verify.* The best candidates are shown to the language model again, with the question
   itself. It must answer yes, and its probability of "yes" is the confidence shown.

A time in the question ("yesterday", "last night", "on Monday", "on October 5th", "this
morning") limits the search to that range. Matching moments that overlap are joined into
one result, and that piece of the recording is cut out as its own file next to a strip
of its key frames.

See [docs/CONCEPTS.md](docs/CONCEPTS.md) for why it is built this way and
[docs/CODE_GUIDE.md](docs/CODE_GUIDE.md) for every file and function.

## Profiles

| Profile | GPU memory | Detector | Language model | Embeddings | Key frames per moment |
|---------|------------|----------|----------------|------------|-----------------------|
| `laptop` | 6 GB | YOLOE-26m | SmolVLM2-500M | none | 4 |
| `colab-t4` | 12 to 16 GB | YOLOE-26l | Qwen2.5-VL-3B, half precision | SigLIP 2 base | 6 |
| `colab` | 22 GB or more | YOLOE-26x | Qwen2.5-VL-7B, half precision | SigLIP 2 so400m | 8 |

`--profile auto` (the default) picks by GPU memory. The `laptop` profile only uses models
that are already on disk and refuses to download anything unless you pass
`--allow-download`.

The index is portable: build it on Colab with the large models, download the `index`
folder, and ask questions on the laptop. The captions and tags in it are then the large
models' work; only the final check uses the laptop's small model.

## Measured so far

Status: this has **not yet been run on real camera footage**, and the Colab
profiles have not been run at all. What has been measured is a small check that the
whole pipeline works, on the `laptop` profile (GTX 1660 Ti 6 GB, SmolVLM2-500M), using
two unrelated stock clips renamed to look like camera exports: 36 seconds of video, 5
moments.

| | Result |
|-|--------|
| Indexing speed | 39 s for a 25 s 4K clip, 14 s for an 11 s 4K clip (about 1.4 times the clip length) |
| Questions whose answer is in the footage (4) | 3 answered correctly; all 4 had the right moment ranked first by retrieval |
| Questions whose answer is not in the footage (6), including the eggs question | 6 correctly answered "no" (highest confidence 0.31) |
| Unit tests | 35, no GPU needed |

Recording from a stream has been tested without a camera: with a video file standing in
for the stream, and with the same file served over the local network as a stream that
ends and restarts. In both, clips were saved under the right names, the recorder
reconnected by itself, clips were indexed in the background while recording continued,
and stopping closed the open clip. **It has not been run against a real Tapo camera.**

The one miss: "are people sitting on a sofa" also matched people sitting on a bench
outdoors, with a higher confidence than the real sofa. The questions and the full report
are in `outputs/sample_labels.json` and `outputs/eval_samples_laptop.json`.

Ten questions on two clips say the plumbing works. They say nothing reliable about
accuracy on a kitchen camera. `tools/eval_search.py` exists so that can be measured on
your own recordings; see [Measuring it on your footage](#measuring-it-on-your-footage).

## What to expect, and known limits

- **The small model is the weak point of the laptop profile.** Its captions are vague
  (it described game controllers as "a remote control and a smartphone"), and it knows
  little about fine actions. Expect the laptop profile to be usable for coarse questions
  (is someone there, is there a dog, did someone sit down) and unreliable for fine ones.
- **"Took eggs out of the fridge" is a hard question.** Eggs are a few pixels across
  from a ceiling camera and are in a hand for a second or two. The design gives it the
  best chance (full-resolution crops around the person, several frames in time order, a
  second look with the actual question), but it needs the 3B or 7B model on Colab, and
  even then the honest answer may be "someone opened the fridge and took something out".
- **A "no" is only as good as the candidates that were checked.** By default the best 24
  (laptop) are checked; the output says how many were not. `--verify all` checks every
  moment in the range.
- **The yes/no check has a threshold.** 0.5 by default. Small models lean towards "yes";
  raise `--threshold` for fewer false matches.
- **It does not know who a person is.** There is no face recognition, by design.
- **No audio in the cut clips**, because OpenCV writes video only. The original
  recording is untouched and its path is printed.
- **Times are only as good as the source.** Check `python recall.py stats`: if it
  reports `file-modified` for your recordings, the app did not put the time in the file
  name or the file, and times will be when the file was saved, not when it was recorded.
- Only English questions.

## Getting footage, way 1: record from the camera (recommended)

HomeRecall connects to the camera's video stream over your home network, looks for a
person twice a second, and saves a clip from 3 seconds before the person appeared until
5 seconds after they left. Each clip is named by its start time and indexed in the
background as soon as it is saved, so there is nothing to export and times are exact.
The camera's own recording to its SD card or cloud carries on unaffected.

**One-time setup in the Tapo app** (steps from
[Tapo's RTSP guide](https://www.tapo.com/en/faq/34/); menu names can change between app
versions):

1. Open the camera's live view and tap the settings icon in the top right corner.
2. Tap Advanced Settings, then Camera Account. Accept the notice and create a username
   and password, 6 to 32 characters each. This is a separate login for the stream, not
   your Tapo account. Do not reuse your Tapo password for it.
3. Still in the camera's settings, find the camera's IP address (under the device
   information). It looks like `192.168.1.50`.
4. Set the video quality to the best setting, because on some models the stream's
   resolution follows it.

**Then on the PC:**

```powershell
cd C:\Users\hamee\vscode\homerecall
$py = "..\livescene\.venv\Scripts\python.exe"

& $py recall.py camera add kitchen       # asks for the IP address, username and password
& $py recall.py camera check kitchen     # connects once and saves a snapshot to results\
& $py recall.py watch kitchen            # records and indexes until you press Ctrl+C
```

`camera add` stores the camera's address in `cameras.json` in the project folder. That
file contains the camera account's password in plain text. It is excluded from git and
never printed: wherever the address is shown, the login is replaced by `***`.

Leave `watch` running in its own PowerShell window. In a second window you can `ask`
at any time, including about a clip saved a minute ago.

| `watch` option | Default | Meaning |
|----------------|---------|---------|
| `--pre` | 3 | Seconds kept from before the person appeared |
| `--post` | 5 | Seconds recorded after the person was last seen |
| `--max-clip` | 120 | Longest clip in seconds; longer activity continues in a new clip |
| `--min-free-gb` | 10 | Recording pauses while the disk has less free space than this |
| `--recordings` | `recordings\<camera>` | Where clips are saved |
| `--no-index` | off | Only record; index later with `index` |
| `--source` | | Use a stream address, a webcam number (`0`) or a video file instead of a named camera |

Things to know:

- **The PC has to be on and `watch` running.** Nothing is recorded while it is not, and
  it cannot reach footage from before it started (use way 2 for that).
- **Disk use.** Clips are re-encoded as MPEG-4. A test at 2560x1440 and 15 frames per
  second on busy footage came to about 50 MB per minute of recorded activity; a mostly
  still indoor scene should be smaller. Nothing is deleted automatically.
- **No sound.** OpenCV records video only.
- **If it does not connect:** check the IP address (it can change when the router
  restarts; reserving it in the router's settings stops that), the username and
  password, that the PC and camera are on the same network, and that the camera is not
  in privacy mode. Battery-powered Tapo models do not offer this stream. Some newer
  versions of the Tapo app are reported to have an extra third-party access switch;
  that has not been confirmed for this project.
- **To try it without a camera:** `& $py recall.py watch --source 0` uses the laptop's
  webcam.

## Getting footage, way 2: a folder of exported recordings

HomeRecall reads video files (`.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`, `.ts`) from a
folder, including subfolders. Two ways to fill that folder from a Tapo camera:

- Download recordings in the Tapo app's playback view and copy them to the PC. Keep the
  file names the app gives them, since that is usually where the date and time are.
- Take the microSD card out of the camera and copy its contents. This only works when
  card encryption is off; an encrypted card is unreadable on a computer
  ([Tapo's note on card encryption](https://www.tapo.com/en/faq/745/)). Reading a Tapo
  card's files has not been tried with this project yet.

Then `& $py recall.py index D:\tapo`.

## Getting it back on a PC from GitHub

The repository holds the code, documents, notebook and test reports. It does not hold a
Python environment, models, recordings, the index or results. (The local copy at
`C:\Users\hamee\vscode\homerecall` was removed on 2026-11-03 to free disk space, together
with LiveScene's, whose environment and models it used; see
[docs/INSTALLED.md](docs/INSTALLED.md).)

**Way A: next to LiveScene (no second environment, recommended).**

1. Get LiveScene back first, by following "Getting it back on a PC from GitHub" in
   [LiveScene's README](https://github.com/HameesSajid/livescene#getting-it-back-on-a-pc-from-github).
   That rebuilds `livescene\.venv` and downloads the models into `livescene\.cache`,
   including `yoloe-26m-seg.pt`, which HomeRecall's `laptop` profile needs.
2. Clone HomeRecall into the same parent folder (the repository is private, so Git uses
   your saved GitHub sign-in):

   ```powershell
   cd C:\Users\hamee\vscode
   git clone https://github.com/HameesSajid/homerecall.git
   cd homerecall
   $py = "..\livescene\.venv\Scripts\python.exe"
   & $py tests\test_core.py
   ```

   If `git` is not found, use `& "C:\Program Files\Git\cmd\git.exe"`.
3. Test footage: the old `recordings\` folder held copies of LiveScene's two sample clips
   renamed `20261006_193005.mp4` (`dog_square.mp4`) and `20261007_101500.mp4`
   (`gaming_sofa.mp4`). Copy them in under those names and run `& $py recall.py index
   recordings` to rebuild `index\`.
4. Cameras: `cameras.json` is never in git. Run `& $py recall.py camera add <name>` again.

**Way B: on its own.** Follow "Setting it up without LiveScene" below; the first
`--allow-download` run fetches the models (about 2.3 GB) into `homerecall\.cache`.

**Colab** needs nothing on the PC: see [Run on Google Colab](#run-on-google-colab).

## Run locally (Windows, NVIDIA GPU)

HomeRecall uses the same libraries as LiveScene. If LiveScene is set up in the folder
next to this one, there is nothing to install: run HomeRecall with LiveScene's Python,
and it finds LiveScene's downloaded models by itself.

```powershell
cd C:\Users\hamee\vscode\homerecall
$py = "..\livescene\.venv\Scripts\python.exe"

& $py recall.py watch kitchen                  # way 1: record from the camera and index
& $py recall.py index D:\tapo                 # way 2: index a folder (new files only)
& $py recall.py ask "did anyone take eggs out of the fridge yesterday"
& $py recall.py ask                            # keep asking; models load once
& $py recall.py log --date 2026-10-07          # everything the camera saw that day
& $py recall.py stats                          # what is indexed, and where times came from
```

Useful options for `ask`:

| Option | Meaning |
|--------|---------|
| `--date 2026-10-07`, `--since`, `--until` | Limit the time range (a time in the question does the same) |
| `--verify 50`, `--verify all` | How many candidates the language model checks |
| `--threshold 0.7` | Confidence needed to count as a match |
| `--top 10` | Results to show |
| `--open` | Open the best matching clip when done |
| `--no-clips` | Do not cut matching clips out (faster) |
| `--no-verify` | Retrieval only; results are labelled as not verified |

Results are written to `results\<time>_<question>\`: `1.mp4`, `1.jpg`, ... and
`result.json`.

**Setting it up without LiveScene.** Python 3.10 or newer and an NVIDIA GPU:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
.\.venv\Scripts\python.exe recall.py --allow-download index D:\tapo
```

Models (about 2.3 GB for the laptop profile) then download into `.cache` in this folder.

**Tests:**

```powershell
& $py tests\test_core.py
& $py tests\test_recorder.py
```

## Run on Google Colab

Colab is where the accurate profiles run. The notebook
[notebooks/homerecall_colab.ipynb](notebooks/homerecall_colab.ipynb) has every step as a
cell. It has not been run on Colab yet; if a cell fails, the error message is what is
needed to fix it.

Using Colab means uploading your home recordings to Google (Drive or the Colab session).
That is your decision to make; the notebook does not send them anywhere else.

1. Open [colab.research.google.com](https://colab.research.google.com), File, Upload
   notebook, choose `notebooks/homerecall_colab.ipynb`.
2. Runtime, Change runtime type, pick a GPU. A T4 selects `colab-t4`; an L4 or A100
   selects `colab`.
3. Get the code in: the repository is private, so download it as a ZIP from GitHub
   (Code, Download ZIP) and upload it when the second cell asks.
4. Put your recordings in a Google Drive folder and set `RECORDINGS` in the notebook, or
   upload files directly with the upload cell.
5. Run the index cell, then ask questions in the question cell.
6. Run the last cell to download `index.zip`. Unzip it into this project's folder on the
   PC (so that `index\index.db` exists) to ask questions locally against the index the
   large models built. Clips are cut from the original recordings, so on the PC the
   recordings need to be at the same path as on Colab for the cut to work; the dates,
   times, captions and key frames work regardless.

## Measuring it on your footage

Write down a few questions and which recordings answer them, including some questions
whose true answer is "no":

```json
[
  {"question": "did anyone take eggs out of the fridge", "clips": ["20261007_074210.mp4"]},
  {"question": "did a dog come into the kitchen", "clips": []}
]
```

```powershell
& $py tools\eval_search.py my_labels.json --out outputs\eval_mine.json
```

It reports whether each yes/no was right, how often the right recording was ranked
first, in the top 5 and in the top 10 before verification, and how many wrong moments
were accepted.

## Privacy

`recordings/`, `index/` and `results/` are excluded from git, because they contain
footage and frames from inside your home. `cameras.json` is excluded because it contains
the camera account's password. Do not remove those lines from `.gitignore`.

## Roadmap

1. Run against the real camera and on real kitchen recordings, on Colab and locally, and
   replace the table above with real numbers.
2. A small local web page: type a question, watch the matching clip.
3. Hand and held-object crops for fine actions such as taking a small item.
4. Save the camera's own compressed video instead of re-encoding it (smaller files, and
   sound). This needs ffmpeg installed.
5. Automatic clean-up of recordings older than a chosen number of days.
6. Reading the camera's on-screen clock as a fourth time source.

## Repository layout

```
recall.py                 command line: camera, watch, index, ask, log, stats
cameras.example.json      the format of cameras.json (which is private and not in git)
homerecall/cameras.py     camera addresses; hides the login wherever one is shown
homerecall/recorder.py    recording from a stream when a person appears; background indexing
homerecall/models.py      loads each model on first use
homerecall/config.py      model profiles (laptop, colab-t4, colab)
homerecall/paths.py       where models, the index and results live; download guard
homerecall/timestamps.py  when a recording started (file name, MP4 header, file date)
homerecall/segments.py    cutting a recording into moments; the activity crop
homerecall/vocab.py       word lists given to the detector
homerecall/detector.py    YOLOE open-vocabulary detector
homerecall/vlm.py         language model backends, including the yes probability
homerecall/embed.py       SigLIP 2 embeddings (Colab profiles)
homerecall/text.py        prompts, reply parsing, keywords (BM25), time phrases, fusion
homerecall/store.py       the SQLite index
homerecall/indexer.py     the indexing pass
homerecall/search.py      retrieve, verify, merge
homerecall/report.py      the printed answer, contact sheet, cut clip
tools/eval_search.py      accuracy on your own labelled questions
tests/test_core.py        25 unit tests, no GPU needed
tests/test_recorder.py    10 unit tests for recording, no camera or GPU needed
notebooks/                Colab notebook
docs/                     concepts, code guide, install log
outputs/                  evaluation labels and reports
```

## Credits

Built on [Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/),
[SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct),
[Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and
[SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-384). Check each model's
licence before commercial use (Ultralytics is AGPL-3.0).
