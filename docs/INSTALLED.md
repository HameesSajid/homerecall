# What was installed, where, and how to remove it

Last updated: 2026-10-08. Sizes are as measured on that date.

## Summary

**Nothing was installed and nothing was downloaded for HomeRecall.** No Python package,
no model, no program. It runs with LiveScene's existing Python environment and
LiveScene's existing model files, and only added the files listed below.

## 1. What HomeRecall added to this PC

Everything is in one folder, `C:\Users\hamee\vscode\homerecall\`. It is not inside
OneDrive, so none of it syncs.

| Path inside the project folder | Size | Contents | On GitHub? |
|--------------------------------|------|----------|------------|
| `recall.py`, `homerecall\`, `tests\`, `tools\`, `notebooks\` | under 1 MB | Source code and the Colab notebook | Yes |
| `README.md`, `docs\`, `requirements.txt` | under 1 MB | Documentation | Yes |
| `outputs\sample_labels.json`, `outputs\eval_samples_laptop.json` | under 1 MB | The ten test questions and their report | Yes |
| `recordings\` | 104 MB | Copies of LiveScene's two sample clips, renamed `20261006_193005.mp4` and `20261007_101500.mp4` to look like camera exports, for testing. Safe to delete | No |
| `index\` | 1.4 MB | The search index built from those two clips: `index.db` and 25 key frame images. Safe to delete; `recall.py index` rebuilds it | No |
| `results\` | 12 MB | Answers from test questions: cut clips, frame strips, `result.json`. Safe to delete | No |
| `cameras.example.json` | under 1 KB | Shows the format of `cameras.json`, with placeholder values | Yes |
| `.git\` (hidden) | under 1 MB | Git history | It is the link to GitHub |

The "No" rows are excluded by `.gitignore`. `recordings\`, `index\` and `results\` will
hold images from inside your home once real footage is indexed, which is why they must
stay out of git.

### Files that appear when you start using the camera

Neither exists yet; both are created by commands you run.

| Path inside the project folder | Created by | Contents | On GitHub? |
|--------------------------------|------------|----------|------------|
| `cameras.json` | `recall.py camera add <name>` | The camera's network address, including the camera account's username and **password in plain text**. Delete the file, or run `recall.py camera remove <name>`, to forget it | No |
| `recordings\<camera name>\` | `recall.py watch <name>` | One MPEG-4 clip per visit, named by start time. About 50 MB per minute of recorded activity at 2560x1440 (measured on busy test footage; a still room should be less). **Nothing deletes old clips automatically**, so this folder grows. Recording pauses when the disk has under 10 GB free | No |
| `recordings\<camera name>\.incoming\` | `recall.py watch <name>` | The clip being written right now; it moves up one folder when finished. A file left here after a crash is an unfinished clip and can be deleted | No |
| `results\camera_<name>.jpg` | `recall.py camera check <name>` | One snapshot from the camera | No |

## 2. What HomeRecall uses but did not add

These belong to LiveScene and are documented in LiveScene's own `docs\INSTALLED.md`.
HomeRecall only reads them. **Deleting them breaks HomeRecall as well as LiveScene.**

| Path | Used for |
|------|----------|
| `C:\Users\hamee\vscode\livescene\.venv\` | The Python environment (PyTorch with CUDA, ultralytics, transformers, OpenCV). HomeRecall is run with `..\livescene\.venv\Scripts\python.exe` |
| `C:\Users\hamee\vscode\livescene\.cache\weights\yoloe-26m-seg.pt` (67 MB) | The detector |
| `C:\Users\hamee\vscode\livescene\.cache\weights\mobileclip2_b.ts` (242 MB) | The detector's text encoder |
| `C:\Users\hamee\vscode\livescene\.cache\hf\hub\models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct\` (1.9 GB) | The language model |

How it finds them: `homerecall\paths.py` uses `..\livescene\.cache` when this project has
no `.cache` folder of its own. If a `.cache` folder is ever created inside
`homerecall\`, that one is used instead.

Nothing was written into LiveScene's folder. Its code, environment and cache are
unchanged.

## 3. Protection against accidental downloads

The `laptop` profile sets Hugging Face's offline switch and checks that the detector
file exists before loading it. If a model is missing, HomeRecall stops with an error
instead of downloading. Downloads only happen if you pass `--allow-download`, or select
a Colab profile on this PC with `--profile colab-t4` or `--profile colab` (do not; those
are 8 GB and 18 GB of models).

## 4. Deliberately not installed on this PC

| Item | Size it would take | Where it is used instead |
|------|--------------------|--------------------------|
| SigLIP 2 embedding models (base / so400m) | 0.4 GB / 1.1 GB (approximate) | Colab only |
| Qwen2.5-VL-3B / 7B language models | 7.5 GB / 17 GB | Colab only |
| YOLOE-26l / 26x detectors | 76 MB / 164 MB | Colab only |
| `sentencepiece`, `protobuf` (listed in `requirements.txt` for SigLIP 2) | a few MB | Colab only; not needed by the laptop profile |
| ffmpeg | about 100 MB | Not needed. Clips are recorded and cut with OpenCV, which is why they have no sound and are larger than the camera's own files |

## 5. Removing things

Remove HomeRecall completely (the code stays on GitHub once pushed; LiveScene is not
affected):

```powershell
Remove-Item -Recurse -Force C:\Users\hamee\vscode\homerecall
```

Remove only the test data (frees about 118 MB):

```powershell
cd C:\Users\hamee\vscode\homerecall
Remove-Item -Recurse -Force recordings, index, results
```

Forget what has been indexed, keeping your recordings:

```powershell
Remove-Item -Recurse -Force C:\Users\hamee\vscode\homerecall\index
```

## 6. Git and GitHub

The repository is https://github.com/HameesSajid/homerecall (private, branch `main`).
Your name and email are set for this repository only. It uses the Git for Windows you
installed at `C:\Program Files\Git\` and the GitHub sign-in already saved in Windows
Credential Manager; no new credential was stored.

## Change log

| Date | Change |
|------|--------|
| 2026-10-08 | Project created at `C:\Users\hamee\vscode\homerecall\`. No packages installed, no models downloaded |
| 2026-10-08 | Copied LiveScene's two sample clips into `recordings\` under camera-style names for testing; built `index\`; test questions wrote `results\` |
| 2026-10-08 | Git repository initialised locally |
| 2026-10-08 | Private GitHub repository `HameesSajid/homerecall` created through the GitHub website and the project pushed to it |
| 2026-10-08 | Added recording from a camera stream (`watch`, `camera`). No packages installed, no models downloaded: the stream is read and clips are written with the OpenCV already in LiveScene's environment. Test clips made while building it were written to a temporary folder and deleted |
