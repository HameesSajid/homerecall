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
| `.git\` (hidden) | under 1 MB | Git history | It is the link to GitHub |

The "No" rows are excluded by `.gitignore`. `recordings\`, `index\` and `results\` will
hold images from inside your home once real footage is indexed, which is why they must
stay out of git.

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
| ffmpeg | about 100 MB | Not needed. Clips are cut with OpenCV, which is why cut clips have no sound |

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

The repository was initialised locally with your name and email set for this repository
only. It uses the Git for Windows you installed at `C:\Program Files\Git\`. See the
change log for whether it has been pushed.

## Change log

| Date | Change |
|------|--------|
| 2026-10-08 | Project created at `C:\Users\hamee\vscode\homerecall\`. No packages installed, no models downloaded |
| 2026-10-08 | Copied LiveScene's two sample clips into `recordings\` under camera-style names for testing; built `index\`; test questions wrote `results\` |
| 2026-10-08 | Git repository initialised locally; not yet pushed to GitHub |
