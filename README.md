# ClearCut — Offline Background Remover

Removes backgrounds from images and animations (GIF / WebP / APNG), fully offline.
Runs on any PC (CPU) and uses the GPU automatically when one is set up.

---

## 1. Setup (Windows, PowerShell)

Python 3.12 is recommended. Check which versions you have with `py -0`.

```
cd "path\to\ClearCut"
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell says scripts are disabled, run this once, then activate again:
```
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install the common packages:
```
pip uninstall -y opencv-python opencv-python-headless
pip install -r requirements.txt
```
(The first line only matters if you're updating an older venv: the app now needs
`opencv-contrib-python-headless`, and it can't be installed next to plain OpenCV.)

Then pick **one** of the options below. Only one onnxruntime package can be
installed in a venv at a time, so each GPU option starts by removing the CPU one.

### Option A — CPU (works everywhere)
Nothing more to do. `requirements.txt` already installed the CPU version.

### Option B — NVIDIA GPU (CUDA, fastest)
```
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu[cuda,cudnn]"
```
If pip complains about `[cuda,cudnn]`, use this instead:
```
pip install onnxruntime-gpu nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
```
- `onnxruntime-gpu` needs the **CUDA 12** libraries. Never install packages ending
  in `-cu13` alongside it; they overwrite files and break loading.
- A system-wide CUDA install is not needed. The app finds the libraries from
  these pip packages by itself.
- These libraries are large (1–2 GB).

### Option C — Any GPU on Windows (DirectML: NVIDIA, AMD, Intel)
```
pip uninstall -y onnxruntime
pip install onnxruntime-directml
```
One small package, no CUDA needed. A bit slower than CUDA on NVIDIA cards, but
the easiest GPU option and the best choice for a version you share with others.

### Check what's detected
```
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
- `CUDAExecutionProvider` → Option B is working
- `DmlExecutionProvider` → Option C is working
- only `CPUExecutionProvider` → CPU mode

## 2. Run
```
.\venv\Scripts\Activate.ps1
python main.py
```
When the AI model loads, the log panel shows which device it's using:
- `model running on GPU (CUDA)` or `(Dml)`: GPU is working
- `GPU not usable, running on CPU instead (...)`: the GPU setup has a problem.
  The app keeps working on CPU; the message in brackets says what failed.

---

## How to use

- **Add files:** drag and drop images, GIFs, folders or ZIPs, or use the buttons.
  Select several files in the list to process only those.
- **Automatic:** just press *Remove backgrounds*. The AI finds the main subject.
- **Choose what to keep:** select an image and click *Choose what to keep…*
  (or double-click it). Subjects are detected and highlighted automatically:
  - click a subject to keep / remove it
  - drag a box to add something it missed
  - right-click to delete a subject
  - plain one-colour backgrounds are removed exactly without AI; busy
    backgrounds use the AI (*Re-detect with AI* forces it)
- **Background setting:**
  - *Auto-detect*: green/blue screens are keyed out, everything else uses AI
  - *Green / blue screen*: always colour-key
  - *Pick a colour…*: remove any solid colour. Use the swatch or *Pick from
    image*, adjust *Tolerance*, and watch the live preview. *Keep matching
    areas inside the subject* keeps e.g. white eyes on a white background.
  - *AI only*: never colour-key
- **Remove watermark / logo / text:** select a file and click the button. Paint
  over the watermark (or use *Box*), right-click to erase. Painting once covers
  every frame of an animation. *Watermark fill*:
  - *Best (AI)*: LaMa inpainting model, rebuilds the texture underneath so the
    area looks untouched. Downloads once (~200 MB).
  - *Fast (no AI)*: instant, fine on plain areas, can look smudged on patterns.
  Fills are smoothed across frames (no flicker) and matched to the image's grain.
  Pick *Keep background (clean-up only)* to only remove the watermark.
- **Remove leftover specks:** drops small floating bits of background the AI left.
- **Animations** keep their frame timing and looping. Output as WebP (default,
  smooth edges, small), APNG (smooth edges, larger) or GIF (works everywhere,
  hard edges only).
- **Output:** `name_no_bg.png` for stills, in the chosen output folder
  (default: `no_bg` next to the first file).

## Models and offline use

Models are stored in `models\` next to `main.py`, not in the venv, so deleting
or rebuilding the venv doesn't re-download them. The first run downloads the
model you use once:
- *Best quality* (BiRefNet): about 970 MB
- *Fast* (BiRefNet Lite): about 225 MB
- Watermark AI (LaMa): about 200 MB, stored in `models\lama\`

After that no internet is needed. To ship the app fully offline, run each model
once and include the `models\` folder.

**RAM:** the AI needs roughly 4–6 GB of free RAM on CPU. Use *Fast* on weaker
machines. Colour keying needs almost nothing.

## Troubleshooting

| Problem | Fix |
|---|---|
| `cudnn64_9.dll` or `cublasLt64_12.dll` missing | Uninstall every `onnxruntime*` and `nvidia-*` package, then redo Option B |
| Log says GPU not usable | App still works on CPU. Check the provider list above, or switch to Option C |
| Very slow on CPU | Use *Fast* model, or set up Option B / C |
| Out of memory | Use *Fast* model and close other programs |

## Licenses (all allow commercial use and redistribution)

| Component | License |
|---|---|
| BiRefNet weights, rembg, onnxruntime | MIT |
| LaMa inpainting model | Apache 2.0 |
| OpenCV | Apache 2.0 |
| Pillow | MIT-CMU (HPND) |
| NumPy | BSD |
| PySide6 (Qt) | LGPL v3 — fine when shipped as separate libraries (normal pip / PyInstaller one-folder builds); include the LGPL notice |
| NVIDIA CUDA / cuDNN (Option B only) | NVIDIA license — allows redistribution of these runtime libraries, but check the terms if you bundle them |

The code never uses rembg's default model (`bria-rmbg`), which is non-commercial.