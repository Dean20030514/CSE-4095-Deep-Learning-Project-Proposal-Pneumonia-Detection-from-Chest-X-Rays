# CXR Pneumonia Detection

A compact PyTorch project for pneumonia detection from chest X-rays. It includes training/validation scripts, Grad-CAM explainability, and a Streamlit demo app.

> For research and education only. Not for clinical use.

## Structure

```text
cxr-pneumonia/
├─ data/
│  ├─ raw/                # raw data (read-only)
│  ├─ processed/          # cleaned/resampled data
│  └─ external/           # optional external sources (e.g., NIH)
├─ src/
│  ├─ configs/            # training configs (YAML)
│  ├─ datasets.py
│  ├─ transforms.py
│  ├─ models.py
│  ├─ losses.py
│  ├─ train.py
│  ├─ validate.py
│  ├─ infer.py
│  ├─ explain.py          # Grad-CAM
│  └─ utils.py
├─ experiments/           # logs, checkpoints, curves
├─ app/
│  └─ streamlit_app.py    # demo app
├─ requirements.txt
└─ README.md / MODEL_CARD.md
```

## Setup

- Python 3.10+
- Install dependencies (choose CPU-only or matching your CUDA version).

```bash
pip install -r cxr-pneumonia/requirements.txt
# For torch with CUDA, see: https://pytorch.org/get-started/locally/
```

## Data preparation

You can use either CSV files listing images and labels or a folder with two subfolders `normal/` and `pneumonia/`.

- CSV format (recommended): columns `path,label` (0=normal, 1=pneumonia).
- Folder format: `data/processed/images/normal/*.png` and `data/processed/images/pneumonia/*.png`.

Update paths in `src/configs/default.yaml` accordingly.

### Datasets (not in Git)

This repository does not version datasets. Place data locally under `cxr-pneumonia/data/`, which is ignored by Git:

- `cxr-pneumonia/data/raw/` — original datasets (read-only)
- `cxr-pneumonia/data/processed/` — preprocessed files (generated)

For example, if you use Kaggle "Chest X-Ray Images (Pneumonia)", unpack it so that you get:

```text
cxr-pneumonia/data/raw/chest_xray/
├─ train/
│  ├─ NORMAL/*.jpeg
│  └─ PNEUMONIA/*.jpeg
├─ val/
│  ├─ NORMAL/*.jpeg
│  └─ PNEUMONIA/*.jpeg
└─ test/
  ├─ NORMAL/*.jpeg
  └─ PNEUMONIA/*.jpeg
```

Then point the config or CLI flags to the proper paths (see `src/configs/default.yaml`).

### Download from Kaggle (Windows PowerShell)

If you prefer to download automatically via Kaggle API (requires a Kaggle account):

1) Create API token on Kaggle: Account settings → "Create New Token". This downloads `kaggle.json`.
2) Place the file at `%USERPROFILE%\.kaggle\kaggle.json` and ensure permissions are user-only.
3) Install the CLI and download/unzip into `data/raw/`:

```powershell
pip install kaggle ; `
mkdir -Force cxr-pneumonia\data\raw ; `
cd cxr-pneumonia\data\raw ; `
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -f chest_xray.zip -p . ; `
tar -xf chest_xray.zip ; `
del chest_xray.zip
```

You should end up with the structure:

```text
cxr-pneumonia/data/raw/chest_xray/
├─ train/NORMAL, train/PNEUMONIA
├─ val/NORMAL,   val/PNEUMONIA
└─ test/NORMAL,  test/PNEUMONIA
```

Now you can proceed with the quick EDA and split steps below.

## Train

```bash
python cxr-pneumonia/src/train.py --config cxr-pneumonia/src/configs/default.yaml
```

Checkpoints and logs will be saved under `cxr-pneumonia/experiments/run-YYYYmmdd-HHMMSS/`.

## Validate

```bash
python cxr-pneumonia/src/validate.py --config cxr-pneumonia/src/configs/default.yaml \
  --weights cxr-pneumonia/experiments/run-*/best.ckpt \
  --csv data/processed/val.csv
```

## Inference

```bash
python cxr-pneumonia/src/infer.py --config cxr-pneumonia/src/configs/default.yaml \
  --weights cxr-pneumonia/experiments/run-*/best.ckpt \
  --input path/to/image_or_folder \
  --output predictions.csv
```

## Grad-CAM

```bash
python cxr-pneumonia/src/explain.py --config cxr-pneumonia/src/configs/default.yaml \
  --weights cxr-pneumonia/experiments/run-*/best.ckpt \
  --image path/to/image.png
```

This writes `*_gradcam.png` next to the input image.

## Streamlit demo

```bash
streamlit run cxr-pneumonia/app/streamlit_app.py
```

In the left sidebar, set the weights path (e.g., `cxr-pneumonia/experiments/run-*/best.ckpt`), choose backbone, and upload an image.

## Tips

- Models: `resnet18`, `resnet34`, `efficientnet_b0`.
- Loss: BCEWithLogits (default) or Focal loss.
- Image size: default 224; for higher fidelity try 320–384.
- If you don't have a CSV for validation, the training script will split 10% from the training data automatically.

## License & Responsible AI

Data may be subject to their own licenses (e.g., Kaggle datasets, NIH CXR14). Ensure compliance. The provided code is not a medical device and should not be used for diagnosis.

For model details and intended use, see `MODEL_CARD.md`.
