# Data Preprocessing

This directory contains preprocessing scripts for all datasets used in DeepSparse.

## Structure

```
data/
  base/           Shared utilities (projector, dataset, saver, utils)
  atlas-mini/     Pretraining dataset (AbdomenAtlas1.0Mini)
  PANORAMA/       Finetuning — abdomen (PANORAMA challenge)
  PENGWIN/        Finetuning — pelvis  (PENGWIN challenge)
  LUNA16_v2/      Finetuning — lung    (LUNA16)
  ToothFairy/     Finetuning — tooth   (ToothFairy challenge)
```

## Shared Utilities (`base/`)

- `projector.py` — TIGRE-based cone-beam CT forward projector
- `dataset.py`   — Generic Dataset class: resample → crop/pad → normalize → block conversion
- `saver.py`     — Saves processed CT, blocks, and projections with consistent folder structure
- `utils.py`     — SimpleITK load/save helpers

## Dataset Naming in Training

The `dataset_img_dir_dict` in `code/datasets/base.py` maps training dataset names to these folders:

| Training name | Folder        |
|---------------|---------------|
| `atlas-mini`  | `atlas-mini/` |
| `abdomen`     | `PANORAMA/`   |
| `pelvis`      | `PENGWIN/`    |
| `luna`        | `LUNA16_v2/`  |
| `tooth`       | `ToothFairy/` |

## Dependencies

```
pip install SimpleITK scipy numpy tigre matplotlib tqdm
```

See each dataset's `README.md` for dataset-specific download and preprocessing instructions.
