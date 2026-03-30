# PENGWIN (Finetuning — Pelvis)

PENGWIN challenge dataset — pelvic CT scans used for finetuning DeepSparse.

- **Source**: [PENGWIN Grand Challenge](https://pengwin.grand-challenge.org/)
- **Training name**: `pelvis`

## Data Preparation

### 1 — Download raw data

Download `PENGWIN_CT_train_images_part1.zip`, `PENGWIN_CT_train_images_part2.zip`, and `PENGWIN_CT_train_labels.zip` from the challenge. Unzip to obtain `.mha` files.

### 2 — Convert `.mha` to `.nii.gz`

```bash
python mha_to_nii_gz.py \
    --input_dir ./raw_mha \
    --output_dir ./raw \
    --mode convert_format \
    --num_workers 16
```

This converts the challenge `.mha` files to `.nii.gz` and organizes them into:

```
PENGWIN/raw/
  images/{case_id}.nii.gz
  labels/{case_id}.nii.gz
  names.txt
```

### 3 — Process all cases

```bash
bash run.sh
```

This loops over `raw/names.txt` and calls `main.py -n {case_id}` per case (one at a time due to a known TIGRE memory issue). After all cases are processed, `meta_info.json` is updated automatically.

## Output File Structure

```
PENGWIN/
  config.yaml               Projector and dataset configuration
  meta_info.json            Split lists and relative data paths
  splits.json               Train/eval/test case ID lists
  processed/
    images/                 Processed CT volumes (.nii.gz, uint8)
    projections/            Cone-beam projections (.pickle)
    projections_vis/        Projection visualizations (.png)
    blocks/                 64³ block values + coords
```
