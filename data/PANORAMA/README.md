# PANORAMA (Finetuning — Abdomen)

PANORAMA challenge dataset — abdominal CT scans used for finetuning DeepSparse.

- **Source**: [PANORAMA Grand Challenge](https://panorama.grand-challenge.org/)
- **Training name**: `abdomen`

## Data Preparation

### 1 — Download and organize raw data

After downloading the PANORAMA challenge data, organize as:

```
PANORAMA/raw/
  images/{case_id}.nii.gz
  labels/{case_id}.nii.gz
  names.txt                   list of all case IDs (one per line)
```

If starting from scratch, run `generate_splits.py` to create `splits.json` and `raw/names.txt` from the challenge's original split file (`train_test_old_splits.json`):

```bash
python generate_splits.py
```

### 2 — Process all cases

Run `main.py` per case (recommended due to a known TIGRE memory issue where repeated calls slow down):

```bash
bash run.sh
```

This loops over `raw/names.txt` and calls:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py -n {case_id}
```

After all cases are processed, `meta_info.json` is automatically updated with the split lists and relative data paths.

## Output File Structure

```
PANORAMA/
  config.yaml               Projector and dataset configuration
  meta_info.json            Split lists and relative data paths
  splits.json               Train/eval/test case ID lists
  processed/
    images/                 Processed CT volumes (.nii.gz, uint8)
    projections/            Cone-beam projections (.pickle)
    projections_vis/        Projection visualizations (.png)
    blocks/                 64³ block values + coords
```
