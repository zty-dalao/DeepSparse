# ToothFairy (Finetuning — Tooth)

ToothFairy challenge dataset — dental CT scans used for finetuning DeepSparse.

- **Source**: [ToothFairy Grand Challenge](https://toothfairy.grand-challenge.org/)
- **Training name**: `tooth`

## Data Preparation

### 1 — Download and organize raw data

Download the ToothFairy dataset. Each case is a folder containing `data.npy`:

```
ToothFairy/raw/
  {case_id}/
    data.npy    CT volume as numpy array, shape [D, H, W], spacing 0.3mm isotropic
```

### 2 — Process all cases

```bash
bash run.sh
```

This loops over all subdirectories in `raw/` and calls `main.py -n {case_id}` per case (one at a time due to a known TIGRE memory issue). After all cases are processed, `meta_info.json` is updated automatically.

## Output File Structure

```
ToothFairy/
  config.yaml               Projector and dataset configuration
  meta_info.json            Split lists and relative data paths
  splits.json               Train/eval/test case ID lists
  processed/
    images/                 Processed CT volumes (.nii.gz, uint8)
    projections/            Cone-beam projections (.pickle)
    projections_vis/        Projection visualizations (.png)
    blocks/                 64³ block values + coords
```
