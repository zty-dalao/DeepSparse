# LUNA16_v2 (Finetuning — Lung)

LUNA16 dataset — lung CT scans used for finetuning DeepSparse.

- **Source**: [LUNA16 Challenge](https://luna16.grand-challenge.org/)
- **Training name**: `luna`

## Data Preparation

### 1 — Download and organize raw data

Download the LUNA16 dataset (10 subsets). Organize as:

```
LUNA16_v2/raw/
  subset0/*.mhd  (+ *.raw files)
  subset1/*.mhd
  ...
  subset9/*.mhd
  names.txt       list of all case names (one per line, format: subset{i}_{uid})
```

`names.txt` is auto-generated from `splits.json` or can be produced by listing all `.mhd` files across subsets.

### 2 — Process all cases

```bash
bash run.sh
```

This loops over `raw/names.txt` and calls `main.py -n {name}` per case (one at a time due to a known TIGRE memory issue). After all cases are processed, `meta_info.json` is updated automatically.

## Output File Structure

```
LUNA16_v2/
  config.yaml               Projector and dataset configuration
  meta_info.json            Split lists and relative data paths
  splits.json               Train/eval/test case ID lists
  processed/
    images/                 Processed CT volumes (.nii.gz, uint8)
    projections/            Cone-beam projections (.pickle)
    blocks/                 64³ block values + coords
```
