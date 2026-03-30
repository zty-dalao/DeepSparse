# atlas-mini (Pretraining Dataset)

AbdomenAtlas1.0Mini — a large-scale public abdominal CT dataset used for pretraining DeepSparse.

- **Source**: [AbdomenAtlas1.0Mini](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini)
- **Training name**: `atlas-mini`

## Preprocessing Pipeline

Run the following 4 steps from inside the `preprocess/` directory. Intermediate data is stored under `preprocess/`.

### Step 1 — Resample raw CTs to 256³ at 1.5 mm spacing

```bash
python resample.py \
    --data_dir /path/to/AbdomenAtlas1.0Mini \
    --save_dir ./resampled
```

Long volumes are automatically split into overlapping 256³ chunks (named `{case}#{i}.nii.gz`).

### Step 2 — Convert to uint8 and generate 64³ blocks

```bash
python convert_blocks.py \
    --data_dir ./resampled \
    --save_dir ./resampled_v2 \
    --block_dir ./blocks
```

Outputs:
- `resampled_v2/{name}.nii.gz` — uint8 CT at 1.6 mm spacing
- `blocks/{name}_block-{i}.npy` — 64³ block values (uint8)
- `blocks/blocks_coords.npy` — shared block coordinates (float32)

### Step 3 — Generate cone-beam X-ray projections

```bash
cd preprocess/
bash run_project.sh
```

Or per-case (recommended for large datasets):

```bash
CUDA_VISIBLE_DEVICES=0 python project.py \
    -n {name}.nii.gz \
    --data_dir ./resampled \
    --save_dir ./projections \
    --vis_dir ./projections_vis \
    --config ../config.yaml
```

Uses TIGRE for GPU-accelerated cone-beam CT forward projection (200 angles, 0–360°).

### Step 4 — Create train/eval/test splits and write `meta_info.json`

```bash
python split.py \
    --raw_dir /path/to/AbdomenAtlas1.0Mini \
    --resampled_dir ./resampled_v2 \
    --save_path ../meta_info.json
```

Splits: 10% test, 30 eval, remainder train (seed=0).

## Output File Structure

```
atlas-mini/
  config.yaml               Projector and dataset configuration
  meta_info.json            Split lists and relative data paths
  preprocess/
    resampled/              Float32 CTs at 1.5mm (intermediate)
    resampled_v2/           Uint8 CTs at 1.6mm
    blocks/                 64³ blocks + coords
    projections/            .pickle files (projs, projs_max, angles)
    projections_vis/        Projection visualizations (.png)
```

## Projector Configuration (`config.yaml`)

Cone-beam CT geometry: DSD=1200mm, DSO=1000mm, 256×256 detector, 200 projections over 360°.
