import os
import argparse
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk


def sitk_save(path, image, spacing):
    spacing = np.array(spacing).astype(np.float64)
    out = sitk.GetImageFromArray(image)
    out.SetSpacing(spacing)
    sitk.WriteImage(out, path)


def generate_blocks(resolution, block_size):
    resolution = np.array(resolution)
    block_size = np.array(block_size)
    nx, ny, nz = block_size
    offsets = (resolution / block_size).astype(int)

    base = np.mgrid[:nx, :ny, :nz]
    base = base.reshape(3, -1).transpose(1, 0)
    base = base * offsets

    block_list = []
    for x in range(offsets[0]):
        for y in range(offsets[1]):
            for z in range(offsets[2]):
                block = base + np.array([x, y, z])
                block_list.append(block)

    blocks_coords = np.stack(block_list, axis=0)
    blocks_coords = blocks_coords / (resolution - 1)
    blocks_coords = blocks_coords.astype(np.float32)
    return blocks_coords, block_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 2: Convert resampled CTs to uint8, generate 64^3 blocks.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to resampled/ directory (output of resample.py)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Output directory for resampled_v2/ images (uint8, 1.6mm)')
    parser.add_argument('--block_dir', type=str, required=True,
                        help='Output directory for blocks/')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='1-based starting index for resume. E.g. --start_idx 400 will skip the first '
                             '399 sorted files and start from the 400th. Default: 0 (process all).')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.block_dir, exist_ok=True)

    resolution = [256, 256, 256]
    block_size = [64, 64, 64]
    spacing_out = [1.6, 1.6, 1.6]

    # Sort for deterministic ordering; resume relies on stable indices.
    file_names = sorted(os.listdir(args.data_dir))
    total = len(file_names)

    if args.start_idx > 0:
        if args.start_idx > total:
            raise ValueError(f'--start_idx {args.start_idx} exceeds total files ({total})')
        file_names = file_names[args.start_idx - 1:]
        print(f'Resume: skipping first {args.start_idx - 1} files, '
              f'processing {len(file_names)} remaining (indices {args.start_idx}–{total}).')
    else:
        print(f'Processing all {total} files (use --start_idx N to resume from the N-th file).')

    blocks_coords, block_list = generate_blocks(resolution, block_size)
    blocks_coords_saved = os.path.exists(os.path.join(args.block_dir, 'blocks_coords.npy'))

    for name in tqdm(file_names, ncols=50):
        stem = name.split('.')[0]

        # -- skip if already fully processed (resampled_v2 + all blocks exist) --
        save_path = os.path.join(args.save_dir, f'{stem}.nii.gz')
        first_block_path = os.path.join(args.block_dir, f'{stem}_block-0.npy')
        last_block_path = os.path.join(args.block_dir, f'{stem}_block-{len(block_list) - 1}.npy')
        if os.path.exists(save_path) and os.path.exists(first_block_path) and os.path.exists(last_block_path):
            continue

        path = os.path.join(args.data_dir, name)
        itk_img = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(itk_img)
        image = image.astype(np.float32)
        image = (image + 1000) / 2000
        image = (image * 255).astype(np.uint8)
        image = image.transpose(2, 1, 0)  # [x, y, z]

        # save block coordinates once
        if not blocks_coords_saved:
            np.save(os.path.join(args.block_dir, 'blocks_coords.npy'), blocks_coords)
            blocks_coords_saved = True

        # save block values
        for i, block in enumerate(block_list):
            block_vals = image[block[:, 0], block[:, 1], block[:, 2]]
            np.save(os.path.join(args.block_dir, f'{stem}_block-{i}.npy'), block_vals)

        # save resampled_v2 image (uint8, transposed back)
        sitk_save(save_path, image.transpose(2, 1, 0), spacing_out)
