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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.block_dir, exist_ok=True)

    resolution = [256, 256, 256]
    block_size = [64, 64, 64]
    spacing_out = [1.6, 1.6, 1.6]

    blocks_coords, block_list = generate_blocks(resolution, block_size)
    blocks_coords_saved = False

    for name in tqdm(os.listdir(args.data_dir), ncols=50):
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
        stem = name.split('.')[0]
        for i, block in enumerate(block_list):
            block_vals = image[block[:, 0], block[:, 1], block[:, 2]]
            np.save(os.path.join(args.block_dir, f'{stem}_block-{i}.npy'), block_vals)

        # save resampled_v2 image (uint8, transposed back)
        save_path = os.path.join(args.save_dir, f'{stem}.nii.gz')
        sitk_save(save_path, image.transpose(2, 1, 0), spacing_out)
