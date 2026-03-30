import os
import scipy
import argparse
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk


def sitk_save(path, image, spacing):
    spacing = np.array(spacing).astype(np.float64)
    out = sitk.GetImageFromArray(image)
    out.SetSpacing(spacing)
    sitk.WriteImage(out, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 1: Resample raw CTs to 256^3 at 1.5mm spacing.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to raw AbdomenAtlas1.0Mini/ directory')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Output directory for resampled volumes')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    resample_size = [256, 256, 256]
    resample_spacing = [1.5, 1.5, 1.5]

    for name in tqdm(os.listdir(args.data_dir), ncols=50):
        path = os.path.join(args.data_dir, name, 'ct.nii.gz')
        try:
            itk_img = sitk.ReadImage(path)
        except Exception:
            print(f'Cannot open file {name}, skipping.')
            continue

        itk_img = sitk.DICOMOrient(itk_img, 'LPS')
        image = sitk.GetArrayFromImage(itk_img)
        image = image.transpose(2, 1, 0)  # [x, y, z]

        # for x/y: scale to fit
        scale_x = resample_size[0] / image.shape[0]
        scale_y = resample_size[1] / image.shape[1]

        # for z: crop/pad/split to 256
        image_list = []
        if image.shape[2] <= resample_size[2]:
            image_list.append(image)
            scale_z = resample_size[2] / image.shape[2]
        elif image.shape[2] <= resample_size[2] * 1.3:
            z_start = (image.shape[2] - resample_size[2]) // 2
            image_list.append(image[..., z_start:z_start + resample_size[2]])
            scale_z = 1
        else:
            cnt = int(np.ceil(image.shape[2] / resample_size[2]))
            offset = resample_size[2] - (resample_size[2] * cnt - image.shape[2]) / (cnt - 1)
            offset = int(offset)
            for i in range(cnt):
                z_start = offset * i
                image_list.append(image[..., z_start:z_start + resample_size[2]])
            scale_z = 1

        scale = [scale_x, scale_y, scale_z]
        for i, im in enumerate(image_list):
            im = scipy.ndimage.zoom(im, scale, order=3, prefilter=False)
            if len(image_list) == 1:
                save_path = os.path.join(args.save_dir, f'{name}.nii.gz')
            else:
                save_path = os.path.join(args.save_dir, f'{name}#{i}.nii.gz')
            im = im.transpose(2, 1, 0)
            sitk_save(save_path, im, resample_spacing)
