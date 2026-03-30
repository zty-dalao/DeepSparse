import os
import sys
import yaml
import pickle
import argparse
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from base.projector import Projector, visualize_projections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 3: Generate cone-beam X-ray projections.')
    parser.add_argument('-n', '--name', type=str, required=True,
                        help='Filename (e.g. patient001.nii.gz) in data_dir')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to resampled/ directory (output of resample.py)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Output directory for projection .pickle files')
    parser.add_argument('--vis_dir', type=str, default=None,
                        help='Output directory for projection visualizations (optional)')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config.yaml')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    if args.vis_dir is not None:
        os.makedirs(args.vis_dir, exist_ok=True)

    stem = args.name.split('.')[0]

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    projector = Projector(config=config['projector'])

    path = os.path.join(args.data_dir, args.name)
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0)
    image = image.astype(np.float32)
    image = (image + 1000) / 2000

    data = projector(image)

    projs = data['projs']
    projs_max = projs.max()
    projs_max = np.ceil(projs_max * 100) / 100
    projs = (projs / projs_max * 255).astype(np.uint8)

    with open(os.path.join(args.save_dir, f'{stem}.pickle'), 'wb') as f:
        pickle.dump({
            'projs': projs,
            'projs_max': projs_max,
            'angles': data['angles']
        }, f, pickle.HIGHEST_PROTOCOL)

    if args.vis_dir is not None:
        idx = np.linspace(1, len(data['angles']), 50).astype(int) - 1
        visualize_projections(
            os.path.join(args.vis_dir, f'{stem}.png'),
            data['projs'][idx], data['angles'][idx]
        )
