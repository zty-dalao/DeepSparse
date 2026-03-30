import os
import sys
import yaml
import json
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset import Dataset_LUNA16
from base.projector import Projector
from base.saver import Saver


PATH_DICT = {
    'image': 'images/{}.nii.gz',
    'projs': 'projections/{}.pickle',
    'blocks_vals': 'blocks/{}_block-{}.npy',
    'blocks_coords': 'blocks/blocks_coords.npy',
}

META_PATHS = {
    'dataset_config': './config.yaml',
    'image': 'processed/images/{}.nii.gz',
    'projs': 'processed/projections/{}.pickle',
    'blocks_vals': 'processed/blocks/{}_block-{}.npy',
    'blocks_coords': 'processed/blocks/blocks_coords.npy',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process LUNA16 dataset.')
    parser.add_argument('-n', '--name', type=str, default=None,
                        help='Process a single case by name (recommended due to TIGRE memory constraints)')
    parser.add_argument('--root_dir', type=str, default='./',
                        help='Dataset root directory')
    args = parser.parse_args()

    root_dir = args.root_dir
    processed_dir = os.path.join(root_dir, 'processed')
    config_path = os.path.join(root_dir, 'config.yaml')

    saver = Saver(
        root_dir=processed_dir,
        path_dict=PATH_DICT,
        projs_vis=False
    )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset = Dataset_LUNA16(
        root_dir=root_dir,
        config=config['dataset']
    ).init_projector(Projector(config=config['projector']))

    if args.name is not None:
        dataset.filter_names([args.name])
        saver.save(dataset[0])
    else:
        for data in tqdm(dataset, ncols=50):
            saver.save(data)

    with open(os.path.join(root_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)

    info = {}
    info.update(META_PATHS)
    info['train'] = splits['train']
    info['eval'] = splits['eval']
    info['test'] = splits['test']

    with open(os.path.join(root_dir, 'meta_info.json'), 'w') as f:
        json.dump(info, f, indent=4)
