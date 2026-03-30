import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base.dataset import Dataset



class Dataset_PANORAMA(Dataset):
    def __init__(self, root_dir, config, process_mask=True):
        super().__init__(config, process_mask=process_mask)

        with open(os.path.join(root_dir, 'raw/names.txt'), 'r') as f:
            names = f.read().splitlines()

        self._data_list = []
        for name in names:
            self._data_list.append({
                'name': name,
                'path': os.path.join(root_dir, 'raw', 'images', f'{name}.nii.gz'),
                'mask_path': os.path.join(root_dir, 'raw', 'labels', f'{name}.nii.gz')
            })
