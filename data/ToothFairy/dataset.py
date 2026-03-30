import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base.dataset import Dataset



class Dataset_Tooth(Dataset):
    def __init__(self, root_dir, config):
        super().__init__(config)

        names = os.listdir(os.path.join(root_dir, 'raw'))

        self._data_list = []
        for name in names:
            self._data_list.append({
                'name': name,
                'path': os.path.join(root_dir, f'raw/{name}/data.npy')
            })

    def _load_raw(self, data):
        image = np.load(data['path'])
        image = image.transpose(1, 2, 0)
        spacing = np.array([0.3, 0.3, 0.3])
        return {
            'name': data['name'],
            'image': image,
            'mask': np.ones_like(image),
            'spacing': spacing
        }
