import os
import sys
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base.dataset import Dataset



class Dataset_LUNA16(Dataset):
    def __init__(self, root_dir, config):
        super().__init__(config)

        self._data_list = []
        for sub_id in range(10):
            tag = f'subset{sub_id}'
            for path in glob(os.path.join(root_dir, 'raw', tag, '*.mhd')):
                name = os.path.basename(path).replace('.mhd', '')
                name = f'{tag}_{name}'
                self._data_list.append({
                    'name': name,
                    'path': path
                })
