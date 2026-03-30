import os
import pickle
import numpy as np
from copy import deepcopy

from base.utils import sitk_save
from base.projector import visualize_projections



PATH_DICT = {
    'image': 'images/{}.nii.gz',
    'projs': 'projections/{}.pickle',
    'projs_vis': 'projections_vis/{}.png',
    'blocks_vals': 'blocks/{}_block-{}.npy',
    'blocks_coords': 'blocks/blocks_coords.npy'
}


class Saver:
    def __init__(self, root_dir, path_dict=None, projs_vis=True, save_mask=False):
        self._root_dir = root_dir

        if path_dict is None:
            path_dict = PATH_DICT
        self._rel_path_dict = deepcopy(path_dict)
        self._path_dict = deepcopy(path_dict)
        
        self._projs_vis = projs_vis
        self._is_blocks_coords_saved = False
        self._save_mask_flag = save_mask

        # create sub-folders and resolve to absolute paths for I/O
        for key in self._path_dict.keys():
            if key == 'projs_vis' and not self._projs_vis:
                continue
            path = os.path.join(root_dir, self._path_dict[key])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._path_dict[key] = path

    @property
    def path_dict(self):
        return self._rel_path_dict

    def _save_CT(self, data):
        name = data['name']
        sitk_save(
            self._path_dict['image'].format(name), 
            image=data['image'],
            spacing=data['spacing'],
            origin=data['origin'],
            uint8=True
        )
        
    def _save_mask(self, data):
        name = data['name']
        sitk_save(
            self._path_dict['seg_mask'].format(name), 
            image=data['mask'],
            spacing=data['spacing'],
            origin=data['origin'],
            uint8=False,
            image_type=np.uint8
        )

    def _save_blocks(self, data):
        if not self._is_blocks_coords_saved:
            np.save(self._path_dict['blocks_coords'], data['blocks_coords'])
            self._is_blocks_coords_saved = True

        for i, block in enumerate(data['blocks_vals']):
            save_path = self._path_dict['blocks_vals'].format(data['name'], i)
            block = (block * 255).astype(np.uint8)
            np.save(save_path, block)

    def _save_projs(self, data):
        projs = data['projs']
        projs_max = projs.max()

        projs = (projs / projs_max * 255).astype(np.uint8)
        with open(self._path_dict['projs'].format(data['name']), 'wb') as f:
            pickle.dump({
                'projs': projs,
                'projs_max': projs_max,
                'angles': data['angles']
            }, f, pickle.HIGHEST_PROTOCOL)

        if self._projs_vis:
            visualize_projections(
                self._path_dict['projs_vis'].format(data['name']), 
                data['projs'], data['angles']
            )

    def save(self, data):
        self._save_CT(data)
        if self._save_mask_flag:
            self._save_mask(data)
        self._save_blocks(data)
        self._save_projs(data)
