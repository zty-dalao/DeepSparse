import scipy
import numpy as np
from copy import deepcopy

from base.utils import sitk_load



class Dataset:
    def __init__(self, config, process_mask= False):
        self._data_list = []

        # processed: consistent spacing, resolution. this means the image also has the consistent size (mm)
        # 2 options: specify the spacing, or the image size (mm)
        # TODO: image size
        self._spacing = np.array(config['spacing'])

        # for cropping/padding
        # currently we process all data(sets) to 256^3
        self._resolution = np.array(config['resolution'])

        # for data normalization
        # generally, the minimum value is set to -1000 (air), 
        # the maximum value is chosen based on the value range of dataset
        self._value_range = np.array(config['value_range'])

        # to save volumetric images (CT) as blocks for faster data loading
        # not necessary
        self._block_size = np.array(config['block_size'])
        self._block_info = None

        # if true, return data without processing (i.e., return raw data)
        self._return_raw = False

        # if true, process the mask (e.g., resample, crop/pad, normalize)
        self._process_mask = process_mask

        # projector to simulate 2d projections
        # if not None, the projector will be applied to all data in the dataset
        self._projector = None

    def filter_names(self, names):
        new_list = []
        for item in self._data_list:
            if item['name'] in names:
                new_list.append(item)
        self._data_list = new_list
        return self

    def return_raw(self, flag):
        self._return_raw = flag
        return self
    
    def init_projector(self, projector):
        # if the projector is initialized here, it will be applied to all data in this dataset.
        self._projector = projector
        return self
    
    def _generate_blocks(self):
        if self._block_info is not None:
            return self._block_info
        
        nx, ny, nz = self._block_size
        assert (self._resolution % self._block_size).sum() == 0, \
            f'resolution {self._resolution} is not divisible by block_size {self._block_size}'
        offsets = (self._resolution / self._block_size).astype(int)

        base = np.mgrid[:nx, :ny, :nz] # [3, nx, ny, nz]
        base = base.reshape(3, -1).transpose(1, 0) # [*, 3]
        base = base * offsets
        
        block_list = []
        for x in range(offsets[0]):
            for y in range(offsets[1]):
                for z in range(offsets[2]):
                    block = base + np.array([x, y, z])
                    block_list.append(block)
        
        blocks_coords = np.stack(block_list, axis=0) # [N, *, 3]
        blocks_coords = blocks_coords / (self._resolution - 1) # coords starts from 0
        blocks_coords = blocks_coords.astype(np.float32)

        self._block_info = {
            'coords': blocks_coords,
            'list': block_list
        }
        return self._block_info
    
    def _convert_blocks(self, data):
        block_info = self._generate_blocks()
        blocks_vals = [
            data['image'][b[:, 0], b[:, 1], b[:, 2]]
            for b in block_info['list']
        ]
        data['blocks_vals'] = blocks_vals
        data['blocks_coords'] = block_info['coords']
        return data
    
    def _process(self, data):
        # data -> resample -> crop/pad -> normalize
        return self._normalize(
            self._crop_pad(
                self._resample(data)
            )
        )

    def _resample(self, data):
        # resample data (spacing)
        resample_ratio = data['spacing'] / self._spacing
        data['image'] = scipy.ndimage.zoom( 
            # numerical offsets depending on the value range 
            # NOTE: normalization first or later will (slightly) affect this
            data['image'], 
            resample_ratio, 
            order=3, 
            prefilter=False
        )
        data['spacing'] = deepcopy(self._spacing)
        if self._process_mask:
            data['mask'] = scipy.ndimage.zoom( 
            # numerical offsets depending on the value range 
            # NOTE: normalization first or later will (slightly) affect this
                data['mask'], 
                resample_ratio, 
                order=0, 
                prefilter=False
            )
        return data

    def _crop_pad(self, data):
        # crop or add padding (resolution)
        processed = []
        original = []
        if not self._process_mask:
            shape = data['image'].shape
            origin = []
            for i in range(3):
                if shape[i] >= self._resolution[i]:
                    # center crop
                    offset = (shape[i] - self._resolution[i]) // 2
                    origin.append(offset)
                    processed.append({
                        'left': 0,
                        'right': self._resolution[i]
                    })
                    original.append({
                        'left': offset,
                        'right': offset + self._resolution[i]
                    })
                else:
                    # padding
                    offset = (self._resolution[i] - shape[i]) // 2
                    origin.append(-offset)
                    processed.append({
                        'left': offset,
                        'right': offset + shape[i]
                    })
                    original.append({
                        'left': 0,
                        'right': shape[i]
                    })

            def slice_array(a, index_a, b, index_b):
                a[
                    index_a[0]['left']:index_a[0]['right'],
                    index_a[1]['left']:index_a[1]['right'],
                    index_a[2]['left']:index_a[2]['right']
                ] = b[
                    index_b[0]['left']:index_b[0]['right'],
                    index_b[1]['left']:index_b[1]['right'],
                    index_b[2]['left']:index_b[2]['right']
                ]
                return a
        else:
            # Process mask is True: crop/pad centered at foreground mask bbox ROI center
            mask = data['mask']
            shape = data['image'].shape
            
            # Find bounding box of foreground mask
            foreground_indices = np.where(mask > 0)
            if len(foreground_indices[0]) == 0:
                # No foreground, fall back to center crop/pad
                bbox_min = np.array([0, 0, 0])
                bbox_max = np.array(shape)
            else:
                bbox_min = np.array([
                    foreground_indices[0].min(),
                    foreground_indices[1].min(),
                    foreground_indices[2].min()
                ])
                bbox_max = np.array([
                    foreground_indices[0].max() + 1,
                    foreground_indices[1].max() + 1,
                    foreground_indices[2].max() + 1
                ])
            
            # Calculate ROI center
            roi_center = (bbox_min + bbox_max) // 2
            
            origin = []
            for i in range(3):
                # Calculate crop/pad centered at ROI center
                center_offset = roi_center[i] - self._resolution[i] // 2
                
                if center_offset < 0:
                    # Need padding on the left
                    pad_left = -center_offset
                    crop_start = 0
                    crop_end = min(shape[i], self._resolution[i] - pad_left)
                    
                    origin.append(0)
                    processed.append({
                        'left': pad_left,
                        'right': pad_left + (crop_end - crop_start)
                    })
                    original.append({
                        'left': crop_start,
                        'right': crop_end
                    })
                elif center_offset + self._resolution[i] > shape[i]:
                    # Need padding on the right
                    crop_start = max(0, shape[i] - self._resolution[i])
                    crop_end = shape[i]
                    pad_left = 0 if crop_start > 0 else (self._resolution[i] - shape[i])
                    
                    origin.append(crop_start)
                    processed.append({
                        'left': pad_left,
                        'right': pad_left + (crop_end - crop_start)
                    })
                    original.append({
                        'left': crop_start,
                        'right': crop_end
                    })
                else:
                    # Pure crop, no padding needed
                    crop_start = center_offset
                    crop_end = center_offset + self._resolution[i]
                    
                    origin.append(crop_start)
                    processed.append({
                        'left': 0,
                        'right': self._resolution[i]
                    })
                    original.append({
                        'left': crop_start,
                        'right': crop_end
                    })
            
            def slice_array(a, index_a, b, index_b):
                a[
                    index_a[0]['left']:index_a[0]['right'],
                    index_a[1]['left']:index_a[1]['right'],
                    index_a[2]['left']:index_a[2]['right']
                ] = b[
                    index_b[0]['left']:index_b[0]['right'],
                    index_b[1]['left']:index_b[1]['right'],
                    index_b[2]['left']:index_b[2]['right']
                ]
                return a
        
        # NOTE: 'mask' is used for evaluation with masked region, not used currently
        data['mask'] = slice_array( 
            np.zeros(self._resolution),
            processed,
            data['mask'] if self._process_mask else np.ones_like(data['image']),
            original
        )
        data['image'] = slice_array(
            np.full(self._resolution, fill_value=self._value_range[0], dtype=np.float32),
            processed,
            data['image'],
            original
        )
        # used as offsets to align the processed CT with original CT's annotations (e.g., segmentation masks, mesh, ...)
        data['origin'] = np.array(origin) * data['spacing'] 
        return data

    def _normalize(self, data):
        # NOTE: CT is measured in HU scale. air: -1000, water: 0
        # we do not convert HU to attenuation, mu = HU / 1000 * (mu_water - mu_air) + mu_water,
        # because linear operations will not affect the normalization.
        min_value, max_value = self._value_range
        image = data['image']
        image = np.clip(image, a_min=min_value, a_max=max_value)
        image = (image - min_value) / (max_value - min_value)
        data['image'] = image
        return data
    
    def _load_raw(self, data):
        image, spacing, _ = sitk_load(data['path'])
        
        if self._process_mask:
            mask, _, _ = sitk_load(data['mask_path'], image_type=np.uint8)
            data['mask'] = mask
            
        data = {
            'name': data['name'],
            'image': image,
            'mask': data.get('mask', None),
            'spacing': spacing
        }
        return data

    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, index):
        item = self._data_list[index]
        data = self._load_raw(item)

        if not self._return_raw:
            data = self._process(data)
            data = self._convert_blocks(data)

            if self._projector is not None:
                # NOTE: the (case-specific) projector can be defined outside the dataloader (not applied to all data)
                projs = self._projector(data['image'])
                data.update(projs) # keys: ['projs', 'angles']

        return data
