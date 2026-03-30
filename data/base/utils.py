import numpy as np
import SimpleITK as sitk



def sitk_load(path, uint8=False, spacing_unit='mm', image_type=np.float32):
    # load as float32
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)
    origin = np.array(itk_img.GetOrigin(), dtype=np.float32)
    if spacing_unit == 'm':
        spacing *= 1000.
        origin *= 1000
    elif spacing_unit != 'mm':
        raise ValueError
    image = sitk.GetArrayFromImage(itk_img)
    image = image.transpose(2, 1, 0) # to [x, y, z]
    image = image.astype(image_type)
    if uint8:
        # if data is saved as uint8, [0, 255] => [0, 1]
        image /= 255.
    return image, spacing, origin


def sitk_save(path, image, spacing=None, origin=None, uint8=False, image_type=np.float32):
    # default: float32 (input)
    image = image.astype(image_type)
    image = image.transpose(2, 1, 0)
    if uint8:
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)
    if spacing is not None:
        out.SetSpacing(spacing.astype(np.float64)) # unit: mm
    if origin is not None:
        out.SetOrigin(origin.astype(np.float64)) # unit: mm
    sitk.WriteImage(out, path)


def check_range(dataset):
    from tqdm import tqdm
    
    # NOTE: set dataset.return_raw(True) to load raw data
    # just to check the range of the image size
    size_list = []
    for item in tqdm(dataset, ncols=50):
        image = item['image']
        spacing = item['spacing']
        shape = np.array(image.shape)
        image_size = shape * spacing
        size_list.append(image_size)
    
    size_list = np.stack(size_list, axis=0) # [N, 3]
    size_min = np.min(size_list, axis=0)
    size_max = np.max(size_list, axis=0)
    # TODO: histgram
    return size_min, size_max
