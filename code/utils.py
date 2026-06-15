import yaml
import numpy as np
import SimpleITK as sitk        # 医学图像读写、空间信息处理。
from easydict import EasyDict   # 把字典转换成可以用 cfg.xxx 方式访问的对象



def load_config(path):
    ''' 读取 指定path下的yaml文件内容并解析成 Python 字典 '''
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return EasyDict(cfg)


def convert_cuda(item):
    '''定义一个函数，将字典中除指定键外的张量转成 CUDA 浮点张量'''
    for key in item.keys():
        if key not in ['name', 'dst_name']:
            item[key] = item[key].float().cuda()
    return item


def count_parameters(model):
    '''定义一个函数，用来统计模型参数数量'''
    return sum(p.numel() for p in model.parameters())   # model.parameters()：遍历模型所有参数张量。
                                                        # p.numel()：计算每个参数张量的元素个数。
                                                        # sum(...)：将所有参数元素数累加，得到模型总参数量。


def sitk_load(path, uint8=False, spacing_unit='mm'):
    '''
    定义一个函数，用 SimpleITK 加载医学图像，并返回图像数组、spacing、origin。

    uint8=False：如果为 True，表示输入图像是 uint8，会做归一化。
    spacing_unit='mm'：默认单位是毫米，支持 'm' 表示米。

    返回值：
    image: ITK对象，numpy 数组，shape [x, y, z]，dtype float32
    spacing: 体素间距数组，numpy 数组，shape [3]，dtype float32，单位 mm
    origin: 物理坐标原点数组，numpy 数组，shape [3]，dtype float32，单位 mm
    '''
    # load as float32
    itk_img = sitk.ReadImage(path)                              # 返回 ITK image 对象。
    spacing = np.array(itk_img.GetSpacing(), dtype=np.float32)  # 获取图像 spacing（体素间距），转换成 float32 数组。单位默认为 mm。
    origin = np.array(itk_img.GetOrigin(), dtype=np.float32)    # 获取图像 origin（物理坐标原点），转换成 float32 数组。
    if spacing_unit == 'm':                                     # 果指定单位是米，则把 spacing、origin 转换为毫米
        spacing *= 1000.
        origin *= 1000
    elif spacing_unit != 'mm':
        raise ValueError
    image = sitk.GetArrayFromImage(itk_img)                     # 从 ITK image 对象中提取图像数据，得到一个 numpy 数组，默认 shape 是 [z, y, x]。
    image = image.transpose(2, 1, 0)                            # to [x, y, z]
    image = image.astype(np.float32)                            # 转换成 float32 类型，方便后续处理。
    if uint8:
        # if data is saved as uint8, [0, 255] => [0, 1]，如果输入是 uint8 数据，则把像素值从 [0,255] 归一化到 [0,1]。
        image /= 255.
    return image, spacing, origin


def sitk_save(path, image, spacing=None, origin=None, uint8=False):
    '''
    定义一个函数，把 NumPy 图像写回文件。

    spacing、origin 可选，用于设置输出图像的空间信息。
    uint8=False：如果为 True，输出会转换回 uint8。
    '''
    # default: float32 (input)
    image = image.astype(np.float32)                            # 先把输入图像转成 float32。
    image = image.transpose(2, 1, 0)                            # 把数组从 [x, y, z] 变回 SimpleITK 常用的 [z, y, x] 顺序。

    if uint8:                                                   # 如果需要保存 uint8，则先把 [0,1] 还原到 [0,255]，再转成 uint8。
        # value range should be [0, 1]
        image = (image * 255).astype(np.uint8)
    out = sitk.GetImageFromArray(image)                         # 从 NumPy 数组创建一个 SimpleITK 图像对象。
    if spacing is not None:                                     # 如果提供了 spacing，则把它设置到输出图像。
        out.SetSpacing(spacing.astype(np.float64))              # unit: mm,转换成 float64，符合 SimpleITK 要求
    if origin is not None:
        out.SetOrigin(origin.astype(np.float64))                # unit: mm,如果提供了 origin，则设置输出图像的原点信息
    sitk.WriteImage(out, path)                                  # 把 SimpleITK 图像对象写回文件。
