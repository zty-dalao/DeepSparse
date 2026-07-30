"""
计算 overlap 数据中 CBCT 和 CT 之间的 PSNR。
1. 不用 mask，计算 30 个 case 的 PSNR 并取平均
2. 用 mask（cbct_mask & ct_mask 的交集），计算 30 个 case 的 PSNR 并取平均
"""

import numpy as np
import nibabel as nib
import os
import glob

OVERLAP_DIR = "/home/zty20020112/workspace/DeepSparse/data/paired/processed/overlap"
NUM_CASES = 30


def compute_psnr(img1, img2, mask=None, data_range=None):
    """
    计算 PSNR。
    PSNR = 10 * log10(data_range^2 / MSE)

    Args:
        img1, img2: numpy arrays (CT reference, CBCT)
        mask: optional binary mask
        data_range: max possible value range. If None, computed as max(img1) - min(img1)
    """
    if mask is not None:
        img1 = img1[mask > 0]
        img2 = img2[mask > 0]

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

    if data_range is None:
        data_range = img1.max() - img1.min()

    if mse == 0:
        return float("inf")

    psnr = 10.0 * np.log10((data_range ** 2) / mse)
    return psnr


def main():
    # 获取所有唯一的 case ID
    all_files = sorted(glob.glob(os.path.join(OVERLAP_DIR, "*_ct.nii.gz")))
    case_ids = []
    for f in all_files:
        basename = os.path.basename(f)
        case_id = basename.replace("_ct.nii.gz", "")
        case_ids.append(case_id)

    print(f"总共有 {len(case_ids)} 个 case")
    print(f"取前 {NUM_CASES} 个进行计算\n")

    case_ids = case_ids[:NUM_CASES]

    psnr_no_mask_list = []
    psnr_with_mask_list = []

    # 用于统计 data_range
    data_ranges = []

    for i, case_id in enumerate(case_ids):
        cbct_path = os.path.join(OVERLAP_DIR, f"{case_id}_cbct.nii.gz")
        ct_path = os.path.join(OVERLAP_DIR, f"{case_id}_ct.nii.gz")
        cbct_mask_path = os.path.join(OVERLAP_DIR, f"{case_id}_cbct_mask.nii.gz")
        ct_mask_path = os.path.join(OVERLAP_DIR, f"{case_id}_ct_mask.nii.gz")

        # 加载数据
        cbct = nib.load(cbct_path).get_fdata().astype(np.float32)
        ct = nib.load(ct_path).get_fdata().astype(np.float32)
        cbct_mask = nib.load(cbct_mask_path).get_fdata().astype(np.uint8)
        ct_mask = nib.load(ct_mask_path).get_fdata().astype(np.uint8)

        print(f"[{i+1:2d}/{NUM_CASES}] {case_id}")
        print(f"    CT  range: [{ct.min():.1f}, {ct.max():.1f}]")
        print(f"    CBCT range: [{cbct.min():.1f}, {cbct.max():.1f}]")

        # 使用 CT 的 data_range 作为参考
        data_range = ct.max() - ct.min()
        data_ranges.append(data_range)

        # 1. 不用 mask 的 PSNR
        psnr_no_mask = compute_psnr(ct, cbct, mask=None, data_range=data_range)
        psnr_no_mask_list.append(psnr_no_mask)
        print(f"    PSNR (no mask):     {psnr_no_mask:.4f} dB")

        # 2. 用 mask（cbct_mask & ct_mask 的交集）
        combined_mask = (cbct_mask > 0) & (ct_mask > 0)
        mask_ratio = combined_mask.sum() / combined_mask.size
        print(f"    Mask 覆盖率: {mask_ratio:.4f} ({combined_mask.sum()}/{combined_mask.size})")

        psnr_with_mask = compute_psnr(ct, cbct, mask=combined_mask, data_range=data_range)
        psnr_with_mask_list.append(psnr_with_mask)
        print(f"    PSNR (with mask):  {psnr_with_mask:.4f} dB")
        print()

    # 汇总结果
    print("=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"Data range 平均值 (CT): {np.mean(data_ranges):.2f} ± {np.std(data_ranges):.2f}")

    # 过滤掉 inf（如果有完全相同的图像）
    psnr_no_mask_arr = np.array([p for p in psnr_no_mask_list if np.isfinite(p)])
    psnr_with_mask_arr = np.array([p for p in psnr_with_mask_list if np.isfinite(p)])

    print(f"\n无 Mask PSNR ({len(psnr_no_mask_arr)} cases):")
    print(f"  Mean:  {np.mean(psnr_no_mask_arr):.4f} dB")
    print(f"  Std:   {np.std(psnr_no_mask_arr):.4f} dB")
    print(f"  Min:   {np.min(psnr_no_mask_arr):.4f} dB")
    print(f"  Max:   {np.max(psnr_no_mask_arr):.4f} dB")

    print(f"\n有 Mask PSNR ({len(psnr_with_mask_arr)} cases):")
    print(f"  Mean:  {np.mean(psnr_with_mask_arr):.4f} dB")
    print(f"  Std:   {np.std(psnr_with_mask_arr):.4f} dB")
    print(f"  Min:   {np.min(psnr_with_mask_arr):.4f} dB")
    print(f"  Max:   {np.max(psnr_with_mask_arr):.4f} dB")

    # 额外：使用固定 MAX=2000（HU 范围[-1000, 1000]）
    print(f"\n--- 使用固定 data_range=2000 (HU [-1000,1000]) ---")
    psnr_no_mask_2000 = []
    psnr_with_mask_2000 = []
    for case_id in case_ids:
        ct_path = os.path.join(OVERLAP_DIR, f"{case_id}_ct.nii.gz")
        cbct_path = os.path.join(OVERLAP_DIR, f"{case_id}_cbct.nii.gz")
        ct = nib.load(ct_path).get_fdata().astype(np.float32)
        cbct = nib.load(cbct_path).get_fdata().astype(np.float32)
        cbct_mask = nib.load(os.path.join(OVERLAP_DIR, f"{case_id}_cbct_mask.nii.gz")).get_fdata().astype(np.uint8)
        ct_mask = nib.load(os.path.join(OVERLAP_DIR, f"{case_id}_ct_mask.nii.gz")).get_fdata().astype(np.uint8)
        combined_mask = (cbct_mask > 0) & (ct_mask > 0)

        psnr_no_mask_2000.append(compute_psnr(ct, cbct, mask=None, data_range=2000))
        psnr_with_mask_2000.append(compute_psnr(ct, cbct, mask=combined_mask, data_range=2000))

    psnr_no_mask_2000 = np.array([p for p in psnr_no_mask_2000 if np.isfinite(p)])
    psnr_with_mask_2000 = np.array([p for p in psnr_with_mask_2000 if np.isfinite(p)])

    print(f"无 Mask (data_range=2000): Mean={np.mean(psnr_no_mask_2000):.4f} dB")
    print(f"有 Mask (data_range=2000): Mean={np.mean(psnr_with_mask_2000):.4f} dB")


if __name__ == "__main__":
    main()
