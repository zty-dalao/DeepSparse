"""
重新计算 PSNR：使用训练时的真实预处理 —— clip[-1000,1000] → normalize[0,1]。
这样算出的 PSNR 才对应模型实际看到的信号。

同时对比:
  1. 原始 HU（不裁剪，不归一化）—— 之前的计算方式
  2. 训练预处理（clip + normalize）—— 模型实际看到的
"""

import numpy as np
import nibabel as nib
import os
import glob

OVERLAP_DIR = "/home/zty20020112/workspace/DeepSparse/data/paired/processed/overlap"
NUM_CASES = 30
VALUE_RANGE = (-1000, 1000)  # HU clip range


def compute_psnr(img1, img2, mask=None, data_range=1.0):
    """PSNR = 10 * log10(data_range^2 / MSE)"""
    if mask is not None:
        img1 = img1[mask > 0]
        img2 = img2[mask > 0]
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def preprocess(arr, value_range):
    """训练时的预处理：clip + normalize to [0,1]"""
    min_v, max_v = value_range
    arr = np.clip(arr, min_v, max_v)
    arr = (arr - min_v) / (max_v - min_v)
    return arr


def main():
    all_files = sorted(glob.glob(os.path.join(OVERLAP_DIR, "*_ct.nii.gz")))
    case_ids = [os.path.basename(f).replace("_ct.nii.gz", "") for f in all_files]
    case_ids = case_ids[:NUM_CASES]

    print(f"{'Case':<30} {'Raw HU':>10} {'Raw HU':>10}  |  {'Train':>10} {'Train':>10}")
    print(f"{'':30} {'no mask':>10} {'w/ mask':>10}  |  {'no mask':>10} {'w/ mask':>10}")
    print("-" * 80)

    raw_no_mask = []
    raw_with_mask = []
    train_no_mask = []
    train_with_mask = []

    for i, case_id in enumerate(case_ids):
        ct = nib.load(os.path.join(OVERLAP_DIR, f"{case_id}_ct.nii.gz")).get_fdata().astype(np.float32)
        cbct = nib.load(os.path.join(OVERLAP_DIR, f"{case_id}_cbct.nii.gz")).get_fdata().astype(np.float32)
        cbct_mask = nib.load(os.path.join(OVERLAP_DIR, f"{case_id}_cbct_mask.nii.gz")).get_fdata().astype(np.uint8)
        ct_mask = nib.load(os.path.join(OVERLAP_DIR, f"{case_id}_ct_mask.nii.gz")).get_fdata().astype(np.uint8)
        combined_mask = (cbct_mask > 0) & (ct_mask > 0)

        # ── 方式1: 原始 HU ──
        data_range_raw = ct.max() - ct.min()
        r_nm = compute_psnr(ct, cbct, mask=None, data_range=data_range_raw)
        r_wm = compute_psnr(ct, cbct, mask=combined_mask, data_range=data_range_raw)
        raw_no_mask.append(r_nm)
        raw_with_mask.append(r_wm)

        # ── 方式2: 训练预处理 (clip[-1000,1000] → [0,1]) ──
        ct_norm = preprocess(ct, VALUE_RANGE)
        cbct_norm = preprocess(cbct, VALUE_RANGE)
        t_nm = compute_psnr(ct_norm, cbct_norm, mask=None, data_range=1.0)
        t_wm = compute_psnr(ct_norm, cbct_norm, mask=combined_mask, data_range=1.0)
        train_no_mask.append(t_nm)
        train_with_mask.append(t_wm)

        print(f"{case_id:<30} {r_nm:10.4f} {r_wm:10.4f}  |  {t_nm:10.4f} {t_wm:10.4f}")

    # ── 汇总 ──
    raw_nm_arr = np.array(raw_no_mask)
    raw_wm_arr = np.array(raw_with_mask)
    train_nm_arr = np.array(train_no_mask)
    train_wm_arr = np.array(train_with_mask)

    print("\n" + "=" * 80)
    print(f"{'汇总 (30 cases)':<30} {'原始 HU':>10} {'原始 HU':>10}  |  {'训练预处理':>10} {'训练预处理':>10}")
    print(f"{'':30} {'no mask':>10} {'w/ mask':>10}  |  {'no mask':>10} {'w/ mask':>10}")
    print("-" * 80)
    print(f"{'Mean':<30} {raw_nm_arr.mean():10.4f} {raw_wm_arr.mean():10.4f}  |  {train_nm_arr.mean():10.4f} {train_wm_arr.mean():10.4f}")
    print(f"{'Std':<30} {raw_nm_arr.std():10.4f} {raw_wm_arr.std():10.4f}  |  {train_nm_arr.std():10.4f} {train_wm_arr.std():10.4f}")
    print(f"{'Min':<30} {raw_nm_arr.min():10.4f} {raw_wm_arr.min():10.4f}  |  {train_nm_arr.min():10.4f} {train_wm_arr.min():10.4f}")
    print(f"{'Max':<30} {raw_nm_arr.max():10.4f} {raw_wm_arr.max():10.4f}  |  {train_nm_arr.max():10.4f} {train_wm_arr.max():10.4f}")

    print(f"\n差异 (训练预处理 - 原始 HU):")
    print(f"  No mask:  {train_nm_arr.mean() - raw_nm_arr.mean():+.4f} dB")
    print(f"  With mask: {train_wm_arr.mean() - raw_wm_arr.mean():+.4f} dB")


if __name__ == "__main__":
    main()
