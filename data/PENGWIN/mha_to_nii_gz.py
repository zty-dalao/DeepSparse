import SimpleITK as sitk
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

log_lock = Lock()


def safe_log(level, message):
    with log_lock:
        if level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)


def check_metadata_consistency(image, segmentation, case_id):
    inconsistencies = []
    
    img_spacing = image.GetSpacing()
    seg_spacing = segmentation.GetSpacing()
    if not np.allclose(img_spacing, seg_spacing, rtol=1e-5, atol=1e-8):
        inconsistencies.append(f"Spacing mismatch: image={img_spacing}, seg={seg_spacing}")
    
    img_origin = image.GetOrigin()
    seg_origin = segmentation.GetOrigin()
    if not np.allclose(img_origin, seg_origin, rtol=1e-5, atol=1e-8):
        inconsistencies.append(f"Origin mismatch: image={img_origin}, seg={seg_origin}")
    
    img_direction = image.GetDirection()
    seg_direction = segmentation.GetDirection()
    if not np.allclose(img_direction, seg_direction, rtol=1e-5, atol=1e-8):
        inconsistencies.append(f"Direction mismatch: image={img_direction}, seg={seg_direction}")
    
    img_size = image.GetSize()
    seg_size = segmentation.GetSize()
    if img_size != seg_size:
        inconsistencies.append(f"Size mismatch: image={img_size}, seg={seg_size}")
    
    return inconsistencies


def process_case_check(case_id, input_dir):
    try:
        image_path = input_dir / 'images' / f'{case_id}.mha'
        label_path = input_dir / 'labels' / f'{case_id}.mha'
        
        if not image_path.exists():
            safe_log('error', f"[{case_id}] Image file not found: {image_path}")
            return False
        
        if not label_path.exists():
            safe_log('error', f"[{case_id}] Label file not found: {label_path}")
            return False
        
        image = sitk.ReadImage(str(image_path))
        segmentation = sitk.ReadImage(str(label_path))
        
        inconsistencies = check_metadata_consistency(image, segmentation, case_id)
        
        if inconsistencies:
            safe_log('warning', f"[{case_id}] Metadata inconsistencies found:")
            for inc in inconsistencies:
                safe_log('warning', f"  - {inc}")
            return False
        else:
            safe_log('info', f"[{case_id}] Metadata consistent")
            return True
            
    except Exception as e:
        safe_log('error', f"[{case_id}] Error processing: {str(e)}")
        return False


def process_case_convert(case_id, input_dir, output_dir):
    try:
        image_path = input_dir / 'images' / f'{case_id}.mha'
        label_path = input_dir / 'labels' / f'{case_id}.mha'
        
        if not image_path.exists():
            safe_log('error', f"[{case_id}] Image file not found: {image_path}")
            return False
        
        if not label_path.exists():
            safe_log('error', f"[{case_id}] Label file not found: {label_path}")
            return False
        
        image = sitk.ReadImage(str(image_path))
        segmentation = sitk.ReadImage(str(label_path))
        
        inconsistencies = check_metadata_consistency(image, segmentation, case_id)
        
        if inconsistencies:
            safe_log('warning', f"[{case_id}] Metadata inconsistencies detected:")
            for inc in inconsistencies:
                safe_log('warning', f"  - {inc}")
            safe_log('warning', f"[{case_id}] Converting anyway, but please review!")
        
        output_image_dir = output_dir / 'images'
        output_label_dir = output_dir / 'labels'
        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        
        output_image_path = output_image_dir / f'{case_id}.nii.gz'
        output_label_path = output_label_dir / f'{case_id}.nii.gz'
        
        sitk.WriteImage(image, str(output_image_path))
        sitk.WriteImage(segmentation, str(output_label_path))
        
        safe_log('info', f"[{case_id}] Converted successfully")
        return True
        
    except Exception as e:
        safe_log('error', f"[{case_id}] Error converting: {str(e)}")
        return False


def get_case_ids(input_dir):
    images_dir = input_dir / 'images'
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    case_ids = []
    for mha_file in images_dir.glob('*.mha'):
        case_ids.append(mha_file.stem)
    
    return sorted(case_ids)


def main():
    parser = argparse.ArgumentParser(description='Convert MHA files to NII.GZ with metadata checking')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images/ and labels/ subdirectories')
    parser.add_argument('--output_dir', type=str, help='Output directory for converted files (required for convert_format mode)')
    parser.add_argument('--mode', type=str, required=True, choices=['check_meta', 'convert_format'], help='Operation mode')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    if args.mode == 'convert_format':
        if not args.output_dir:
            logger.error("--output_dir is required for convert_format mode")
            return
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    try:
        case_ids = get_case_ids(input_dir)
        logger.info(f"Found {len(case_ids)} cases to process")
        
        if len(case_ids) == 0:
            logger.warning("No .mha files found in images directory")
            return
        
        success_count = 0
        total_count = len(case_ids)
        
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            if args.mode == 'check_meta':
                futures = {executor.submit(process_case_check, case_id, input_dir): case_id 
                          for case_id in case_ids}
            else:
                futures = {executor.submit(process_case_convert, case_id, input_dir, output_dir): case_id 
                          for case_id in case_ids}
            
            for future in as_completed(futures):
                case_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                except Exception as e:
                    safe_log('error', f"[{case_id}] Unexpected error: {str(e)}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing complete: {success_count}/{total_count} cases successful")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
