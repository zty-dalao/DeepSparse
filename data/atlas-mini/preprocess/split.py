import os
import json
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 4: Create train/eval/test splits and write meta_info.json.')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Path to raw AbdomenAtlas1.0Mini/ directory (used to enumerate case names)')
    parser.add_argument('--resampled_dir', type=str, required=True,
                        help='Path to resampled_v2/ directory (to enumerate processed names with # splits)')
    parser.add_argument('--save_path', type=str, default='../meta_info.json',
                        help='Output path for meta_info.json')
    args = parser.parse_args()

    np.random.seed(0)

    names = os.listdir(args.raw_dir)
    np.random.shuffle(names)

    num_test = int(len(names) * 0.1)
    num_eval = 30

    test_list = names[:num_test]
    eval_list = names[num_test:num_test + num_eval]

    test_ids, eval_ids, train_ids = [], [], []
    for fname in os.listdir(args.resampled_dir):
        stem = fname.split('.')[0]
        base_name = stem.split('#')[0]
        if base_name in test_list:
            test_ids.append(stem)
        elif base_name in eval_list:
            eval_ids.append(stem)
        else:
            train_ids.append(stem)

    test_ids = sorted(test_ids)
    eval_ids = sorted(eval_ids)
    train_ids = sorted(train_ids)

    print(f'train: {len(train_ids)}, eval: {len(eval_ids)}, test: {len(test_ids)}')

    info = {
        'dataset_config': './config.yaml',
        'image': 'preprocess/resampled_v2/{}.nii.gz',
        'projs': 'preprocess/projections/{}.pickle',
        'blocks_vals': 'preprocess/blocks/{}_block-{}.npy',
        'blocks_coords': 'preprocess/blocks/blocks_coords.npy',
        'train': train_ids,
        'eval': eval_ids,
        'test': test_ids,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(info, f, indent=4)

    print(f'Saved meta_info.json to {args.save_path}')
