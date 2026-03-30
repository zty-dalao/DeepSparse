import json
import os
import numpy as np
from pathlib import Path


def generate_splits():
    base_dir = Path(__file__).parent
    old_splits_file = base_dir / "train_test_old_splits.json"
    output_splits_file = base_dir / "splits.json"
    raw_dir = base_dir / "raw"
    names_file = raw_dir / "names.txt"

    raw_dir.mkdir(exist_ok=True)

    with open(old_splits_file, 'r') as f:
        old_splits = json.load(f)

    train_ids = old_splits["train_ids"]
    test_ids = old_splits["test_ids"]
    all_case_ids = train_ids + test_ids

    print(f"Total cases: {len(all_case_ids)} (old train: {len(train_ids)}, old test: {len(test_ids)})")

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(all_case_ids))
    shuffled_case_ids = [all_case_ids[i] for i in shuffled_indices]

    total_cases = len(shuffled_case_ids)
    eval_count = 200
    test_count = 600
    train_count = total_cases - eval_count - test_count

    new_train_ids = shuffled_case_ids[:train_count]
    new_eval_ids = shuffled_case_ids[train_count:train_count + eval_count]
    new_test_ids = shuffled_case_ids[train_count + eval_count:]

    print(f"New splits: train={len(new_train_ids)}, eval={len(new_eval_ids)}, test={len(new_test_ids)}")

    new_splits = {
        "train": new_train_ids,
        "eval": new_eval_ids,
        "test": new_test_ids,
        "seed": 42,
        "num_train": len(new_train_ids),
        "num_eval": len(new_eval_ids),
        "num_test": len(new_test_ids),
        "total": total_cases
    }

    with open(output_splits_file, 'w') as f:
        json.dump(new_splits, f, indent=2)
    print(f"Saved splits to: {output_splits_file}")

    with open(names_file, 'w') as f:
        for case_id in all_case_ids:
            f.write(f"{case_id}\n")
    print(f"Saved names to: {names_file}")


if __name__ == "__main__":
    generate_splits()
