from pathlib import Path


def get_data_dirs(input_dir, size_name="large"):
    # For training, we only need train and validation
    # Use validation as test if actual test set is not available
    test_path = input_dir / "ebnerd_testset" / "ebnerd_testset" / "test"
    if not test_path.exists():
        # Use validation as test for development
        test_path = input_dir / f"ebnerd_{size_name}" / "validation"
    
    return {
        "train": input_dir / f"ebnerd_{size_name}" / "train",
        "validation": input_dir / f"ebnerd_{size_name}" / "validation",
        "test": test_path,
    }
