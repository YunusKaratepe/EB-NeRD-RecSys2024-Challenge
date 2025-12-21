import polars as pl
from pathlib import Path

dataset_path = Path("D:/ITU/phd/data-mining/project/EB-NeRD/dataset/ebnerd_large")

print("Checking date ranges...\n")

# Training
train = pl.read_parquet(dataset_path / "train" / "behaviors.parquet")
train_min = train["impression_time"].min()
train_max = train["impression_time"].max()
print(f"TRAINING:")
print(f"  Min: {train_min}")
print(f"  Max: {train_max}")
print(f"  Count: {len(train):,} impressions\n")

# Validation
val = pl.read_parquet(dataset_path / "validation" / "behaviors.parquet")
val_min = val["impression_time"].min()
val_max = val["impression_time"].max()
print(f"VALIDATION:")
print(f"  Min: {val_min}")
print(f"  Max: {val_max}")
print(f"  Count: {len(val):,} impressions\n")

# Check overlap
if val_min > train_max:
    print(f"✓ Validation comes AFTER training (no overlap)")
    print(f"  Time gap: {(val_min - train_max)}")
elif train_min > val_max:
    print(f"✗ Training comes AFTER validation (unusual!)")
else:
    print(f"⚠ Training and validation periods OVERLAP")
