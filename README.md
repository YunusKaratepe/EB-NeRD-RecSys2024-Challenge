# News Recommendation with Semantic Clustering

RecSys 2024 News Recommendation Challenge - Enhanced with Semantic Clustering Features

## Project Overview

This project implements a news recommendation system based on the 1st place solution of the RecSys 2024 Challenge, with additional semantic clustering features to improve recommendation quality. The system uses LightGBM with 100+ features including TF-IDF similarities, statistical features, and semantic clusters.

## Directory Structure

```
./                                # Project root (work from here)
├── run.bat                          # Command runner
├── run_multiple_seeds.bat           # Run experiments with multiple seeds
├── tasks.py                         # Task definitions
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── PROJE_GELISIM_RAPORU.md          # Project development report (Turkish)
├── experiments/                     # Training scripts
│   └── 015_train_third/            # Main experiment
│       ├── run.py                  # Training script with temporal split & seed management
│       ├── graph_features.py       # Graph-based features (disabled)
│       ├── semantic_cluster_features.py  # BERT-based clustering
│       ├── GRAPH_FEATURES_ANALYSIS.md    # Why graph features failed (Turkish)
│       └── exp/                    # Experiment configs
│           ├── base.yaml           # Base configuration
│           ├── medium067_001.yaml  # Medium baseline config
│           ├── small067_001.yaml   # Small dataset config
│           └── large067_001.yaml   # Large dataset config
├── features/                        # Feature extraction scripts (30+ feature types)
├── preprocess/                      # Data preprocessing
├── yamls/                           # Global configurations
│   ├── dir/
│   │   └── local.yaml              # Path configurations (relative paths)
│   └── exp/
│       └── base.yaml               # Base experiment settings
├── input/                           # Raw datasets (relative to project root)
│   ├── ebnerd_small/               # Small dataset
│   │   ├── articles.parquet
│   │   ├── train/
│   │   │   ├── behaviors.parquet
│   │   │   └── history.parquet
│   │   └── validation/
│   │       ├── behaviors.parquet
│   │       └── history.parquet
│   ├── ebnerd_medium/              # Medium dataset (10% of large)
│   └── ebnerd_large/               # Full large dataset
└── output/                          # Generated outputs
    ├── features/                   # Extracted features
    ├── preprocess/                 # Preprocessed datasets
    │   ├── make_candidate/        # Candidate generation results
    │   └── dataset067/            # Final datasets for training
    └── experiments/                # Training results
        └── 015_train_third/       # Organized by experiment
            ├── medium067_001_seed7_20251221_143045/    # Timestamped + seed
            ├── medium067_001_seed42_20251221_150122/   # Different seed
            └── medium067_001_seed123_20251221_153018/  # Another seed
```

## Setup Instructions

### 1. Environment Setup

**Conda Environment** (recommended):

```bash
# Navigate to project root
cd "Path to the project"

# Activate conda environment
.\activate.bat

# Or manually:
conda activate ./conda
```

**Required Packages**:

- Python 3.10+
- polars
- lightgbm
- scikit-learn
- numpy, pandas
- hydra-core

### 2. Dataset Preparation

**All datasets are in the input folder (relative to main directory):**

```
./input/
├── ebnerd_small/       # Small dataset (~260K train impressions)
├── ebnerd_medium/      # Medium dataset (~1.2M train impressions, 10% of large)
└── ebnerd_large/       # Large dataset (~12M train impressions)
```

**Dataset Location**: All datasets should be placed in `kami/input/` folder. The configuration file `yamls/dir/local.yaml` is set to use these paths.

**Option A: Use Small Dataset** (fastest, for testing)

- Path: `input/ebnerd_small/`
- Training time: ~5 minutes
- RAM: ~4-8GB

**Option B: Use Medium Dataset** (recommended for research)

- Path: `input/ebnerd_medium/`
- Training time: ~20-30 minutes
- RAM: ~8-16GB

**Option C: Use Full Large Dataset** (production, requires high-spec machine)

- Path: `input/ebnerd_large/`
- Training time: ~6-12 hours
- RAM: ~768GB

### 3. Path Configuration

All paths are configured in `yamls/dir/local.yaml` using **relative paths**:

```yaml
input_dir: input              # Relative to project root
output_dir: output            # Relative to project root
exp_dir: output/experiments
features_dir: output/features
preprocess_dir: output/preprocess
candidate_dir: output/preprocess/make_candidate
```

**Important**: All commands should be run from the project root directory.

**For Small Dataset**:

```bash
.\run.bat create-candidates --debug
.\run.bat create-features --debug
.\run.bat create-datasets --debug
.\run.bat train --debug
```

**For Medium Dataset**:

```bash
.\run.bat create-candidates --exp=medium
.\run.bat create-features --exp=medium
.\run.bat create-datasets --exp=medium
.\run.bat train --exp=medium067_001
```

**For Large Dataset**:

```bash
.\run.bat create-candidates
.\run.bat create-features
.\run.bat create-datasets
.\run.bat train
```

### Running with Different Seeds (for Significance Testing)

**Single run with custom seed**:

```bash
.\run.bat train --exp=medium067_001 --seed=42
```

**Multiple seeds for statistical significance**:

```bash
.\run_multiple_seeds.bat medium067_001 7 42 123
```

This will run the experiment 3 times with seeds 7, 42, and 123. Each run will be saved to a separate timestamped folder:

- `output/experiments/015_train_third/medium067_001_seed7_20251221_143045/`
- `output/experiments/015_train_third/medium067_001_seed42_20251221_150122/`
- `output/experiments/015_train_third/medium067_001_seed123_20251221_153018/`

## Dataset

| Dataset | Train Impressions | Validation Impressions | Articles | Users (train) | Users (val) |
| ------- | ----------------- | ---------------------- | -------- | ------------- | ----------- |
| Small   | ~260K             | ~270K                  | 125K     | ~69K          | ~77K        |
| Medium  | ~1.2M             | ~1.25M                 | 64K      | 342K          | 384K        |
| Large   | ~12M              | ~12.5M                 | 125K     | ~3.4M         | ~3.8M       |

* You can download the dataset from this drive link: https://drive.google.com/drive/folders/1G1iLHNx2enkdYMeTusaGgt8_rv8fSzFi?usp=sharing

* After you download the files, unzip them into the `input` folder in the project workspace.

## Recommended Workflow for Research

1. **Quick Testing** (30 minutes):
   
   - Use small dataset with `--debug` flag
   - Test different semantic clustering configs
   - Iterate quickly

2. **Full Evaluation** (2-3 hours):
   
   - Use medium dataset with `--exp=medium`
   - Train baseline + semantic variants
   - Compare results

## Additional Files

* `create_medium_dataset.py` file is used to create medium dataset. Since you are given this dataset you don't have to run this.

* Medium dataset that is provided is created to include 10% of the Large dataset.

* You can run `create_medium_dataset.py` with different `sample_ratio` 

## Citation & References

- **RecSys 2024 Challenge**: [Competition Link](https://www.recsyschallenge.com/2024/)
- **Original 1st Place Solution**: [GitHub](https://github.com/...)
- **Dataset**: Ekstra Bladet News Recommendation Dataset (EB-NeRD)

## Contact & Support

For any questions regarding the project, you can send an email to:

karatepe22@itu.edu.tr

---

**Last Updated**: December 21, 2025
