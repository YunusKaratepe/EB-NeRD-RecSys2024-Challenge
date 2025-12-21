# News Recommendation with Semantic Clustering

RecSys 2024 News Recommendation Challenge - Enhanced with Semantic Clustering Features

## Project Overview

This project implements a news recommendation system based on the 1st place solution of the RecSys 2024 Challenge, with additional semantic clustering features to improve recommendation quality. The system uses LightGBM with 100+ features including TF-IDF similarities, statistical features, and semantic clusters.

## Results Summary

### Small Dataset (ebnerd_small)

| Experiment                    | Test AUC | Test nDCG@10 | Test MRR | Improvement |
| ----------------------------- | -------- | ------------ | -------- | ----------- |
| Baseline (No Clustering)      | 0.845008 | 0.759527     | 0.689138 | -           |
| Title-based Clustering (K=30) | 0.845862 | 0.759873     | 0.689472 | **+0.10%**  |
| Body-based Clustering (K=30)  | 0.843266 | 0.757545     | 0.686831 | -0.21%      |

### Medium Dataset (10% of ebnerd_large, ~1.2M impressions)

| Experiment                    | Test AUC | Test nDCG@10 | Test MRR | Improvement |
| ----------------------------- | -------- | ------------ | -------- | ----------- |
| Baseline (No Clustering)      | 0.865739 | 0.759978     | 0.684697 | -           |
| Title-based Clustering (K=30) | 0.865871 | 0.760190     | 0.684956 | **+0.015%** |
| Body-based Clustering (K=30)  | 0.865991 | 0.760280     | 0.684946 | **+0.029%** |

**Key Findings**: 

- Semantic clustering improves performance on medium dataset with proper temporal split
- Body clustering shows slightly higher AUC improvement (+0.029%) than title clustering (+0.015%)
- Both clustering methods improve nDCG@10 and MRR metrics
- Title-based semantic clustering is still effective across different dataset sizes

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
- sentence-transformers (for semantic clustering)
- wandb (optional, for experiment tracking)

### 2. Dataset Preparation

**All datasets are in the input folder (relative to kami/):**

```
kami/input/
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

**Benefits of Relative Paths**:

- ✓ Portable across different machines
- ✓ No need to update paths when moving project
- ✓ Works on any directory structure

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

**For Large Dataset**:

```bash
.\run.bat create-candidates
.\run.bat create-features
.\run.bat create-datasets
.\run.bat train
```

### Step-by-Step Explanation

#### Step 1: Create Candidates

**What it does**: Generates candidate article-user pairs for each impression.

**Command**:

```bash
.\run.bat create-candidates --exp=medium
```

**Input**: 

- `input/ebnerd_medium/train/behaviors.parquet`
- `input/ebnerd_medium/validation/behaviors.parquet`

**Output**: 

- `output/preprocess/make_candidate/medium/train_candidate.parquet` (~13.4M rows)
- `output/preprocess/make_candidate/medium/validation_candidate.parquet` (~14.2M rows)
- `output/preprocess/make_candidate/medium/test_candidate.parquet` (same as validation)

**Time**: ~5 seconds for medium dataset

#### Step 2: Create Features

**What it does**: Extracts 100+ features for each candidate (TF-IDF similarities, statistics, temporal features, etc.)

**Command**:

```bash
.\run.bat create-features --exp=medium
```

**Input**: 

- Candidate files from Step 1
- `input/ebnerd_medium/articles.parquet`
- `input/ebnerd_medium/train/history.parquet`

**Output**: 

- `output/features/*/medium/` (30+ feature folders)

**Time**: ~30-60 minutes for medium dataset

**Feature Categories**:

- **a_base**: Article base features (total_inviews, pageviews, etc.)
- **a_click_ranking**: Article click ranking features
- **c_topics_sim_count_svd**: Topic similarity using SVD
- **c_title_tfidf_svd_sim**: Title TF-IDF similarity
- **c_body_tfidf_svd_sim**: Body text TF-IDF similarity
- **c_category_tfidf_sim**: Category similarity
- **c_is_already_clicked**: Whether user already clicked article
- **i_base_feat**: Impression-level features
- **u_stat_history**: User history statistics
- And 20+ more feature types

#### Step 3: Create Datasets

**What it does**: Combines all features with candidates into final training/validation/test datasets.

**Command**:

```bash
.\run.bat create-datasets --exp=medium
```

**Input**: 

- Candidate files from Step 1
- Feature files from Step 2

**Output**: 

- `output/preprocess/dataset067/medium/train_dataset.parquet` (~103 columns)
- `output/preprocess/dataset067/medium/validation_dataset.parquet`
- `output/preprocess/dataset067/medium/test_dataset.parquet`

**Time**: ~5-10 minutes for medium dataset

**Important**: This step may fail with RAM errors on large datasets (requires ~32GB+ for medium, ~768GB for large)

#### Step 4: Train Model

**What it does**: Trains LightGBM model with optional semantic clustering features.

**Command**:

```bash
.\run.bat train --exp=medium067_001
```

**Input**: 

- `output/preprocess/dataset067/medium/train_dataset.parquet`
- `output/preprocess/dataset067/medium/validation_dataset.parquet`
- `output/preprocess/dataset067/medium/test_dataset.parquet`

**Output**: 

- `output/experiments/[date]-medium/medium067_001/`
  - `model_dict_model.pkl` - Trained model
  - `results.txt` - Evaluation metrics
  - `validation_result.parquet` - Validation predictions
  - `test_result.parquet` - Test predictions
  - `importance_model.png` - Feature importance plot
  - `run.log` - Training log

**Time**: ~10-30 minutes for medium dataset

*eed: 7                        # Random seed (can override via --seed=42)
size_name: medium              # Dataset size: small/medium/large
sampling_rate: 0.1             # Subsample rate (for debugging, null for full data)
dataset_path: output/preprocess/dataset067  # Path to preprocessed dataset

## Graph-based features (DISABLED - performance degradation)

use_graph_features: false          # Set to true to enable (not recommended)
graph_embedding_dim: 64        # Node2Vec embedding dimensions
graph_walk_length: 30                # Random walk length
graph_num_walks: 200               # Number of walks per node

## Semantic clustering features (ENABLED - slight improvement)

use_semantic_clusters: true     # Enable BERT-based clustering
semantic_n_clusters: 30            # Number of clusters (K)
semantic_text_column: body   # Text to cluster: title/body/subtitle

lgbm:
  num_boost_round: 1200        # Training iterations (increased for slower learning)
  early_stopping_round: 100    # Early stopping pdifferent settings**:

1. Copy base config:
   
   ```bash
   cd experiments/015_train_third/exp
   copy medium067_001.yaml medium067_002.yaml
   ```

2. Edit the config:
   
   ```yaml
   seed: 42                       # Different seed
   size_name: medium
   use_semantic_clusters: true
   semantic_text_column: body     # Change from title to body
   semantic_n_clusters: 50        # Try more clusters
   ```

3. Run training:
   
   ```bash
   .\run.bat train --exp=medium067_002
   ```

4. Or override parameters via command line:
   
   ```bash
   .\run.bat train --exp=medium067_001 --seed=4er of clusters (K)
   semantic_text_column: title    # Text to cluster: title/body/subtitle
   
   ```

lgbm:
  num_boost_round: 1200        # Training iterations
  early_stopping_round: 100    # Early stopping patience
  params:
    learning_rate: 0.03
    max_depth: 10
    lambda_l2: 0.5
    num_leaves: 256

```

### Creating New Experiment Configs

**Example: Create config for medium dataset with body clustering**:

1. Copy base config:

   ```bash
   cd experiments/015_train_third/exp
   copy medium067_001.yaml medium067_002.yaml
```

2. Edit the config:
   
   ```yaml
   size_name: medium
   use_semantic_clusters: true
   semantic_text_column: body     # Change from title to body
   ```

3. Run training:
   
   ```bash
   .\run.bat train --exp=medium067_002
   Feature Engineering Details
   
   ```

### Semantic Clustering Implementation (BERT-Based)

**Status**: ✓ Enabled by default, provides small improvement

**How It Works**:

1. **Text Processing**: During training, article texts (title/body/subtitle) are extracted
2. **BERT Embeddings**: Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
3. **K-Means Clustering**: Articles grouped into K semantic clusters (default K=30)
4. **Feature Generation**: For each candidate, creates 33 features:
   - Cluster membership (one-hot encoded)
   - Cluster-based statistics (user's cluster preference, article's cluster popularity)

**Code Location**:

- Implementation: `experiments/015_train_third/semantic_cluster_features.py`
- Training integration: `experiments/015_train_third/run.py`
- Feature columns added: `semantic_cluster_*` (33 features)

**Configuration**:

```yaml
use_semantic_clusters: true       # Enable/disable
semantic_n_clusters: 30           # Number of clusters (K)
semantic_text_column: body        # title/body/subtitle
```

**Performance Impact**: +0.03% AUC on medium dataset

### Graph-Based Features (Node2Vec) - DISABLED

**Status**: ✗ Disabled due to performance degradation

**Why Disabled**:

- Performance drop: -11.2% AUC with graph features
- Cold-start problem: 90% of validation/test users not in training graph
- Zero embeddings for unknown users → meaningless features
- Domain characteristics: News articles have high turnover, graph becomes stale quickly

**Detailed Analysis**: See `experiments/015_train_third/GRAPH_FEATURES_ANALYSIS.md`

**Configuration**:

```Reproducibility

### Why Seed Management Matters

For statistical significance testing and result reproducibility, all random number generators are seeded:

- Python's `random` module
- NumPy's random functions
- LightGBM's internal randomness
- Python hash randomization (PYTHONHASHSEED)

### How Seeds Are Set

**In configuration**:
```yaml
seed: 7  # Default seed, can be overridden
```

**Via command line**:

```bash
.\run.bat train --exp=medium067_001 --seed=42
```

**In code** (automatically applied):

```python
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
# LightGBM params['seed'] = seed
```

### Output Organization

Each run is saved with **seed + timestamp** in the folder name:

```
output/experiments/015_train_third/
├── medium067_001_seed7_20251221_143045/
├── medium067_001_seed42_20251221_150122/
└── medium067_001_seed123_20251221_153018/
```

This ensures:

- ✓ No overwriting of previous results
- ✓ Easy comparison across seeds
- ✓ Traceability of when experiments were run
- ✓ Statistical significance testing with multiple seeds

### Running Multiple Seeds for Significance Analysis

```bash
.\run_multiple_seeds.bat medium067_001 7 42 123
```

This automates running the same experiment with 3 different seeds, useful for:

- Computing mean and standard deviation of metrics
- Statistical hypothesis testing
- Verifying result stability

## Common Issues & Solutions

### Issue 1: RAM Errors During create-datasets

**Symptom**: Process killed or out of memory error  
**Solution**: Use smaller dataset (small or medium instead of large)

### Issue 2: Path not found errors

**Symptom**: `FileNotFoundError: The system cannot find the path specified`  
**Solution**: 

- Make sure you're running commands from project root directory
- Check `yamls/dir/local.yaml` - paths should be relative (`input`, `output`)
- Verify datasets exist in `input/ebnerd_medium/` or `input/ebnerd_small/`

### Issue 3: Conda environment not found

**Symptom**: `EnvironmentLocationNotFound: Not a conda environment`  
**Solution**: 

- Activate the correct environment: `conda activate .\\.conda`
- Or use the absolute path to the environment
- Check that environment exists in `.conda` folder

### Issue 4: "Could not find 'exp/medium_067_001'"

**Symptom**: Config file not found error  
**Solution**: 

- Config files use no underscore: use `medium067_001` not `medium_067_001`
- Check available configs in `experiments/015_train_third/exp/`
- Config name must match the YAML filename (without .yaml extension)

### Issue 5: Semantic clustering degrades performance

**Symptom**: Lower AUC with clustering enabled  
**Solution**: 

- Try different text columns (body works slightly better than title)
- Adjust K (try 20, 30, 40, 50)
- Improvement is small (~0.03%), within statistical noise on small samples
  Based on LightGBM feature importance:
1. `c_time_min_diff` (28.5%) - How recent is the article?
2. `i_impression_times_in_1h` (12.3%) - User activity level
3. `a_total_inviews` (8.7%) - Article popularity
4. `c_user_count_past_1h_ratio` (6.4%) - User behavior pattern
5. `u_total_read_time_mean` (5.2%) - User engagement level
6. `c_topics_count_svd_sim` (4.1%) - Content similarity ✓
7. `i_total_pageviews_mean` (3.8%) - Impression quality
8. `c_title_tfidf_svd_sim` (2.9%) - Title similarity ✓
9. `a_sentiment_score` (2.7%) - Article sentiment
10. `c_body_tfidf_svd_sim` (2.1%) - Body similarity ✓

**Key Insight**: Temporal and behavioral features dominate (>60% importance), content features are secondary but still valuable.heck results.txt files in each experiment folder
type output\experiments\2025-12-21-medium\medium067_001\results.txt
type output\experiments\2025-12-21-medium-clustering-tfidf-title\medium067_001\results.txt
type output\experiments\2025-12-21-medium-clustering-tfidf-body\medium067_001\results.txt

```

**Current Results** (from output/experiments/):

- **Baseline**: AUC 0.865739, nDCG@10 0.759978, MRR 0.684697
- **Title Clustering**: AUC 0.865871 (+0.015%), nDCG@10 0.760190, MRR 0.684956
- **Body Clustering**: AUC 0.865991 (+0.029%), nDCG@10 0.760280, MRR 0.684946

### 3. Feature Importance

View feature importance plots in experiment output folders:

```
output/experiments/[date]-medium/medium067_001/importance_model.png
```

## Analysis Tools

### Cold-Start Performance Analysis

Analyze how well the model handles unpopular (cold-start) articles that have sparse interaction history.

**Basic Usage**:
```bash
# Analyze small dataset experiment
python analyze_cold_start.py --experiment output/experiments/015_train_third/small067_001_seed7_20251221_143045 --size small

# Analyze medium dataset experiment
python analyze_cold_start.py --experiment output/experiments/015_train_third/medium067_001_seed7_20251221_143045 --size medium
```

**Custom Cold-Start Threshold**:
```bash
# Use bottom 20% as cold-start (default is 10%)
python analyze_cold_start.py --experiment output/experiments/015_train_third/medium067_001_seed7_20251221_143045 --size medium --percentile 0.2

# Use bottom 5% as cold-start
python analyze_cold_start.py --experiment output/experiments/015_train_third/medium067_001_seed7_20251221_143045 --size medium --percentile 0.05
```

**What It Reports**:
- Overall metrics (AUC, nDCG@10, MRR) on all test items
- Metrics specifically on impressions containing cold-start items
- Performance gap between popular and cold-start items
- Percentage of test set affected by cold-start

**Parameters**:
- `--experiment`: Path to experiment output folder (required)
- `--size`: Dataset size - `small` or `medium` (default: `small`)
- `--percentile`: Bottom percentile to consider as cold-start (default: `0.1` = bottom 10%)

**Example Output**:
```
================================================================================
COLD-START ANALYSIS: Bottom 10% Items
Experiment: output/experiments/015_train_third/medium067_001_seed7_20251221_143045
================================================================================

1. Loading articles and identifying cold-start items...
   Total articles: 64,000
   Popularity threshold (inviews): 150.0
   Cold-start articles: 6,400 (10.0%)

   ALL ITEMS:
   AUC       : 0.865739
   nDCG@5    : 0.714002
   nDCG@10   : 0.759978
   MRR       : 0.684697

   IMPRESSIONS WITH COLD-START ITEMS:
   AUC       : 0.853421
   nDCG@5    : 0.701234
   nDCG@10   : 0.745678
   MRR       : 0.671234

7. Performance Gap (All Items vs Impressions with Cold-Start):
   Δ AUC     : 0.012318 (1.42%)
   Δ nDCG@10 : 0.014300 (1.88%)
```

This helps you understand:
- How well your model generalizes to unpopular content
- Whether content-based features (TF-IDF, BERT clustering) help with cold-start
- If there's a significant performance drop on cold-start items

## Feature Engineering Details

### Semantic Clustering Implementation

### How It Works

1. **Text Processing**: During training, article texts (title/body/subtitle) are extracted
2. **TF-IDF Vectorization**: Texts converted to TF-IDF vectors
3. **K-Means Clustering**: Articles grouped into K clusters (default K=30)
4. **Feature Generation**: For each candidate, creates 33 features:
   - Cluster membership (one-hot encoded)
   - Cluster-based statistics (user's cluster preference, article's cluster popularity)

### Code Location

- Implementation: `kami/semantic_cluster_features.py`
- Training integration: `experiments/015_train_third/run.py` (lines 350-400, 520-570)
- Feature columns added: `semantic_cluster_*` (33 features)

### Disabling Semantic Clustering

Set in experiment config:

```yaml
use_semantic_clusters: false
```

## Common Issues & Solutions

### Issue 1: RAM Errors During create-datasets

**Symptom**: Process killed or out of memory error  
**Solution**: Use smaller dataset (small or medium instead of large)

### Issue 2: impression_time column not found

**Symptom**: Error during training about missing impression_time  
**Solution**: The code now uses temporal splits. Make sure you're using the updated run.py

### Issue 3: Can't find articles.parquet

**Symptom**: FileNotFoundError for articles.parquet  
**Solution**: Check symlinks in `input/` folder, or update paths in configs

### Issue 4: Semantic clustering degrades performance

**Symptom**: Lower AUC with clustering enabled  
**Solution**: 

- Try different text columns (title works better than body)
- Adjust K (try 20, 30, 40, 50)
- Use more training data (medium or large dataset)

## Performance Benchmarks

### Processing Time (Medium Dataset, ~1.2M train + 1.25M validation impressions)

| Step                  | Time   | Memory       |
| --------------------- | ------ | ------------ |
| create-candidates     | ~5s    | ~1.2GB       |
| create-features       | ~40min | ~3-4GB peak  |
| create-datasets       | ~5min  | ~8-16GB peak |
| train (w/ clustering) | ~20min | ~4-8GB       |

### Dataset Statistics

| Dataset | Train Impressions | Validation Impressions | Articles | Users (train) | Users (val) |
| ------- | ----------------- | ---------------------- | -------- | ------------- | ----------- |
| Small   | ~260K             | ~270K                  | 125K     | ~69K          | ~77K        |
| Medium  | ~1.2M             | ~1.25M                 | 64K      | 342K          | 384K        |
| Large   | ~12M              | ~12.5M                 | 125K     | ~3.4M         | ~3.8M       |

## Recommended Workflow for Research

1. **Quick Testing** (30 minutes):
   
   - Use small dataset with `--debug` flag
   - Test different semantic clustering configs
   - Iterate quickly

2. **Full Evaluation** (2-3 hours):
   
   - Use medium dataset with `--exp=medium`
   - Train baseline + semantic variants
   - Compare results

3. **Production Training** (6-12 hours, requires 768GB RAM):
   
   - Use large dataset
   - Train final model
   - Submit to competition

## Citation & References

- **RecSys 2024 Challenge**: [Competition Link](https://www.recsyschallenge.com/2024/)
- **Original 1st Place Solution**: [GitHub](https://github.com/...)
- **Dataset**: Ekstra Bladet News Recommendation Dataset (EB-NeRD)

## Contact & Support

For any questions regarding the project, you can send an email to:

karatepe22@itu.edu.tr

---

**Last Updated**: December 21, 2025
