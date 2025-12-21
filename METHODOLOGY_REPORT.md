# Methodology Report: RecSys Challenge 2024 - News Recommendation System

## 1. Overview

This report documents the implementation and evaluation of a news recommendation system based on the 1st place solution from the RecSys Challenge 2024. The work includes preprocessing, feature engineering, model training, and baseline comparisons on the EB-NeRD (Ekstra Bladet News Recommendation Dataset).

## 2. Dataset

### 2.1 Dataset Description
- **Source**: EB-NeRD (Ekstra Bladet News Recommendation Dataset)
- **Dataset Sizes**:
  - **Small**: ~240k validation impressions, 2.9M candidate pairs
  - **Large**: 12M+ candidate pairs (memory-constrained, not fully processed)
- **Articles**: 125,541 unique articles with metadata (title, body, category, subcategory, published_time, total_pageviews, etc.)

### 2.2 Dataset Splits
- **Training Set**: Historical user-article interactions with click labels
- **Validation Set**: Used for model evaluation and hyperparameter tuning
- **Test Set**: Articles metadata available, but ground truth labels not provided (for competition submission only)

### 2.3 Data Structure
Each impression contains:
- `impression_id`: Unique identifier for the impression
- `user_id`: Anonymized user identifier
- `article_id`: List of candidate articles shown to the user
- `label`: Binary labels indicating clicks (1) or no-clicks (0)
- Additional temporal and contextual features

## 3. Preprocessing Pipeline

### 3.1 Candidate Generation
**Location**: `preprocess/make_candidate/run.py`

The first step creates user-article candidate pairs from raw behavior data:

1. **Input**: Raw user behavior logs (impressions, clicks, article views)
2. **Process**:
   - Explodes impression data to create individual user-article pairs
   - Each impression generates multiple candidates (one per article shown)
   - Preserves temporal ordering and user context
3. **Output**: Candidate dataset with (user_id, article_id, impression_id, label) tuples

**Memory Optimization**: 
- Added sampling capability (`sample_fraction` parameter) to handle large datasets
- For large dataset: `sample_fraction: 0.25` (25% sampling) to fit within 32GB RAM constraints
- Small dataset: `sample_fraction: 1.0` (full data)

### 3.2 Feature Engineering
**Location**: `preprocess/dataset067/run.py`

Comprehensive feature engineering pipeline creating 100+ features across multiple categories:

#### 3.2.1 Base Features (a_base)
- Article metadata: category, subcategory, sentiment scores
- User demographics and interaction history
- Temporal features: time of day, day of week

#### 3.2.2 Article Statistics (i_article_stat_v2)
- Global article popularity metrics
- Engagement rates: inviews per pageview, read time per pageview
- Temporal statistics: time since publication

#### 3.2.3 User Statistics (u_click_article_stat_v2, u_stat_history)
- User historical behavior patterns
- Click-through rates across categories
- User engagement metrics (total inviews, pageviews, read time)

#### 3.2.4 Content Similarity Features
- **TF-IDF + SVD**: Text similarity for titles, subtitles, body (c_title_tfidf_svd_sim, c_subtitle_tfidf_svd_sim, c_body_tfidf_svd_sim)
- **Category Similarity**: TF-IDF similarity for categories, subcategories, entity groups (c_category_tfidf_sim, c_subcategory_tfidf_sim)
- **Topic Similarity**: SVD-based topic modeling (c_topics_sim_count_svd, ua_topics_sim_count_svd_feat)

#### 3.2.5 Temporal Features
- Article publish time relative to impression (c_article_publish_time_v5)
- Time difference features (i_viewtime_diff)
- Impression count features with read time (c_appear_imp_count_read_time_per_inview_v7)

#### 3.2.6 Click Ranking Features
- Click position in impression (a_click_ranking)
- Click ratio among candidates (a_click_ratio, a_click_ratio_multi)
- Whether article was already clicked (c_is_already_clicked)

#### 3.2.7 Transition Probability Features
- Probability of transitioning from first article to candidate (y_transition_prob_from_first)

**Total Features**: ~102 features in final dataset

### 3.3 Dataset Output
**Location**: `output/preprocess/dataset067/small/`

Generated files:
- `train_dataset.parquet`: Training data with all features and labels
- `validation_dataset.parquet`: Validation data with all features and labels
- `test_dataset.parquet`: Test data with all features (no labels)

## 4. Model Architecture and Training

### 4.1 Model Selection
**Model**: LightGBM (Gradient Boosting Decision Tree)
**Objective**: LambdaRank (learning-to-rank objective optimized for nDCG)

### 4.2 Training Strategy
**Location**: `experiments/015_train_third/run.py`

The winning solution uses a **three-stage training approach**:

#### Stage 1: Train & Validate
1. Train on training set
2. Validate on validation set
3. Early stopping based on validation nDCG@10
4. Save best iteration number

#### Stage 2: Retrain on Validation (not executed in our experiments)
1. Train on validation set only
2. Use iteration count from Stage 1
3. Generate predictions for test set

#### Stage 3: Retrain on Full Data (not executed in our experiments)
1. Combine training + validation sets
2. Train for same number of iterations as Stage 1
3. Generate final test predictions

**Note**: In our experiments, we only executed Stage 1 (train on train, evaluate on validation) as test labels are not available for evaluation.

### 4.3 Model Configuration
Key hyperparameters (from `experiments/015_train_third/exp/small.yaml`):

```yaml
lgbm:
  params:
    objective: lambdarank  # Ranking objective
    metric: ndcg
    ndcg_eval_at: [10]
    learning_rate: 0.1
    num_leaves: 255
    min_data_in_leaf: 20
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    max_depth: -1
    two_rounds: true  # Use two-round training for memory efficiency
  
  num_boost_round: 10000
  early_stopping_round: 100
  verbose_eval: 100
```

### 4.4 Feature Selection
**Unuse Columns**: impression_id, user_id, article_id, label (excluded from training)

**Feature Handling**:
- Categorical features explicitly defined for proper encoding
- Boolean features converted to int8 for compatibility
- Null values handled during feature engineering phase

### 4.5 Evaluation Metrics
**Library**: ebrec (custom evaluation library from RecSys challenge)

**Primary Metrics**:
- **AUC** (Area Under ROC Curve): Binary classification performance
- **nDCG@5** (Normalized Discounted Cumulative Gain at 5): Ranking quality for top 5
- **nDCG@10**: Ranking quality for top 10
- **MRR** (Mean Reciprocal Rank): Average rank of first relevant item

**Evaluation Process**:
1. Group predictions by impression_id
2. Rank articles within each impression based on model scores
3. Compare ranks against ground truth click labels
4. Compute metrics across all impressions

## 5. Baseline Methods

### 5.1 Motivation
To establish performance floors and validate the value of complex feature engineering and ML models, three simple non-ML baseline methods were implemented.

### 5.2 Baseline Implementations
**Location**: `experiments/000_simple_baselines/run.py`

#### 5.2.1 Popularity Baseline
**Method**: Rank articles by global popularity (total_pageviews)

**Rationale**: Popular articles are more likely to be clicked regardless of user context.

**Implementation**:
```python
- Join candidates with articles on article_id
- Use total_pageviews as ranking score
- Fill missing values with 0
- Rank within each impression (descending order)
```

#### 5.2.2 Recency Baseline
**Method**: Rank articles by recency (published_time)

**Rationale**: Users may prefer newer articles in news recommendation.

**Implementation**:
```python
- Join candidates with articles on article_id
- Use published_time as ranking score
- Fill missing values with minimum timestamp
- Rank within each impression (descending order)
```

#### 5.2.3 Combined Baseline (Popularity + Recency)
**Method**: Weighted combination of normalized popularity and recency

**Rationale**: Balance between popular articles and fresh content.

**Implementation**:
```python
- Normalize popularity to [0, 1]: popularity / max_popularity
- Normalize recency to [0, 1]: (time - min_time) / (max_time - min_time)
- Combined score = 0.7 × popularity + 0.3 × recency
- Rank within each impression (descending order)
```

**Weight Configuration**: 
- `popularity_weight: 0.7` (configurable via YAML)
- `recency_weight: 0.3` (1 - popularity_weight)

### 5.3 Baseline Evaluation Process
1. Load validation dataset and articles metadata
2. For each baseline:
   - Generate scores for all candidates
   - Rank articles within impressions
   - Prepare labels and predictions in ebrec format
   - Compute AUC, nDCG@5, nDCG@10, MRR
3. Save results to parquet files
4. Generate summary comparison table

## 6. Results

### 6.1 Baseline Performance
**Evaluation Set**: Validation set (244,647 impressions, 2.9M candidate pairs)

| Baseline | AUC | nDCG@5 | nDCG@10 | MRR |
|----------|-----|--------|---------|-----|
| Popularity | 0.5966 | 0.4214 | 0.4950 | 0.3769 |
| Recency | 0.5009 | 0.3365 | 0.4215 | 0.3076 |
| Popularity + Recency (70/30) | 0.5971 | 0.4219 | 0.4952 | 0.3772 |

**Key Observations**:
1. **Popularity is a strong signal**: Achieves AUC of 0.5966, significantly above random (0.5)
2. **Recency alone is weak**: AUC of 0.5009, barely better than random baseline
3. **Minimal improvement from combining**: Combined baseline only marginally improves over popularity alone (0.5971 vs 0.5966)
4. **nDCG@10 performance**: Popularity achieves 0.4950, indicating reasonable ranking quality

### 6.2 LightGBM Performance (To Be Compared)
**Status**: Model training completed on small dataset
**Evaluation**: Results to be extracted from experiment logs

Expected improvements based on competition results:
- Significant AUC improvement (expected: 0.70-0.80+)
- Better nDCG scores due to personalization
- Higher MRR from learning user-article interactions

### 6.3 Performance Analysis

#### 6.3.1 Why Popularity Works
- News articles have inherent quality signals
- Popular articles are generally more engaging
- Popularity captures implicit editorial quality
- Simple heuristic provides reasonable recommendations

#### 6.3.2 Why Recency Fails
- Not all users prefer newest content
- Quality matters more than freshness in news
- Recency doesn't account for article relevance
- Temporal biases in validation set

#### 6.3.3 Value of Machine Learning
The baseline results establish that:
1. Simple heuristics can achieve ~0.60 AUC
2. Personalization and complex features are needed to exceed this floor
3. LightGBM with 100+ features should significantly outperform baselines
4. Feature engineering effort is justified if model exceeds 0.60+ AUC

## 7. Technical Implementation Details

### 7.1 Software Stack
- **Language**: Python 3.9+
- **Data Processing**: Polars (high-performance DataFrame library)
- **Machine Learning**: LightGBM 4.x
- **Configuration**: Hydra (hierarchical configuration management)
- **Evaluation**: ebrec (custom RecSys evaluation library)
- **Experiment Tracking**: Weights & Biases (wandb)

### 7.2 Computational Environment
- **OS**: Windows
- **RAM**: 32GB (constraint for large dataset)
- **Storage**: SSD for fast I/O
- **Environment Manager**: Conda

### 7.3 Memory Optimization Strategies
1. **Sampling**: Reduced large dataset to 25% for memory constraints
2. **Two-round training**: LightGBM feature to save memory during training
3. **Parquet format**: Efficient columnar storage with compression
4. **Garbage collection**: Explicit memory cleanup between stages
5. **Feature dtype optimization**: Boolean → int8, float64 → float32 where appropriate

### 7.4 Code Organization
```
recsys-challenge-2024-1st-place-master/kami/
├── preprocess/
│   ├── make_candidate/       # Candidate generation
│   └── dataset067/           # Feature engineering
├── experiments/
│   ├── 015_train_third/      # LightGBM training (3-stage)
│   ├── 016_catboost/         # Alternative model (CatBoost)
│   └── 000_simple_baselines/ # Baseline implementations
├── features/                 # Feature-specific implementations
├── ebrec/                    # Evaluation library
├── utils/                    # Helper functions
└── yamls/                    # Configuration files
    ├── dir/                  # Directory paths
    └── exp/                  # Experiment configs
```

### 7.5 Configuration Management
**Hydra Framework**: Hierarchical YAML-based configuration

Example structure:
```yaml
config.yaml           # Main config, references others
├── dir: local        # from yamls/dir/local.yaml
└── exp: small        # from exp/small.yaml
    └── base.yaml     # Base experiment parameters
```

Benefits:
- Easy experimentation with different configurations
- Version control for hyperparameters
- Reproducibility through config logging
- Override parameters from command line

## 8. Reproducibility

### 8.1 Running Candidate Generation
```bash
cd recsys-challenge-2024-1st-place-master/kami
python preprocess/make_candidate/run.py exp=small
```

### 8.2 Running Feature Engineering
```bash
python preprocess/dataset067/run.py exp=small
```

### 8.3 Running Baseline Evaluation
```bash
python experiments/000_simple_baselines/run.py exp=small
```

Output location: `output/experiments/000_simple_baselines/small/`
- `validation_result_popularity.parquet`
- `validation_result_recency.parquet`
- `validation_result_popularity_recency.parquet`
- `baseline_results.txt` (summary)

### 8.4 Running LightGBM Training
```bash
python experiments/015_train_third/run.py exp=small
```

Output location: `output/experiments/015_train_third/small/`
- `model_dict_first_stage.pkl` (trained model)
- `importance_first_stage.png` (feature importance plot)
- `validation_result_first.parquet` (predictions)
- `run.log` (training logs)

### 8.5 Seeds for Reproducibility
- Random seed: 42 (set in all configurations)
- Sampling seed: 42 (for consistent subsampling)
- LightGBM seed: Set via `params.seed` in config

## 9. Challenges and Solutions

### 9.1 Memory Constraints
**Problem**: Large dataset (12M rows) exceeds 32GB RAM during preprocessing

**Attempted Solutions**:
1. Checkpointing (save intermediate results)
2. Chunked processing (process in batches)
3. Garbage collection (explicit memory cleanup)
4. Data type optimization (use smaller dtypes)

**Final Solution**: 
- Implemented sampling capability in candidate generation
- Focus on small dataset for complete pipeline execution
- Large dataset requires 64GB+ RAM or distributed processing

### 9.2 Evaluation API Inconsistency
**Problem**: `MetricEvaluator.evaluate()` returns self, not dict

**Solution**: 
- Changed from `results = evaluator.evaluate()` 
- To `evaluator.evaluate()` then `results = evaluator.evaluations`
- Matches implementation in winning solution

### 9.3 Path Resolution Issues
**Problem**: Articles.parquet location varies by dataset size

**Solution**: 
- Implemented fallback path checking
- Try multiple possible locations in order
- Use first existing path

### 9.4 Configuration Complexity
**Problem**: Hydra override syntax errors in baseline implementation

**Solution**: 
- Simplified config files
- Removed problematic override directives
- Used minimal required configuration

## 10. Future Work

### 10.1 Model Improvements
1. **Hyperparameter Tuning**: Grid search or Bayesian optimization for LightGBM
2. **Feature Selection**: Identify most important features via SHAP or feature importance
3. **Model Ensemble**: Combine LightGBM + CatBoost predictions
4. **Neural Networks**: Implement deep learning baselines (e.g., BERT for text features)

### 10.2 Feature Engineering
1. **User Embeddings**: Learn dense representations of user preferences
2. **Article Embeddings**: Use pre-trained language models (BERT, RoBERTa)
3. **Graph Features**: User-article interaction graphs
4. **Temporal Dynamics**: Time-decay features, trending signals
5. **Cross-features**: Explicit feature interactions

### 10.3 Evaluation
1. **Cross-validation**: K-fold CV for more robust metrics
2. **Temporal Validation**: Time-based splits to simulate production
3. **User Segmentation**: Analyze performance by user types
4. **Cold Start Analysis**: Evaluate on new users/articles

### 10.4 Scalability
1. **Distributed Processing**: Use Spark or Dask for large dataset
2. **Feature Store**: Pre-compute and cache features
3. **Online Learning**: Incremental model updates
4. **Model Serving**: Deploy for real-time inference

### 10.5 Large Dataset Processing
1. **Cloud Computing**: Use AWS/GCP with high-memory instances
2. **Incremental Processing**: Process in chunks with persistent storage
3. **Feature Sampling**: Use most important features only
4. **Data Reduction**: Dimensionality reduction techniques

## 11. Conclusions

### 11.1 Summary of Work
1. ✅ Successfully replicated RecSys 2024 winning solution preprocessing pipeline
2. ✅ Generated 100+ engineered features from raw data
3. ✅ Implemented and evaluated three non-ML baseline methods
4. ✅ Trained LightGBM model on small dataset with LambdaRank objective
5. ✅ Established performance floor: Popularity baseline achieves 0.60 AUC

### 11.2 Key Findings
1. **Popularity is a strong baseline**: Simple heuristic achieves reasonable performance
2. **Feature engineering is extensive**: 100+ features across multiple categories
3. **Evaluation is consistent**: Using same metrics and library as competition
4. **Memory is a constraint**: Large dataset requires significant computational resources

### 11.3 Methodology Validation
The implemented approach follows best practices:
- ✅ Clear train/validation/test splits
- ✅ No data leakage (test labels not used)
- ✅ Reproducible with seeds
- ✅ Comparable baselines established
- ✅ Standard evaluation metrics (AUC, nDCG, MRR)
- ✅ Feature engineering based on domain knowledge

### 11.4 Next Steps
1. Extract LightGBM validation results from logs
2. Compare ML model vs baselines
3. Analyze feature importance
4. Document performance improvements
5. Prepare final report with complete results

---

## Appendix A: File Locations

### Datasets
- Small dataset: `dataset/ebnerd_small/`
- Large dataset: `dataset/ebnerd_large/`
- Articles metadata: `dataset/ebnerd_large/articles.parquet`

### Preprocessing Outputs
- Candidates: `output/preprocess/make_candidate/`
- Features: `output/features/`
- Final datasets: `output/preprocess/dataset067/small/`

### Experiment Outputs
- Baselines: `output/experiments/000_simple_baselines/small/`
- LightGBM: `output/experiments/015_train_third/small/`

### Configuration Files
- Main configs: `experiments/{experiment_name}/config.yaml`
- Experiment configs: `experiments/{experiment_name}/exp/`
- Global configs: `yamls/dir/local.yaml`

## Appendix B: Key Code Snippets

### Baseline Ranking Function
```python
def make_result_df(df: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """Create result dataframe with rankings based on scores"""
    result_df = (
        df.select(["impression_id", "user_id", "article_id", score_col])
        .with_columns(
            pl.col(score_col)
            .rank(method="ordinal", descending=True)
            .over("impression_id")
            .alias("rank")
        )
        .group_by(["impression_id", "user_id"], maintain_order=True)
        .agg([pl.col("article_id"), pl.col("rank")])
    )
    return result_df
```

### Evaluation Setup
```python
from ebrec.evaluation import MetricEvaluator, AucScore, NdcgScore, MrrScore

metrics = [AucScore(), NdcgScore(k=5), NdcgScore(k=10), MrrScore()]
evaluator = MetricEvaluator(
    labels=labels,
    predictions=predictions,
    metric_functions=metrics
)
evaluator.evaluate()
results = evaluator.evaluations
```

### LambdaRank Configuration
```python
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10],
    'learning_rate': 0.1,
    'num_leaves': 255,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
}
```

---

**Document Version**: 1.0  
**Last Updated**: December 1, 2025  
**Author**: Research Team  
**Project**: RecSys Challenge 2024 - News Recommendation System
