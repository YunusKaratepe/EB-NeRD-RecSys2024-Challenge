# EB-NeRD-RecSys2024-Challenge
A Recommendation System using the dataset for RecSys2024 Challenge.
This repository is created for Data Mining lecture project.

## Project Overview
This project implements a news recommendation system based on the 1st place solution from the RecSys 2024 Challenge. The system uses LightGBM with LambdaRank objective for learning-to-rank on news articles.

## Model Architecture
- **Algorithm**: LightGBM with LambdaRank objective (listwise ranking)
- **Loss Function**: LambdaRank - optimizes for ranking metrics directly
- **Features**: 103 features including:
  - Time-based features (publish time, impression time differences)
  - User behavior features (click history, read time patterns)
  - Article features (topics, categories, sentiment)
  - Similarity features (TF-IDF based similarities)
  - Statistical features (impression counts, pageviews)

## Training Setup
### Data Split
- **Training Set**: Full training dataset (`train_dataset.parquet`)
- **Validation Set**: First 50% of validation dataset (used for early stopping)
- **Test Set**: Second 50% of validation dataset (held-out for final evaluation)

### Training Configuration
- **Objective**: lambdarank (listwise ranking)
- **Learning Rate**: 0.1
- **Max Depth**: 8
- **Num Leaves**: 128
- **Early Stopping**: 40 rounds
- **Max Iterations**: 400
- **Evaluation Metrics**: AUC, nDCG@5, nDCG@10, MRR

### Key Design Decisions
1. **Single-stage training**: Simplified from original 3-stage approach to prevent data leakage
2. **Strict data separation**: Train data never touches validation/test data
3. **Impression-level splitting**: Validation/test split by impression IDs to ensure no overlap
4. **Sampling**: 10% sampling rate for small dataset experiments

## Evaluation
The model is evaluated on both validation and test sets using:
- **AUC**: Area Under the ROC Curve
- **nDCG@5**: Normalized Discounted Cumulative Gain at position 5
- **nDCG@10**: Normalized Discounted Cumulative Gain at position 10
- **MRR**: Mean Reciprocal Rank

## Feature Importance
The training pipeline automatically generates three feature importance visualizations:
1. `importance_model.png` - Top 100 features
2. `importance_top20_model.png` - Top 20 most important features
3. `importance_bottom20_model.png` - Bottom 20 least important features

## Code Structure
```
experiments/015_train_third/
├── run.py              # Main training script (LightGBM only)
├── config.yaml         # Training configuration
└── output/             # Training outputs (models, logs, charts)
```

## Running the Code
```bash
cd recsys-challenge-2024-1st-place-master/kami
run.bat train --debug
```

## Changes from Original Implementation
1. **Removed multi-stage training**: Simplified from 3-stage to single-stage
2. **Removed submission code**: No longer generates competition submission files
3. **Added feature importance charts**: Top 20 and bottom 20 feature visualizations
4. **Always compute AUC**: Removed conditional AUC computation
5. **Removed other models**: Only LightGBM training remains

## Output Files
After training completes, the following files are generated:
- `model_dict_model.pkl` - Trained LightGBM model
- `validation_result.parquet` - Predictions on validation set
- `test_result.parquet` - Predictions on test set
- `importance_*.png` - Feature importance visualizations
- `run.log` - Training logs with metrics

## Results
Check the log file for VALIDATION and TEST metrics:
```
VALIDATION: {'auc': X.XXXX, 'ndcg@5': X.XXXX, 'ndcg@10': X.XXXX, 'mrr': X.XXXX}
TEST: {'auc': X.XXXX, 'ndcg@5': X.XXXX, 'ndcg@10': X.XXXX, 'mrr': X.XXXX}
```