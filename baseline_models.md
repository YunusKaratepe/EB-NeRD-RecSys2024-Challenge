# Baseline Models for EB-NeRD News Recommendation

## Overview
This document defines baseline models to compare against the winner's complex feature engineering approach.

## 1. Simple Baselines (Non-ML)

### 1.1 Random Baseline
**Description**: Randomly rank articles in each impression
- **Purpose**: Lower bound performance
- **Implementation**: Random shuffle of inview articles
- **Expected nDCG@5**: ~0.20-0.25
- **Expected nDCG@10**: ~0.25-0.30

### 1.2 Popularity Baseline
**Description**: Rank by global article popularity
- **Features**: total_pageviews or total_inviews
- **Purpose**: Simple but often strong baseline
- **Expected nDCG@5**: ~0.35-0.45
- **Expected nDCG@10**: ~0.40-0.50

### 1.3 Recency Baseline  
**Description**: Rank by article freshness (newest first)
- **Features**: published_time
- **Purpose**: Test temporal bias
- **Expected nDCG@5**: ~0.30-0.40
- **Expected nDCG@10**: ~0.35-0.45

### 1.4 Popularity + Recency
**Description**: Weighted combination
- **Formula**: `score = 0.7 * popularity_rank + 0.3 * recency_rank`
- **Purpose**: Combining two strong signals
- **Expected nDCG@5**: ~0.40-0.50
- **Expected nDCG@10**: ~0.45-0.55

## 2. Content-Based Baselines

### 2.1 Category Match
**Description**: Prioritize articles matching user's top categories from history
- **Features**: 
  - User's top 3 categories from history
  - Article category
  - Category match score
- **Purpose**: Test simple personalization
- **Expected nDCG@5**: ~0.45-0.55
- **Expected nDCG@10**: ~0.50-0.60

### 2.2 TF-IDF Similarity
**Description**: Content similarity using article text (title + body)
- **Method**: 
  - TF-IDF vectors for articles
  - User profile = average TF-IDF of history
  - Cosine similarity for ranking
- **Purpose**: Test semantic matching
- **Expected nDCG@5**: ~0.50-0.60
- **Expected nDCG@10**: ~0.55-0.65

## 3. Simple ML Baselines

### 3.1 Logistic Regression
**Description**: Linear model with basic features
- **Features** (15-20):
  - Article: popularity, CTR, freshness, category, premium, sentiment
  - User: activity_level (from history length), top_category
  - Context: hour, day_of_week, device_type
  - Match: category_match_binary, category_affinity_score
- **Purpose**: Linear baseline
- **Expected nDCG@5**: ~0.55-0.65
- **Expected nDCG@10**: ~0.60-0.70

### 3.2 LightGBM with Basic Features
**Description**: Gradient boosting with ~50 features
- **Features**:
  - All Logistic Regression features
  - + Article engagement metrics (avg_read_time, avg_scroll)
  - + User aggregations (avg_read_time, category_diversity)
  - + Temporal features (time_since_last_read)
  - + Statistical features (popularity_rank, CTR_rank)
- **Purpose**: Simple tree-based model
- **Hyperparameters**: Default or light tuning
- **Expected nDCG@5**: ~0.60-0.70
- **Expected nDCG@10**: ~0.65-0.75

### 3.3 Matrix Factorization (Implicit Feedback)
**Description**: Collaborative filtering using implicit package
- **Method**: Alternating Least Squares (ALS)
- **Input**: User-Article interaction matrix (from history + behaviors)
- **Purpose**: Test pure collaborative filtering
- **Expected nDCG@5**: ~0.45-0.55
- **Expected nDCG@10**: ~0.50-0.60
- **Note**: May struggle with cold-start

## 4. Hybrid Baseline

### 4.1 Ensemble of Simple Models
**Description**: Weighted average of best baselines
- **Components**:
  - Popularity (weight: 0.2)
  - Category Match (weight: 0.3)
  - LightGBM Basic (weight: 0.5)
- **Purpose**: Simple ensemble approach
- **Expected nDCG@5**: ~0.65-0.72
- **Expected nDCG@10**: ~0.70-0.77

## 5. Comparison Table

| Model | Features | Complexity | Expected nDCG@5 | Expected nDCG@10 | Implementation Time |
|-------|----------|------------|-----------------|------------------|---------------------|
| Random | 0 | Very Low | 0.20-0.25 | 0.25-0.30 | 10 min |
| Popularity | 1 | Very Low | 0.35-0.45 | 0.40-0.50 | 30 min |
| Recency | 1 | Very Low | 0.30-0.40 | 0.35-0.45 | 30 min |
| Pop + Recency | 2 | Low | 0.40-0.50 | 0.45-0.55 | 1 hour |
| Category Match | 3-5 | Low | 0.45-0.55 | 0.50-0.60 | 2 hours |
| TF-IDF | Dense | Medium | 0.50-0.60 | 0.55-0.65 | 3-4 hours |
| Logistic Reg. | 15-20 | Low-Medium | 0.55-0.65 | 0.60-0.70 | 4-5 hours |
| LightGBM Basic | ~50 | Medium | 0.60-0.70 | 0.65-0.75 | 6-8 hours |
| Matrix Factor. | Latent | Medium | 0.45-0.55 | 0.50-0.60 | 4-6 hours |
| Simple Ensemble | ~50 | Medium | 0.65-0.72 | 0.70-0.77 | 8-10 hours |
| **Winner (Full)** | **~250** | **Very High** | **~0.80-0.85** | **~0.82-0.87** | **~100+ hours** |

## 6. Recommended Implementation Order

### Phase 1: Quick Baselines (Day 1)
1. ✓ Random baseline
2. ✓ Popularity baseline  
3. ✓ Recency baseline
4. ✓ Popularity + Recency

**Goal**: Establish lower bounds quickly

### Phase 2: Personalization (Day 2-3)
5. ✓ Category Match baseline
6. ✓ Logistic Regression with basic features

**Goal**: Test simple personalization impact

### Phase 3: Advanced Baselines (Day 4-5)
7. ✓ LightGBM with ~50 features
8. ✓ TF-IDF similarity (if time permits)
9. ✓ Matrix Factorization (if time permits)

**Goal**: Establish strong baseline for comparison

### Phase 4: Ensemble (Day 6)
10. ✓ Simple ensemble of top 3 models

**Goal**: Best baseline performance

## 7. Evaluation Metrics

### Primary Metrics (from competition)
- **nDCG@5**: Main metric
- **nDCG@10**: Secondary metric
- **MRR (Mean Reciprocal Rank)**: Click position

### Secondary Metrics
- **AUC**: Overall ranking quality
- **Precision@k**: Click prediction accuracy
- **Recall@k**: Coverage of clicked articles
- **Coverage**: % of articles recommended
- **Diversity**: Category distribution in recommendations

## 8. Data Preparation for Baselines

### Training Data Format
```python
# For each impression, create candidate pairs
impression_id | user_id | article_id | clicked | features...
```

### Required Preprocessing
1. **Candidate generation**: Explode inview articles
2. **Label creation**: clicked = 1 if in article_ids_clicked, else 0
3. **Feature engineering**: Based on model complexity
4. **Train/validation split**: Time-based (last 20% of time for validation)

## 9. Success Criteria

### Baseline Success
- Random < Popularity < Category Match < LightGBM Basic < Winner
- LightGBM Basic should reach ~0.65-0.70 nDCG@10
- Gap to winner's approach (~0.82-0.87) shows value of extensive feature engineering

### Analysis Goals
1. **Feature importance**: Which simple features matter most?
2. **Cold-start**: How do baselines handle new users?
3. **Category bias**: Are recommendations too category-focused?
4. **Efficiency**: Training time vs performance trade-off
5. **Interpretability**: Can explain simpler models better

## 10. Code Structure

```
baselines/
├── data/
│   ├── prepare_candidates.py      # Generate candidate pairs
│   └── feature_engineering.py     # Basic feature extraction
├── models/
│   ├── random_baseline.py         # Random ranking
│   ├── popularity_baseline.py     # Popularity-based
│   ├── category_match.py          # Category personalization
│   ├── logistic_regression.py     # Linear model
│   ├── lightgbm_basic.py          # Simple GBDT
│   ├── tfidf_similarity.py        # Content-based
│   └── matrix_factorization.py    # Collaborative filtering
├── ensemble/
│   └── simple_ensemble.py         # Weighted combination
├── evaluation/
│   └── metrics.py                 # nDCG, MRR, AUC calculation
└── run_all_baselines.py           # Main pipeline
```

## 11. Notes

### Dataset Considerations
- Use **ebnerd_small** for fast iteration
- Test on **ebnerd_large** only after validating approach
- Memory constraints: 32GB RAM limits complexity

### Winner's Approach Comparison
- Winner used ~250 features with extensive feature engineering
- Our baselines: 0-50 features
- **Gap analysis** will show value of complex feature engineering
- Goal: Demonstrate diminishing returns of complexity

### Timeline
- **Week 1**: Implement all baselines
- **Week 2**: Run experiments, collect results
- **Week 3**: Analysis and documentation
- **Week 4**: Compare with winner's approach (if time permits)
