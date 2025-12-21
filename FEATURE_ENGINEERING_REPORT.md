# Feature Engineering Report: RecSys Challenge 2024

## Table of Contents
1. [Overview](#overview)
2. [Feature Organization](#feature-organization)
3. [Article Features (a_*)](#article-features-a)
4. [User Features (u_*)](#user-features-u)
5. [Impression Features (i_*)](#impression-features-i)
6. [Candidate Features (c_*)](#candidate-features-c)
7. [User-Article Features (y_*)](#user-article-features-y)
8. [Feature Importance & Selection](#feature-importance--selection)
9. [Implementation Details](#implementation-details)

---

## Overview

The feature engineering pipeline creates **100+ features** across multiple categories to capture different aspects of the news recommendation problem. Features are designed to model:
- Article content and popularity
- User preferences and behavior patterns
- Temporal dynamics
- Content similarity
- Click patterns and sequences

**Total Feature Count**: ~102 features (varies by configuration)

**Feature Naming Convention**:
- Prefix indicates scope/join key:
  - `a_`: Article-level (keyed by article_id)
  - `u_`: User-level (keyed by user_id)
  - `i_`: Impression-level (keyed by impression_id + user_id)
  - `c_`: Candidate-level (same length as candidates, direct join)
  - `y_`: User-Article interaction (keyed by user_id + article_id)

---

## Feature Organization

### Feature Categories

```
features/
├── a_base/                      # Article base features
├── a_additional_feature/        # Additional article metadata
├── a_click_ranking/             # Click popularity ranking
├── a_click_ratio/               # Click-through rates
├── a_click_ratio_multi/         # Multi-dimensional CTR
├── i_article_stat_v2/           # Impression article statistics
├── i_base_feat/                 # Impression base features
├── i_stat_feat/                 # Impression statistics
├── i_viewtime_diff/             # View time differences
├── u_click_article_stat_v2/     # User click statistics
├── u_stat_history/              # User historical patterns
├── c_title_tfidf_svd_sim/       # Title similarity (TF-IDF+SVD)
├── c_subtitle_tfidf_svd_sim/    # Subtitle similarity
├── c_body_tfidf_svd_sim/        # Body text similarity
├── c_category_tfidf_sim/        # Category similarity
├── c_subcategory_tfidf_sim/     # Subcategory similarity
├── c_entity_groups_tfidf_sim/   # Entity similarity
├── c_ner_clusters_tfidf_sim/    # Named entity similarity
├── c_topics_sim_count_svd/      # Topic similarity
├── c_article_publish_time_v5/   # Temporal features
├── c_is_already_clicked/        # Click history flag
├── c_appear_imp_count_v7/       # Impression counts
├── ua_topics_sim_count_svd_feat/ # User-article topic similarity
└── y_transition_prob_from_first/ # Article transition probabilities
```

---

## Article Features (a_*)

### a_base (Base Article Metadata)
**Purpose**: Capture intrinsic article properties
**Key**: article_id

**Features** (7 features):
```python
USE_COLUMNS = [
    "premium",                      # Premium content flag
    "category_article_type",        # Encoded article type (ordinal)
    "total_inviews",                # Total impressions served
    "total_pageviews",              # Total page views received
    "total_read_time",              # Total reading time
    "sentiment_score",              # Sentiment analysis score
    "ordinal_sentiment_label",      # Encoded sentiment (Negative/Neutral/Positive)
]
```

**Rationale**:
- **Premium**: Premium content often has higher engagement
- **Article Type**: Different types (news, opinion, feature) have different appeal
- **Total Inviews/Pageviews**: Popularity signals quality and relevance
- **Read Time**: Longer reads indicate engagement depth
- **Sentiment**: Emotional tone affects click likelihood (positive/negative news)

**Preprocessing**:
- OrdinalEncoder for article_type (categorical → integer)
- OrdinalEncoder for sentiment_label (ordered: Negative < Neutral < Positive)

---

### a_click_ranking (Click Popularity Features)
**Purpose**: Global click popularity ranking
**Key**: article_id

**Features** (2 features):
```python
USE_COLUMNS = [
    "click_rank",      # Rank by click frequency (1 = most clicked)
    "click_count",     # Absolute click count
]
```

**Rationale**:
- **Click Rank**: Articles with more historical clicks are more likely to be clicked again
- **Click Count**: Absolute popularity measure
- Used to identify "viral" or trending articles

**Calculation**:
```python
# Count clicks per article from history
count_df = history_df.explode("article_id_fixed")
    .value_counts()
    .sort("count", descending=True)
    .with_row_index("rank")
```

**Deduplication**: Optional `is_user_unique` flag to count each user only once per article

---

### a_click_ratio (Click-Through Rate)
**Purpose**: Article-level CTR metrics
**Key**: article_id

**Features** (1-2 features):
```python
USE_COLUMNS = [
    "click_ratio",     # CTR = clicks / impressions
]
```

**Rationale**:
- CTR measures article appeal relative to exposure
- High CTR = compelling headlines/topics
- More robust than raw click counts (controls for exposure)

---

### a_click_ratio_multi (Multi-dimensional CTR)
**Purpose**: CTR broken down by context
**Key**: article_id

**Features** (multiple):
```python
# CTR stratified by:
- Device type
- Time of day
- Category
- User segment
```

**Rationale**:
- Different articles perform differently in different contexts
- Mobile vs desktop reading patterns
- Morning news vs evening entertainment

---

## User Features (u_*)

### u_click_article_stat_v2 (User Click Statistics)
**Purpose**: Aggregate statistics of articles users have clicked
**Key**: user_id

**Features** (10 features):
```python
USE_COLUMNS = [
    "time_min_diff_click_publish_mean",  # Avg time between click and publish
    "time_min_diff_click_publish_std",   # Std dev of above
    "total_inviews_mean",                # Avg inviews of clicked articles
    "total_inviews_std",                 # Std dev of inviews
    "total_pageviews_mean",              # Avg pageviews of clicked articles
    "total_pageviews_std",               # Std dev of pageviews
    "total_read_time_mean",              # Avg read time of clicked articles
    "total_read_time_std",               # Std dev of read time
    "sentiment_score_mean",              # Avg sentiment of clicked articles
    "sentiment_score_std",               # Std dev of sentiment
]
```

**Rationale**:
- **Mean statistics**: User preferences (do they like popular articles? long reads?)
- **Std statistics**: User consistency (narrow vs broad interests)
- **Time diff**: Does user prefer breaking news or evergreen content?
- **Sentiment**: User preference for positive/negative news

**Calculation**:
```python
# For each user, explode click history and join with article metadata
explode_df = history_df.explode(["article_id_fixed", "impression_time_fixed"])
    .join(articles_df, on="article_id")
    .with_columns(
        (impression_time - published_time).dt.total_minutes()
    )
# Aggregate by user_id
stats_df = explode_df.group_by("user_id").agg([
    pl.mean("total_inviews"), 
    pl.std("total_inviews"),
    ...
])
```

**Memory Optimization**: Chunked processing for large datasets

---

### u_stat_history (User Historical Patterns)
**Purpose**: User behavior patterns over time
**Key**: user_id

**Features** (varies):
```python
# Temporal patterns:
- Click frequency (daily/weekly)
- Session patterns (length, frequency)
- Diversity metrics (category spread)
- Recency of last interaction
```

**Rationale**:
- **Frequency**: Active users have different behavior than occasional users
- **Session patterns**: Binge readers vs quick checkers
- **Diversity**: Niche vs broad interests
- **Recency**: Account for user lifecycle (new/returning/churned)

---

## Impression Features (i_*)

### i_article_stat_v2 (Impression-level Article Statistics)
**Purpose**: Aggregate statistics of articles in each impression
**Key**: impression_id + user_id

**Features** (10 features):
```python
USE_COLUMNS = [
    "time_min_diff_mean",        # Avg time since publish for articles in impression
    "time_min_diff_std",         # Std dev of above
    "total_inviews_mean",        # Avg inviews of articles in impression
    "total_inviews_std",         # Std dev of inviews
    "total_pageviews_mean",      # Avg pageviews of articles in impression
    "total_pageviews_std",       # Std dev of pageviews
    "total_read_time_mean",      # Avg read time of articles in impression
    "total_read_time_std",       # Std dev of read time
    "sentiment_score_mean",      # Avg sentiment of articles in impression
    "sentiment_score_std",       # Std dev of sentiment
]
```

**Rationale**:
- **Context matters**: Article appeal depends on what else is shown
- **Mean values**: Overall quality of impression candidate set
- **Std values**: Diversity of options (mixed vs homogeneous)
- **Temporal spread**: Are all articles recent or mixed ages?

**Calculation**:
```python
# For each impression, explode article_ids_inview
candidate_df = behaviors_df
    .select(["impression_id", "article_ids_inview", "user_id", "impression_time"])
    .explode("article_ids_inview")
    .join(articles_df, on="article_id")
    .with_columns(
        (impression_time - published_time).dt.total_minutes().alias("time_min_diff")
    )
# Aggregate by impression_id
stats_df = candidate_df.group_by(["impression_id", "user_id"]).agg([
    pl.mean("total_inviews"),
    pl.std("total_inviews"),
    ...
])
```

---

### i_viewtime_diff (View Time Differences)
**Purpose**: Temporal patterns within impressions
**Key**: impression_id

**Features** (varies):
```python
- Time between first and last article in impression
- Sequential viewing patterns
- Dwell time distributions
```

**Rationale**:
- Long sessions indicate engaged browsing
- Quick sessions indicate targeted searching
- Sequential patterns reveal exploration strategies

---

### i_base_feat (Base Impression Features)
**Purpose**: Core impression metadata
**Key**: impression_id

**Features**:
```python
- Session ID
- Device type
- Time of day
- Day of week
- Number of articles shown
```

**Rationale**:
- Session context affects recommendations
- Device affects interaction patterns
- Temporal patterns in news consumption

---

### i_stat_feat (Statistical Impression Features)
**Purpose**: Derived statistics from impression
**Key**: impression_id

**Features**:
```python
- Candidate set size
- Diversity measures
- Quality metrics
```

---

## Candidate Features (c_*)

### Content Similarity Features (TF-IDF + SVD)

#### c_title_tfidf_svd_sim (Title Similarity)
**Purpose**: Measure how well article titles match user preferences
**Key**: Candidate-level (impression_id + article_id)

**Features** (2 features):
```python
USE_COLUMNS = [
    "title_count_svd_sim",    # Cosine similarity score
    "title_count_svd_rn",     # Rank within impression (by similarity)
]
```

**Algorithm**:
1. **TF-IDF Vectorization**: Convert all article titles to TF-IDF vectors
2. **SVD Dimensionality Reduction**: Reduce to 50 components (n_components=50)
3. **User Embedding**: Average vectors of articles user has clicked
4. **Similarity Calculation**: Cosine similarity between article and user embedding

**Code**:
```python
# Train TF-IDF + SVD on all article titles
vectorizer = TfidfVectorizer(max_features=10000)
article_tfidf = vectorizer.fit_transform(articles_df["title"])
svd = TruncatedSVD(n_components=50)
article_embeddings = svd.fit_transform(article_tfidf)

# Create user embeddings from click history
for user_clicked_articles in history:
    user_embedding = article_embeddings[clicked_articles].mean(axis=0)

# Calculate similarity for each candidate
similarity = (article_embeddings[article_id] * user_embeddings[user_id]).sum(axis=1)
```

**Rationale**:
- **Titles are key**: Users judge articles primarily by title
- **SVD reduces noise**: Captures semantic topics, not just word matches
- **Personalization**: User embedding captures individual preferences
- **Relative ranking**: Rank matters more than absolute score

**Memory Optimization**: Chunked processing (1M candidates at a time)

---

#### c_subtitle_tfidf_svd_sim (Subtitle Similarity)
**Purpose**: Same as title similarity but for subtitles
**Features**: Similar to title features

**Rationale**:
- Subtitles provide additional context
- Different from title (more descriptive)
- Some articles have compelling subtitles

---

#### c_body_tfidf_svd_sim (Body Text Similarity)
**Purpose**: Full article content similarity
**Features**: Similar to title features

**Rationale**:
- Captures deeper content themes
- Important for engaged readers who read full articles
- Complements headline-based features

**Challenge**: Body text is much longer (requires more aggressive SVD compression)

---

#### c_category_tfidf_sim (Category Similarity)
**Purpose**: Match article categories to user preferences
**Features** (2 features):
```python
- category_tfidf_sim     # Similarity score
- category_tfidf_rn      # Rank within impression
```

**Rationale**:
- Category indicates topic area (politics, sports, entertainment)
- Users often have strong category preferences
- Simpler than full text similarity

---

#### c_subcategory_tfidf_sim (Subcategory Similarity)
**Purpose**: Fine-grained category matching
**Features**: Similar to category features

**Rationale**:
- More specific than main category
- Example: Sports → Football vs Basketball
- Captures niche interests

---

#### c_entity_groups_tfidf_sim (Entity Similarity)
**Purpose**: Match based on named entities (people, places, organizations)
**Features**: Similar structure

**Rationale**:
- Entities indicate specific topics
- Example: "Joe Biden", "Apple Inc", "Ukraine"
- Users follow specific entities (celebrities, companies, locations)

---

#### c_ner_clusters_tfidf_sim (NER Cluster Similarity)
**Purpose**: Clustered named entity similarity
**Features**: Similar structure

**Rationale**:
- Entities grouped by type (PERSON, ORG, LOC)
- Captures entity-type preferences
- More robust than individual entities

---

### Topic Similarity Features

#### c_topics_sim_count_svd (Topic Similarity)
**Purpose**: Abstract topic matching using SVD
**Features**:
```python
- topics_sim          # Topic similarity score
- topics_count_rn     # Rank by similarity
```

**Algorithm**:
1. Create topic distribution for each article (likely from LDA or similar)
2. Apply SVD for dimensionality reduction
3. Compare with user's topic preferences

**Rationale**:
- Topics are higher-level than categories
- Capture semantic themes across articles
- Example topics: "Economy", "Technology", "Health"

---

#### ua_topics_sim_count_svd_feat (User-Article Topic Features)
**Purpose**: User-specific topic preferences
**Key**: user_id + article_id

**Features**:
```python
- User's topic profile
- Article's topic distribution
- Similarity between user and article topics
```

**Rationale**:
- Persistent user-article features
- Can be precomputed and cached
- Used across multiple impressions

---

### Temporal Features

#### c_article_publish_time_v5 (Temporal Features)
**Purpose**: Capture recency and temporal patterns
**Features** (6 features):
```python
USE_COLUMNS = [
    "time_sec_diff",             # Seconds since published
    "time_sec_diff_rn",          # Rank by recency (within impression)
    "time_min_diff",             # Minutes since published
    "time_hour_diff",            # Hours since published
    "time_day_diff",             # Days since published
    "time_sec_diff_per_fastest", # Ratio to newest article in impression
]
```

**Calculation**:
```python
candidate_df = candidate_df.with_columns([
    (impression_time - published_time).dt.total_seconds().alias("time_sec_diff"),
    (impression_time - published_time).dt.total_minutes().alias("time_min_diff"),
    (impression_time - published_time).dt.total_hours().alias("time_hour_diff"),
    (impression_time - published_time).dt.total_days().alias("time_day_diff"),
]).with_columns([
    pl.col("time_sec_diff").rank().over(["impression_id", "user_id"]).alias("time_sec_diff_rn"),
    (pl.col("time_sec_diff") / pl.col("time_sec_diff").min().over(["impression_id", "user_id"]))
        .alias("time_sec_diff_per_fastest"),
])
```

**Rationale**:
- **Breaking news**: Very recent articles have special appeal
- **Evergreen content**: Some articles remain relevant over time
- **Multiple timescales**: Some patterns at minute-level, others at day-level
- **Relative recency**: Newest article in impression context matters
- **Rank**: Non-linear recency effects (newest >> 2nd newest)

---

### Behavioral Features

#### c_is_already_clicked (Click History Flag)
**Purpose**: Has user already clicked this article?
**Features** (1 feature):
```python
USE_COLUMNS = [
    "is_already_clicked",  # Boolean: True if user clicked this article before
]
```

**Calculation**:
```python
candidate_df = candidate_df.join(
    history_df.select(["user_id", "article_id_fixed"]),
    on="user_id"
).with_columns(
    pl.col("article_id").is_in(pl.col("article_id_fixed")).alias("is_already_clicked")
)
```

**Rationale**:
- **Repeat clicks are rare**: Users typically don't re-read articles
- **Strong negative signal**: Already-clicked articles should be deprioritized
- **Data leakage note**: This is technically a "leak" feature (uses future data in training)
  - Included because competition allows it
  - In production, would need careful handling

---

#### c_appear_imp_count_v7 (Impression Count Features)
**Purpose**: How often has article appeared in impressions?
**Features**:
```python
- appear_imp_count           # Total times shown
- appear_imp_count_rank      # Rank by frequency
- appear_imp_count_ratio     # Ratio to max in impression
```

**Rationale**:
- Frequently shown articles might be "pushed" content
- Or they might be genuinely popular
- Helps model learn impression serving strategy

---

#### c_appear_imp_count_read_time_per_inview_v7 (Engagement Metrics)
**Purpose**: Combine impression counts with engagement
**Features**:
```python
- appear_imp_count_with_read_time      # Weighted by read time
- read_time_per_inview                 # Engagement rate
```

**Rationale**:
- Impressions with engagement > impressions alone
- Read time indicates content quality
- Helps identify "clickbait" (high impressions, low read time)

---

## User-Article Features (y_*)

### y_transition_prob_from_first (Article Transition Probabilities)
**Purpose**: Model sequential article reading patterns
**Key**: user_id + article_id

**Features** (2 features):
```python
USE_COLUMNS = [
    "transition_prob_from_first",    # P(read article | last read article by user)
    "transition_count_from_first",   # Count of transitions
]
```

**Algorithm**:
1. **Extract Sequences**: From user history, create article reading sequences
2. **Build Transition Matrix**: Count transitions from article A → article B
3. **Calculate Probabilities**: P(B|A) = count(A→B) / count(A→any)
4. **Join with Candidates**: For each user's last-read article, get transition probs to candidates

**Code**:
```python
# Create transitions
explode_df = history_df.explode("article_id_fixed").sort(["user_id", "impression_time"])
explode_df = explode_df.with_columns([
    pl.col("article_id").alias("from_article_id"),
    pl.col("article_id").shift(-1).over("user_id").alias("to_article_id"),
])

# Count transitions
transition_df = explode_df.group_by(["from_article_id", "to_article_id"]).agg(
    pl.count().alias("from_to_count")
).with_columns(
    pl.col("from_to_count").sum().over("from_article_id").alias("from_count")
).with_columns(
    (pl.col("from_to_count") / pl.col("from_count")).alias("transition_prob")
)

# Get user's last article
last_article_df = history_df.group_by("user_id").agg(
    pl.col("article_id_fixed").sort_by("impression_time").last()
)

# Join to get transition probabilities for candidates
```

**Rationale**:
- **Sequential reading patterns**: Users follow topical threads
- **Example**: Read article about "iPhone" → likely to read "Apple earnings"
- **Markov assumption**: Next article depends on last article
- **Personalization**: Different users have different reading patterns

**Deduplication**: Optional `is_user_unique` to avoid counting repeated user transitions

---

## Feature Importance & Selection

### Feature Categories by Importance (Based on RecSys Competition)

**Tier 1 - Critical Features** (Top 10-15% contribution):
1. **Content Similarity** (c_title_tfidf_svd_sim, c_body_tfidf_svd_sim)
   - Direct relevance to user interests
2. **Click History** (c_is_already_clicked, a_click_ranking)
   - Strong behavioral signals
3. **Temporal Features** (c_article_publish_time_v5)
   - Recency is crucial in news

**Tier 2 - Important Features** (Next 20-30% contribution):
1. **Article Popularity** (a_base: total_pageviews, total_inviews)
2. **User Statistics** (u_click_article_stat_v2)
3. **Impression Context** (i_article_stat_v2)
4. **Transition Probabilities** (y_transition_prob_from_first)

**Tier 3 - Supporting Features** (Remaining contribution):
1. **Category/Entity Similarity** (c_category_tfidf_sim, c_entity_groups_tfidf_sim)
2. **Engagement Metrics** (c_appear_imp_count_read_time_per_inview_v7)
3. **Topic Features** (c_topics_sim_count_svd)

### Feature Selection Strategy

**Leak Features** (Not used in production):
- `c_is_already_clicked` (uses future data)
- Future impression features marked in config

**Configuration** (from config.yaml):
```yaml
exp:
  article_stats_cols: true/false      # Enable/disable article stats
  past_impression_cols: true/false    # Enable past impression features
  future_impression_cols: false       # Typically disabled (leakage)
```

---

## Implementation Details

### Memory Optimization Techniques

1. **Chunked Processing**:
```python
chunk_size = 1000000  # Process 1M rows at a time
for start_idx in range(0, total_rows, chunk_size):
    chunk_df = df[start_idx:end_idx]
    # Process chunk
    chunks.append(result)
combined = pl.concat(chunks)
```

2. **Garbage Collection**:
```python
del intermediate_df
gc.collect()
```

3. **Data Type Optimization**:
```python
# Use smaller dtypes where possible
.cast(pl.Int32)   # Instead of Int64
.cast(pl.Float32) # Instead of Float64
```

4. **Lazy Evaluation** (Polars):
```python
# Operations are optimized and executed together
df.lazy().filter(...).select(...).collect()
```

### Performance Considerations

**Feature Generation Time** (Small Dataset):
- Simple features (a_base, i_base_feat): < 1 minute
- Statistical features (i_article_stat_v2, u_click_article_stat_v2): 5-10 minutes
- Similarity features (c_title_tfidf_svd_sim): 10-20 minutes
- Transition features (y_transition_prob_from_first): 5-10 minutes

**Total Pipeline Time**: ~2-3 hours for small dataset

**Large Dataset Challenges**:
- Memory requirements: 32-64GB RAM
- Processing time: 10-20 hours
- Requires chunking and careful memory management

### Code Structure

Each feature has its own directory with:
```
feature_name/
├── run.py           # Main feature generation script
├── config.yaml      # Hydra configuration
└── exp/
    ├── base.yaml    # Base parameters
    └── small.yaml   # Dataset-specific overrides
```

**Standard Feature Template**:
```python
PREFIX = "c"  # or a, i, u, y
KEY_COLUMNS = ["impression_id", "article_id"]  # Join keys
USE_COLUMNS = ["feature1", "feature2"]         # Output features

def process_df(cfg, input_df):
    # Feature generation logic
    return output_df

def create_feature(cfg, output_path):
    for data_name in ["train", "validation", "test"]:
        # Load data
        # Process
        # Save to output_path/f"{data_name}_feat.parquet"
```

### Configuration Management

Features are configured via Hydra:
```yaml
# features/{feature_name}/config.yaml
defaults:
  - _self_
  - dir: local
  - exp: small

exp:
  seed: 42
  size_name: small
  # Feature-specific parameters
  n_components: 50  # For SVD
  is_user_unique: true  # Deduplicate by user
```

### Feature Joining

In dataset creation (`preprocess/dataset067/run.py`):
```python
# Start with candidates
df = candidate_df

# Join article features (a_*)
df = df.join(a_base_df, on="article_id", how="left")
df = df.join(a_click_ranking_df, on="article_id", how="left")

# Join user features (u_*)
df = df.join(u_click_article_stat_df, on="user_id", how="left")

# Join impression features (i_*)
df = df.join(i_article_stat_df, on=["impression_id", "user_id"], how="left")

# Join user-article features (y_*)
df = df.join(y_transition_prob_df, on=["user_id", "article_id"], how="left")

# Candidate features (c_*) are already same length, just concat
df = pl.concat([df, c_title_sim_df, c_publish_time_df, ...], how="horizontal")
```

---

## Feature Summary Table

| Category | # Features | Purpose | Key Rationale |
|----------|-----------|---------|---------------|
| Article Base (a_base) | 7 | Intrinsic article properties | Quality, sentiment, metadata |
| Article Popularity (a_click_*) | 4 | Click patterns | Viral/trending signals |
| User Click Stats (u_click_article_stat_v2) | 10 | User preferences | Personalization |
| User History (u_stat_history) | 5-10 | Behavioral patterns | Engagement style |
| Impression Stats (i_article_stat_v2) | 10 | Impression context | Candidate set quality |
| Impression Base (i_base_feat, i_stat_feat) | 5-10 | Impression metadata | Context awareness |
| Title Similarity (c_title_tfidf_svd_sim) | 2 | Content matching | Relevance |
| Subtitle Similarity (c_subtitle_tfidf_svd_sim) | 2 | Content matching | Additional context |
| Body Similarity (c_body_tfidf_svd_sim) | 2 | Deep content matching | Semantic relevance |
| Category Similarity (c_category_tfidf_sim) | 2 | Topic matching | Broad interests |
| Subcategory Similarity (c_subcategory_tfidf_sim) | 2 | Fine-grained topics | Niche interests |
| Entity Similarity (c_entity_groups_tfidf_sim) | 2 | Entity following | Specific topics |
| NER Similarity (c_ner_clusters_tfidf_sim) | 2 | Named entity matching | Entity types |
| Topic Similarity (c_topics_sim_count_svd) | 2 | Abstract topics | Semantic themes |
| Temporal Features (c_article_publish_time_v5) | 6 | Recency | Breaking news vs evergreen |
| Click History (c_is_already_clicked) | 1 | Re-click prevention | Avoid duplicates |
| Impression Counts (c_appear_imp_count_v7) | 3 | Exposure metrics | Serving strategy |
| Engagement (c_appear_imp_count_read_time_per_inview_v7) | 2 | Quality metrics | Engagement depth |
| Transitions (y_transition_prob_from_first) | 2 | Sequential patterns | Reading paths |
| **TOTAL** | **~102** | | |

---

## Key Insights & Best Practices

### 1. Content Similarity is King
- TF-IDF + SVD on titles/body is most important feature group
- 50 components captures semantic meaning without noise
- User embeddings enable personalization

### 2. Temporal Features are Critical for News
- Recency affects click probability dramatically
- Multiple timescales needed (seconds to days)
- Relative recency (rank) often better than absolute

### 3. User Features Enable Personalization
- User click statistics reveal preferences
- Standard deviation captures consistency
- Transition probabilities model reading paths

### 4. Feature Engineering > Model Complexity
- 100+ well-designed features with LightGBM beats complex deep learning
- Domain knowledge crucial (news-specific patterns)
- Interpretability important for production

### 5. Memory Management is Essential
- Large datasets require chunking
- Polars is much faster than Pandas
- Explicit gc.collect() helps

### 6. Leakage Features Must Be Handled Carefully
- `c_is_already_clicked` uses future data
- Only valid for offline evaluation
- Must be disabled for production

---

## Appendix: Feature Engineering Pipeline Flow

```
Raw Data
    ↓
[1] Candidate Generation
    ├── Explode impressions → candidates
    └── Output: candidate.parquet
    ↓
[2] Article Features (a_*)
    ├── Load articles.parquet
    ├── Process metadata
    ├── Calculate click statistics
    └── Output: {train,val,test}_feat.parquet
    ↓
[3] User Features (u_*)
    ├── Load history.parquet
    ├── Aggregate user behavior
    ├── Calculate statistics
    └── Output: {train,val,test}_feat.parquet
    ↓
[4] Impression Features (i_*)
    ├── Load behaviors.parquet + articles.parquet
    ├── Aggregate per impression
    ├── Calculate context statistics
    └── Output: {train,val,test}_feat.parquet
    ↓
[5] Candidate Features (c_*)
    ├── Load candidates + articles + history
    ├── Calculate similarities (TF-IDF+SVD)
    ├── Calculate temporal features
    ├── Calculate behavioral features
    └── Output: {train,val,test}_feat.parquet
    ↓
[6] User-Article Features (y_*)
    ├── Load history.parquet
    ├── Build transition matrix
    ├── Calculate probabilities
    └── Output: {train,val,test}_feat.parquet
    ↓
[7] Dataset Assembly
    ├── Load candidates
    ├── Join all feature parquets
    ├── Handle missing values
    ├── Apply feature selection
    └── Output: {train,val,test}_dataset.parquet
    ↓
[8] Model Training
    └── Use assembled datasets with LightGBM
```

---

**Document Version**: 1.0  
**Last Updated**: December 1, 2025  
**Total Features**: ~102  
**Processing Time**: 2-3 hours (small dataset)  
**Memory Requirement**: 16-32GB (small), 64GB+ (large)
