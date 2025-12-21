# BERT-Based Semantic Clustering Features

Bu modül, makale içeriğinden BERT embeddings çıkarıp K-Means ile semantic cluster'lar oluşturur ve kullanıcı ilgi dağılımı feature'ları üretir.

## Yöntem

1. **Global Clustering**: Tüm makaleler BERT ile embed edilir, K-Means ile K cluster oluşturulur
2. **User Profiling**: Her kullanıcının okuma geçmişinden cluster distribution hesaplanır

## Gereksinimler

```bash
pip install sentence-transformers
```

## Kullanım

```python
from bert_cluster_features import BERTClusterFeatureExtractor

# Initialize
extractor = BERTClusterFeatureExtractor(
    n_clusters=50,  # Number of semantic clusters
    model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
)

# Train global clusters on all articles
extractor.fit(articles_df, text_column='title')  # or 'body'

# Extract features for each user-article pair
features_df = extractor.transform(behaviors_df, history_df)
```

## Features Generated

- `bert_user_cluster_X`: User's click distribution across K clusters (K features)
- `bert_article_cluster`: Article's assigned cluster ID (1 feature)
- `bert_user_article_cluster_match`: Binary: user's favorite cluster == article cluster (1 feature)
- `bert_user_article_cluster_affinity`: User's click rate on article's cluster (1 feature)

**Total**: K + 3 features
