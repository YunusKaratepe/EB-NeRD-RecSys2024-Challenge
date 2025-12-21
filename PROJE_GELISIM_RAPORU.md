# Haber Öneri Sistemi: Proje Gelişim Raporu

**Proje**: EB-NeRD Haber Öneri Yarışması  
**Tarih**: Aralık 2025  
**Veri Seti**: EB-NeRD Medium (ebnerd_medium)

---

## 1. Proje Özeti ve Mevcut Durum

### 1.1. Problem Tanımı

Bir haber platformunda, kullanıcılara içerik önerileri yapmak için bir makine öğrenmesi modeli geliştiriyoruz. Problem, **ranking** problemi olarak tanımlanmıştır:

- Her impression (gösterim) için birden fazla aday makale var
- Bu makaleleri kullanıcının tıklama ihtimaline göre sıralamak gerekiyor
- Metrikler: AUC, nDCG@5, nDCG@10, MRR

### 1.2. Mevcut Model Başarısı

**Baseline Model Performansı** (LightGBM + 103 feature):

- **AUC**: 0.845
- **nDCG@10**: 0.730
- **MRR**: 0.651

**Kullanılan Yaklaşım**: 

- Model: LightGBM (Gradient Boosting)
- Objective: LambdaRank (learning-to-rank)
- Feature Sayısı: 103 özellik
- Veri Boyutu: Medium dataset, %10 sampling

---

## 2. Denenen Yaklaşımlar ve Sonuçları

### 2.1. TF-IDF Tabanlı Benzerlik Özellikleri (Mevcut Başarılı Yaklaşım)

#### Yöntem Detayı

Makale içerikleri (başlık, alt başlık, gövde metni) ile kullanıcı geçmişi arasında **metin benzerliği** hesaplanıyor.

**Pipeline Akışı**:

1. **TF-IDF Vektörizasyonu**: Her makale metni TF-IDF vektörüne dönüştürülür
2. **SVD (Truncated SVD)**: Boyut indirgeme ile gürültü azaltılır ve hesaplama verimliliği sağlanır
3. **Kullanıcı Profili Oluşturma**: Kullanıcının geçmişte okuduğu makalelerin vektörlerinin ortalaması alınır
4. **Cosine Similarity**: Aday makale ile kullanıcı profili arasında benzerlik hesaplanır

**Detaylı İşlem Adımları**:

```python
# Adım 1: TF-IDF ile makale vektörleri oluştur
vectorizer = TfidfVectorizer(
    max_features=10000,      # Top 10K kelime
    ngram_range=(1, 2),      # Unigram + bigram
    min_df=2,                # En az 2 dokümanda geç
    max_df=0.8,              # Max %80 dokümanda
    stop_words='english'     # İngilizce stop words filtrele
)
article_matrix = vectorizer.fit_transform(articles['title'])
# Boyut: [N_articles, 10000]

# Adım 2: SVD ile boyut indirgeme
svd = TruncatedSVD(n_components=50)
article_embeddings = svd.fit_transform(article_matrix)
# Boyut: [N_articles, 50] (10000 -> 50 boyuta indirildi)

# Adım 3: L2 normalizasyon (cosine similarity için)
article_embeddings = normalize(article_embeddings, norm='l2')

# Adım 4: Kullanıcı profili = okuduğu makalelerin ortalaması
user_history_articles = [a1, a2, a3, ...]  # Kullanıcının geçmişi
user_embedding = article_embeddings[user_history_articles].mean(axis=0)
user_embedding = normalize(user_embedding, norm='l2')
# Boyut: [50]

# Adım 5: Cosine similarity hesapla
candidate_article_embedding = article_embeddings[candidate_article_id]
similarity = (user_embedding * candidate_article_embedding).sum()
# Değer: [-1, 1] arası, 1 = çok benzer
```

#### Bizim Yaklaşımımızın Özgün Yönleri

**1. SVD Boyut İndirgeme Kullanımı**:
- Standart TF-IDF: Doğrudan yüksek boyutlu sparse vektörlerle çalışır (10K-50K boyut)
- Bizim yaklaşım: SVD ile **50-100 boyuta** indirgeme yapıyoruz
- **Avantajı**: 
  - Gürültü azaltır (rare words'ler filtrelenir)
  - Hesaplama hızı 100x artar
  - Latent semantic analysis etkisi (benzer anlamlı kelimeler yaklaşır)

**2. Kullanıcı Profili Olarak Ortalama Embedding**:
- Standart yaklaşım: Her makale için ayrı ayrı similarity hesapla, topla
- Bizim yaklaşım: Önce kullanıcı embedding'i oluştur (tek seferlik), sonra similarity hesapla
- **Avantajı**:
  - Çok daha verimli (O(N) yerine O(1) per candidate)
  - Kullanıcının genel ilgi alanını temsil eder
  - Sparse history problemine dayanıklı (birkaç makale yeterli)

**3. Normalizasyon Stratejisi**:
- SVD sonrası L2 normalizasyonu uyguluyoruz
- **Neden**: Cosine similarity doğrudan dot product ile hesaplanabilir
- **Avantaj**: Numpy broadcast ile vectorized computation, çok hızlı

**4. Chunk-based Processing**:
```python
# Memory efficient processing
chunk_size = 1_000_000  # 1M candidate at a time
for chunk in chunks:
    similarities = (article_embeddings[chunk] * user_embeddings[chunk]).sum(axis=1)
```
- **Neden**: 13M+ candidate için memory explosion önlenir
- **Avantaj**: Büyük veri setlerinde çalışır

**5. Multi-Text Fusion**:
- Sadece title değil, body, subtitle, category, entities için ayrı ayrı TF-IDF
- Her metin tipi için ayrı özellik üretilir
- Model kendisi öğrenir hangi metin tipi ne kadar önemli
- **Bulgu**: Title > Topics > Body sıralaması (feature importance'a göre)

#### Parametreler

```yaml
# TF-IDF konfigürasyonu
max_features: 10000           # Kelime dağarcığı boyutu
ngram_range: (1, 2)          # Unigram ve bigram kullan
min_df: 2                     # En az 2 dokümanda geçmeli
max_df: 0.8                   # En fazla %80 dokümanda (çok yaygın kelimeleri filtrele)
stop_words: 'english'        # İngilizce stop words

# SVD boyut indirgeme
n_components: 50              # 50 boyuta indir (title için)
n_components: 100             # 100 boyut (body için, daha zengin içerik)

# Normalization
norm: 'l2'                    # L2 normalizasyon (cosine similarity için)
```

**Neden Bu Parametreler?**

- **max_features=10000**: Vocabulary çok büyürse (50K+) sparse olur, çok küçükse (1K) bilgi kaybı
- **ngram_range=(1,2)**: Unigram tek kelimeler, bigram "machine learning" gibi phrase'ler yakalar
- **min_df=2**: Sadece 1 dokümanda geçen kelimeler noise, filtrele
- **max_df=0.8**: "the", "is" gibi her yerde geçen kelimeler discriminative değil
- **n_components=50**: 10000 → 50 yeterli (explained variance ~%70-80)

#### Üretilen Özellikler

Her aday (candidate) makale için 8 ayrı benzerlik özelliği:

1. **c_title_tfidf_svd_sim**: Başlık benzerliği
   - Kullanıcının geçmişte okuduğu makalelerin başlıkları
   - Aday makalenin başlığı  
   - TF-IDF (10K vocab) → SVD (50 dim) → User Profile → Cosine Similarity
   - **En etkili**: Feature importance %2.9

2. **c_subtitle_tfidf_svd_sim**: Alt başlık benzerliği
   - Alt başlıklar genelde daha açıklayıcı
   - Ancak her makalede yok (sparse)

3. **c_body_tfidf_svd_sim**: Gövde metni benzerliği
   - En zengin içerik ama en gürültülü
   - SVD burada daha kritik (100 component kullanıyoruz)
   - Feature importance: %2.1

4. **c_category_tfidf_sim**: Kategori benzerliği
   - "Sports", "Politics", "Technology" gibi
   - Küçük vocabulary, SVD gereksiz

5. **c_subcategory_tfidf_sim**: Alt kategori benzerliği
   - Daha granular: "Football", "Basketball"

6. **c_entity_groups_tfidf_sim**: Varlık grubu benzerliği
   - Named entities: "Biden", "Apple", "Istanbul"
   - Specific interest'leri yakalar

7. **c_ner_clusters_tfidf_sim**: İsimlendirilmiş varlık benzerliği
   - NER clustered entities

8. **c_topics_count_svd_sim**: Konu benzerliği
   - Topic modeling sonuçları
   - **En etkili TF-IDF özelliği**: Feature importance %4.1

**Toplam**: 8 benzerlik özelliği

**Özellik Çeşitliliğinin Önemi**:
- Her metin tipi farklı aspect'leri yakalar
- Title: Kısa, özlü, SEO-optimized
- Body: Derin içerik, bağlam
- Category: High-level interest
- Entities: Specific topics
- LightGBM bu çeşitliliği öğrenip optimal combination'ı bulur

#### Sonuçlar

**✓ Avantajlar**:

- Cold-start problemine dayanıklı (yeni kullanıcılar için de çalışıyor)
- İçerik tabanlı, her makale için geçerli
- Hesaplama maliyeti düşük (SVD sayesinde)
- Multiple text types → diverse signals
- Real-time inference ready (embedding'ler önceden hesaplanabilir)

**✗ Dezavantajlar**:

- **Sınırlı etki**: Feature importance analizinde orta-düşük önem
- Model performansına katkısı **%2-3 civarında**
- Davranışsal özellikler (tıklama, zaman) çok daha etkili
- Semantic anlam sınırlı (BERT vs TF-IDF)
- Synonyms yakalanmıyor ("car" ≠ "automobile")

#### Feature Importance Sıralaması

```
Top 10 Most Important Features:
1. c_time_min_diff                    (28.5%) ← Temporal
2. i_impression_times_in_1h           (12.3%) ← Behavioral
3. a_total_inviews                     (8.7%) ← Popularity
4. c_user_count_past_1h_ratio          (6.4%) ← Behavioral
5. u_total_read_time_mean              (5.2%) ← User profile
6. c_topics_count_svd_sim              (4.1%) ← TF-IDF ✓
7. i_total_pageviews_mean              (3.8%) ← Impression
8. c_title_tfidf_svd_sim               (2.9%) ← TF-IDF ✓
9. a_sentiment_score                   (2.7%) ← Content
10. c_body_tfidf_svd_sim               (2.1%) ← TF-IDF ✓
```

**Yorum**: TF-IDF özellikleri top 20'de yer alıyor ama **temporal ve behavioral features çok daha baskın**.

#### Neden Sınırlı Etki?

1. **Domain özellikleri**: Haber domain'i çok hızlı değişiyor
   
   - Bugünün haberleri yarın irrelevant
   - İçerik benzerliği statik, trend'leri yakalamıyor

2. **Zamansallık baskın**: Kullanıcılar **yeni** haberleri tercih ediyor
   
   - `c_time_min_diff` (makale ne kadar yeni) en önemli feature
   - İçerik benzerliği ikincil kalıyor

3. **Behavioral signals güçlü**: Geçmiş tıklama davranışı içerikten daha iyi predictor
   
   - Kullanıcının son 1 saatteki davranışı
   - Makale ne kadar impression almış
   - Impression içindeki sıra (rank)

---

### 2.1.2. Semantic Clustering Özellikleri (Deneysel)

**NOT**: Bu yaklaşım **TF-IDF similarity features'dan (bölüm 2.1) tamamen farklı bir yöntem**. Similarity features sadece cosine similarity hesaplarken, bu yaklaşım clustering + user profiling yapıyor.

#### Yöntem Detayı

Makaleleri **semantik cluster'lara** ayırıp, her kullanıcı için **cluster distribution profili** oluşturuyoruz. Bu bir **içerik tabanlı collaborative filtering** yaklaşımı.

**Pipeline Akışı (Tek Bir Method İçinde)**:

```
1. TF-IDF Vectorization  → Makale vektörleri oluştur
         ↓
2. K-Means Clustering    → Makaleleri K cluster'a ayır
         ↓
3. User Profiling        → Her kullanıcının cluster distribution'ını hesapla
         ↓
4. Feature Extraction    → Aday makale için cluster-based features üret
```

**Detaylı İşlem Adımları**:

```python
# === ADIM 1: TF-IDF EMBEDDING ===
# Not: Bu similarity features'dan AYRI bir işlem
# Orada similarity için kullanıyorduk, burada clustering için kullanıyoruz
vectorizer = TfidfVectorizer(
    max_features=5000,           # Clustering için 5K yeterli
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    stop_words='english'
)
article_matrix = vectorizer.fit_transform(articles['title'])
article_embeddings = normalize(article_matrix, norm='l2')
# Boyut: [N_articles, 5000]

# === ADIM 2: K-MEANS CLUSTERING ===
# Bu adım embedding'lerin üzerine uygulanıyor
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(
    n_clusters=K,                # K = 100 optimal
    random_state=42,
    batch_size=10000,            # Memory efficiency için
    max_iter=100
)
article_clusters = kmeans.fit_predict(article_embeddings)
# Çıktı: article_clusters[article_id] = cluster_id (0 to K-1)
# Örnek: article_12345 → cluster 47 (Sports haberleri)

# === ADIM 3: USER CLUSTER PROFILING ===
def get_user_cluster_distribution(user_history_articles):
    """
    Kullanıcının geçmişindeki makalelerin cluster dağılımı
    Bu kullanıcının hangi konulara ilgili olduğunu gösterir
    """
    history_clusters = article_clusters[user_history_articles]
    cluster_counts = np.bincount(history_clusters, minlength=K)
    cluster_distribution = cluster_counts / cluster_counts.sum()
    return cluster_distribution

# Örnek: Kullanıcı 20 makale okumuş
# 10 makale → cluster 47 (Sports)
# 7 makale  → cluster 12 (Tech)
# 3 makale  → cluster 89 (Politics)
# Distribution: [0, 0, ..., 0.50(Sports), ..., 0.35(Tech), ..., 0.15(Politics), ...]
# Bu kullanıcı %50 sports, %35 tech, %15 politics ilgili

# === ADIM 4: FEATURE EXTRACTION ===
candidate_cluster_id = article_clusters[candidate_article_id]
user_cluster_dist = get_user_cluster_distribution(user_history_articles)

features = {
    # Ana özellik: Aday makalenin cluster'ına kullanıcının ilgisi
    'user_interest_in_candidate_cluster': user_cluster_dist[candidate_cluster_id],
    
    # Aday makale kullanıcının en ilgili olduğu cluster'da mı?
    'user_top_cluster_match': 1 if candidate_cluster_id == user_cluster_dist.argmax() else 0,
    
    # Kullanıcı ne kadar focused/diverse?
    'user_cluster_entropy': -sum(user_cluster_dist * np.log(user_cluster_dist + 1e-10)),
    
    # Tüm cluster distribution (model kendisi öğrensin)
    # user_cluster_dist_0, user_cluster_dist_1, ..., user_cluster_dist_99
}
```

**BERT Entegrasyonu (Gelecek İyileştirme)**:

Şu an TF-IDF ile embedding oluşturuyoruz. Sistem modular olduğu için **BERT kolayca entegre edilebilir**:

```python
# Sadece Adım 1'i değiştirmek yeterli:
if embedding_type == 'bert':
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    article_embeddings = model.encode(articles['title'], show_progress_bar=True)
    # Boyut: [N_articles, 384]
    # Daha zengin semantic representation
else:  # 'tfidf' (şu an aktif)
    # Yukarıdaki TF-IDF kodu
    
# Adım 2-3-4 AYNI KALIR! (clustering + profiling + features)
# Bu yüzden BERT'i entegre etmek çok kolay
```

#### Standart Clustering'den Farkımız

**1. User-Centric Approach**:
- Standart: Dokümanlara cluster label'ı ver, iş bitti
- Bizim: Kullanıcı geçmişinden **cluster distribution profili** çıkar
- **Avantaj**: "Bu kullanıcının hangi cluster'lara ne kadar ilgisi var?" bilgisi

**2. Distribution-based Features**:
- Sadece "aday makale hangi cluster'da?" değil
- "Kullanıcının bu cluster'a ne kadar ilgisi var?" özelliği üretiyoruz
- Entropy feature: Kullanıcı focused mı (düşük entropy) yoksa diverse mi (yüksek entropy)?

**3. Granular K Selection**:
```python
# Farklı K değerleri denedik
K_values = [50, 100, 200, 500]
# Her biri farklı granularity sağlıyor
# K=50  → Coarse topics (Sports, Tech, Politics)
# K=500 → Fine-grained topics (Football transfer news, iPhone reviews)
```

**4. Modular Embedding Strategy**:
- TF-IDF (şu an) veya BERT (gelecekte) kullanılabilir
- Embedding yöntemi değişse de clustering pipeline aynı
- Config'den tek satırla değiştirilebilir

#### Üretilen Özellikler (K=100 için)

**103 Cluster Feature** (her aday makale için):

1. **user_cluster_distribution_{0...99}**: Kullanıcının her cluster'a ilgi oranı (100 özellik)
   - `user_cluster_distribution_47 = 0.50` → Kullanıcı geçmişindeki makalelerin %50'si cluster 47'de (Sports)
   
2. **user_interest_in_candidate_cluster**: Aday makalenin cluster'ına kullanıcının ilgisi (1 özellik)
   - En direkt özellik, model için çok bilgilendirici

3. **user_cluster_entropy**: Kullanıcı ilgi dağılımının entropy'si (1 özellik)
   - Düşük: Focused user (birkaç cluster'a çok ilgili)
   - Yüksek: Diverse user (her konudan okur)

4. **is_user_top_cluster**: Aday makale kullanıcının en ilgili olduğu cluster'da mı? (1 özellik)
   - Binary feature: 1 = evet, 0 = hayır

**Toplam**: 103 özellik (K=100 için)

#### Parametre Deneyimi

```python
# Denenen konfigürasyonlar
configs = [
    {'K': 50,  'embedding': 'tfidf', 'max_features': 5000},
    {'K': 100, 'embedding': 'tfidf', 'max_features': 5000},
    {'K': 200, 'embedding': 'tfidf', 'max_features': 5000},
    {'K': 500, 'embedding': 'tfidf', 'max_features': 10000},
]
```

**Bulgular**:
- **K=100** optimal (K=50 çok coarse, K=500 overfitting)
- **max_features=5000**: Yeterli (10K ile fark minimal)

#### Sonuçlar

**✓ Avantajlar**:

- Kullanıcı ilgi alanlarını **daha structured** şekilde temsil eder
- Cold-start için faydalı (cluster distribution hızla oluşur)
- Interpretability: "Bu kullanıcı %50 Sports, %35 Tech ilgili"
- Model için yüksek boyutlu (103 feature) ama informative signal
- BERT'e geçiş kolay (sadece embedding yöntemi değişir)

**✗ Dezavantajlar**:

- **Performans etkisi çok sınırlı**: Model AUC'ye katkı ~**0.0005-0.001**
- **Hesaplama maliyeti yüksek**: K-Means fitting + prediction
- **Memory overhead**: 103 extra feature per candidate → 13M candidates için büyük
- TF-IDF similarity features zaten benzer bilgiyi veriyor
- **Trade-off**: Çok feature eklemek → training yavaşlıyor, minimal gain

#### Karar

**Semantic clustering features şu an KULLANILMIYOR** (`use_semantic_clusters=false`).

**Neden?**:
- Maliyet/fayda oranı kötü
- Similarity features yeterli coverage sağlıyor
- Training time 2x uzuyor, performance gain negligible (~0.001 AUC)
- Production'da inference latency artırır

**Ne zaman kullanılabilir?**:
- Eğer cold-start performance kritik olursa
- Eğer interpretability önemliyse (cluster analysis için)
- Eğer compute budget sınırsızsa
- BERT embeddings kullanılabilirse (semantic quality artar)

#### Kod Konumu

```bash
experiments/015_train_third/semantic_cluster_features.py
```

**Açma/Kapama**:
```yaml
# experiments/015_train_third/config.yaml
use_semantic_clusters: false  # true yaparsan aktif olur
semantic_cluster_config:
  n_clusters: 100
  embedding_type: 'tfidf'  # Şu an TF-IDF, ileride 'bert' olabilir
  max_features: 5000
```

---

### 2.2. Bipartite Graph + Node2Vec Embeddings (BAŞARISIZ YAKLAŞIM)

#### Yöntem

Kullanıcı-makale etkileşimlerinden **iki parçalı çizge (bipartite graph)** oluşturup, **Node2Vec** algoritması ile node embedding'leri öğreniyoruz.

**Çizge Yapısı**:

```
Nodes:
- Kullanıcılar: {u1, u2, u3, ..., u_N}
- Makaleler: {a1, a2, a3, ..., a_M}

Edges:
- Kullanıcı-makale tıklama etkileşimleri (label=1 olanlar)

Örnek:
u1 --tıkladı--> a5
u1 --tıkladı--> a12
u2 --tıkladı--> a5
u2 --tıkladı--> a8
```

**Node2Vec Parametreleri**:

```yaml
graph_embedding_dim: 64      # Embedding boyutu
graph_walk_length: 30        # Random walk uzunluğu
graph_num_walks: 200         # Her node için walk sayısı
graph_workers: 4             # Paralel işlem sayısı
```

**Algorithm Flow**:

1. Train verisi üzerinden graph oluştur
2. Node2Vec ile random walk'lar üret
3. Word2Vec ile embedding'leri öğren
4. Her kullanıcı ve makale için 64-boyutlu vektör elde et

#### Üretilen Özellikler

**Yaklaşım A - Tüm Embeddings (131 feature)**:

- User embedding: 64 boyut
- Article embedding: 64 boyut
- Interaction features: 3 (dot product, cosine sim, euclidean distance)

**Yaklaşım B - Sadece Interaction Features (3 feature)**:

- `g_dot_product`: user_emb · article_emb
- `g_cosine_sim`: cos(user_emb, article_emb)
- `g_euclidean_dist`: ||user_emb - article_emb||

#### Deneysel Sonuçlar

| Konfigürasyon            | AUC        | nDCG@5     | nDCG@10    | MRR        | Feature Sayısı |
| ------------------------ | ---------- | ---------- | ---------- | ---------- | -------------- |
| **Baseline (Graph yok)** | **0.8453** | **0.7140** | **0.7305** | **0.6507** | 103            |
| Graph + Tüm Embeddings   | 0.7299     | 0.5624     | 0.6035     | 0.4954     | 234            |
| Graph + Interaction Only | 0.7508     | 0.5900     | 0.6241     | 0.5221     | 106            |

**Performans Düşüşü**:

- AUC: **-11.2%** (Yaklaşım B) ile **-11.6%** (Yaklaşım A)
- nDCG@10: **-14.6%** ile **-17.4%**
- MRR: **-19.8%** ile **-23.9%**

**✗ Sonuç**: Graph-based features **ciddi performans kaybına** yol açtı!

#### Başarısızlık Nedenleri: Detaylı Analiz

##### 1. Cold-Start Problemi (ANA NEDEN)

**Problem**: Train ve validation/test setlerinde **farklı kullanıcılar** var!

```
Train Set:
- Kullanıcılar: {u1, u2, ..., u8844}
- Bu kullanıcılar üzerinden graph oluşturuldu
- Node2Vec ile embedding'ler öğrenildi

Validation Set:
- Kullanıcılar: {u8845, u8846, ..., u16000}
- Bu kullanıcılar graph'ta YOK!
```

**Coverage Analizi**:

| Set        | Unique Users | Graph'ta Bulunan | Coverage |
| ---------- | ------------ | ---------------- | -------- |
| Train      | 8,844        | 8,844            | 100%     |
| Validation | ~7,500       | ~800             | **~10%** |
| Test       | ~7,600       | ~750             | **~10%** |

**Sonuç**: Validation/test kullanıcılarının **%90'ı** train graph'ında yok!

##### 2. Zero Embedding Problemi

Graph'ta olmayan kullanıcılar için kod şöyle çalışıyor:

```python
def get_user_embedding(self, user_id: str) -> np.ndarray:
    return self.user_embeddings.get(
        str(user_id), 
        np.zeros(self.embedding_dim)  # ← Bulunamazsa SIFIR VEKTÖR!
    )
```

**Validation/test için**:

```python
user_embedding = [0, 0, 0, ..., 0]  # 64 sıfır
article_embedding = [0.3, -0.1, 0.5, ...]  # Gerçek değerler

# Interaction features:
g_dot_product = [0,0,...,0] · [0.3,-0.1,...] = 0
g_cosine_sim = 0 / (||0|| * ||article||) = 0
g_euclidean_dist = ||[0,0,...,0] - article|| = ||article|| = sabit
```

**Sonuç**: %90 örnekte graph features **sıfır veya sabit**! Model hiçbir şey öğrenemiyor.

##### 3. Noise Injection ve Overfitting

131 yeni feature eklendi ama bunların %90'ı **bilgi taşımıyor** (sıfır):

| Feature Count  | Training AUC | Validation AUC | Overfitting Gap |
| -------------- | ------------ | -------------- | --------------- |
| 103 (baseline) | 0.866        | 0.845          | **0.021** ✓     |
| 234 (graph)    | 0.901        | 0.730          | **0.171** ✗     |

Overfitting gap **8 kat arttı**!

**Sebep**: Model train setinde graph features'a fit oldu (çünkü train'de coverage %100), ama validation'da bu features anlamlı değil.

##### 4. Dataset Split Stratejisi

**Mevcut strateji**: Temporal split (zaman bazlı)

```python
# Train: İlk N gün
# Validation: Son M gün

# Problem: Farklı günlerde farklı kullanıcılar aktif!
```

**Neden bu strateji seçildi?**

- Gerçek dünya senaryosunu simüle eder (gelecek tahmini)
- Yeni kullanıcılar sürekli geliyor (cold-start realistic)

**Graph methods için ideal olmaması**:

- Graph methods **tüm node'ları** bilmek ister
- Cold-start'ta başarısız oluyorlar

##### 5. Domain Özellikleri: Hızlı Değişen Haber Ortamı

Haber domain'i statik değil, **son derece dinamik**:

**Makale Yaşam Döngüsü**:

```
Publish → [Peak 0-6 saat] → [Decline 6-24 saat] → [Dead 24+ saat]

Ortalama makale ömrü: ~12 saat
Train-test arasındaki zaman: ~7 gün
```

**Train ile Test Arasında Makale Overlap**:

- Train'deki makalelerin %5'i test setinde var
- Test'teki makalelerin %95'i train'de görmediğimiz yeni makaleler

**Graph için sonuç**:

- Article embeddings de çoğunlukla cold-start
- Hem kullanıcı hem makale yeni → double cold-start!

#### Teorik Beklenti vs Gerçeklik

**Teorik (Node2Vec paper)**:

```
user_A → article_1 ← user_B → article_2

user_A için article_2 önerisinde:
user_B üzerinden dolaylı bağlantı yakalanır
İkinci derece komşuluk işe yarar
```

**Bizim durumumuzda**:

```
Train:  user_A → article_1
Test:   user_NEW → article_2

user_NEW graph'ta yok!
article_2 graph'ta yok!
→ Hiçbir bağlantı yakalanamıyor
```

#### Neden Graph Yöntemleri Bu Domain'de Başarısız?

**1. Temporal Dynamics Baskın**:

- Graph statik bir yapı, zamanı modellemiyor
- Haber domain'inde **"ne kadar yeni"** en önemli faktör
- Graph embedding'leri bu bilgiyi yakalayamıyor

**2. Content Heterogeneity**:

- Her makale benzersiz içerik (günlük haberler)
- User-article etkileşimleri tekrar etmiyor
- Graph'ın gücü **recurring patterns**, burada yok

**3. Cold-Start Kaçınılmaz**:

- Yeni kullanıcılar sürekli geliyor
- Yeni makaleler sürekli yayınlanıyor
- Graph her zaman incomplete olacak

**4. Sparse Interactions**:

- Her kullanıcı ortalama 5-10 makale okuyor
- Graph çok seyrek (sparse)
- Meaningful embedding öğrenmek zor

---

### 2.3. Feature Selection: En Önemli Özellikleri Seçme (BAŞARISIZ)

#### Yöntem

Graph features'ların çok fazla olması nedeniyle (131 feature), **sadece en yüksek importance'a sahip olanları** seçmeyi denedik.

**Strateji**:

1. Tüm features ile bir model eğit
2. LightGBM feature importance hesapla
3. Top-K (örn: K=20) graph feature seç
4. Sadece bu features ile yeni model eğit

#### Parametreler

```python
# Feature selection
selection_method: "importance"
top_k_features: 20
threshold: 0.01  # %1'den düşük importance → kes
```

#### Sonuçlar

| Approach                | Features Used | AUC       | nDCG@10   | Performans                 |
| ----------------------- | ------------- | --------- | --------- | -------------------------- |
| All Graph Features      | 131           | 0.730     | 0.604     | ✗ Kötü                     |
| Top-20 Graph Features   | 20            | 0.753     | 0.624     | ✗ Hala kötü                |
| Top-10 Graph Features   | 10            | 0.768     | 0.641     | ✗ Hala baseline'ın altında |
| **No Graph (Baseline)** | **0**         | **0.845** | **0.730** | **✓ En iyi**               |

**Sonuç**: Feature selection yardımcı olmadı, **graph features'ın tamamının çıkarılması** en iyi sonucu verdi.

#### Neden Feature Selection İşe Yaramadı?

**1. Zero Embedding problemi düzelmiyor**:

- Top-K feature seçmek zero embedding'leri düzeltmiyor
- Hala %90 örnekte sıfır veya sabit değerler

**2. Information loss minimal**:

- 131 feature'ın hepsi az bilgi taşıyor
- En iyi 20'si de yeterli bilgi sağlamıyor
- Kayıp: çok az, kazanç: yok

**3. Overfitting devam ediyor**:

- Feature sayısı azaldı (234 → 123)
- Ama model hala train setine overfit oluyor
- Çünkü **feature quality** değişmiyor, sadece **quantity**

**4. Root cause çözülmüyor**:

- Asıl problem: cold-start ve zero embeddings
- Feature selection bu problemi çözmüyor
- Sadece semptomları hafifletiyor

---

### 2.4. BERT-Based Semantic Clustering (DENENIYOR)

#### Yöntem

TF-IDF'in sınırlamalarını aşmak için **BERT embeddings** kullanarak semantic clustering.

**Yaklaşım**:

1. **Global Clustering**: Tüm makaleleri BERT ile embed et, K-Means ile cluster'la
2. **User Profiling**: Her kullanıcının okuma geçmişinden cluster distribution hesapla
3. **Match Features**: Kullanıcı profili ile aday makale cluster'ı arasında benzerlik

#### Parametreler

```yaml
use_semantic_clusters: true
semantic_n_clusters: 30           # K-Means cluster sayısı
semantic_text_column: body        # Hangi metin: title/body/subtitle
bert_model: paraphrase-multilingual-MiniLM-L12-v2
```

#### Üretilen Özellikler

1. **bert_user_cluster_X** (30 feature): Kullanıcının cluster dağılımı
2. **bert_article_cluster** (1 feature): Makalenin cluster ID'si
3. **bert_user_article_cluster_match** (1 feature): En sevilen cluster match mi?
4. **bert_user_article_cluster_affinity** (1 feature): O cluster'a tıklama oranı

**Toplam**: 33 feature

#### Beklenen Avantajlar

**✓ TF-IDF'e göre**:

- Semantic anlam yakalıyor (kelime bazında değil)
- Synonyms ve paraphrases'i anlıyor
- Multilingual (Türkçe/İngilizce karışık içerik)

**✓ Graph'a göre**:

- Cold-start problemi yok (content-based)
- Her makale için cluster assignment var
- Yeni kullanıcılar için geçmişten profil oluşturuyor

#### Mevcut Durum

**Henüz tam test edilmedi**, ama initial signs:

- Hesaplama maliyeti yüksek (BERT inference)
- TF-IDF'den daha iyi sonuç bekleniyor
- %5-7 performans artışı hedefleniyor

---

## 3. Model Mimarisi ve Parametreler

### 3.1. Algoritma Seçimi

**Model**: LightGBM (Light Gradient Boosting Machine)

**Neden LightGBM?**

- ✓ Ranking task'leri için optimize (LambdaRank objective)
- ✓ Büyük veri setlerinde hızlı
- ✓ Categorical features native support
- ✓ Memory efficient
- ✓ Interpretable (feature importance)

**Alternatifler değerlendirildi**:

- ✗ XGBoost: Daha yavaş, benzer performans
- ✗ CatBoost: Denendi (experiments/016_catboost/), LightGBM'den marginally worse
- ✗ Neural Networks: Overfitting, interpretability düşük

### 3.2. LightGBM Hiperparametreler

```yaml
lgbm:
  objective: lambdarank              # Learning-to-rank için

  # Eğitim parametreleri
  num_boost_round: 1200              # İterasyon sayısı (800'den artırıldı)
  early_stopping_round: 100          # Patience (80'den artırıldı)
  learning_rate: 0.03                # Öğrenme hızı (0.05'ten düşürüldü)

  # Tree parametreleri
  max_depth: 10                      # Ağaç derinliği (8'den artırıldı)
  num_leaves: 256                    # Yaprak sayısı (128'den artırıldı)
  min_child_samples: 10              # Minimum örneklem sayısı

  # Regularization
  lambda_l2: 0.5                     # L2 regularization (0.1'den artırıldı)
  feature_fraction: 0.7              # Her ağaçta feature'ların %70'i (0.8'den düşürüldü)
  bagging_fraction: 0.8              # Bootstrap sampling oranı
  bagging_freq: 1                    # Her iterasyonda bagging

  # Evaluation
  metric: [ndcg, auc]                # İzlenen metrikler
  ndcg_eval_at: [5, 10]              # nDCG@5 ve nDCG@10
  first_metric_only: true            # İlk metriğe göre early stop
```

### 3.3. Parametre Tuning Mantığı

#### Overfitting'i Önleme

**Problem**: İlk denemelerde train AUC: 0.90, validation AUC: 0.78 (overfitting)

**Çözümler**:

1. **Learning rate düşürüldü** (0.05 → 0.03):
   
   - Daha yavaş öğrenme
   - Daha stabil gradients
   - Better generalization

2. **Regularization artırıldı** (lambda_l2: 0.1 → 0.5):
   
   - Ağırlıklara penalty
   - Karmaşık modelleri cezalandır

3. **Feature fraction azaltıldı** (0.8 → 0.7):
   
   - Her ağaç feature'ların subset'ini görüyor
   - Ensemble diversity artıyor

4. **Early stopping artırıldı** (80 → 100):
   
   - Daha sabırlı
   - Overfitting'e izin vermeden optimum nokta

#### Expressiveness Artırma

**Problem**: Model yeterince karmaşık pattern'leri yakalayamıyor

**Çözümler**:

1. **Max depth artırıldı** (8 → 10):
   
   - Daha derin ağaçlar
   - Daha karmaşık etkileşimler

2. **Num leaves artırıldı** (128 → 256):
   
   - Daha ince partition'lar
   - Better fit on training data

3. **Num boost round artırıldı** (800 → 1200):
   
   - Daha fazla iterasyon
   - Early stopping zaten var, fazladan zarar yok

**Trade-off**: Expressiveness ↑ + Regularization ↑ = Balanced model

### 3.4. Feature Engineering Detayları

#### Categorical Features

```yaml
cat_cols:
  - device_type          # mobile/desktop/tablet
```

**Neden sadece 1 categorical?**

- Impression_id, article_id, user_id: ID'ler, categorical olarak anlamsız
- Diğer categoricals: Ordinal encode edildi (sentiment, article_type)
- Device_type: Gerçek categorical (sıralama yok)

#### Derived Features

```yaml
# Multiplication features (interaction)
mul_cols_dict:
  topics_sim_mul_a_total_inviews:
    - c_topics_count_svd_sim          # TF-IDF similarity
    - a_total_inviews                 # Popularity

  topics_sim_mul_a_total_pageviews:
    - c_topics_count_svd_sim
    - a_total_pageviews

  topics_sim_mul_c_time_min_diff:
    - c_topics_count_svd_sim
    - c_time_min_diff                 # Recency

# Division features (normalization)
div_cols_dict:
  c_time_min_diff_imp_rate:
    - c_time_min_diff                 # Absolute time
    - i_time_min_diff_mean            # Impression average

  a_total_pageviews_imp_rate:
    - a_total_pageviews               # Article pageviews
    - i_total_pageviews_mean          # Impression average pageviews
```

**Mantık**:

- **Multiplication**: Feature interaction (similarity × popularity = "benzer VE popüler")
- **Division**: Normalization (article metric / impression average = "bu impression'da ne kadar iyi?")

#### Unused Columns

```yaml
unuse_cols:
  - impression_id       # ID, model için anlamsız
  - article_id          # ID, model için anlamsız
  - user_id             # ID, model için anlamsız
  - label               # Target değişken
```

---

## 4. Veri Stratejisi

### 4.1. Dataset Split

**Mevcut strateji**: Time-based validation split

```python
# Train set: Tüm train verisi
train_df = pl.read_parquet("train_dataset.parquet")

# Validation set: Validation verisinin ilk %50'si (TIME bazlı)
validation_impression_ids = sorted(validation_impression_ids)
split_idx = len(validation_impression_ids) // 2
validation_df = validation_data[impression_id in validation_impression_ids[:split_idx]]

# Test set: Validation verisinin son %50'si
test_df = validation_data[impression_id in validation_impression_ids[split_idx:]]
```

**Neden bu strateji?**

- ✓ Gerçek dünya simülasyonu: Gelecek tahmini
- ✓ Temporal leak prevention: Test setindeki impression'lar train'den sonra
- ✓ Cold-start realistic: Yeni kullanıcılar ve makaleler

**Trade-off**:

- ✗ Graph methods için uygun değil (farklı kullanıcılar)
- ✓ Production'a daha yakın (yeni kullanıcılar gelecek)

### 4.2. Sampling Strategy

```yaml
sampling_rate: 0.1        # Train verisinin %10'u
```

**Neden sampling?**

- **Computational efficiency**: Full train ~130M satır, sampling ile ~13M
- **Iteration speed**: Model geliştirme hızlandı (1 saat → 10 dakika)
- **Validation**: Small subset'te validate ettikten sonra full data'da test

**Sampling yöntemi**: Impression-level random sampling

```python
random.seed(cfg.exp.seed)
train_impression_ids = sorted(train_df["impression_id"].unique())
use_train_impression_ids = random.sample(
    train_impression_ids,
    int(len(train_impression_ids) * sampling_rate)
)
```

**Avantajlar**:

- Impression bazlı → her impression'ın tüm candidates'leri korunuyor
- Seed-based → Reproducible

### 4.3. Seed Management

**Reproducibility için tüm random sources seed'leniyor**:

```python
seed = cfg.exp.seed  # Default: 7

# Python random
random.seed(seed)

# NumPy
np.random.seed(seed)

# LightGBM
params['seed'] = seed

# Python hash (dict order)
os.environ['PYTHONHASHSEED'] = str(seed)
```

**Multiple seed çalıştırma** (significance analysis için):

```bash
# 3 farklı seed ile çalıştır
run_multiple_seeds.bat medium067_001 7 42 123
```

**Output organizasyonu**:

```
output/experiments/015_train_third/
├── medium067_001_seed7_20251221_143045/
├── medium067_001_seed42_20251221_150122/
└── medium067_001_seed123_20251221_153018/
```

Her seed için ayrı klasör → karşılaştırma ve istatistiksel test için

---

## 5. Sebep-Sonuç İlişkileri: Neden Böyle Gelişti?

### 5.1. TF-IDF'in Sınırlı Etkisi

**SEBEP 1: Domain Dynamics**

```
Haber yaşam döngüsü: [Publish] → [0-6h: peak] → [6-24h: decline] → [24h+: dead]

→ "Yeni mi?" sorusu "Benzer mi?"den daha önemli
→ Temporal features (c_time_min_diff) >> Content features (TF-IDF)
```

**SEBEP 2: Behavioral Signals**

```
Kullanıcı tıklama geçmişi:
- Son 1 saatte kaç makale gördü?
- Son impression'da ne yaptı?
- Kaç kere bu makaleyi gördü?

→ Davranışsal context > İçerik benzerliği
→ "Daha önce gördüm" > "Benzer içerik"
```

**SEBEP 3: Sparse User History**

```
Ortalama kullanıcı: 5-10 makale okuyor
TF-IDF profile: Az sayıda dokümandan oluşuyor

→ Profile güvenilirliği düşük
→ User representation zayıf
```

**SONUÇ**: TF-IDF çalışıyor ama **temporal ve behavioral features çok daha güçlü**.

### 5.2. Graph'ın Başarısızlığı

**SEBEP 1: Dataset Split → Cold-Start**

```
Time-based split seçimi:
  ↓
Train: İlk N gün → {users: U1}
Test: Son M gün → {users: U2}
  ↓
U1 ∩ U2 ≈ %10 overlap
  ↓
Graph'ta olmayan kullanıcılar → Zero embeddings
  ↓
%90 örnekte anlamlı feature yok
  ↓
Model öğrenememiş → Performans düşüşü
```

**SEBEP 2: Domain Nature → High Turnover**

```
Haber domain özellikleri:
  ↓
Yeni makaleler sürekli yayınlanıyor (günde ~1000 makale)
Yeni kullanıcılar sürekli geliyor (günlük %20 yeni)
  ↓
Graph sürekli değişiyor, static değil
  ↓
Learned embeddings hızla obsolete oluyor
  ↓
Historical graph → Current prediction: Weak correlation
```

**SEBEP 3: Sparse Interactions → Weak Embeddings**

```
Bipartite graph density:
  ↓
Users: 8,844
Articles: 15,000
Edges: 45,000 (clicks)
  ↓
Density: 45K / (8.8K × 15K) = 0.034%
  ↓
Çok seyrek graph → Node2Vec weak embeddings
  ↓
Random walk'lar meaning ful path bulamıyor
```

**SONUÇ**: Graph theoretically good, ama bizim **domain ve split stratejisi için uygun değil**.

### 5.3. Neden TF-IDF Yine de Kullanıyoruz?

**SEBEP 1: Cold-Start Coverage**

```
TF-IDF content-based:
  ↓
Her makale için hesaplanabiliyor (text her zaman var)
  ↓
Yeni kullanıcılar için de çalışıyor (geçmiş profile üzerinden)
  ↓
%100 coverage → Graph'tan çok daha iyi
```

**SEBEP 2: Incremental Gain**

```
Baseline (temporal + behavioral): AUC 0.829
+ TF-IDF features: AUC 0.845
  ↓
+1.6% absolute, +1.9% relative improvement
  ↓
Küçük ama anlamlı kazanç
```

**SEBEP 3: Low Cost**

```
TF-IDF hesaplama: ~5 dakika (offline)
Inference time: +0.01ms per prediction
  ↓
Cost-benefit ratio iyi
```

**SEBEP 4: Ensemble Benefit**

```
Model ensembling theory:
  ↓
Farklı signal'ler (temporal + behavioral + content) → Diverse features
  ↓
LightGBM feature interaction learning
  ↓
(TF-IDF × popularity) gibi combinations öğreniyor
  ↓
Synergy effects
```

**SONUÇ**: TF-IDF tek başına weak, ama **ensemble'da value-add**.

---

## 6. Sonuçlar ve Öneriler

### 6.1. Başarılı Yaklaşımlar

**✓ Temporal Features** (En Güçlü)

- `c_time_min_diff`: Makale ne kadar yeni?
- Feature importance: %28.5
- **Öneri**: Bu feature'ı mutlaka koru, daha fazla temporal feature eklenebilir

**✓ Behavioral Features** (Çok Güçlü)

- Kullanıcının son impression'larındaki davranışı
- Makaleyi daha önce gördü mü?
- Feature importance: %35 (toplamda)
- **Öneri**: Real-time behavioral signals ekle (son 5 dakika, son 1 saat)

**✓ TF-IDF Content Similarity** (Yardımcı)

- Topics, title, body similarity
- Feature importance: %8 (toplamda)
- **Öneri**: Koru, ama daha fazla invest etme

**✓ BERT Semantic Clustering** (Test ediliyor)

- TF-IDF'ten daha iyi semantic understanding bekleniyor
- **Öneri**: Test et, %5-7 gain hedefle

### 6.2. Başarısız Yaklaşımlar

**✗ Graph-Based Embeddings**

- Performance: -11.2% AUC
- **Sonuç**: Kullanma
- **Neden**: Cold-start coverage %10, domain dynamics uygun değil

**✗ Graph Feature Selection**

- Performance: Hala baseline'ın altında
- **Sonuç**: Root cause çözmüyor, vazgeç

### 6.3. Gelecek Adımlar

**Kısa Vadeli (1-2 hafta)**:

1. **BERT Clustering Test**
   
   - Full test ve evaluation yap
   - Baseline'la karşılaştır
   - %5+ gain ise production'a al

2. **Hyperparameter Tuning**
   
   ```yaml
   # Denenecek değerler
   learning_rate: [0.01, 0.03, 0.05]
   max_depth: [8, 10, 12]
   num_leaves: [256, 512]
   lambda_l2: [0.3, 0.5, 0.7]
   ```
   
   - Bayesian optimization kullan
   - Target: +2-3% gain

3. **Feature Engineering**
   
   - Daha fazla temporal feature:
     - `publish_hour`, `publish_day_of_week`
     - `time_since_last_click`
   - User behavior sequences:
     - `last_3_clicks_categories`
     - `click_pattern_vector`

**Orta Vadeli (1 ay)**:

4. **Ensemble Methods**
   
   - LightGBM + CatBoost ensemble
   - Stacking with linear meta-learner
   - Target: +3-5% gain

5. **Neural Network Ranker**
   
   - DeepFM veya Wide & Deep
   - Embeddings + behavioral features
   - Benchmark against LightGBM

**Uzun Vadeli (2-3 ay)**:

6. **Real-time Features**
   
   - Streaming features (son 5 dakika davranış)
   - Real-time popularity signals
   - Requires infrastructure change

7. **User Cold-Start Strategy**
   
   - Yeni kullanıcılar için özel model
   - Content-only model → Hybrid model transition
   - Adaptive weighting

---

## 7. Teknik Detaylar: Reproducibility

### 7.1. Environment

```yaml
Python: 3.10+
Key packages:
  - lightgbm==4.0.0
  - polars==0.19.0
  - scikit-learn==1.3.0
  - sentence-transformers==2.2.2  # BERT için
```

### 7.2. Run Commands

**Single run**:

```bash
run.bat train --exp=medium067_001 --seed=7
```

**Multiple seeds** (significance testing):

```bash
run_multiple_seeds.bat medium067_001 7 42 123
```

### 7.3. Output Structure

```
output/experiments/015_train_third/medium067_001_seed7_20251221_143045/
├── model_dict_model.pkl              # Trained model
├── importance_model.png              # Feature importance plot
├── validation_result.parquet         # Validation predictions
├── test_result.parquet               # Test predictions
├── results.txt                       # Metrics summary
└── run.log                           # Detailed logs
```

### 7.4. Metrics Reported

```
Validation Set:
- AUC: 0.845
- nDCG@5: 0.714
- nDCG@10: 0.730
- MRR: 0.651

Test Set:
- AUC: 0.842
- nDCG@5: 0.709
- nDCG@10: 0.728
- MRR: 0.647
```

---

## 8. Öğrenilen Dersler

### 8.1. Domain Knowledge Kritik

**Ders**: Algoritma seçiminde domain özelliklerini anlamak şart.

**Örnek**: 

- Academic papers: "Graph embeddings work great for recommendations"
- Bizim domain: "Haber çok hızlı değişiyor, graph static kalmış"
- **Sonuç**: Paper'da iyi ≠ Bizim problem'de iyi

### 8.2. Cold-Start Prevention > Cold-Start Solution

**Ders**: Cold-start'tan kaçınmak, çözmekten daha kolay.

**Örnek**:

- Graph: Cold-start'ı çözmek için embedding learning
- TF-IDF: Cold-start yok, çünkü content-based
- **Sonuç**: Prevention-first approach kazandı

### 8.3. Interpretability Matters

**Ders**: Model debugging ve improvement için interpretability şart.

**Örnek**:

- Graph başarısız olunca: Feature importance sayesinde hızlıca anladık
- LightGBM: Hangi features önemli açıkça görülüyor
- Neural network olsaydı: Debug etmek daha zor olurdu

### 8.4. Temporal Dynamics > Structural Patterns

**Ders**: Haber domain'inde "ne zaman" "ne" den önemli.

**Örnek**:

- Graph: "Bu makaleyi okuyanlar bunları da okudu" (structural)
- Temporal: "Son 1 saatte yayınlanan en popüler" (temporal)
- **Sonuç**: Temporal features %28.5, graph %0 importance

### 8.5. Incremental Improvements Sum Up

**Ders**: Birçok küçük improvement > Tek büyük breakthrough

**Timeline**:

```
Week 1: Baseline → AUC 0.78
Week 2: + Temporal features → AUC 0.82 (+4%)
Week 3: + Behavioral features → AUC 0.83 (+1%)
Week 4: + TF-IDF similarity → AUC 0.845 (+1.5%)
Week 5: + Hyperparameter tuning → AUC 0.845 (+0%)
Week 6: - Graph features → AUC 0.845 (0%, ama daha hızlı)
```

Total: **+6.5% improvement** through incremental steps

---

## Ekler

### Ek A: Feature Listesi (Tüm 103 Feature)

```
# Article features (a_*): 15 features
a_premium, a_category_article_type, a_total_inviews, 
a_total_pageviews, a_total_read_time, a_sentiment_score, 
a_ordinal_sentiment_label, a_click_rank, a_click_count,
a_click_ratio, a_inviews_per_pageviews, 
a_read_time_per_pageviews, a_read_time_per_inviews, ...

# User features (u_*): 12 features
u_total_inviews_mean, u_total_inviews_std, 
u_total_pageviews_mean, u_total_pageviews_std,
u_total_read_time_mean, u_total_read_time_std, ...

# Impression features (i_*): 18 features
i_impression_times_in_1h, i_impression_times_in_24h,
i_elapsed_time_since_last_impression, i_time_min_diff_mean, ...

# Candidate features (c_*): 50 features
c_time_min_diff, c_topics_count_svd_sim, c_title_tfidf_svd_sim,
c_body_tfidf_svd_sim, c_category_tfidf_sim, 
c_is_already_clicked, c_user_count_past_1h_ratio, ...

# Derived features: 8 features
topics_sim_mul_a_total_inviews, 
topics_sim_mul_a_total_pageviews,
c_time_min_diff_imp_rate, ...
```

### Ek B: Deneysel Sonuçlar Tablosu

| Experiment       | Config       | AUC   | nDCG@10 | MRR   | Note     |
| ---------------- | ------------ | ----- | ------- | ----- | -------- |
| Baseline         | 103 features | 0.845 | 0.730   | 0.651 | ✓ Best   |
| + Graph (all)    | 234 features | 0.730 | 0.604   | 0.495 | ✗ -11.6% |
| + Graph (top-20) | 123 features | 0.753 | 0.624   | 0.522 | ✗ -9.2%  |
| + BERT (pending) | 136 features | TBD   | TBD     | TBD   | Testing  |

### Ek C: Computational Requirements

**Training Time**:

- Full data (~130M rows): ~6 saat
- Sampled data (%10, ~13M rows): ~35 dakika
- Graph embedding (Node2Vec): +2 saat (unused)
- BERT clustering: +1 saat (testing)

**Memory**:

- Peak RAM: ~32GB (full data)
- Peak RAM: ~8GB (sampled data)
- LightGBM model size: ~150MB

**Infrastructure**:

- CPU: 40 cores kullanılıyor
- GPU: Kullanılmıyor (LightGBM CPU-only)
- Storage: ~50GB (preprocessed features)

---

**Son Güncelleme**: Aralık 2025  
**Rapor Versiyonu**: 1.0  
**Yazar**: EB-NeRD Proje Ekibi
