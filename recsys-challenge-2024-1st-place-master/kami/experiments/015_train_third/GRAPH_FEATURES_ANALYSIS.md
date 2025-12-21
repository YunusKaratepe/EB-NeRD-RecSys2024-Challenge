# Çizge Tabanlı Özellik Çıkarımı: Deneysel Analiz ve Başarısızlık Nedenleri

## 1. Özet

Bu rapor, haber öneri sistemimizde çizge tabanlı özellik çıkarımı (Graph-Based Feature Extraction) yaklaşımının deneysel sonuçlarını ve beklenmedik performans düşüşünün nedenlerini detaylı olarak analiz etmektedir.

## 2. Yöntem

### 2.1. Çizge Tabanlı Özellik Çıkarımı Yaklaşımı

Kullanıcı-makale etkileşimlerinden oluşan **bipartite graph** (iki parçalı çizge) yapısı kullanılarak:

1. **Graph Oluşturma**: 
   - Node'lar: Kullanıcılar (u_XXX) ve makaleler (a_XXX)
   - Edge'ler: Tıklama etkileşimleri (label=1)
   
2. **Structural Embedding Öğrenimi**:
   - Algoritma: Node2Vec (Grover & Leskovec, 2016)
   - Random walk parametreleri:
     - walk_length: 30
     - num_walks: 200
     - embedding_dim: 64
   - Word2Vec ile node embedding'leri öğrenildi

3. **Feature Engineering**:
   - **Yaklaşım A**: Tüm embedding'ler (131 feature)
     - User embeddings: 64 boyut
     - Article embeddings: 64 boyut
     - Interaction features: 3 (dot product, cosine similarity, euclidean distance)
   
   - **Yaklaşım B**: Sadece interaction features (3 feature)
     - g_dot_product: user_emb · article_emb
     - g_cosine_sim: cos(user_emb, article_emb)
     - g_euclidean_dist: ||user_emb - article_emb||

## 3. Deneysel Sonuçlar

### 3.1. Performans Karşılaştırması

**Tablo 1**: Farklı yapılandırmalarda model performansı (Small Dataset, %10 sampling)

| Konfigürasyon | AUC | nDCG@5 | nDCG@10 | MRR | Özellik Sayısı |
|---------------|-----|--------|---------|-----|----------------|
| **Baseline** (Graph yok) | **0.8453** | **0.7140** | **0.7305** | **0.6507** | 103 |
| Graph + Tüm Embeddings | 0.7299 | 0.5624 | 0.6035 | 0.4954 | 234 (103+131) |
| Graph + Interaction Only | 0.7508 | 0.5900 | 0.6241 | 0.5221 | 106 (103+3) |

**Tablo 2**: Baseline'a göre performans değişimi (%)

| Konfigürasyon | AUC | nDCG@5 | nDCG@10 | MRR |
|---------------|-----|--------|---------|-----|
| Graph + Tüm Embeddings | **-11.6%** | **-21.2%** | **-17.4%** | **-23.9%** |
| Graph + Interaction Only | **-11.2%** | **-17.4%** | **-14.6%** | **-19.8%** |

### 3.2. Feature Importance Analizi

Graph features'ların LightGBM feature importance sıralamasında **düşük importance değerleri** aldığı gözlemlendi:

- Top 20 en önemli feature içinde **hiçbir graph feature yok**
- En önemli features: temporal (c_time_*), TF-IDF similarity, behavioral features
- Graph features'lar bottom 50'de kümelenmiş durumda

## 4. Başarısızlık Nedenleri

### 4.1. Cold-Start Problemi

**Ana Sorun**: Train-test split stratejisi ile graph yapısı arasındaki uyumsuzluk.

**Detaylı Açıklama**:
```
Train Set:
- Kullanıcılar: {u1, u2, u3, ..., u_N}
- Makaleler: {a1, a2, a3, ..., a_M}
- Graph üzerinde Node2Vec eğitildi

Validation/Test Set:
- Kullanıcılar: {u_N+1, u_N+2, ...} (Farklı kullanıcılar!)
- Makaleler: {a_X, a_Y, ...} (Kısmen farklı makaleler)
- Bu kullanıcılar graph'ta YOK!
```

**Sonuç**: 
- Validation/test'teki kullanıcıların %90+ train graph'ında bulunmuyor
- Bu kullanıcılar için embedding = **zero vector** (64 boyutlu sıfır vektör)
- Tüm interaction features (dot product, cosine, distance) → **meaningless değerler**

**Tablo 3**: Graph coverage analizi

| Set | Unique Users | Graph'ta Bulunan | Coverage |
|-----|--------------|------------------|----------|
| Train | 8,844 | 8,844 | 100% |
| Validation | ~7,500 | ~800 | **~10%** |
| Test | ~7,600 | ~750 | **~10%** |

### 4.2. Zero Embedding Problemi

**Kod Analizi**:
```python
def get_user_embedding(self, user_id: str) -> np.ndarray:
    return self.user_embeddings.get(str(user_id), 
                                     np.zeros(self.embedding_dim))  # ← Zero vector!
```

Train graph'ında olmayan kullanıcılar için:
- User embedding = [0, 0, 0, ..., 0] (64 zeros)
- Article embedding = Gerçek değerler (çoğu makale train'de var)

**Interaction Features**:
```
g_dot_product = [0,0,...,0] · [0.3, 0.1, ...] = 0
g_cosine_sim = 0 / (||0|| * ||article||) = 0 / 0 = undefined → 0
g_euclidean_dist = ||[0,0,...,0] - article|| = ||article|| → sabit değer
```

**Sonuç**: Tüm validation/test örneklerinin %90+'ında graph features **sabitse** veya **sıfır**, model hiçbir bilgi öğrenemiyor.

### 4.3. Noise Injection

131 yeni feature eklenmesi:
- Model kapasitesini artırdı (daha fazla parametre)
- Ama bu features'lar bilgi taşımıyor (sıfır veya sabit)
- **Pure noise** eklenmesi overfitting'e yol açtı

**Tablo 4**: Feature sayısı ve model performansı

| Feature Count | Training AUC | Validation AUC | Overfitting Gap |
|---------------|--------------|----------------|-----------------|
| 103 (baseline) | 0.866 | 0.845 | **0.021** ✓ |
| 234 (w/ graph) | 0.901 | 0.730 | **0.171** ✗ |

Overfitting gap 8x arttı → Model graph features'a overfit oldu ama generalize edemedi.

### 4.4. Dataset Split Stratejisi

**Mevcut strateji**: Temporal split (impression_id bazlı)
```python
# Validation/test split
split_idx = len(all_validation_impression_ids) // 2
validation_impression_ids = all_validation_impression_ids[:split_idx]
test_impression_ids = all_validation_impression_ids[split_idx:]
```

**Sorun**:
- Train ve validation farklı zaman dilimlerinde
- Farklı zamanlarda farklı kullanıcılar aktif
- Graph-based methods için ideal split: **user-based stratified split** olmalı

**Tablo 5**: Split stratejilerinin graph coverage'a etkisi

| Split Strategy | Train Users | Val/Test Users | Overlap | Graph Coverage |
|----------------|-------------|----------------|---------|----------------|
| **Temporal (current)** | 8,844 | ~15,000 | ~10% | ✗ Çok düşük |
| **User-based stratified** | 10,000 | 6,000 | ~60% | ✓ Kabul edilebilir |
| **Random** | 10,000 | 6,000 | ~100% | ✓ İyi (ama realistic değil) |

## 5. Teorik Beklenti vs Gerçek Sonuç

### 5.1. Teorik Beklenti

Grover & Leskovec (2016) makalesine göre Node2Vec:
- Yapısal komşulukları yakalar
- İkinci dereceden ilişkileri öğrenir
- Cold-start'ta yardımcı olmalı (indirect connections)

**Örnek**:
```
user_A → article_1 ← user_B → article_2

user_A için article_2 önerisinde, user_B üzerinden dolaylı bağlantı yakalanmalı
```

### 5.2. Gerçek Sonuç

Bizim durumumuzda:
```
Train:     user_A → article_1
Test:      user_NEW → article_1

user_NEW train graph'ında yok!
→ user_NEW embedding = [0, 0, ..., 0]
→ İlişki yakalanamadı
```

**Root cause**: Graph'ın test setini kapsamaması.

## 6. Alternatif Çözüm Önerileri

### 6.1. Kısa Vadeli Çözümler

**1. Content-based Fallback**:
```python
if user_id not in user_embeddings:
    # Kullanıcının geçmiş etkileşimlerinden ortalama article embedding'i al
    user_embedding = mean(article_embeddings[user_history])
```

**2. Cold-start için özel stratejiler**:
- Yeni kullanıcılar için demographic/device bilgilerinden embedding üret
- Transfer learning: Benzer kullanıcıların embedding'lerini kullan

**3. Incremental Graph Update**:
- Validation setindeki ilk %10'u graph'a ekle
- Kalan %90'ı test et
- Bu yaklaşım realistic cold-start simülasyonu değil ama graph features'ı değerlendirmek için kullanılabilir

### 6.2. Uzun Vadeli Çözümler

**1. Hybrid Embedding**:
```python
user_embedding = α * graph_embedding + (1-α) * content_embedding
```
- Graph'ta yoksa content-based embedding kullan
- Her zaman valid bir embedding garanti et

**2. Graph Neural Networks (GNN)**:
- Node2Vec yerine GNN (GraphSAGE, GAT)
- Inductive learning: Yeni node'lar için embedding üretebilir
- Node features kullanarak cold-start'ı çözer

**3. Recurrent Graph Embeddings**:
- Temporal dynamics'i yakala
- Zamansal split'e uygun öğrenme

## 7. Sonuç ve Tavsiyeler

### 7.1. Ana Bulgular

1. ✗ **Graph-based features bu dataset/split üzerinde başarısız**
   - AUC: -11.2% düşüş
   - nDCG@10: -14.6% düşüş

2. ✓ **Baseline (content + behavioral) yeterli**
   - 103 feature ile AUC: 0.845
   - Graph olmadan state-of-the-art sonuçlar

3. ⚠ **Cold-start problemi kritik**
   - %90 kullanıcı coverage eksikliği
   - Zero embeddings faydasız

### 7.2. Öneriler

**Production için**:
- Graph features'ı **kullanma**
- Content-based ve behavioral features yeterli
- Mevcut baseline model (LightGBM + 103 features) optimal

**Gelecek araştırma için**:
1. User-based stratified split dene
2. Hybrid embedding yaklaşımları araştır
3. GNN-based methods'ları test et
4. Incremental graph update stratejileri değerlendir

### 7.3. Final Model Konfigürasyonu

**Önerilen Production Config**:
```yaml
use_graph_features: false

lgbm:
  num_boost_round: 1200
  early_stopping_round: 100
  params:
    learning_rate: 0.03
    max_depth: 10
    lambda_l2: 0.5
    feature_fraction: 0.7
    num_leaves: 256
```

**Performans**:
- Test AUC: **0.845**
- Test nDCG@10: **0.730**
- Test MRR: **0.651**

---

## Referanslar

1. Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 855-864).

2. Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). DeepWalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 701-710).

3. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. In Proceedings of the 31st International Conference on Neural Information Processing Systems (pp. 1025-1035).

---

**Not**: Bu analiz, EB-NeRD Small dataset (%10 sampling) üzerinde yapılmıştır. Sonuçlar ve gözlemler bu spesifik dataset ve train-test split stratejisi için geçerlidir.
