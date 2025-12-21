# Graph-Based Feature Extraction - Kullanım Kılavuzu

Bu modül, raporunuzda (Section 4.4.1) bahsedilen **Çizge Tabanlı Özellik Çıkarımı** (Graph-Based Feature Extraction) özelliğini implement eder.

## Özet

Kullanıcı-makale etkileşimlerinden bipartite graph oluşturur ve Node2Vec algoritması ile yapısal embedding'ler çıkarır. Bu embedding'ler LightGBM modeline ek özellikler olarak eklenir.

## Kurulum

```bash
cd recsys-challenge-2024-1st-place-master/kami
pip install -r requirements_graph.txt
```

## Kullanım

### 1. Graph Features'ı Aktifleştirme

Config dosyasını düzenleyin (`experiments/015_train_third/exp/small067_001.yaml`):

```yaml
use_graph_features: true  # false -> true yapın
graph_embedding_dim: 64    # Embedding boyutu (varsayılan: 64)
graph_walk_length: 30      # Random walk uzunluğu (varsayılan: 30)
graph_num_walks: 200       # Walk sayısı (varsayılan: 200)
graph_workers: 4           # Paralel worker sayısı (varsayılan: 4)
```

### 2. Modeli Çalıştırma

```bash
run.bat train --debug
```

İlk çalıştırmada:
1. Bipartite graph oluşturulur (kullanıcı-makale etkileşimlerinden)
2. Node2Vec modeli eğitilir (~10-30 dakika, veri boyutuna bağlı)
3. Embedding'ler çıkarılır ve kaydedilir
4. Özellikler eklenerek LightGBM eğitilir

Sonraki çalıştırmalarda:
- Önceden eğitilmiş graph model yüklenir (hızlı)
- Sadece özellik ekleme yapılır

### 3. Çıktılar

```
output/experiments/015_train_third/small067_001/
├── graph_model/
│   ├── bipartite_graph.gpickle      # Bipartite graph
│   ├── node2vec.model                # Eğitilmiş Node2Vec modeli
│   ├── user_embeddings.pkl           # Kullanıcı embedding'leri
│   └── article_embeddings.pkl        # Makale embedding'leri
├── model_dict_model.pkl
├── importance_*.png
└── run.log
```

## Eklenen Özellikler

Her veri satırına aşağıdaki özellikler eklenir:

### User Embeddings (64 özellik)
- `graph_user_emb_0` ... `graph_user_emb_63`: Kullanıcının yapısal embedding'i

### Article Embeddings (64 özellik)
- `graph_article_emb_0` ... `graph_article_emb_63`: Makalenin yapısal embedding'i

### Interaction Features (3 özellik)
- `graph_dot_product`: Kullanıcı ve makale embedding'lerinin dot product'ı
- `graph_cosine_sim`: Cosine similarity
- `graph_euclidean_dist`: Euclidean distance

**Toplam**: 131 yeni özellik

## Parametrelerin Anlamı

### `embedding_dim` (64)
- Embedding vektörlerinin boyutu
- Daha büyük değer → Daha zengin temsil, daha uzun eğitim
- Önerilen: 32-128 arası

### `walk_length` (30)
- Her random walk'ın adım sayısı
- Daha uzun → Daha uzak komşuları yakalar
- Önerilen: 20-50 arası

### `num_walks` (200)
- Her node başına kaç walk yapılacağı
- Daha fazla → Daha iyi temsil, daha uzun eğitim
- Önerilen: 100-300 arası

### `p` ve `q` (varsayılan: 1.0)
- Node2Vec'in keşif parametreleri
- `p`: Return parameter (önceki node'a dönme olasılığı)
- `q`: In-out parameter (keşif vs. derinleşme dengesi)
- `q < 1`: BFS benzeri (geniş keşif)
- `q > 1`: DFS benzeri (derin keşif)

## Performans Notları

### Hesaplama Maliyeti
- **İlk eğitim**: ~10-30 dakika (small dataset için)
- **Sonraki eğitimler**: ~1-2 dakika (sadece feature ekleme)

### Bellek Kullanımı
- Graph oluşturma: ~2-4 GB
- Node2Vec eğitimi: ~4-8 GB
- Toplam: 32GB RAM yeterli

### Cold-Start İyileştirmesi
Graph features özellikle şu durumlarda etkilidir:
- Yeni makaleler (soğuk başlangıç)
- Az etkileşimli kullanıcılar
- Dolaylı ilişkilerin önemli olduğu durumlar

## Beklenen Başarı İyileştirmesi

Rapordaki hipoteze göre:
- **nDCG@5/10**: +2-5% iyileşme bekleniyor
- **MRR**: +1-3% iyileşme bekleniyor
- **Cold-start performansı**: +5-10% iyileşme bekleniyor

## Sorun Giderme

### "Behaviors file not found" hatası
Eğitim verisi yolunu kontrol edin:
```yaml
dataset_path: output/preprocess/dataset067
```

### Bellek yetersizliği
Parametreleri azaltın:
```yaml
graph_num_walks: 100      # 200 -> 100
graph_embedding_dim: 32   # 64 -> 32
```

### Node2Vec çok yavaş
Worker sayısını artırın:
```yaml
graph_workers: 8  # 4 -> 8 (CPU çekirdek sayınıza göre)
```

## Teorik Arka Plan

Rapordaki Section 4.4.1'de detaylandırıldığı gibi:

> "Kullanıcı-Makale etkileşimlerinden oluşan iki parçalı bir çizge (Bipartite Graph) oluşturulması hedeflenmektedir. Bu çizge üzerinde Node2Vec veya DeepWalk algoritmaları çalıştırılarak, her kullanıcı ve makale için 'yapısal gömme vektörleri' (structural embeddings) elde edilecektir."

Bu implementation tam olarak bu metodolojiye uygundur.

## Sonraki Adımlar

1. **Baseline ile karşılaştırma**: 
   ```bash
   # Graph features olmadan
   use_graph_features: false
   
   # Graph features ile  
   use_graph_features: true
   ```
   
2. **Parametre optimizasyonu**: Grid search ile optimal parametreleri bulma

3. **DeepWalk alternatifi**: Node2Vec yerine DeepWalk deneme

4. **Temporal graphs**: Zamansal graph yapıları deneme

## Referanslar

- Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. KDD.
- Perozzi, B., et al. (2014). DeepWalk: Online learning of social representations. KDD.
