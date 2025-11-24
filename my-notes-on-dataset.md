# Veri Seti Üzerinde Değerlendirmeler

Author: Yunus Karatepe

Date: 2025.11.24

## 1. Veri setinden çıkarımlar

3 farklı veri seti var. Bunlar:

- articles

- history

- behaviors

### 1.1 Articles veri seti

- 125,541 rows × 21 columns

- Haberin kendisini içeren veri setidir.

- Haberin **title** ve **body** sini de içermektedir.

- image_ids alanı var, bu alan ile image embeddings dosyalarını eşleyebiliriz.

- 

### 1.2 History veri seti

- 788,090 rows × 5 columns

- Bir x kullanıcısının haber ile olan ilişkisini içerir.

- Aşağıdaki özellikleri user_id alanı ile beraber içerir.
  
  - impression_time_fixed: Array
  
  - scroll_percentage_fixed: Array
  
  - article_id_fixed: Array
  
  - read_time_fixed: Array

- Her bir user için dizi olarak makale idleri ile beraber diğer özellikleri de tutar.

### 1.3 Behaviors veri seti


