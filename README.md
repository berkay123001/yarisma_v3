# Loan Payback Prediction (Kaggle Top 25% - Solo Entry)

Bu proje, Kaggle Playground Series S5E11 yarışması için geliştirilmiş, yüksek performanslı bir kredi geri ödeme tahminleme boru hattıdır. **3724 takım arasından tek başına (solo entrant) katılarak ilk %25'lik dilime (918. sıra)** girilmesini sağlayan matematiksel optimizasyon ve ensemble tekniklerini içerir.

## 📊 Başarı Metrikleri
*   **En Yüksek Skor (Private LB):** 0.92460 (submission_ensemble_v2)
*   **Birinci Skoru:** 0.92939
*   **Makas:** Sadece 0.00479 (Bu kadar küçük bir farkın tek başına ve kısıtlı donanımla korunması, model çeşitliliği ve ağırlık optimizasyonunun bir sonucudur).

## 🧠 Mühendislik Yaklaşımı ve Teknik Strateji

### 1. Model Çeşitliliği (Diversification)
Tek bir güçlü model yerine, verinin farklı yönlerini öğrenen heterojen bir yapı kurulmuştur:
*   **Gradient Boosting:** LightGBM, CatBoost ve XGBoost'un farklı hiperparametre varyasyonları.
*   **Deep Learning:** TabNet ve Denoising AutoEncoder (DAE) ile gürültüden arındırılmış öznitelik çıkarımı (`src/train_dae_boosted.py`).
*   **TabPFN:** Küçük ve orta ölçekli tabular verilerde devrim yaratan "Prior-Data Fitted Networks" entegrasyonu.

### 2. Hill Climbing Ağırlık Optimizasyonu
Modelleri birleştirirken (Ensemble) manuel ağırlık vermek yerine, Out-of-Fold (OOF) tahminleri üzerinde **Hill Climbing** algoritması çalıştırılmıştır.
*   **Neden?** Basit ortalama (average) yerine her modelin hata payına göre matematiksel olarak en ideal ağırlığı bulmak, skoru 0.921 bandından 0.924 bandına taşımıştır.
*   **Kod:** `src/optimize_ensemble_weights.py`

### 3. "Magic Features" ve Öznitelik Mühendisliği
Standart verilerin ötesine geçmek için şu teknikler uygulandı:
*   **Golden Features:** Genetik algoritmalar ve brute-force kombinasyonlarla en yüksek korelasyona sahip yeni değişkenler türetildi. (`src/find_magic_features.py`)
*   **Target Encoding & Binning:** Kategorik verilerdeki sinyali güçlendirmek için hedef tabanlı kodlama yapıldı.

### 4. Neden Bazı Denemeler Başarısız Oldu?
*   **Overfitting Sorunu:** Pseudo-labeling (sahte etiketleme) aşırı kullanıldığında model test verisindeki gürültüye fazla odaklandı ve skor düştü.
*   **Çözüm:** Eğitim sürecine "Adversarial Validation" ekleyerek eğitim ve test verisi arasındaki dağılım farkı (drift) kontrol altına alındı.

## 🛠️ Teknik Yığın (Tech Stack)
*   **Modeller:** CatBoost, LightGBM, XGBoost, TabNet, TabPFN.
*   **Kütüphaneler:** Scikit-learn, Optuna (Hiperparametre tuning), Pandas, NumPy.
*   **Teknikler:** Hill Climbing Ensemble, DAE Feature Extraction, Stacking.

---
*Not: Bu proje, kısıtlı GPU kaynaklarına rağmen model mimarisi ve matematiksel ağırlıklandırma optimizasyonu ile rekabetçi skorlar elde edilebileceğini kanıtlamaktadır.*
