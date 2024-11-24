# Kalp Hastalığı Tahmini

Bu proje, Spyder IDE kullanarak, kalp hastalıkları riski taşıyan bireylerin tespit edilmesi amacıyla makine öğrenmesi tekniklerini kullanarak kalp hastalığı tahmini yapmayı hedeflemektedir. Proje, UCI Heart Disease veri setini kullanarak, çeşitli özellikler (yaş, kan basıncı, kolesterol seviyesi vb.) ile bir kişinin kalp hastalığına sahip olup olmadığını tahmin etmeye yönelik bir model geliştirmektedir.

## Kullanılan Veri Seti

- **Veri Seti:** UCI Heart Disease Dataset
- **Kaynak:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Hedef Değişken:** `num` (0 = Hastalık yok, 1 = Hastalık mevcut)
- **Özellikler:** Yaş, cinsiyet, kan basıncı, kolesterol seviyesi, vb.

## Proje Adımları

1. **Veri Yükleme ve Ön İşleme:**
   - Veri seti pandas ile yüklenir ve eksik veriler işlenir.
   - Kategorik veriler, One-Hot Encoding yöntemi ile sayısal verilere dönüştürülür.
   - Sayısal veriler, StandardScaler ile normalize edilerek modelin daha iyi performans göstermesi sağlanır.

2. **Veri Setinin Eğitim ve Test Olarak Bölünmesi:**
   - Eğitim ve test verileri, `train_test_split` fonksiyonu ile %75 eğitim ve %25 test oranında bölünür.

3. **Model Eğitimi:**
   - Projede, kalp hastalığı tahmini için çeşitli makine öğrenmesi modelleri (örneğin: Lojistik Regresyon, Karar Ağaçları vb.) eğitilmiştir.

4. **Model Değerlendirme:**
   - Modelin başarısı, doğruluk, hassasiyet, hatırlama, F1 skoru gibi metriklerle değerlendirilmiştir.

5. **Sonuçlar:**
   - Modelin başarısı çeşitli metriklerle ölçülüp karşılaştırılmış ve en iyi performansı veren model seçilmiştir.

## Kullanılan Teknolojiler

- **Python:** Programlama dili
- **Spyder IDE:** Geliştirme ortamı
- **Pandas:** Veri işleme ve analiz
- **Scikit-learn:** Makine öğrenmesi modelleri ve araçları
- **Seaborn ve Matplotlib:** Veri görselleştirme
- **NumPy:** Sayısal hesaplamalar

## Kurulum ve Kullanım

1. **Gereksinimler:**
   - Python 3.x
   - Gerekli kütüphanelerin yüklenmesi için:
   ```bash
   pip install pandas scikit-learn seaborn matplotlib
