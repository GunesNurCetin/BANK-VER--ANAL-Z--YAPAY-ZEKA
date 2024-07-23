BANKA - VERİ ANALİZİ 
Bu kod, bir veri seti üzerinde Mixed Naive Bayes sınıflandırma algoritmasını kullanarak model eğitimi ve test işlemlerini gerçekleştirir.
İşte kodun detaylı işleyişi:

1. Veri Setinin Okunması ve İncelenmesi
Veri seti bank-full.csv dosyasından yüklenir ve genel istatistikleri ekrana yazdırılır.
2. Veriyi Anlama ve Hazırlama
Veri türleri incelenir.
Kategorik veriler sayısal verilere dönüştürülür.
3. Eğitim ve Test Veri Setlerinin Oluşturulması
Veri seti, eğitim (%70) ve test (%30) setlerine ayrılır.
4. Modelleme
Mixed Naive Bayes sınıflandırıcısı oluşturulur ve eğitim verileri ile model eğitilir.
Test verileri üzerinde tahminler yapılır ve etiketler geri dönüştürülür.
5. Performans Değerlendirme
Modelin performansı karmaşıklık matrisi ile değerlendirilir.
Performans metrikleri (True Negatives, False Positives, False Negatives, True Positives) hesaplanır ve yazdırılır.
Sınıflandırma raporu ile modelin precision, recall, F1-score gibi performans metrikleri özetlenir.
Özet
Bu kod, bir banka veri seti üzerinde Mixed Naive Bayes sınıflandırıcısını kullanarak model eğitir ve test eder.
Verinin kategorik nitelikleri sayısal değerlere dönüştürülür, veri eğitim ve test setlerine ayrılır, model eğitilir,
tahminler yapılır ve modelin performansı karmaşıklık matrisi ve sınıflandırma raporu ile değerlendirilir.

BANKA MARKETİNG

Bu kod, Naive Bayes algoritmasını kullanarak bir veri setinin (bank marketing veri seti) 
sınıflandırma performansını değerlendiren bir örnek çalışmadır.
Kodun detaylı işleyişi ve amacı aşağıda açıklanmıştır:

### 1. **Meta Bilgiler**
Kodun başında, dosya hakkında bazı meta bilgileri bulunur:
- Yazar: Elif Kartal & M. Erdal Balaban
- Oluşturulma Tarihi: Ocak 2024
- Veri Seti Kaynağı: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

### 2. **Kütüphanelerin İçe Aktarılması**
- `numpy` ve `pandas` veri işleme ve analiz için kullanılır.
- `LabelEncoder` kategorik verilerin sayısal verilere dönüştürülmesi için kullanılır.
- `StratifiedKFold` çapraz doğrulama için kullanılır.
- `MixedNB` Mixed Naive Bayes sınıflandırıcısını temsil eder.
- `classification_report` model performansını değerlendirmek için kullanılır.

### 3. **Veri Setinin Okunması ve İncelenmesi**
Veri seti CSV dosyasından okunur ve veri türleri kontrol edilir.

### 4. **Veri Hazırlama**
Kategorik veriler sayısal verilere dönüştürülür ve son sütun (hedef değişken) `kategorikNitelikler` listesinden çıkarılır.

### 5. **Çapraz Doğrulama (Cross-Validation)**
- Veri seti 5 katlı çapraz doğrulama ile ayrılır. Her iterasyonda eğitim ve test setlerinin indeksleri yazdırılır.
- Model, eğitim verileri ile eğitilir ve test verileri üzerinde tahminler yapılır. Performans metrikleri olan doğruluk ve F1-skoru hesaplanır.

### 6. **Sonuçların Hesaplanması ve Yazdırılması**
Ortalama doğruluk ve F1-skoru hesaplanır ve yazdırılır.

### Özet
Bu kod, bank marketing veri setini kullanarak Mixed Naive Bayes sınıflandırma algoritmasını uygulayan bir örnektir. Veri seti kategorik verilerin sayısal verilere dönüştürülmesi, çapraz doğrulama ile modelin eğitilmesi ve test edilmesi gibi adımları içerir. Modelin performansı doğruluk ve F1-skoru gibi metriklerle değerlendirilir.
