# Expected F1

Formula 1 yarış tahmin ve oran analiz motoru. Her tahmin, tur tur koşturulan **20.000 yarış simülasyonundan** gelir.

## Doğruluk oranları (2025 + 2026, 36 yarış)

Her yarış **yalnızca kendinden önceki verilerle** tahmin edildi; hiçbir tahmin sonucu önceden görmedi. 2025 sezonu
aynı kodla, yalnızca o sezonun öncesini görerek ayrıca tahmin edildi ve ayarlar iki sezonun ortak sonucuna göre seçildi.

### Sıralama turundan sonra

| Metrik | 2026 (15 yarış) | 2025 (21 yarış) | **Toplam (36)** |
|---|---|---|---|
| Kazanan, modelin ilk 3 favorisi arasında | %93 (14/15) | %100 (21/21) | **%97** (35/36) |
| Modelin favorisi yarışı kazandı | %80 (12/15) | %48 (10/21) | **%61** (22/36) |
| Podyum isabeti (tahmin edilen ilk 3 ∩ gerçek ilk 3) | %62 | %76 | **%70** |
| İlk 10 isabeti | %79 | %76 | **%77** |
| Takım arkadaşı düellosu | %72 | %72 | **%72** (374 düello) |
| Kazanan log loss (rastgele: 3.09, düşük = iyi) | 1.22 | 1.04 | **1.12** |

Favorinin kazanma oranı sezonun yapısına bağlı: 2025'te McLaren'ın iki pilotu ve Verstappen her hafta çekişti
(kazanan yine de her yarışta ilk 3 favori arasındaydı), 2026'da bir takım daha baskın.

**Kazanma olasılığı kalibrasyonu:** simülasyonlar kazananın olasılığına göre yeniden ağırlıklanır (katsayı yalnızca
geçmiş yarışlardan). 36 yarışta favoriye verilen ortalama olasılık **%62**, favorinin gerçek kazanma oranı **%61** —
olasılıklar gerçekleşen oranla örtüşüyor. Kazanan log loss 1.117 → **1.090** (ileriye dönük).

### Sıralama turundan önce (grid de tahmin ediliyor)

| Metrik | 2026 (FIA antrenman verisiyle) | 2025 (antrenmansız) | **Toplam (36)** |
|---|---|---|---|
| Kazanan, ilk 3 favori arasında | %67 | %86 | **%78** |
| Favori kazandı | %53 | %29 | **%39** |
| Podyum isabeti | %49 | %63 | **%57** |
| İlk 10 isabeti | %78 | %71 | **%74** |
| Takım arkadaşı düellosu | %64 | %70 | **%67** |

2025 antrenman sınıflandırmalarının çoğu FIA'nın istek sınırı nedeniyle indirilemediği için 2025 sıralama öncesi
tahminleri antrenman verisi olmadan yapıldı.

## Motor

| | |
|---|---|
| Simülasyon / tahmin | 20.000 yarış × ~60 tur × 22 araç ≈ **25 milyon araç-tur** |
| Eğitim verisi | **88 yarış**, **98.000+ tur**, **3.200+ pit stop**, **26.000+ sollama denemesi** (2023–2026), **FIA antrenman sınıflandırmaları** |
| Model katmanları | Robust tur regresyonu · olay tespiti · Gauss süreci (Kalman araç gelişimi, ampirik Bayes) · antrenman yığınlaması · sıralama yığınlaması · hiyerarşik lojistik sollama · hiyerarşik büzülme · start kalibrasyonu · elemeli sıralama · strateji dengesi · vektörize Monte Carlo · ileriye dönük kalibrasyon |
| Piyasalar | Kazanan, podyum, ilk 6/10, pole, DNF, en hızlı tur, ilk tur lideri, takım arkadaşı düellosu, favori hariç kazanan, kazanan takım, güvenlik aracı sayısı, kazanma farkı, bitiren araç sayısı, şampiyonluk |
| Oran analizi | Marj (overround), seçim başına marj, Shin / power / orantılı adil oran, beklenen getiri |

## Çalıştırma

```bash
pip install -r requirements.txt
python -m f1tahmin.laps 2026 2025 2024 2023
python -m f1tahmin.fia
python server.py
```

İkinci komut tur verisini indirir (Jolpica'nın saatlik sınırı yüzünden ~1.5 saat), üçüncüsü FIA antrenman sınıflandırmalarını
okur. İkisi de kesilirse kaldığı yerden devam eder.
Ardından http://localhost:8765 adresini aç. Yeni yarış sonucu çıkınca **Verileri yenile** düğmesine bas.

Veri kaynakları ücretsizdir ve API anahtarı gerekmez: [Jolpica-F1](https://github.com/jolpica/jolpica-f1) (sonuçlar, tur süreleri, pit stoplar),
[FIA belgeleri](https://www.fia.com/documents/championships/fia-formula-one-world-championship-14) (antrenman sınıflandırmaları),
[Open-Meteo](https://open-meteo.com) (yağmur tahmini).

## Güvenlik

Sunucu yalnızca `127.0.0.1` üzerinde dinler. `Host` başlığı denetimi (DNS rebinding), POST'larda JSON + aynı köken
zorunluluğu (CSRF), gövde boyutu sınırı, girdi doğrulama ve CSP / nosniff / frame başlıkları vardır. Hata ayrıntıları
yalnızca sunucu konsoluna yazılır. İndirilen veri (`cache/`) repoya dahil edilmez.

---

Bilgi ve analiz amaçlıdır, bahis tavsiyesi değildir. Formula 1, FIA veya herhangi bir takımla bağlantılı değildir;
"F1" ve "Formula 1" ilgili sahiplerinin markalarıdır.
