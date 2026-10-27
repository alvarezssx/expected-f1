# Expected F1

Formula 1 yarış tahmin ve oran analiz motoru. Her tahmin, tur tur koşturulan **20.000 yarış simülasyonundan** gelir.

## 2026 sezonu doğruluk oranları

15 yarış (4.–18. yarışlar). Her yarış **yalnızca kendinden önceki verilerle** tahmin edildi; hiçbir tahmin sonucu önceden görmedi.

### Sıralama turundan sonra

| Metrik | Expected F1 |
|---|---|
| Kazanan, modelin ilk 3 favorisi arasında | **%93** (14/15) |
| Modelin favorisi yarışı kazandı | **%80** (12/15) |
| Podyum isabeti (tahmin edilen ilk 3 ∩ gerçek ilk 3) | **%62** |
| İlk 10 isabeti | **%79** |
| Takım arkadaşı düellosu (kim önde bitirir) | **%72** (164 düello) |
| Kazanan log loss (rastgele tahmin: 3.09, düşük = iyi) | **1.14** (kalibrasyonlu) |
| Favoriye verilen ortalama kazanma olasılığı | **%55** (kalibrasyondan önce %38) |

### Sıralama turundan önce (grid de tahmin ediliyor, FIA antrenman sonuçlarıyla)

| Metrik | Expected F1 |
|---|---|
| Kazanan, ilk 3 favori arasında | **%73** |
| Favori kazandı | **%53** |
| Podyum isabeti | **%49** |
| İlk 10 isabeti | **%78** |
| Takım arkadaşı düellosu | **%64** |

Antrenman verisinin eklenmesi sıralama öncesi birleşik log loss'u 17.45'ten **16.56**'ya, podyum Brier'ı 0.081'den
**0.076**'ya indirdi (aynı 15 yarış, aynı ayarlar).

### Yarış yarış (sıralama turundan sonra)

| Yarış | Kazanan | Modelin favorisi | Kazanana verilen olasılık |
|---|---|---|---|
| 4 · Miami | Antonelli | ✅ Antonelli | %50 |
| 5 · Kanada | Antonelli | ✅ Antonelli | %31 |
| 6 · Monaco | Antonelli | ✅ Antonelli | %51 |
| 7 · Barselona | Hamilton | Russell | %25 |
| 8 · Avusturya | Russell | ✅ Russell | %36 |
| 9 · Britanya | Leclerc | Antonelli | %15 |
| 10 · Belçika | Antonelli | ✅ Antonelli | %49 |
| 11 · Macaristan | Norris | ✅ Norris | %26 |
| 12 · Hollanda | Norris | ✅ Norris | %38 |
| 13 · İtalya | Antonelli | Russell | %4 |
| 14 · İspanya (Madrid) | Antonelli | ✅ Antonelli | %31 |
| 15 · Azerbaycan | Russell | ✅ Russell | %50 |
| 16 · Bahreyn (Sepang) | Verstappen | ✅ Verstappen | %40 |
| 17 · Singapur | Verstappen | ✅ Verstappen | %35 |
| 18 · ABD | Antonelli | ✅ Antonelli | %36 |

Ölçek için: 22 araçlı alanda rastgele bir tahmin kazanana %4.5 verir.

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
