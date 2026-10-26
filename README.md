# Pit Oranı — F1 tahmin ve oran analiz aracı

Her yarış 20.000 kez, **tur tur** simüle edilir. Tüm bahis türleri (kazanan, podyum, ilk 6/10, DNF, pole,
en hızlı tur, ilk tur lideri, takım arkadaşı düellosu, "favori hariç kazanan", kazanan takım, güvenlik aracı
sayısı, kazanma farkı, bitiren araç sayısı, şampiyonluk) aynı simülasyondan, birbiriyle tutarlı çıkar.

## Çalıştırma

```bash
pip install -r requirements.txt
python -m f1tahmin.laps 2026 2025 2024 2023
python server.py
```

İkinci komut tur tur veriyi indirir (Jolpica saatlik sınırı yüzünden ~1.5 saat, kaldığı yerden devam eder).
Sonra http://localhost:8765 adresini aç. Yeni yarış sonucu çıkınca **Verileri yenile** düğmesine bas; yeni yarışın
tur verisini de indirir. Model kodu değişince `f1tahmin/engine.py` içindeki `V2_VERSION`'ı artır (backtest yeniden hesaplanır).

## Veri kaynakları (hepsi ücretsiz, anahtarsız)

| Kaynak | Kullanım |
|---|---|
| Jolpica-F1 | 2022–2026 sonuçlar, sıralama süreleri (Q1/Q2/Q3), **2023–2026 tur tur süreler, pozisyonlar, pit stoplar** |
| Open-Meteo | Yarış saatindeki yağmur olasılığı (16 gün ileriye kadar) |
| OpenF1 | İsteğe bağlı; şu an erişim sorunlu, kullanılmıyor |

## Model (v2, tur tur)

| Dosya | İçerik |
|---|---|
| `f1tahmin/laps.py` | Tur verisi indirme (sınıra uyan, devam edebilen) |
| `f1tahmin/features.py` | Bir yarıştan ölçümler: sürücü temposu (Huber-robust regresyon), yakıt, lastik aşınması, kirli hava, SC/VSC tespiti, pit kaybı, ilk tur, sollama denemeleri |
| `f1tahmin/pace.py` | Gauss süreci tempo modeli: araç rastgele yürüyüşü + sürücü farkı, hiperparametreler marjinal olabilirlikle |
| `f1tahmin/track.py` | Pist parametreleri (hiyerarşik büzülme), hiyerarşik lojistik sollama modeli, grid sırasına göre start gürültüsü |
| `f1tahmin/racesim.py` | Vektörize tur tur simülatör: elemeli sıralama, strateji, SC/VSC, pit, sollama, DNF |
| `f1tahmin/lapengine.py` | Hepsini bağlar; sıralama → yarış yığınlama regresyonu |
| `f1tahmin/engine.py` | Tahmin, ileriye dönük backtest (v1 vs v2), kalibrasyon, şampiyonluk |
| `f1tahmin/odds.py` | Marj, Shin / power / orantılı adil oran, beklenen getiri |

v1 (sıralama modeli, `model.py` + `simulate.py`) tur verisi yetersizse yedek olarak ve karşılaştırma için durur.

## Backtest (2026, 4.–17. yarışlar, ileriye dönük)

| | v1 | v2 |
|---|---|---|
| Sıralama öncesi kazanan log loss | 1.916 | **1.644** |
| Sıralama sonrası kazanan log loss | 1.512 | **1.227** |
| Sıralama sonrası podyum Brier | 0.0800 | **0.0700** |
| Sıralama sonrası favori kazandı | %64 | **%79** |

## Elle ayarlanmış varsayımlar

`racesim.py` başındaki sabitler (strateji sapması, SC'de pit kararı, undercut savunması, yağmur çarpanları,
DNF→SC olasılıkları). Değiştirince `V2_VERSION`'ı artırıp **Model Performansı** sekmesinden etkisini kontrol et.

## Bilinen sınırlar

- Lastik hamuru verisi yok (F1 arşivi 401): aşınma tek ortalama hızla, strateji farkları zamanca dengelenerek modelleniyor.
- Antrenman (FP) long-run temposu yok; Jolpica antrenman turu vermiyor.
- Pole'dan başlayanın kazanma şansı düşük tahmin ediliyor (2026'da gerçek %71, model ~%35).
- Grid cezaları bilinmiyor; şampiyonlukta sezon içi gelişim yalnızca Gauss sürecinin belirsizliği olarak var.

## Güvenlik

Sunucu yalnızca `127.0.0.1` üzerinde dinler; `Host` başlığı denetimi (DNS rebinding), POST'larda JSON + aynı köken
zorunluluğu (CSRF), gövde boyutu sınırı, girdi doğrulama ve CSP / nosniff / frame başlıkları vardır. Hata ayrıntıları
yalnızca sunucu konsoluna yazılır. Proje API anahtarı kullanmaz; indirilen veri (`cache/`) repoya dahil edilmez.

Bilgi ve analiz amaçlıdır, bahis tavsiyesi değildir. Formula 1, FIA veya herhangi bir takımla bağlantılı değildir;
"F1" ve "Formula 1" ilgili sahiplerinin markalarıdır.
