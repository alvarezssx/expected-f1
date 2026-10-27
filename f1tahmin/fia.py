"""FIA resmi belgeleri: serbest antrenman (FP1/FP2/FP3) sınıflandırmaları.

fia.com her hafta sonu için zaman çizelgesi PDF'leri yayınlar. Burada yalnızca antrenman
sınıflandırmaları okunur: her sürücünün o seanstaki en iyi turu. Etkinlik, PDF başlığındaki
tarih aralığının son günü Jolpica'daki yarış tarihiyle eşleştirilerek yarışa bağlanır
(FIA ve Jolpica yarış adları her zaman aynı değildir). PDF'ler saklanmaz; yalnızca
ayrıştırılmış sonuç cache/fia/ altına yazılır.

Kullanım:  python -m f1tahmin.fia
"""
from __future__ import annotations

import io
import json
import re
import time
import unicodedata
from datetime import date
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pypdf import PdfReader

from . import data

BASE = "https://www.fia.com"
INDEX = f"{BASE}/documents/championships/fia-formula-one-world-championship-14"
FIA_DIR = data.CACHE / "fia"
FIA_DIR.mkdir(exist_ok=True)
MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                          "august", "september", "october", "november", "december"])}
FP_RE = re.compile(r"/system/files/decision-document/(\d{4})_[a-z0-9_\-]+?_-_free_practice_([123])_classification\.pdf")
ROW_RE = re.compile(r"([A-Z][A-Za-zÀ-ÿ\-]+(?: [A-Z][A-Za-zÀ-ÿ\-]+)*) ([A-ZÀ-Ý][A-ZÀ-Ý'\-]+(?: [A-ZÀ-Ý][A-ZÀ-Ý'\-]+)*)\n(.+?)(\d:\d{2}\.\d{3})")

_last = 0.0


def _get(url: str) -> bytes:
    """FIA sunucusuna nazik istek: aralarında 3 sn, 429 gelirse giderek uzayan bekleme."""
    global _last
    for attempt in range(6):
        wait = 3.0 - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.time()
        try:
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0 (expected-f1)"}), timeout=60) as r:
                return r.read()
        except HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            pause = 60 * (attempt + 1)
            print(f"  429: {pause} sn bekleniyor", flush=True)
            time.sleep(pause)
    raise RuntimeError(url)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def _secs(t: str) -> float:
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def parse_classification(pdf: bytes) -> tuple[date | None, list[tuple[str, str, float]]]:
    """(belgenin yayın tarihi, [(ad, SOYAD, en iyi tur sn), ...]) — sıralı.

    Başlıktaki etkinlik tarih aralığı her zaman doğru değil; "Date" satırındaki yayın tarihi kullanılır."""
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf)).pages)
    text = text.replace("\u00a0", " ").replace("\ufffd", " ")
    doc_date = None
    m = re.search(r"Date\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", text)
    if m:
        month = next((v for k, v in MONTHS.items() if k.startswith(m.group(2).lower()[:3])), None)
        if month:
            doc_date = date(int(m.group(3)), month, int(m.group(1)))
    rows = []
    body = text[text.find("NODRIVER"):] if "NODRIVER" in text else text
    for first, last, _team, t in ROW_RE.findall(body):
        rows.append((first.strip(), last.strip(), _secs(t)))
    return doc_date, rows


def event_slugs(season: int) -> list[str]:
    """Belge listesindeki etkinlik adlarından PDF adres parçaları.

    Etkinlik sayfaları belge listesini JavaScript ile yüklediği için PDF adresleri doğrudan
    kalıptan kurulur: {sezon}_{etkinlik}_-_free_practice_{n}_classification.pdf
    Liste en yeni sezonla başlar; güncel sezonda ilk tekrar eden ad önceki sezona geçildiğini gösterir.
    Geçmiş sezonlar için tüm adlar denenir (404'ler hatırlanır).
    """
    html = _get(INDEX).decode("utf-8", "ignore")
    slugs = [m.group(0).split("/")[-1].split("_-_")[0][5:]
             for m in FP_RE.finditer(html) if m.group(1) == str(season)]  # güncel etkinlik
    seen = set()
    for name in re.findall(r'href="/decision-document-list/nojs/\d+"[^>]*>\s*([^<]*Grand Prix[^<]*)</a>', html):
        name = name.strip()
        if name in seen:
            if season == data.CURRENT_SEASON:
                break
            continue
        seen.add(name)
        base = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
        slugs += [base.replace(" ", "_"), base.replace(" ", "_").replace("-", "_")]
    # Jolpica yarış adları da aday (ör. "Bahrain Grand Prix in Malaysia")
    for r in data.schedule(season):
        slugs.append(_norm_slug(r["raceName"]))
    return list(dict.fromkeys(slugs))


def _norm_slug(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().replace(" ", "_")


def fetch_season(season: int = data.CURRENT_SEASON, refresh: bool = False) -> dict:
    """{tur: {"FP1": {sürücü_id: sn}, ...}} — bitmiş ve güncel etkinlikler.

    Denenmiş adresler hatırlanır; kesilirse kaldığı yerden sürer. refresh=True yalnızca
    bulunamayan (404) adresleri yeniden dener (yeni etkinliklerin belgeleri sonradan çıkar)."""
    out_path = FIA_DIR / f"fp_{season}.json"
    tried_path = FIA_DIR / f"fp_{season}_tried.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    tried = json.loads(tried_path.read_text(encoding="utf-8")) if tried_path.exists() else {}
    if refresh:
        tried = {u: s for u, s in tried.items() if s != "404"}
    sched = {date.fromisoformat(r["date"]): int(r["round"]) for r in data.schedule(season)}
    entries = {}
    for r in data.races_compact(season):
        entries[r["round"]] = {_norm(x["name"].split(" ", 1)[-1]): x["driver"] for x in r["results"]}
    last_entries = entries[max(entries)] if entries else {}

    # FIA belge adı sezona göre değişiyor: 2026 "free_practice_1", 2025 "p1"
    styles = ["free_practice_{n}", "p{n}"] if season >= 2026 else ["p{n}", "free_practice_{n}"]
    for slug in event_slugs(season):
        for n in (1, 2, 3):
            sess = f"FP{n}"
            pdf = path = None
            for style in styles:
                cand = f"/system/files/decision-document/{season}_{slug}_-_{style.format(n=n)}_classification.pdf"
                if cand in tried:
                    continue
                try:
                    pdf, path = _get(BASE + cand), cand
                    break
                except HTTPError as e:
                    if e.code == 404:  # sprint hafta sonunda FP2/FP3 yok ya da ad varyantı tutmadı
                        tried[cand] = "404"
                        tried_path.write_text(json.dumps(tried), encoding="utf-8")
                        continue
                    raise
            if pdf is None:
                continue
            doc_date, rows = parse_classification(pdf)
            # antrenman belgesi yarıştan 0-3 gün önce yayınlanır
            rnd = next((r for d, r in sched.items() if doc_date and 0 <= (d - doc_date).days <= 3), None)
            tried[path] = "ok" if rnd is not None and rows else "unmatched"
            tried_path.write_text(json.dumps(tried), encoding="utf-8")
            if rnd is None or not rows:
                print(f"  eşleşmedi: {path.rsplit('/', 1)[-1]} (belge tarihi {doc_date})", flush=True)
                continue
            names = entries.get(rnd, last_entries)
            best = {}
            for first, last, t in rows:
                d = names.get(_norm(last)) or names.get(_norm(first + last))
                if d:
                    best[d] = t
            out.setdefault(str(rnd), {})[sess] = best
            out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            print(f"  {rnd}. yarış {sess}: {len(best)}/{len(rows)} sürücü eşleşti", flush=True)
    return out


def load(season: int = data.CURRENT_SEASON) -> dict[int, dict[str, dict[str, float]]]:
    p = FIA_DIR / f"fp_{season}.json"
    raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {int(k): v for k, v in raw.items()}


if __name__ == "__main__":
    fetch_season()
    print("BİTTİ", flush=True)
