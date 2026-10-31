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


TIME_RE = re.compile(r"\d:\d{2}\.\d{3}")


def parse_qualifying(pdf: bytes) -> list[dict]:
    """FIA sıralama sınıflandırması -> [{pos, number, first, last, entrant, q1, q2, q3}]."""
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(pdf)).pages)
    lines = [l.replace("\u00a0", " ").rstrip() for l in text.split("\n")]
    rows = []
    for i, line in enumerate(lines[:-1]):
        m = re.match(r"^(\d{1,2}) (\d{1,2}) ([A-Z][A-Za-zÀ-ÿ\-]+(?: [A-Z][A-Za-zÀ-ÿ\-]+)*) ([A-ZÀ-Ý'\-]+(?: [A-ZÀ-Ý'\-]+)*)$",
                     line.strip())
        if not m:
            continue
        nxt = lines[i + 1].strip()
        times = TIME_RE.findall(nxt)
        first_t = TIME_RE.search(nxt)
        entrant = nxt[:first_t.start()].strip() if first_t else nxt
        rows.append({"pos": int(m.group(1)), "number": m.group(2), "first": m.group(3), "last": m.group(4),
                     "entrant": entrant, "q": (times + [None, None, None])[:3]})
    return rows


def fetch_qualifying(season: int, rnd: int) -> list[dict] | None:
    """Jolpica'da henüz olmayan sıralama sonucunu FIA PDF'inden Jolpica biçiminde üretir.

    Önce kesin (final), yoksa geçici (provisional) sınıflandırma denenir. Sürücü/takım kimlikleri
    sezonun önceki yarışlarından eşlenir; yeni sürücü (yedek pilot) için soyadından kimlik üretilir."""
    race = next((r for r in data.schedule(season) if int(r["round"]) == rnd), None)
    if not race:
        return None
    slugs = [_norm_slug(race["raceName"])]
    pdf = None
    for slug in slugs:
        for kind in ("final_qualifying_classification", "provisional_qualifying_classification"):
            try:
                pdf = _get(f"{BASE}/system/files/decision-document/{season}_{slug}_-_{kind}.pdf")
                break
            except HTTPError as e:
                if e.code != 404:
                    raise
        if pdf:
            break
    if not pdf:
        return None
    rows = parse_qualifying(pdf)
    if len(rows) < 10:
        return None
    known_driver, known_team, team_name = {}, {}, {}
    for r in data.races_compact(season):
        for x in r["results"]:
            known_driver[_norm(x["name"].split(" ", 1)[-1])] = (x["driver"], x["code"])
            known_team[x["driver"]] = x["team"]
            team_name[x["team"]] = x["team_name"]
    # yarış kadrosundaki takım: aynı FIA takım adını taşıyan tanınan sürücünün takımı
    entrant_team = {}
    for row in rows:
        hit = known_driver.get(_norm(row["last"]))
        if hit and hit[0] in known_team:
            entrant_team.setdefault(row["entrant"], known_team[hit[0]])
    out = []
    for row in rows:
        did, code = known_driver.get(_norm(row["last"]), (_norm(row["last"]), row["last"][:3].upper()))
        team = entrant_team.get(row["entrant"], _norm(row["entrant"]))
        q1, q2, q3 = row["q"]
        entry = {"number": row["number"], "position": str(row["pos"]),
                 "Driver": {"driverId": did, "code": code, "givenName": row["first"],
                            "familyName": row["last"].title()},
                 "Constructor": {"constructorId": team, "name": team_name.get(team, row["entrant"])}}
        for k, v in (("Q1", q1), ("Q2", q2), ("Q3", q3)):
            if v:
                entry[k] = v
        out.append(entry)
    return out


def merge_qualifying(season: int, rnd: int) -> bool:
    """FIA sıralama sonucunu yerel sıralama önbelleğine ekler (Jolpica'da bu yarış yoksa)."""
    cache = data.CACHE / f"qualifying_{season}.json"
    q = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
    if any(int(r["round"]) == rnd for r in q):
        return False
    res = fetch_qualifying(season, rnd)
    if not res:
        return False
    q.append({"season": str(season), "round": str(rnd), "source": "FIA", "QualifyingResults": res})
    cache.write_text(json.dumps(q, ensure_ascii=False), encoding="utf-8")
    return True


def fetch_event_fp(season: int, rnd: int) -> dict:
    """Tek bir etkinliğin antrenman sınıflandırmalarını hedefli indirir (tüm sezonu taramadan)."""
    race = next((r for r in data.schedule(season) if int(r["round"]) == rnd), None)
    out_path = FIA_DIR / f"fp_{season}.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    if not race:
        return out
    slug = _norm_slug(race["raceName"])
    names = {}
    for r in data.races_compact(season):
        names.update({_norm(x["name"].split(" ", 1)[-1]): x["driver"] for x in r["results"]})
    for n in (1, 2, 3):
        for style in ("free_practice_{n}", "p{n}"):
            try:
                pdf = _get(f"{BASE}/system/files/decision-document/{season}_{slug}_-_{style.format(n=n)}_classification.pdf")
            except HTTPError as e:
                if e.code == 404:
                    continue
                raise
            _, rows = parse_classification(pdf)
            best = {}
            for first, last, t in rows:
                d = names.get(_norm(last)) or names.get(_norm(first + last)) or _norm(last)
                best[d] = t
            if rows:
                out.setdefault(str(rnd), {})[f"FP{n}"] = best
            break
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def load(season: int = data.CURRENT_SEASON) -> dict[int, dict[str, dict[str, float]]]:
    p = FIA_DIR / f"fp_{season}.json"
    raw = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {int(k): v for k, v in raw.items()}


if __name__ == "__main__":
    fetch_season()
    print("BİTTİ", flush=True)
