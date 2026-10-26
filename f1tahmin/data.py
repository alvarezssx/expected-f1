"""Ücretsiz veri kaynakları: Jolpica-F1 (sonuçlar), OpenF1 (güvenlik aracı), Open-Meteo (hava).

Hiçbiri API anahtarı istemez. Tüm yanıtlar cache/ klasörüne yazılır; biten sezonlar
bir kez indirilir, içinde bulunulan sezon `refresh=True` ile yenilenir.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

JOLPICA = "https://api.jolpi.ca/ergast/f1"
OPENF1 = "https://api.openf1.org/v1"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

CURRENT_SEASON = 2026
HISTORY_SEASONS = [2022, 2023, 2024, 2025]  # pist karakteri (grid etkisi, DNF) için

CLASSIFIED = {"Finished", "Lapped"}
EXCLUDED = {"Disqualified", "Did not start", "Withdrew", "Not classified", "Excluded"}


def _get_json(url: str, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "f1-tahmin/0.1"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 + 3 * attempt)
                continue
            raise
        except URLError:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
                continue
            raise
    raise RuntimeError(url)


def _cached(name: str, fetch, refresh: bool = False):
    path = CACHE / f"{name}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch()
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def _jolpica_all(endpoint: str) -> list[dict]:
    """Sayfalı Jolpica uç noktasını sonuna kadar okur ve yarışları birleştirir."""
    races: dict[str, dict] = {}
    offset, total = 0, None
    while total is None or offset < total:
        d = _get_json(f"{JOLPICA}/{endpoint}.json?limit=100&offset={offset}")["MRData"]
        total = int(d["total"])
        table = next(v for k, v in d.items() if k.endswith("Table"))
        list_key = next(k for k in table if isinstance(table[k], list))
        for race in table[list_key]:
            key = race["round"]
            if key in races:
                for k, v in race.items():
                    if isinstance(v, list):
                        races[key].setdefault(k, []).extend(v)
            else:
                races[key] = race
        offset += 100
        time.sleep(0.3)  # Jolpica: 4 istek/sn sınırı
    return sorted(races.values(), key=lambda r: int(r["round"]))


def schedule(season: int = CURRENT_SEASON, refresh: bool = False) -> list[dict]:
    return _cached(f"schedule_{season}", lambda: _jolpica_all(f"{season}/races"), refresh)


def race_results(season: int, refresh: bool = False) -> list[dict]:
    return _cached(f"results_{season}", lambda: _jolpica_all(f"{season}/results"), refresh)


def qualifying(season: int, refresh: bool = False) -> list[dict]:
    return _cached(f"qualifying_{season}", lambda: _jolpica_all(f"{season}/qualifying"), refresh)


def driver_standings(season: int = CURRENT_SEASON, refresh: bool = False) -> list[dict]:
    def fetch():
        d = _get_json(f"{JOLPICA}/{season}/driverStandings.json?limit=100")["MRData"]
        lists = d["StandingsTable"]["StandingsLists"]
        return lists[0]["DriverStandings"] if lists else []
    return _cached(f"standings_{season}", fetch, refresh)


# --------------------------------------------------------------------------- #
# Sadeleştirilmiş kayıtlar: modelin kullandığı tek biçim
# --------------------------------------------------------------------------- #

def _status_kind(status: str) -> str:
    if status in CLASSIFIED or status.startswith("+"):
        return "finished"
    if status in EXCLUDED:
        return "excluded"
    return "dnf"


def races_compact(season: int, refresh: bool = False) -> list[dict]:
    """Her yarış: round, circuit, sonuç listesi (driver, team, grid, pos, kind) ve sıralama turu."""
    quali = {r["round"]: r for r in qualifying(season, refresh)}
    out = []
    for race in race_results(season, refresh):
        rows = []
        for x in race["Results"]:
            grid = int(x["grid"])
            rows.append({
                "driver": x["Driver"]["driverId"],
                "code": x["Driver"].get("code") or x["Driver"]["familyName"][:3].upper(),
                "name": f'{x["Driver"]["givenName"]} {x["Driver"]["familyName"]}',
                "team": x["Constructor"]["constructorId"],
                "team_name": x["Constructor"]["name"],
                "grid": grid if grid > 0 else None,  # 0 = pit çıkışı
                "pos": int(x["position"]),
                "kind": _status_kind(x["status"]),
            })
        q = quali.get(race["round"])
        q_order = [r["Driver"]["driverId"] for r in sorted(q["QualifyingResults"], key=lambda r: int(r["position"]))] if q else []
        out.append({
            "season": season,
            "round": int(race["round"]),
            "name": race["raceName"],
            "circuit": race["Circuit"]["circuitId"],
            "date": race["date"],
            "results": sorted(rows, key=lambda r: r["pos"]),
            "quali": q_order,
        })
    return out


# --------------------------------------------------------------------------- #
# OpenF1: pist bazında güvenlik aracı sıklığı (2023+)
# --------------------------------------------------------------------------- #

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def safety_car_history(calendar: list[dict], years=(2023, 2024, 2025)) -> dict:
    """{circuitId: [0/1, ...]} — yarışta en az bir tam güvenlik aracı çıktı mı.

    OpenF1 canlı seans sırasında anonim erişimi kapatıyor; o durumda boş döner
    ve hiçbir şey cache'lenmez, bir sonraki çağrıda tekrar denenir.
    """
    path = CACHE / "safety_car.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    circuits = {r["Circuit"]["circuitId"]: r["Circuit"] for r in calendar}
    keys = {cid: {_norm(c["Location"]["locality"]), _norm(c["circuitName"])} for cid, c in circuits.items()}

    def match(session: dict) -> str | None:
        loc = _norm(session.get("location", ""))
        short = _norm(session.get("circuit_short_name", ""))
        for cid, names in keys.items():
            if any(loc and (loc in n or n in loc) for n in names) or any(short and short in n for n in names):
                return cid
        return None

    history: dict[str, list[int]] = {}
    try:
        for year in years:
            sessions = _get_json(f"{OPENF1}/sessions?{urlencode({'year': year, 'session_name': 'Race'})}")
            if not isinstance(sessions, list):  # {"detail": "..."} = erişim kısıtlı
                return {}
            for s in sessions:
                cid = match(s)
                if not cid:
                    continue
                msgs = _get_json(f"{OPENF1}/race_control?{urlencode({'session_key': s['session_key'], 'category': 'SafetyCar'})}")
                deployed = any("SAFETY CAR DEPLOYED" in (m.get("message") or "") and "VIRTUAL" not in (m.get("message") or "") for m in msgs)
                history.setdefault(cid, []).append(int(deployed))
                time.sleep(0.4)
    except (HTTPError, URLError, KeyError, TypeError, AttributeError):
        return {}
    path.write_text(json.dumps(history), encoding="utf-8")
    return history


# --------------------------------------------------------------------------- #
# Open-Meteo: yarış saatindeki yağmur olasılığı (en fazla 16 gün ileri)
# --------------------------------------------------------------------------- #

def rain_forecast(race: dict) -> dict | None:
    start = datetime.fromisoformat(f'{race["date"]}T{race.get("time", "13:00:00Z").replace("Z", "+00:00")}')
    days_ahead = (start - datetime.now(timezone.utc)).days
    if days_ahead < -1 or days_ahead > 15:
        return None
    loc = race["Circuit"]["Location"]
    params = {
        "latitude": loc["lat"], "longitude": loc["long"],
        "hourly": "precipitation_probability,precipitation,temperature_2m",
        "timezone": "UTC", "start_date": race["date"], "end_date": race["date"],
    }
    try:
        d = _get_json(f"{OPEN_METEO}?{urlencode(params)}")
    except (HTTPError, URLError):
        return None
    hours = d["hourly"]["time"]
    # yarış ~2 saat sürer: başlangıç saati ve sonraki iki saatin en kötüsü
    i = next((k for k, t in enumerate(hours) if t.startswith(f'{race["date"]}T{start.hour:02d}')), None)
    if i is None:
        return None
    window = slice(i, i + 3)
    probs = [p for p in d["hourly"]["precipitation_probability"][window] if p is not None]
    return {
        "rain_prob": max(probs) / 100 if probs else 0.0,
        "precip_mm": sum(p or 0 for p in d["hourly"]["precipitation"][window]),
        "temp_c": d["hourly"]["temperature_2m"][i],
    }
