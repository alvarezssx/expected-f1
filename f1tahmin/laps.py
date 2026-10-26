"""Tur tur yarış verisi (Jolpica): her sürücünün her turdaki süresi ve pozisyonu, pit stopları.

Jolpica sayfa başına 100 satır ve saatte ~500 istek sınırı koyuyor; bir yarış ~13 istek.
Her yarış cache/laps/{sezon}_{tur}.json dosyasına bir kez yazılır, indirme kaldığı yerden sürer.

Kullanım:  python -m f1tahmin.laps 2026 2025 2024 2023
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .data import CACHE, JOLPICA, race_results

LAPS_DIR = CACHE / "laps"
LAPS_DIR.mkdir(exist_ok=True)

_last_call = 0.0


def _get(url: str) -> dict:
    global _last_call
    for attempt in range(12):
        wait = 0.35 - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            with urlopen(Request(url, headers={"User-Agent": "f1-tahmin/0.2"}), timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))["MRData"]
        except HTTPError as e:
            if e.code == 429:  # saatlik sınır: bekle ve tekrar dene
                print(f"  429 sınırı, {60 * (attempt + 1)} sn bekleniyor", flush=True)
                time.sleep(60 * (attempt + 1))
                continue
            raise
        except URLError:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(url)


def _parse_time(s: str | None) -> float | None:
    if not s:
        return None
    parts = s.split(":")
    try:
        return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))
    except ValueError:
        return None


def fetch_race(season: int, rnd: int) -> dict:
    """{'laps': {driver: [sn, ...]}, 'pos': {driver: [poz, ...]}, 'pits': [[driver, tur, durak, sn], ...]}"""
    path = LAPS_DIR / f"{season}_{rnd}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    laps: dict[str, dict[int, float]] = {}
    pos: dict[str, dict[int, int]] = {}
    offset, total = 0, None
    while total is None or offset < total:
        d = _get(f"{JOLPICA}/{season}/{rnd}/laps.json?limit=100&offset={offset}")
        total = int(d["total"])
        for race in d["RaceTable"]["Races"]:
            for lap in race["Laps"]:
                n = int(lap["number"])
                for t in lap["Timings"]:
                    laps.setdefault(t["driverId"], {})[n] = _parse_time(t.get("time"))
                    pos.setdefault(t["driverId"], {})[n] = int(t["position"])
        offset += 100

    d = _get(f"{JOLPICA}/{season}/{rnd}/pitstops.json?limit=100")
    pits = []
    for race in d["RaceTable"]["Races"]:
        for p in race.get("PitStops", []):
            pits.append([p["driverId"], int(p["lap"]), int(p["stop"]), _parse_time(p.get("duration"))])

    def to_list(m: dict[int, float | int]) -> list:
        n = max(m) if m else 0
        return [m.get(i) for i in range(1, n + 1)]

    out = {
        "season": season, "round": rnd,
        "laps": {k: to_list(v) for k, v in laps.items()},
        "pos": {k: to_list(v) for k, v in pos.items()},
        "pits": pits,
    }
    if laps:  # boş yanıtları cache'leme (veri henüz girilmemiş olabilir)
        path.write_text(json.dumps(out), encoding="utf-8")
    return out


def available(season: int) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in LAPS_DIR.glob(f"{season}_*.json"))


def load(season: int, rnd: int) -> dict | None:
    path = LAPS_DIR / f"{season}_{rnd}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


if __name__ == "__main__":
    seasons = [int(s) for s in sys.argv[1:]] or [2026, 2025, 2024, 2023]
    for season in seasons:
        rounds = [int(r["round"]) for r in race_results(season)]
        for rnd in rounds:
            if (LAPS_DIR / f"{season}_{rnd}.json").exists():
                continue
            t = time.time()
            r = fetch_race(season, rnd)
            print(f"{season} R{rnd}: {len(r['laps'])} sürücü, {sum(len(v) for v in r['laps'].values())} tur, "
                  f"{len(r['pits'])} pit ({time.time() - t:.0f} sn)", flush=True)
    print("BİTTİ", flush=True)
