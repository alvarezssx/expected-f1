"""Monte Carlo yarış hafta sonu simülasyonu.

Her simülasyonda sırasıyla: sıralama turları (grid) -> güvenlik aracı / yağmur ->
DNF'ler -> bitiren sürücülerin sırası. Plackett-Luce'tan örnekleme, log-güce Gumbel
gürültüsü ekleyip sıralamakla birebir aynıdır; bu yüzden tüm bahis türleri aynı
simülasyondan, birbiriyle tutarlı çıkar.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import Ratings, Reliability, TrackStats

POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 20)

# Varsayımlar (veriden öğrenilmedi, backtest ile kontrol edilir):
SC_CHAOS = 0.12      # güvenlik aracı çıkan yarışta sıralama bu oranda daha rastgele
RAIN_CHAOS = 0.35    # yağmurlu yarışta ek rastgelelik
RAIN_DNF = 1.3       # yağmurda DNF çarpanı


@dataclass
class Entry:
    driver: str
    team: str
    code: str
    name: str
    team_name: str


@dataclass
class SimResult:
    entries: list[Entry]
    positions: np.ndarray   # (n_sims, n_drivers), 1 = kazanan
    grid: np.ndarray        # (n_sims, n_drivers)
    dnf: np.ndarray         # (n_sims, n_drivers) bool
    sc: np.ndarray          # (n_sims,) bool
    rain: np.ndarray        # (n_sims,) bool


def simulate(
    entries: list[Entry],
    race_r: Ratings,
    quali_r: Ratings,
    rel: Reliability,
    ts: TrackStats,
    circuit: str,
    n_sims: int = 20000,
    known_grid: list[str] | None = None,
    rain_prob: float = 0.0,
    temperature: float = 1.0,
    seed: int | None = 7,
) -> SimResult:
    rng = np.random.default_rng(seed)
    n = len(entries)
    race_s = np.array([race_r.strength(e.driver, e.team) for e in entries])
    quali_s = np.array([quali_r.strength(e.driver, e.team) for e in entries])

    # 1) Grid
    if known_grid:
        order = {d: i + 1 for i, d in enumerate(known_grid)}
        fixed = np.array([order.get(e.driver, n) for e in entries])
        grid = np.broadcast_to(fixed, (n_sims, n)).copy()
    else:
        q = quali_s + rng.gumbel(size=(n_sims, n))
        grid = np.argsort(np.argsort(-q, axis=1), axis=1) + 1

    # 2) Güvenlik aracı ve yağmur
    p_sc = ts.sc(circuit)
    sc = rng.random(n_sims) < p_sc
    rain = rng.random(n_sims) < rain_prob
    chaos = temperature * (1 + SC_CHAOS * (sc - p_sc)) * (1 + RAIN_CHAOS * rain)

    # 3) DNF
    p_dnf = np.array([rel.team_rate.get(e.team, rel.base) for e in entries]) * ts.dnf(circuit)
    p_dnf = np.clip(p_dnf[None, :] * np.where(rain, RAIN_DNF, 1.0)[:, None], 0, 0.9)
    dnf = rng.random((n_sims, n)) < p_dnf

    # 4) Bitiş sırası: DNF'ler en sona, kendi aralarında rastgele
    score = (race_s[None, :] + race_r.gamma * ts.grid(circuit) * -np.log(grid)) / chaos[:, None]
    score = score + rng.gumbel(size=(n_sims, n))
    score = np.where(dnf, -1e6 + rng.random((n_sims, n)), score)
    positions = np.argsort(np.argsort(-score, axis=1), axis=1) + 1

    return SimResult(entries, positions, grid, dnf, sc, rain)


def summarize(sim: SimResult) -> dict:
    pos, grid = sim.positions, sim.grid
    rows = []
    for j, e in enumerate(sim.entries):
        p = pos[:, j]
        rows.append({
            "driver": e.driver, "code": e.code, "name": e.name,
            "team": e.team, "team_name": e.team_name,
            "win": float(np.mean(p == 1)),
            "podium": float(np.mean(p <= 3)),
            "top6": float(np.mean(p <= 6)),
            "top10": float(np.mean(p <= 10)),
            "dnf": float(np.mean(sim.dnf[:, j])),
            "pole": float(np.mean(grid[:, j] == 1)),
            "exp_pos": float(np.mean(p)),
            "exp_points": float(np.mean(POINTS[p - 1])),
        })
    rows.sort(key=lambda r: (-r["win"], r["exp_pos"]))

    # Takım arkadaşı karşılaştırması (son sıralamada önde bitiren)
    idx = {e.driver: j for j, e in enumerate(sim.entries)}
    h2h, seen = [], set()
    for e in sim.entries:
        mate = next((m for m in sim.entries if m.team == e.team and m.driver != e.driver), None)
        if not mate or e.team in seen:
            continue
        seen.add(e.team)
        a, b = idx[e.driver], idx[mate.driver]
        both = ~sim.dnf[:, a] & ~sim.dnf[:, b]
        h2h.append({
            "team_name": e.team_name,
            "a": e.code, "b": mate.code,
            "a_ahead": float(np.mean(pos[:, a] < pos[:, b])),
            "a_ahead_both_finish": float(np.mean(pos[both, a] < pos[both, b])) if both.any() else None,
        })

    # Favori hariç kazanan
    fav = rows[0]
    fj = idx[fav["driver"]]
    others = np.where(np.arange(len(sim.entries))[None, :] == fj, 999, pos)
    winner_wo = np.argmin(others, axis=1)
    without = sorted(
        ({"code": e.code, "p": float(np.mean(winner_wo == j))} for j, e in enumerate(sim.entries) if j != fj),
        key=lambda r: -r["p"],
    )

    finishers = (~sim.dnf).sum(axis=1)
    return {
        "drivers": rows,
        "h2h": h2h,
        "without_favourite": {"favourite": fav["code"], "rows": without[:10]},
        "race": {
            "safety_car": float(np.mean(sim.sc)),
            "rain": float(np.mean(sim.rain)),
            "exp_finishers": float(np.mean(finishers)),
            "p_finishers_ge_18": float(np.mean(finishers >= 18)),
        },
        "n_sims": int(pos.shape[0]),
    }


def market_probs(sim: SimResult, market: str) -> dict[str, float]:
    """Herhangi bir 'ilk K' piyasası için sürücü -> olasılık."""
    k = {"win": 1, "podium": 3, "top6": 6, "top10": 10}[market]
    return {e.driver: float(np.mean(sim.positions[:, j] <= k)) for j, e in enumerate(sim.entries)}
