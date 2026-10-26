"""Güç puanları: zaman ağırlıklı Plackett-Luce, araç + sürücü ayrıştırmalı.

Bir sürücünün gücü = takım (araç) puanı + sürücü puanı. Takım arkadaşları aynı aracı
paylaştığı için sürücü puanı fiilen takım arkadaşına göre fark olarak öğrenilir; sürücü
puanına dar bir öncül konarak bir-iki iyi yarışın sürücüye değil araca yazılması sağlanır.

Yarış modelinde grid pozisyonu ayrı bir değişken: skor += gamma * pist_katsayısı * -log(grid).
Pist katsayısı geçmiş sezonlardan (2022-2025) grid-finiş korelasyonuyla ölçülür: Monaco
gibi sollamanın zor olduğu yerlerde grid daha belirleyici.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

HALF_LIFE = 6.0       # yarış; eski yarışların ağırlığı bu kadar yarışta yarıya iner
SIGMA_TEAM = 2.0      # takım puanı öncülü (geniş)
SIGMA_DRIVER = 0.6    # sürücü puanı öncülü (dar)
GAMMA_PRIOR = (1.0, 1.0)


# --------------------------------------------------------------------------- #
# Pist karakteri (geçmiş sezonlardan)
# --------------------------------------------------------------------------- #

@dataclass
class TrackStats:
    grid_factor: dict[str, float] = field(default_factory=dict)  # 1.0 = ortalama pist
    dnf_factor: dict[str, float] = field(default_factory=dict)   # 1.0 = ortalama DNF oranı
    sc_prob: dict[str, float] = field(default_factory=dict)
    sc_base: float = 0.55
    sc_source: str = "varsayılan"

    def grid(self, circuit: str) -> float:
        return self.grid_factor.get(circuit, 1.0)

    def dnf(self, circuit: str) -> float:
        return self.dnf_factor.get(circuit, 1.0)

    def sc(self, circuit: str) -> float:
        return self.sc_prob.get(circuit, self.sc_base)


def track_stats(history: list[dict], sc_history: dict[str, list[int]]) -> TrackStats:
    rhos: dict[str, list[float]] = {}
    dnf: dict[str, list[int]] = {}
    for race in history:
        fin = [r for r in race["results"] if r["kind"] == "finished" and r["grid"]]
        if len(fin) >= 8:
            rho = spearmanr([r["grid"] for r in fin], [r["pos"] for r in fin]).statistic
            rhos.setdefault(race["circuit"], []).append(rho)
        starters = [r for r in race["results"] if r["kind"] != "excluded"]
        dnf.setdefault(race["circuit"], []).extend(int(r["kind"] == "dnf") for r in starters)

    ts = TrackStats()
    all_rho = [x for v in rhos.values() for x in v]
    if all_rho:
        mean_rho = float(np.mean(all_rho))
        for c, v in rhos.items():
            k = 2.0  # iki "ortalama" yarışlık çekme: tek sezonluk gürültüyü bastırır
            shrunk = (np.sum(v) + k * mean_rho) / (len(v) + k)
            ts.grid_factor[c] = float(shrunk / mean_rho)

    all_dnf = [x for v in dnf.values() for x in v]
    if all_dnf:
        g = float(np.mean(all_dnf))
        for c, v in dnf.items():
            k = 40.0  # ~2 yarışlık başlangıç
            ts.dnf_factor[c] = float(((np.sum(v) + k * g) / (len(v) + k)) / g)

    flat = [x for v in sc_history.values() for x in v]
    if flat:
        ts.sc_base = float(np.mean(flat))
        ts.sc_source = f"OpenF1 ({len(flat)} yarış)"
        for c, v in sc_history.items():
            ts.sc_prob[c] = float((np.sum(v) + 2 * ts.sc_base) / (len(v) + 2))
    return ts


# --------------------------------------------------------------------------- #
# Plackett-Luce uydurma
# --------------------------------------------------------------------------- #

@dataclass
class Ratings:
    drivers: list[str]
    team_of: dict[str, str]
    car: dict[str, float]
    skill: dict[str, float]
    gamma: float = 0.0

    def strength(self, driver: str, team: str | None = None) -> float:
        team = team or self.team_of.get(driver)
        return self.car.get(team, 0.0) + self.skill.get(driver, 0.0)


def _pl_grad(scores: np.ndarray) -> tuple[float, np.ndarray]:
    """Bitiş sırasına dizilmiş skorlar için log-olabilirlik ve skorlara göre gradyan."""
    e = np.exp(scores - scores.max())
    suffix = np.cumsum(e[::-1])[::-1]
    ll = float(np.sum(scores - scores.max() - np.log(suffix)))
    grad = 1.0 - e * np.cumsum(1.0 / suffix)
    return ll, grad


def fit(observations: list[dict], with_grid: bool) -> Ratings:
    """observations: [{"order": [(driver, team, grid), ...], "weight": w, "tau": pist katsayısı}]"""
    drivers = sorted({d for o in observations for d, _, _ in o["order"]})
    teams = sorted({t for o in observations for _, t, _ in o["order"]})
    di = {d: i for i, d in enumerate(drivers)}
    ti = {t: i for i, t in enumerate(teams)}
    nT, nD = len(teams), len(drivers)

    prepared = []
    for o in observations:
        if len(o["order"]) < 2:
            continue
        n = len(o["order"])
        prepared.append((
            np.array([ti[t] for _, t, _ in o["order"]]),
            np.array([di[d] for d, _, _ in o["order"]]),
            np.array([-np.log(g or n) for _, _, g in o["order"]]) * o.get("tau", 1.0),
            o["weight"],
        ))

    def nll(x):
        car, skill, gamma = x[:nT], x[nT:nT + nD], (x[-1] if with_grid else 0.0)
        g_car, g_skill, g_gamma = np.zeros(nT), np.zeros(nD), 0.0
        total = 0.0
        for t_idx, d_idx, gfeat, w in prepared:
            ll, g = _pl_grad(car[t_idx] + skill[d_idx] + gamma * gfeat)
            total += w * ll
            np.add.at(g_car, t_idx, w * g)
            np.add.at(g_skill, d_idx, w * g)
            g_gamma += w * float(g @ gfeat)
        total -= 0.5 * np.sum(car ** 2) / SIGMA_TEAM ** 2 + 0.5 * np.sum(skill ** 2) / SIGMA_DRIVER ** 2
        g_car -= car / SIGMA_TEAM ** 2
        g_skill -= skill / SIGMA_DRIVER ** 2
        grad = [g_car, g_skill]
        if with_grid:
            mu, sd = GAMMA_PRIOR
            total -= 0.5 * (gamma - mu) ** 2 / sd ** 2
            grad.append([g_gamma - (gamma - mu) / sd ** 2])
        return -total, -np.concatenate(grad)

    x0 = np.zeros(nT + nD + (1 if with_grid else 0))
    res = minimize(nll, x0, jac=True, method="L-BFGS-B")
    x = res.x
    team_of = {}
    for o in observations:  # en son yarıştaki takım geçerli
        for d, t, _ in o["order"]:
            team_of[d] = t
    return Ratings(
        drivers=drivers,
        team_of=team_of,
        car={t: float(x[ti[t]]) for t in teams},
        skill={d: float(x[nT + di[d]]) for d in drivers},
        gamma=float(x[-1]) if with_grid else 0.0,
    )


def _weight(round_: int, upto: int) -> float:
    return 0.5 ** ((upto - round_) / HALF_LIFE)


def fit_season(races: list[dict], upto_round: int, ts: TrackStats) -> tuple[Ratings, Ratings]:
    """upto_round'dan ÖNCEKİ yarışlarla yarış ve sıralama modellerini uydurur."""
    past = [r for r in races if r["round"] < upto_round]
    race_obs, quali_obs = [], []
    for r in past:
        w = _weight(r["round"], upto_round)
        team = {x["driver"]: x["team"] for x in r["results"]}
        fin = [x for x in r["results"] if x["kind"] == "finished"]
        race_obs.append({
            "order": [(x["driver"], x["team"], x["grid"]) for x in fin],
            "weight": w,
            "tau": ts.grid(r["circuit"]),
        })
        if r["quali"]:
            quali_obs.append({
                "order": [(d, team.get(d, "?"), None) for d in r["quali"] if d in team],
                "weight": w,
            })
    return fit(race_obs, with_grid=True), fit(quali_obs, with_grid=False)


@dataclass
class Reliability:
    team_rate: dict[str, float]
    base: float


def reliability(races: list[dict], upto_round: int, k: float = 10.0) -> Reliability:
    """Takım bazında yarış başına DNF olasılığı (Beta öncüllü)."""
    past = [r for r in races if r["round"] < upto_round]
    starts: dict[str, int] = {}
    dnfs: dict[str, int] = {}
    for r in past:
        for x in r["results"]:
            if x["kind"] == "excluded":
                continue
            starts[x["team"]] = starts.get(x["team"], 0) + 1
            dnfs[x["team"]] = dnfs.get(x["team"], 0) + int(x["kind"] == "dnf")
    total_s = sum(starts.values())
    base = sum(dnfs.values()) / total_s if total_s else 0.12
    return Reliability(
        team_rate={t: (dnfs[t] + k * base) / (starts[t] + k) for t in starts},
        base=base,
    )
