"""Sezon tempo modeli: Gauss süreci (araç rastgele yürüyüşü + sürücü farkı).

Gözlem (yüzde cinsinden, tur süresine oranla; negatif = hızlı):
    y[d, r] = araç[takım, r] + sürücü[d] + hafta_sonu_formu + ölçüm_hatası
    araç[t, r] = araç[t, r-1] + gelişim,   gelişim ~ N(0, q²)        (Kalman rastgele yürüyüşü)
    araç[t, 1] ~ N(0, σ0²),  sürücü[d] ~ N(0, σs²),  form ~ N(0, τ²),  hata ~ N(0, se²)

Bu, çekirdeği
    K = [aynı takım]·(σ0² + q²·min(r, r')) + [aynı sürücü]·σs² + [aynı gözlem]·(τ² + se²)
olan bir Gauss sürecine denk. σ0, q, σs, τ log marjinal olabilirliği maksimize edilerek veriden
seçilir (ampirik Bayes). Tahmin, tüm sürücüler için ortak (korelasyonlu) bir normal dağılımdır:
takım arkadaşları aynı araç belirsizliğini paylaşır.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


@dataclass
class Obs:
    driver: str
    team: str
    rnd: int
    y: float      # %
    se: float     # %


@dataclass
class PaceGP:
    obs: list[Obs]
    sigma0: float
    q: float
    sigma_s: float
    tau: float
    alpha: np.ndarray       # K⁻¹ y
    chol: tuple
    loglik: float

    def _k(self, t1, r1, d1, t2, r2, d2):
        same_t = (t1[:, None] == t2[None, :])
        same_d = (d1[:, None] == d2[None, :])
        rmin = np.minimum(r1[:, None], r2[None, :])
        return same_t * (self.sigma0 ** 2 + self.q ** 2 * rmin) + same_d * self.sigma_s ** 2

    def predict(self, entries: list[tuple[str, str]], rnd: int) -> tuple[np.ndarray, np.ndarray]:
        """[(sürücü, takım)] için rnd yarışındaki gizli tempo: ortalama ve kovaryans (form hariç)."""
        o = self.obs
        t_o = np.array([x.team for x in o]); r_o = np.array([x.rnd for x in o]); d_o = np.array([x.driver for x in o])
        t_n = np.array([t for _, t in entries]); d_n = np.array([d for d, _ in entries]); r_n = np.full(len(entries), rnd)
        Ks = self._k(t_o, r_o, d_o, t_n, r_n, d_n)
        Kss = self._k(t_n, r_n, d_n, t_n, r_n, d_n)
        mean = Ks.T @ self.alpha
        v = cho_solve(self.chol, Ks)
        cov = Kss - Ks.T @ v
        return mean, (cov + cov.T) / 2 + 1e-9 * np.eye(len(entries))

    def residuals(self) -> dict[tuple[str, int], float]:
        """Her gözlemin hafta sonu form artığı (y - gizli tempo tahmini)."""
        o = self.obs
        t = np.array([x.team for x in o]); r = np.array([x.rnd for x in o]); d = np.array([x.driver for x in o])
        f = self._k(t, r, d, t, r, d) @ self.alpha
        return {(x.driver, x.rnd): float(x.y - fi) for x, fi in zip(o, f)}


def fit_gp(obs: list[Obs], init=(1.0, 0.15, 0.25, 0.3)) -> PaceGP:
    y = np.array([x.y for x in obs])
    se2 = np.array([x.se for x in obs]) ** 2
    t = np.array([x.team for x in obs]); r = np.array([x.rnd for x in obs]); d = np.array([x.driver for x in obs])
    same_t = (t[:, None] == t[None, :]).astype(float)
    same_d = (d[:, None] == d[None, :]).astype(float)
    rmin = np.minimum(r[:, None], r[None, :]).astype(float)

    def build(p):
        s0, q, ss, tau = np.exp(p)
        return same_t * (s0 ** 2 + q ** 2 * rmin) + same_d * ss ** 2 + np.diag(tau ** 2 + se2)

    def nll(p):
        K = build(p)
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, y)
        logdet = 2 * np.sum(np.log(np.diag(c[0])))
        # zayıf log-normal öncüller: aşırı uç hiperparametreleri engeller
        prior = 0.5 * np.sum(((p - np.log(init)) / 1.5) ** 2)
        return 0.5 * y @ a + 0.5 * logdet + prior

    res = minimize(nll, np.log(init), method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-3})
    p = res.x
    K = build(p)
    c = cho_factor(K, lower=True)
    s0, q, ss, tau = np.exp(p)
    return PaceGP(obs, s0, q, ss, tau, cho_solve(c, y), c, -float(res.fun))


# --------------------------------------------------------------------------- #
# Gözlemleri hazırlama
# --------------------------------------------------------------------------- #

def race_obs(feats: list[dict], upto_round: int) -> list[Obs]:
    out = []
    for f in feats:
        if f["round"] >= upto_round:
            continue
        for d, p in f["pace"].items():
            if p["team"] and p["n"] >= 8:
                out.append(Obs(d, p["team"], f["round"], 100 * p["alpha"] / f["ref"], 100 * p["se"] / f["ref"]))
    return out


def _t(s):
    if not s:
        return None
    m, _, sec = s.rpartition(":")
    try:
        return (float(m) * 60 if m else 0.0) + float(sec)
    except ValueError:
        return None


def quali_times(q_race: dict) -> dict:
    """Sıralama turu: her sürücünün Q1'e eşdeğer en iyi süresi ve seans bazlı ham süreler.

    Pist her seansta hızlandığı için Q2/Q3 süreleri, iki seansta da süre koyan sürücülerin
    medyan iyileşmesi kadar geri kaydırılır.
    """
    rows = []
    for x in q_race["QualifyingResults"]:
        rows.append({"driver": x["Driver"]["driverId"], "team": x["Constructor"]["constructorId"],
                     "q1": _t(x.get("Q1")), "q2": _t(x.get("Q2")), "q3": _t(x.get("Q3"))})

    def evo(a, b):
        diffs = [r[b] - r[a] for r in rows if r[a] and r[b] and abs(r[b] - r[a]) < 3]
        return float(np.median(diffs)) if diffs else 0.0

    e12, e23 = evo("q1", "q2"), evo("q2", "q3")
    best = {}
    for r in rows:
        c = [v for v in (r["q1"], r["q2"] and r["q2"] - e12, r["q3"] and r["q3"] - e12 - e23) if v]
        if c:
            best[r["driver"]] = (r["team"], min(c))
    return {"rows": rows, "best": best, "evo": (e12, e23)}


def quali_obs(q_seasons: list[dict], upto_round: int) -> tuple[list[Obs], float]:
    """Yüzde cinsinden sıralama temposu gözlemleri + seans içi tek tur gürültüsü (σ, %)."""
    out, noise = [], []
    for q in q_seasons:
        rnd = int(q["round"])
        if rnd >= upto_round:
            continue
        qt = quali_times(q)
        if len(qt["best"]) < 10:
            continue
        times = np.array([v for _, v in qt["best"].values()])
        ref = float(np.median(times))
        for d, (team, v) in qt["best"].items():
            g = 100 * (v - ref) / ref
            if abs(g) < 5:  # sorunlu tur / kaza
                out.append(Obs(d, team, rnd, g, 0.05))
        # Q2 -> Q3 tutarlılığı: seans başına tek tur gürültüsü
        e12, e23 = qt["evo"]
        for r in qt["rows"]:
            if r["q2"] and r["q3"]:
                dlt = (r["q3"] - r["q2"] - e23) / ref * 100
                if abs(dlt) < 1.5:
                    noise.append(dlt)
    sigma_lap = float(np.std(noise) / np.sqrt(2)) if len(noise) > 10 else 0.12
    return out, sigma_lap
