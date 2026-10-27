"""Tur tur modelin hazırlanması: veri -> ölçümler -> tempo GP'leri -> pist -> simülasyon -> piyasalar."""
from __future__ import annotations

import warnings

import numpy as np

from . import data, fia, laps
from .features import process_race
from .model import reliability, TrackStats
from .pace import fit_gp, quali_obs, quali_times, race_obs
from .racesim import Inputs, simulate
from .track import build_track_params, fit_overtake, start_profile, team_pit_effects

SEASONS = [2023, 2024, 2025, data.CURRENT_SEASON]
FP_WEIGHTS = {"FP1": 1.0, "FP2": 1.5, "FP3": 2.0}  # geç seanslar sıralama/yarış programına daha yakın


class LapEngine:
    def __init__(self, races_current: list[dict], ts_v1: TrackStats):
        self.races = races_current
        self.ts_v1 = ts_v1
        self.reload()

    def reload(self):
        feats = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for s in SEASONS:
                compact = {r["round"]: r for r in (self.races if s == data.CURRENT_SEASON else data.races_compact(s))}
                for rnd in laps.available(s):
                    if rnd in compact:
                        f = process_race(laps.load(s, rnd), compact[rnd])
                        if f:
                            feats.append(f)
        self.feats = feats
        self.quali = data.qualifying(data.CURRENT_SEASON)
        self.fp = fia.load(data.CURRENT_SEASON)
        self.use_fp = True
        self._ot_cache = {}
        self._gp_cache: dict = {}
        self._wf_errors: dict[int, list[tuple[float, float]]] = {}
        self._fp_rows_cache: dict = {}

    def _gps(self, rnd: int):
        """rnd'den ÖNCEKİ veriyle kurulan yarış ve sıralama GP'leri (önbellekli)."""
        if rnd not in self._gp_cache:
            cur = [f for f in self.feats if f["season"] == data.CURRENT_SEASON]
            q_obs, q_sigma = quali_obs(self.quali, rnd)
            self._gp_cache[rnd] = (fit_gp(race_obs(cur, rnd)), fit_gp(q_obs), q_sigma)
        return self._gp_cache[rnd]

    # ------------------------------------------------------------------ #
    # Antrenman (FIA FP sınıflandırmaları) -> sıralama öncesi tahmin
    # ------------------------------------------------------------------ #

    def fp_pct(self, rnd: int) -> dict[str, float]:
        """Sürücünün antrenman temposu: her seansta seans medyanına göre % fark, seans ağırlıklı ortalama."""
        sessions = self.fp.get(rnd, {})
        acc: dict[str, list[tuple[float, float]]] = {}
        for sess, times in sessions.items():
            if len(times) < 10:
                continue
            ref = float(np.median(list(times.values())))
            for d, t in times.items():
                g = 100 * (t - ref) / ref
                if abs(g) < 4:  # sorunlu seans / yarım program
                    acc.setdefault(d, []).append((g, FP_WEIGHTS.get(sess, 1.0)))
        return {d: float(np.average([g for g, _ in v], weights=[w for _, w in v])) for d, v in acc.items()}

    def _fp_rows(self, r: int) -> list[tuple[float, float, float, float, float]]:
        """r yarışı için (gerçek yarış, GP yarış, GP sıralama, gerçek sıralama, antrenman) — GP'ler yalnızca öncesiyle."""
        if r in self._fp_rows_cache:
            return self._fp_rows_cache[r]
        fp = self.fp_pct(r)
        out = []
        cur = [f for f in self.feats if f["season"] == data.CURRENT_SEASON]
        f_r = next((f for f in cur if f["round"] == r), None)
        qp = self._quali_pct(r)
        hit = next((q for q in self.quali if int(q["round"]) == r), None)
        if fp and hit and len([f for f in cur if f["round"] < r]) >= 3:
            gp_r, gp_q, _ = self._gps(r)
            team = {x["Driver"]["driverId"]: x["Constructor"]["constructorId"] for x in hit["QualifyingResults"]}
            ents = [(d, team[d]) for d in fp if d in team]
            if ents:
                mr, _ = gp_r.predict(ents, r)
                mq, _ = gp_q.predict(ents, r)
                for (d, _), a, b in zip(ents, mr, mq):
                    p = f_r["pace"].get(d) if f_r else None
                    y = 100 * p["alpha"] / f_r["ref"] if p and p["n"] >= 8 else np.nan
                    out.append((y, float(a), float(b), qp.get(d, np.nan), fp[d]))
        self._fp_rows_cache[r] = out
        return out

    def fp_link(self, rnd: int) -> dict | None:
        """Yığınlama: sıralama ≈ a·GP_sıralama + b·antrenman + c  ve  yarış ≈ a'·GP_yarış + b'·antrenman + c'.
        Katsayılar ve belirsizlik daralması yalnızca rnd'den önceki yarışlardan."""
        rows = np.array([p for r in range(4, rnd) for p in self._fp_rows(r)])
        if len(rows) < 60:
            return None
        y, gr, gq, qo, fp = rows.T
        out = {"n": len(rows)}
        for key, target, base in (("quali", qo, gq), ("race", y, gr)):
            m = np.isfinite(target)
            X = np.column_stack([base[m], fp[m], np.ones(m.sum())])
            coef = np.linalg.solve(X.T @ X + np.diag([1e-3, 1e-3, 0]), X.T @ target[m])
            mse_s = float(np.mean((target[m] - X @ coef) ** 2))
            mse_g = float(np.mean((target[m] - base[m]) ** 2))
            out[key] = {"coef": [float(c) for c in coef], "var_ratio": float(np.clip(mse_s / mse_g, 0.2, 1.0)),
                        "rmse_gp": mse_g ** 0.5, "rmse_stack": mse_s ** 0.5}
        return out

    # ------------------------------------------------------------------ #
    # Sıralama turu -> yarış temposu bağlantısı (ileriye dönük hatalardan)
    # ------------------------------------------------------------------ #

    def _quali_pct(self, rnd: int) -> dict[str, float]:
        hit = next((q for q in self.quali if int(q["round"]) == rnd), None)
        if not hit:
            return {}
        qt = quali_times(hit)
        if len(qt["best"]) < 10:
            return {}
        ref = float(np.median([v for _, v in qt["best"].values()]))
        return {d: 100 * (v - ref) / ref for d, (_, v) in qt["best"].items() if abs(100 * (v - ref) / ref) < 5}

    def _errors_for(self, r: int) -> list[tuple[float, float, float, float]]:
        """r yarışı için, yalnızca öncesiyle kurulan modellerden:
        (gerçek yarış temposu, GP yarış tahmini, GP sıralama tahmini, gerçek sıralama temposu)."""
        if r in self._wf_errors:
            return self._wf_errors[r]
        cur = [f for f in self.feats if f["season"] == data.CURRENT_SEASON]
        f_r = next((f for f in cur if f["round"] == r), None)
        qp = self._quali_pct(r)
        out = []
        if f_r and qp and len([f for f in cur if f["round"] < r]) >= 3:
            gp_r, gp_q, _ = self._gps(r)
            ents = [(d, p["team"]) for d, p in f_r["pace"].items() if d in qp and p["n"] >= 8]
            if ents:
                mr, _ = gp_r.predict(ents, r)
                mq, _ = gp_q.predict(ents, r)
                for (d, _), a, b in zip(ents, mr, mq):
                    out.append((100 * f_r["pace"][d]["alpha"] / f_r["ref"], float(a), float(b), qp[d]))
        self._wf_errors[r] = out
        return out

    def quali_link(self, rnd: int) -> dict:
        """Sıralama turu sonrası yarış temposu için yığınlama (stacking) regresyonu:
            yarış ≈ c_gp · GP_yarış_tahmini + c_q · gerçek_sıralama_temposu + c0
        katsayılar yalnızca rnd'den önceki yarışların ileriye dönük tahminlerinden öğrenilir.
        var_ratio: yığınlama hatası / yalnız-GP hatası (belirsizlik bu oranda daralır)."""
        rows = [p for r in range(4, rnd) for p in self._errors_for(r)]
        if len(rows) < 60:
            return {"coef": None, "var_ratio": 0.85, "n": len(rows)}
        y, gr, gq, qo = np.array(rows).T
        X = np.column_stack([gr, qo, np.ones(len(y))])
        coef = np.linalg.solve(X.T @ X + np.diag([1e-3, 1e-3, 0]), X.T @ y)
        mse_stack = float(np.mean((y - X @ coef) ** 2))
        mse_gp = float(np.mean((y - gr) ** 2))
        return {"coef": [float(c) for c in coef], "var_ratio": float(np.clip(mse_stack / mse_gp, 0.2, 1.0)),
                "n": len(rows), "rmse_gp": mse_gp ** 0.5, "rmse_stack": mse_stack ** 0.5}

    def coverage(self) -> dict:
        out = {}
        for f in self.feats:
            out[f["season"]] = out.get(f["season"], 0) + 1
        return out

    def ready_for(self, rnd: int) -> bool:
        cur = [f for f in self.feats if f["season"] == data.CURRENT_SEASON and f["round"] < rnd]
        return len(cur) >= 3

    def _overtake(self, rnd: int):
        """Sollama modeli ve grid sırasına göre start gürültüsü (yalnızca bu yarıştan önceki veriyle)."""
        if rnd not in self._ot_cache:
            past = [f for f in self.feats if (f["season"], f["round"]) < (data.CURRENT_SEASON, rnd)]
            self._ot_cache[rnd] = (fit_overtake(past), start_profile(past))
        return self._ot_cache[rnd]

    def build(self, rnd: int, circuit: str, entries, known_grid: list[str] | None,
              rain_prob: float, pace_scale: float = 1.0, form_scale: float = 1.0) -> tuple[Inputs, dict]:
        cur_feats = [f for f in self.feats if f["season"] == data.CURRENT_SEASON]
        gp_r, gp_q, q_sigma = self._gps(rnd)

        # sıralama ve yarış formu arasındaki korelasyon (artıklardan)
        rr, rq = gp_r.residuals(), gp_q.residuals()
        both = [k for k in rr if k in rq]
        rho = float(np.clip(np.corrcoef([rr[k] for k in both], [rq[k] for k in both])[0, 1], 0.0, 0.9)) if len(both) > 20 else 0.3

        ents = [(e.driver, e.team) for e in entries]
        r_mean, r_cov = gp_r.predict(ents, rnd)
        q_mean, q_cov = gp_q.predict(ents, rnd)
        race_tau, quali_tau = gp_r.tau, gp_q.tau

        # Sıralama turundan önce: antrenman temposu o hafta sonu aracın hızı hakkında kanıttır.
        fp_used = None
        fpp = self.fp_pct(rnd) if (self.use_fp and not known_grid) else {}
        if fpp:
            fl = self.fp_link(rnd)
            if fl:
                fq = np.array([fpp.get(e.driver, np.nan) for e in entries])
                have = np.isfinite(fq)
                a, b, c = fl["quali"]["coef"]
                q_mean = np.where(have, a * q_mean + b * np.nan_to_num(fq) + c, q_mean)
                a, b, c = fl["race"]["coef"]
                r_mean = np.where(have, a * r_mean + b * np.nan_to_num(fq) + c, r_mean)
                q_cov, quali_tau = q_cov * fl["quali"]["var_ratio"], quali_tau * np.sqrt(fl["quali"]["var_ratio"])
                r_cov, race_tau = r_cov * fl["race"]["var_ratio"], race_tau * np.sqrt(fl["race"]["var_ratio"])
                fp_used = {**fl, "sessions": sorted(self.fp.get(rnd, {})), "drivers": int(have.sum())}

        # Sıralama turu bittiyse: gerçek sıralama temposu, o hafta sonunun yarış temposu hakkında kanıttır.
        #   E[yarış | sıralama] = c_gp·GP + c_q·gözlenen_sıralama + c0  (ileriye dönük yığınlama),
        #   belirsizlik × (yığınlama hatası / GP hatası)
        link = None
        qp = self._quali_pct(rnd) if known_grid else {}
        if qp:
            link = self.quali_link(rnd)
            ratio = link["var_ratio"]
            if link["coef"]:
                c_gp, c_q, c0 = link["coef"]
                obs = np.array([qp.get(e.driver, q_mean[j]) for j, e in enumerate(entries)])
                r_mean = c_gp * r_mean + c_q * obs + c0
            r_cov = r_cov * ratio
            race_tau = race_tau * np.sqrt(ratio)

        rel = reliability(self.races, rnd)
        p_dnf = np.array([rel.team_rate.get(e.team, rel.base) for e in entries]) * self.ts_v1.dnf(circuit)

        ot, profile = self._overtake(rnd)
        tp = build_track_params(self.feats, circuit, (data.CURRENT_SEASON, rnd), ot, profile)
        team_eff, p_slow, slow_mean = team_pit_effects([f for f in cur_feats if f["round"] < rnd])

        grid = None
        if known_grid:
            pos = {d: i + 1 for i, d in enumerate(known_grid)}
            grid = np.array([pos.get(e.driver, len(entries)) for e in entries])

        inp = Inputs(
            drivers=[e.driver for e in entries], teams=[e.team for e in entries],
            race_mean=r_mean, race_cov=r_cov, race_tau=race_tau,
            quali_mean=q_mean, quali_cov=q_cov, quali_tau=quali_tau, quali_session_sigma=q_sigma, rho=rho,
            p_dnf=np.clip(p_dnf, 0.01, 0.6),
            team_pit=np.array([team_eff.get(e.team, 0.0) for e in entries]),
            p_slow_pit=p_slow, slow_pit_mean=slow_mean,
            track=tp, b_pace=ot.b_pace, b_gap=ot.b_gap,
            known_grid=grid, rain_prob=rain_prob, pace_scale=pace_scale, form_scale=form_scale,
        )
        meta = {
            "gp_race": {"sigma0": gp_r.sigma0, "q": gp_r.q, "sigma_s": gp_r.sigma_s, "tau": gp_r.tau, "n": len(gp_r.obs)},
            "gp_quali": {"sigma0": gp_q.sigma0, "q": gp_q.q, "sigma_s": gp_q.sigma_s, "tau": gp_q.tau, "n": len(gp_q.obs),
                         "session_sigma": q_sigma},
            "rho": rho,
            "quali_link": link,
            "fp_link": fp_used,
            "overtake": {"a": ot.a, "a_track": tp.overtake_logit, "a_2026": ot.a_2026, "b_pace": ot.b_pace, "b_gap": ot.b_gap},
            "track": {k: v for k, v in tp.__dict__.items() if k not in ("sc_len", "vsc_len", "start_profile")},
            "start_profile": tp.start_profile[:10],
            "pit": {"p_slow": p_slow, "slow_mean": slow_mean},
            "pace": [{"driver": e.driver, "race_pct": float(r_mean[j]), "race_sd": float(np.sqrt(r_cov[j, j])),
                      "race_sec": float(r_mean[j] * tp.ref / 100), "quali_pct": float(q_mean[j]),
                      "quali_sd": float(np.sqrt(q_cov[j, j]))} for j, e in enumerate(entries)],
        }
        return inp, meta

    def run(self, rnd, circuit, entries, known_grid, rain_prob, n=20000, seed=0, pace_scale=1.0, form_scale=1.0):
        inp, meta = self.build(rnd, circuit, entries, known_grid, rain_prob, pace_scale, form_scale)
        return simulate(inp, n=n, seed=seed), meta
