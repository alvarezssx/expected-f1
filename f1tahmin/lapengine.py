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
FP_WEIGHTS = {"FP1": 1.0, "FP2": 1.5, "FP3": 2.0}  # eski yöntem: sabit seans ağırlıkları
SESSIONS = ("FP1", "FP2", "FP3")
FRIDAY = ("FP1", "FP2")       # Cuma akşamı tahmini: yalnızca Cuma seansları
TEAM_BIAS_K = 3.0              # takım sapması öncülü: ~3 yarışlık sıfıra çekme


class LapEngine:
    def __init__(self, races_current: list[dict], ts_v1: TrackStats, season: int = data.CURRENT_SEASON):
        """season: tahmin edilen sezon. Bu sezonun verisi yalnızca ilgili yarıştan öncesiyle kullanılır;
        geçmiş sezonlar pist karakteri ve sollama modeli için. Backtest'te 2025 de bu şekilde tahmin edilir."""
        self.season = season
        self.races = races_current
        self.ts_v1 = ts_v1
        self.reload()

    def reload(self):
        feats = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for s in SEASONS:
                compact = {r["round"]: r for r in (self.races if s == self.season else data.races_compact(s))}
                for rnd in laps.available(s):
                    if rnd in compact:
                        f = process_race(laps.load(s, rnd), compact[rnd])
                        if f:
                            feats.append(f)
        self.feats = feats
        self.quali = data.qualifying(self.season)
        self.fp = fia.load(self.season)
        self.use_fp = True
        self._ot_cache = {}
        self._gp_cache: dict = {}
        self._wf_errors: dict[int, list[tuple[float, float]]] = {}
        self._fp_rows_cache: dict = {}
        self._fp_link_cache: dict = {}
        # Öğrenilmiş seans ağırlıkları ve takım sapması 2026 A/B testinde (15 yarış, ileriye dönük)
        # görülmemiş yarışlardaki hatayı artırdı (sıralama RMSE 0.475% -> 0.483% / 0.493%); kapalı.
        # Daha fazla veri birikince yeniden test edilebilir.
        self.fp_learned = False
        self.fp_team_bias = False

    def _gps(self, rnd: int):
        """rnd'den ÖNCEKİ veriyle kurulan yarış ve sıralama GP'leri (önbellekli)."""
        if rnd not in self._gp_cache:
            cur = [f for f in self.feats if f["season"] == self.season]
            q_obs, q_sigma = quali_obs(self.quali, rnd)
            self._gp_cache[rnd] = (fit_gp(race_obs(cur, rnd)), fit_gp(q_obs), q_sigma)
        return self._gp_cache[rnd]

    # ------------------------------------------------------------------ #
    # Antrenman (FIA FP sınıflandırmaları) -> sıralama öncesi tahmin
    # ------------------------------------------------------------------ #

    def session_gaps(self, rnd: int) -> dict[str, dict[str, float]]:
        """Her antrenman seansında sürücünün seans medyanına göre % farkı (negatif = hızlı)."""
        out = {}
        for sess, times in self.fp.get(rnd, {}).items():
            if len(times) < 10:
                continue
            ref = float(np.median(list(times.values())))
            g = {d: 100 * (t - ref) / ref for d, t in times.items()}
            out[sess] = {d: v for d, v in g.items() if abs(v) < 4}  # sorunlu seans / yarım program
        return out

    @staticmethod
    def _features(gaps: dict[str, float], sessions: tuple[str, ...]):
        """Seans değişkenleri (eksik seans, sürücünün eldeki seanslarının ortalamasıyla) ve sabit ağırlıklı ortalama."""
        vals = [gaps.get(s) for s in sessions]
        avail = [v for v in vals if v is not None]
        if not avail:
            return None, None
        fill = float(np.mean(avail))
        w = [(gaps[s], FP_WEIGHTS[s]) for s in sessions if s in gaps]
        old = float(np.average([g for g, _ in w], weights=[x for _, x in w]))
        return [fill if v is None else v for v in vals], old

    def fp_pct(self, rnd: int, sessions: tuple[str, ...] = SESSIONS) -> dict[str, float]:
        """Sabit ağırlıklı antrenman temposu (gösterim ve eski yöntem için)."""
        by_driver: dict[str, dict[str, float]] = {}
        for sess, g in self.session_gaps(rnd).items():
            for d, v in g.items():
                by_driver.setdefault(d, {})[sess] = v
        out = {}
        for d, gaps in by_driver.items():
            _, old = self._features(gaps, sessions)
            if old is not None:
                out[d] = old
        return out

    def _fp_rows(self, r: int) -> list[dict]:
        """r yarışı için sürücü satırları: gerçek yarış/sıralama temposu, GP tahminleri (yalnızca öncesiyle),
        takım ve seans farkları."""
        if r in self._fp_rows_cache:
            return self._fp_rows_cache[r]
        sg = self.session_gaps(r)
        out = []
        cur = [f for f in self.feats if f["season"] == self.season]
        f_r = next((f for f in cur if f["round"] == r), None)
        qp = self._quali_pct(r)
        hit = next((q for q in self.quali if int(q["round"]) == r), None)
        if sg and hit and len([f for f in cur if f["round"] < r]) >= 3:
            gp_r, gp_q, _ = self._gps(r)
            team = {x["Driver"]["driverId"]: x["Constructor"]["constructorId"] for x in hit["QualifyingResults"]}
            drivers = sorted({d for g in sg.values() for d in g if d in team})
            if drivers:
                ents = [(d, team[d]) for d in drivers]
                mr, _ = gp_r.predict(ents, r)
                mq, _ = gp_q.predict(ents, r)
                for (d, t), a, b in zip(ents, mr, mq):
                    p = f_r["pace"].get(d) if f_r else None
                    y = 100 * p["alpha"] / f_r["ref"] if p and p["n"] >= 8 else np.nan
                    out.append({"round": r, "driver": d, "team": t, "y": y, "gr": float(a), "gq": float(b),
                                "qo": qp.get(d, np.nan),
                                "gaps": {s: g[d] for s, g in sg.items() if d in g}})
        self._fp_rows_cache[r] = out
        return out

    def fp_link(self, rnd: int, sessions: tuple[str, ...] = SESSIONS,
                learned: bool = True, team_bias: bool = True) -> dict | None:
        """Antrenman -> sıralama / yarış temposu yığınlaması (yalnızca rnd'den önceki yarışlardan öğrenilir).

        learned=True:  hedef ≈ a·GP + Σ b_s·seans_s + c   (seans ağırlıkları veriden)
        learned=False: hedef ≈ a·GP + b·sabit_ağırlıklı_ortalama + c
        team_bias:     regresyon artığının takım ortalaması, az verili takım sıfıra çekilerek (her takımın
                       antrenmanda tempo saklama / gösterme alışkanlığı)
        """
        key = (rnd, sessions, learned, team_bias)
        if key in self._fp_link_cache:
            return self._fp_link_cache[key]
        rows = []
        for r in range(4, rnd):
            for row in self._fp_rows(r):
                feats, old = self._features(row["gaps"], sessions)
                if feats is not None:
                    rows.append((row, feats if learned else [old]))
        result = None
        if len(rows) >= 60:
            result = {"n": len(rows), "sessions": list(sessions), "learned": learned}
            for key_t, tcol, bcol in (("quali", "qo", "gq"), ("race", "y", "gr")):
                sel = [(r, f) for r, f in rows if np.isfinite(r[tcol])]
                target = np.array([r[tcol] for r, _ in sel])
                base = np.array([r[bcol] for r, _ in sel])
                X = np.column_stack([base, np.array([f for _, f in sel]), np.ones(len(sel))])
                ridge = np.diag([1e-3] * (X.shape[1] - 1) + [0.0])
                coef = np.linalg.solve(X.T @ X + ridge, X.T @ target)
                resid = target - X @ coef
                bias = {}
                if team_bias:
                    per: dict[str, dict[int, list[float]]] = {}
                    for (r, _), e in zip(sel, resid):
                        per.setdefault(r["team"], {}).setdefault(r["round"], []).append(e)
                    for t, by_r in per.items():
                        means = [float(np.mean(v)) for v in by_r.values()]
                        bias[t] = float(np.sum(means) / (len(means) + TEAM_BIAS_K))
                    resid = resid - np.array([bias.get(r["team"], 0.0) for r, _ in sel])
                mse_s = float(np.mean(resid ** 2))
                mse_g = float(np.mean((target - base) ** 2))
                result[key_t] = {"coef": [float(c) for c in coef], "team_bias": bias,
                                 "var_ratio": float(np.clip(mse_s / mse_g, 0.2, 1.0)),
                                 "rmse_gp": mse_g ** 0.5, "rmse_stack": mse_s ** 0.5}
        self._fp_link_cache[key] = result
        return result

    def fp_predict(self, rnd: int, entries, q_mean, r_mean, sessions, learned, team_bias):
        """Antrenman bilgisiyle güncellenmiş sıralama ve yarış temposu; antrenmanı olmayan sürücü GP'de kalır."""
        fl = self.fp_link(rnd, sessions, learned, team_bias)
        sg = self.session_gaps(rnd)
        if not fl or not any(s in sg for s in sessions):
            return None
        q_new, r_new = q_mean.copy(), r_mean.copy()
        n = 0
        for j, e in enumerate(entries):
            gaps = {s: sg[s][e.driver] for s in sessions if s in sg and e.driver in sg[s]}
            feats, old = self._features(gaps, sessions)
            if feats is None:
                continue
            f = feats if learned else [old]
            n += 1
            for key_t, arr, base in (("quali", q_new, q_mean[j]), ("race", r_new, r_mean[j])):
                c = fl[key_t]["coef"]
                arr[j] = c[0] * base + float(np.dot(c[1:-1], f)) + c[-1] + fl[key_t]["team_bias"].get(e.team, 0.0)
        return q_new, r_new, fl, n

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
        cur = [f for f in self.feats if f["season"] == self.season]
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
        cur = [f for f in self.feats if f["season"] == self.season and f["round"] < rnd]
        return len(cur) >= 3

    def _overtake(self, rnd: int):
        """Sollama modeli ve grid sırasına göre start gürültüsü (yalnızca bu yarıştan önceki veriyle)."""
        if rnd not in self._ot_cache:
            past = [f for f in self.feats if (f["season"], f["round"]) < (self.season, rnd)]
            self._ot_cache[rnd] = (fit_overtake(past), start_profile(past))
        return self._ot_cache[rnd]

    def build(self, rnd: int, circuit: str, entries, known_grid: list[str] | None,
              rain_prob: float, pace_scale: float = 1.0, form_scale: float = 1.0,
              fp_sessions: tuple[str, ...] = SESSIONS) -> tuple[Inputs, dict]:
        cur_feats = [f for f in self.feats if f["season"] == self.season]
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
        if self.use_fp and not known_grid:
            got = self.fp_predict(rnd, entries, q_mean, r_mean, fp_sessions, self.fp_learned, self.fp_team_bias)
            if got:
                q_mean, r_mean, fl, n_fp = got
                q_cov, quali_tau = q_cov * fl["quali"]["var_ratio"], quali_tau * np.sqrt(fl["quali"]["var_ratio"])
                r_cov, race_tau = r_cov * fl["race"]["var_ratio"], race_tau * np.sqrt(fl["race"]["var_ratio"])
                fp_used = {**fl, "sessions": sorted(s for s in self.fp.get(rnd, {}) if s in fp_sessions),
                           "drivers": n_fp}

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
        tp = build_track_params(self.feats, circuit, (self.season, rnd), ot, profile)
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

    def run(self, rnd, circuit, entries, known_grid, rain_prob, n=20000, seed=0, pace_scale=1.0, form_scale=1.0,
            fp_sessions=SESSIONS):
        inp, meta = self.build(rnd, circuit, entries, known_grid, rain_prob, pace_scale, form_scale, fp_sessions)
        return simulate(inp, n=n, seed=seed), meta
