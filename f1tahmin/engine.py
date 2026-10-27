"""Veri, model ve simülasyonu birbirine bağlayan katman: tahmin, backtest, şampiyonluk.

İki model var:
  v1  sıralama modeli (Plackett-Luce): yalnızca bitiş sıralarından öğrenir, hızlı.
  v2  tur tur simülatör (lapengine + racesim): tur süreleri, lastik, pit, SC, sollama.
Tur verisi yeterli olan her yarışta v2 kullanılır; backtest ikisini aynı yarışlarda karşılaştırır.
"""
from __future__ import annotations

import json
import math
import threading

import numpy as np

from . import data, fia, laps, odds
from .lapengine import FRIDAY, SESSIONS, LapEngine
from .model import fit_season, reliability, track_stats
from .simulate import POINTS, Entry, SimResult, simulate, summarize, market_probs

SPRINT_POINTS = np.array([8, 7, 6, 5, 4, 3, 2, 1] + [0] * 30)
# v2 kalibrasyon ızgarası: (tempo ölçeği, form ölçeği)
V2_GRID = [(s, f) for s in (0.8, 1.0, 1.25) for f in (0.0, 0.35, 0.7)]
BACKTEST_EXTRA_SEASONS = [2025]  # doğrulama: bu sezonlar da ayrıca tahmin edilip ayar seçimine katılır
V2_CACHE = data.CACHE / "backtest_v2.json"
V2_VERSION = 15  # simülatör/model kodu değişince artır: diskteki backtest geçersiz olur


class Engine:
    def __init__(self, refresh: bool = False, season: int = data.CURRENT_SEASON, light: bool = False):
        """light=True: yalnızca backtest koşuları üretir (geçmiş sezonu doğrulamak için), kendi
        kalibrasyonunu yapmaz; ayarlar ana motordan gelir."""
        self.season = season
        self.light = light
        self.lock = threading.RLock()
        self.load(refresh)

    def load(self, refresh: bool = False):
        self.calendar = data.schedule(self.season, refresh=refresh)
        self.races = data.races_compact(self.season, refresh=refresh)
        self.standings = data.driver_standings(self.season, refresh=refresh)
        history = [r for s in data.HISTORY_SEASONS if s < self.season for r in data.races_compact(s)]
        self.sc_history = data.safety_car_history(self.calendar)
        self.ts = track_stats(history, self.sc_history)
        self.done = {r["round"] for r in self.races}
        self._pred_cache: dict = {}
        self._sims: dict = {}
        self._backtest = None
        if refresh and self.season == data.CURRENT_SEASON:  # yeni yarışların tur ve antrenman verisi
            have = set(laps.available(self.season))
            for rd in sorted(self.done - have):
                laps.fetch_race(self.season, rd)
            try:
                fia.fetch_season(self.season, refresh=True)
            except Exception as e:  # FIA erişilemezse eldeki antrenman verisiyle devam
                print(f"FIA antrenman verisi alınamadı: {type(e).__name__}", flush=True)
        self.lap = LapEngine(self.races, self.ts, self.season)
        self.temps = {"pre": 1.0, "post": 1.0}
        self.v2_params = {"pre": (1.0, 1.0), "post": (1.0, 1.0)}
        self.win_k = {"pre": 1.0, "post": 1.0}
        self.win_calib = {}
        self._result_map = {(self.season, r["round"]): {x["driver"]: x for x in r["results"]} for r in self.races}
        if self.light:
            # 2025 antrenman verisinin çoğu FIA'nın istek sınırı yüzünden indirilemedi; yarım veriyle
            # tutarsız karışım olmasın diye doğrulama sezonunda sıralama öncesi antrenmansız tahmin edilir.
            self.lap.use_fp = False
            self.extra = []
            return
        # Doğrulama: geçmiş sezon(lar) da aynı kodla, yalnızca o sezonun öncesini görerek tahmin edilir
        self.extra = [Engine(season=s, light=True) for s in BACKTEST_EXTRA_SEASONS]
        for e in self.extra:
            self._result_map.update(e._result_map)
        # v1 kalibrasyonu: birleşik log loss'u en düşük sıcaklık
        bt = self.backtest_v1()
        self.temps = {"pre": bt["modes"]["pre_quali"]["best_temperature"],
                      "post": bt["modes"]["post_quali"]["best_temperature"]}
        # v2 kalibrasyonu: tempo ve form ölçeği (diskte saklanır; yeni veri gelince yeniden hesaplanır)
        bt2 = self.backtest_v2()
        if bt2:
            self.v2_params = {"pre": tuple(bt2["modes"]["pre_quali"]["best"]),
                              "post": tuple(bt2["modes"]["post_quali"]["best"])}
            # kazanma olasılığı kalibrasyonu: seçilen ayarın yarış yarış olasılıklarından
            for stage, mode in (("pre", "pre_quali"), ("post", "post_quali")):
                m = bt2["modes"][mode]
                rounds = _best_entry(m)["per_round"]
                self.win_k[stage] = _fit_win_k(rounds)
                self.win_calib[mode] = _walk_forward_win_calib(rounds)

    # ------------------------------------------------------------------ #

    def race_info(self, round_: int) -> dict:
        return next(r for r in self.calendar if int(r["round"]) == round_)

    def next_round(self) -> int | None:
        future = [int(r["round"]) for r in self.calendar if int(r["round"]) not in self.done]
        return min(future) if future else None

    def calendar_view(self) -> list[dict]:
        nxt = self.next_round()
        return [{
            "round": int(r["round"]), "name": r["raceName"], "date": r["date"],
            "circuit": r["Circuit"]["circuitId"], "locality": r["Circuit"]["Location"]["locality"],
            "sprint": "Sprint" in r,
            "status": "done" if int(r["round"]) in self.done else ("next" if int(r["round"]) == nxt else "upcoming"),
        } for r in self.calendar]

    def entries_for(self, round_: int) -> list[Entry]:
        own = next((r for r in self.races if r["round"] == round_), None)
        src = own or max((r for r in self.races if r["round"] < round_), key=lambda r: r["round"])
        return [Entry(x["driver"], x["team"], x["code"], x["name"], x["team_name"]) for x in src["results"]]

    def _quali_order(self, round_: int) -> list[str] | None:
        race = next((r for r in self.races if r["round"] == round_), None)
        if race and race["quali"]:
            return race["quali"]
        # Sıralama yapıldı ama yarış henüz koşulmadıysa: ayrı uç noktadan bak
        try:
            q = data.qualifying(self.season)
        except Exception:
            return None
        hit = next((r for r in q if int(r["round"]) == round_), None)
        if not hit:
            return None
        return [x["Driver"]["driverId"] for x in sorted(hit["QualifyingResults"], key=lambda x: int(x["position"]))]

    def _grid_for(self, round_: int, grid_mode: str) -> list[str] | None:
        # "actual" ve bitmiş yarışta "auto": gerçek start gridi (cezalar dahil)
        if grid_mode == "actual" or (grid_mode == "auto" and round_ in self.done):
            res = next(r for r in self.races if r["round"] == round_)["results"]
            return [x["driver"] for x in sorted(res, key=lambda x: x["grid"] or 99)]
        if grid_mode == "auto":  # yaklaşan yarış: sıralama yapıldıysa sonucu
            return self._quali_order(round_)
        return None  # "simulated": sıralama öncesi görünüm, grid simüle edilir

    def _weather(self, round_: int, rain_prob: float | None):
        if rain_prob is not None:
            return None, rain_prob
        weather = data.rain_forecast(self.race_info(round_)) if round_ not in self.done else None
        return weather, (weather["rain_prob"] if weather else 0.0)

    def _run(self, round_: int, grid_mode: str = "auto", n_sims: int = 20000,
             temperature: float | None = None, rain_prob: float | None = None, seed: int | None = None):
        """v1: sıralama modeli."""
        race_r, quali_r = fit_season(self.races, round_, self.ts)
        rel = reliability(self.races, round_)
        info = self.race_info(round_)
        entries = self.entries_for(round_)
        known_grid = self._grid_for(round_, grid_mode)
        if temperature is None:
            temperature = self.temps["post" if known_grid else "pre"]
        weather, rain_prob = self._weather(round_, rain_prob)
        sim = simulate(entries, race_r, quali_r, rel, self.ts, info["Circuit"]["circuitId"],
                       n_sims=n_sims, known_grid=known_grid, rain_prob=rain_prob,
                       temperature=temperature, seed=round_ if seed is None else seed)
        return sim, race_r, quali_r, rel, weather, known_grid

    def _run_v2(self, round_: int, grid_mode: str = "auto", n_sims: int = 20000,
                params: tuple[float, float] | None = None, rain_prob: float | None = None, seed: int | None = None,
                calibrate: bool = True):
        """v2: tur tur simülatör. RaceSim, meta, hava, grid ve v1 biçimli özet nesnesi döner.

        grid_mode "friday": Cuma akşamı tahmini — grid simüle edilir, yalnızca FP1 + FP2 kullanılır.
        calibrate: kazanma olasılığı kalibrasyonu (backtest kendi koşularında kapatır)."""
        info = self.race_info(round_)
        entries = self.entries_for(round_)
        known_grid = self._grid_for(round_, grid_mode)
        stage = "post" if known_grid else ("friday" if grid_mode == "friday" else "pre")
        if params is None:
            params = self.v2_params.get(stage, self.v2_params["pre"])
        weather, rain_prob = self._weather(round_, rain_prob)
        rs, meta = self.lap.run(round_, info["Circuit"]["circuitId"], entries, known_grid, rain_prob,
                                n=n_sims, seed=round_ if seed is None else seed,
                                pace_scale=params[0], form_scale=params[1],
                                fp_sessions=FRIDAY if stage == "friday" else SESSIONS)
        k = self.win_k.get("post" if known_grid else "pre", 1.0) if calibrate else 1.0
        rs = rs.calibrated(k, seed=round_)
        meta["win_k"] = k
        v1like = SimResult(entries, rs.positions, rs.grid, rs.dnf, rs.sc_count > 0, rs.rain)
        return rs, meta, weather, known_grid, v1like

    # ------------------------------------------------------------------ #

    def predict(self, round_: int, grid_mode: str = "auto") -> dict:
        key = (round_, grid_mode)
        with self.lock:
            if key in self._pred_cache:
                return self._pred_cache[key]
            info = self.race_info(round_)
            circuit = info["Circuit"]["circuitId"]
            base_meta = {
                "round": round_, "name": info["raceName"], "date": info["date"],
                "circuit": circuit, "locality": info["Circuit"]["Location"]["locality"],
                "trained_on": len([r for r in self.races if r["round"] < round_]),
            }
            if self.lap.ready_for(round_):
                rs, meta, weather, known_grid, sim = self._run_v2(round_, grid_mode)
                out = summarize(sim)
                out.update(_v2_extras(rs, sim.entries, meta))
                stage = "post" if known_grid else ("friday" if grid_mode == "friday" else "pre")
                ps, fs = self.v2_params.get(stage, self.v2_params["pre"])
                out["meta"] = {**base_meta, "model": "v2", "grid_known": bool(known_grid), "weather": weather,
                               "stage": stage, "pace_scale": ps, "form_scale": fs, "v2": meta}
            else:
                sim, race_r, _, rel, weather, known_grid = self._run(round_, grid_mode)
                out = summarize(sim)
                out["meta"] = {**base_meta, "model": "v1", "grid_known": bool(known_grid), "weather": weather,
                               "temperature": self.temps["post" if known_grid else "pre"]}
            rel = reliability(self.races, round_)
            out["meta"]["track_v1"] = {"grid_factor": self.ts.grid(circuit), "dnf_factor": self.ts.dnf(circuit)}
            out["ratings"] = sorted(({"team": t, "dnf_rate": r} for t, r in rel.team_rate.items()
                                     if t in {e.team for e in sim.entries}), key=lambda r: r["dnf_rate"])
            if round_ in self.done:
                res = next(r for r in self.races if r["round"] == round_)["results"]
                out["actual"] = [{"code": x["code"], "pos": x["pos"], "kind": x["kind"]} for x in res]
            self._pred_cache[key] = out
            self._sims[key] = sim
            return out

    def analyze_odds(self, round_: int, market: str, odds_in: dict[str, float]) -> dict:
        self.predict(round_)
        sim = self._sims[(round_, "auto")]
        labels = {e.driver: f"{e.code} · {e.name}" for e in sim.entries}
        return odds.analyze(market, odds_in, market_probs(sim, market), labels)

    # ------------------------------------------------------------------ #
    # Backtest: her yarışı yalnızca ÖNCEKİ yarışlarla tahmin et (sızıntı yok)
    # ------------------------------------------------------------------ #

    def _score(self, runs: list[tuple[int, int, list[str], np.ndarray]]) -> dict:
        """runs: [(sezon, yarış, sürücüler, pozisyon matrisi)] — birden çok sezon birlikte puanlanabilir."""
        stats = {"logloss": [], "ll_place": [], "brier_win": [], "brier_podium": [], "brier_top10": [],
                 "fav_won": [], "winner_in_top3": [], "calib": [], "per_round": [],
                 "podium_hit": [], "top10_hit": [], "h2h": []}
        n = 20
        for season, rd, drivers, pos in runs:
            res = self._result_map[(season, rd)]
            n = len(drivers)
            p_win, p_pod, p_t10 = (np.mean(pos == 1, axis=0), np.mean(pos <= 3, axis=0), np.mean(pos <= 10, axis=0))
            actual = np.array([res[d]["pos"] if d in res else 99 for d in drivers])
            winner = int(np.argmin(actual))
            ll = -math.log(max(p_win[winner], 1e-4))
            stats["logloss"].append(ll)
            stats["per_round"].append({"season": season, "round": rd, "logloss": ll, "p_winner": float(p_win[winner]),
                                       "winner": drivers[winner], "favourite": drivers[int(np.argmax(p_win))],
                                       "p_win": [round(float(x), 5) for x in p_win], "winner_idx": winner})
            stats["ll_place"].append(_binary_ll(p_pod, actual <= 3) + _binary_ll(p_t10, actual <= 10))
            stats["brier_win"].append(float(np.mean((p_win - (actual == 1)) ** 2)))
            stats["brier_podium"].append(float(np.mean((p_pod - (actual <= 3)) ** 2)))
            stats["brier_top10"].append(float(np.mean((p_t10 - (actual <= 10)) ** 2)))
            stats["fav_won"].append(int(np.argmax(p_win) == winner))
            stats["winner_in_top3"].append(int(winner in np.argsort(-p_win)[:3]))
            # Sezgisel isabet oranları: tahmin edilen ilk 3 / ilk 10'un kaçı gerçekten oraya girdi
            stats["podium_hit"].append(len(set(np.argsort(-p_pod)[:3]) & set(np.nonzero(actual <= 3)[0])) / 3)
            stats["top10_hit"].append(len(set(np.argsort(-p_t10)[:10]) & set(np.nonzero(actual <= 10)[0])) / 10)
            # Takım arkadaşı düellosu: önde bitirmesi beklenen gerçekten önde bitirdi mi
            team = {d: res[d]["team"] for d in drivers if d in res}
            idx = {d: j for j, d in enumerate(drivers)}
            for t in set(team.values()):
                pair = [d for d in drivers if team.get(d) == t]
                if len(pair) == 2:
                    a, b = (idx[d] for d in pair)
                    pa = float(np.mean(pos[:, a] < pos[:, b]))
                    if pa != 0.5 and actual[a] != actual[b]:
                        stats["h2h"].append(int((pa > 0.5) == (actual[a] < actual[b])))
            stats["calib"].extend(zip(p_pod.tolist(), (actual <= 3).astype(int).tolist()))
            stats["calib"].extend(zip(p_t10.tolist(), (actual <= 10).astype(int).tolist()))
        return {
            "logloss": float(np.mean(stats["logloss"])),
            "logloss_uniform": math.log(n),
            # seçim ölçütü: kazanan + podyum + ilk 10 log loss (hepsi nat cinsinden)
            "combined": float(np.mean(stats["logloss"]) + np.mean(stats["ll_place"])),
            "brier_win": float(np.mean(stats["brier_win"])),
            "brier_podium": float(np.mean(stats["brier_podium"])),
            "brier_top10": float(np.mean(stats["brier_top10"])),
            "brier_podium_base": float((3 / n) * (1 - 3 / n)),
            "brier_top10_base": float((10 / n) * (1 - 10 / n)),
            "fav_won": float(np.mean(stats["fav_won"])),
            "winner_in_top3": float(np.mean(stats["winner_in_top3"])),
            "podium_hit": float(np.mean(stats["podium_hit"])),
            "top10_hit": float(np.mean(stats["top10_hit"])),
            "h2h_acc": float(np.mean(stats["h2h"])) if stats["h2h"] else None,
            "h2h_n": len(stats["h2h"]),
            "n_races": len(stats["logloss"]),
            "calibration": _calibration(stats["calib"]),
            "per_round": stats["per_round"],
        }

    def _bt_rounds(self, start_round: int = 4) -> list[int]:
        return sorted(r for r in self.done if r >= start_round and self.lap.ready_for(r))

    def _engines(self) -> list["Engine"]:
        """Backtest'te tahmin edilen sezonlar: bu sezon + doğrulama sezonları."""
        return [self] + self.extra

    def _bt_all_rounds(self) -> list[tuple["Engine", int]]:
        return [(e, rd) for e in self._engines() for rd in e._bt_rounds()]

    def backtest_v1(self, temps=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2), n_sims: int = 6000) -> dict:
        todo = self._bt_all_rounds()
        modes = {}
        for mode, grid_mode in (("pre_quali", "simulated"), ("post_quali", "actual")):
            per = {}
            for T in temps:
                runs = []
                for e, rd in todo:
                    sim, *_ = e._run(rd, grid_mode, n_sims=n_sims, temperature=T, rain_prob=0.0)
                    runs.append((e.season, rd, [x.driver for x in sim.entries], sim.positions))
                per[T] = self._score(runs)
            best = min(per, key=lambda t: per[t]["combined"])
            modes[mode] = {"by_temperature": {str(k): v for k, v in per.items()}, "best_temperature": best}
        self._bt_v1 = {"rounds": [[e.season, rd] for e, rd in todo], "modes": modes}
        return self._bt_v1

    def _v2_signature(self) -> str:
        sig = {"grid": V2_GRID, "v": V2_VERSION}
        for e in self._engines():
            fp_cov = {r: sorted(s) for r, s in sorted(e.lap.fp.items())} if e.lap.use_fp else {}
            sig[str(e.season)] = {"cov": e.lap.coverage(), "done": sorted(e.done), "fp": fp_cov}
        return json.dumps(sig)

    def backtest_v2(self, n_sims: int = 3000, force: bool = False) -> dict | None:
        todo = self._bt_all_rounds()
        if not todo:
            return None
        sig = self._v2_signature()
        if V2_CACHE.exists() and not force:
            cached = json.loads(V2_CACHE.read_text(encoding="utf-8"))
            if cached.get("signature") == sig:
                self._bt_v2 = cached
                return cached
        print(f"v2 backtest: {len(todo)} yarış ({', '.join(str(e.season) for e in self._engines())}) "
              f"× 2 mod × {len(V2_GRID)} ayar ...", flush=True)
        modes = {}
        for mode, grid_mode in (("pre_quali", "simulated"), ("post_quali", "actual")):
            per, runs_by = {}, {}
            for params in V2_GRID:
                runs = []
                for e, rd in todo:
                    rs, _, _, _, sim = e._run_v2(rd, grid_mode, n_sims=n_sims, params=params, rain_prob=0.0,
                                                 calibrate=False)
                    runs.append((e.season, rd, [x.driver for x in sim.entries], rs.positions))
                per[params] = self._score(runs)
                runs_by[params] = runs
                print(f"  {mode} tempo×{params[0]} form×{params[1]}: log loss {per[params]['logloss']:.3f}, "
                      f"birleşik {per[params]['combined']:.2f}", flush=True)
            best = min(per, key=lambda k: per[k]["combined"])
            by_season = {}
            for e in self._engines():
                s = self._score([r for r in runs_by[best] if r[0] == e.season])
                by_season[str(e.season)] = {k: v for k, v in s.items() if k not in ("per_round", "calibration")}
            modes[mode] = {"by_params": {f"{k[0]}|{k[1]}": v for k, v in per.items()}, "best": list(best),
                           "by_season": by_season}
        out = {"signature": sig, "rounds": [[e.season, rd] for e, rd in todo], "modes": modes}
        V2_CACHE.write_text(json.dumps(out), encoding="utf-8")
        self._bt_v2 = out
        return out

    def backtest(self) -> dict:
        return {"v1": self._bt_v1, "v2": getattr(self, "_bt_v2", None),
                "win_calibration": {"k": self.win_k, **self.win_calib}}

    # ------------------------------------------------------------------ #
    # Şampiyonluk: kalan yarışları simüle et, mevcut puanlara ekle
    # ------------------------------------------------------------------ #

    def championship(self, n_sims: int = 4000) -> dict:
        key = ("champ",)
        with self.lock:
            if key in self._pred_cache:
                return self._pred_cache[key]
        nxt = self.next_round()
        current = {s["Driver"]["driverId"]: float(s["points"]) for s in self.standings}
        names = {s["Driver"]["driverId"]: s["Driver"].get("code", s["Driver"]["familyName"][:3].upper()) for s in self.standings}
        remaining = [int(r["round"]) for r in self.calendar if int(r["round"]) not in self.done]
        if not remaining:
            return {"remaining": [], "rows": []}
        ids = [e.driver for e in self.entries_for(nxt)]
        totals = np.array([current.get(d, 0.0) for d in ids])[None, :].repeat(n_sims, axis=0)

        def race(rd, seed):
            if self.lap.ready_for(rd):
                rs, _, _, _, sim = self._run_v2(rd, "simulated", n_sims=n_sims, rain_prob=0.0, seed=seed)
            else:
                sim, *_ = self._run(rd, "simulated", n_sims=n_sims, rain_prob=0.0, seed=seed)
            idx = [[e.driver for e in sim.entries].index(d) for d in ids]
            return sim.positions[:, idx]

        for rd in remaining:
            totals += POINTS[race(rd, rd) - 1]
            if "Sprint" in self.race_info(rd):
                # sprint: aynı güçlerle ikinci, bağımsız bir kısa yarış
                totals += SPRINT_POINTS[race(rd, rd + 1000) - 1]
        champ = np.argmax(totals + np.random.default_rng(0).random(totals.shape) * 1e-3, axis=1)
        rows = sorted(({
            "driver": d, "code": names.get(d, d[:3].upper()),
            "points_now": current.get(d, 0.0),
            "exp_points": float(np.mean(totals[:, j])),
            "p10": float(np.percentile(totals[:, j], 10)),
            "p90": float(np.percentile(totals[:, j], 90)),
            "champion": float(np.mean(champ == j)),
        } for j, d in enumerate(ids)), key=lambda r: (-r["champion"], -r["exp_points"]))
        out = {"remaining": remaining, "rows": rows, "n_sims": n_sims}
        with self.lock:
            self._pred_cache[key] = out
        return out


def _v2_extras(rs, entries, meta) -> dict:
    """Tur tur simülasyonun ek piyasaları."""
    n, D = rs.positions.shape
    pace = {p["driver"]: p for p in meta["pace"]}
    extra_rows = {}
    for j, e in enumerate(entries):
        extra_rows[e.driver] = {
            "fastest_lap": float(np.mean(rs.fastest_lap == j)),
            "lap1_lead": float(np.mean(rs.lap1_leader == j)),
            "exp_grid": float(np.mean(rs.grid[:, j])),
            "exp_stops": float(np.mean(rs.stops[:, j])),
            "race_sec": pace[e.driver]["race_sec"],
            "race_pct": pace[e.driver]["race_pct"],
            "race_sd": pace[e.driver]["race_sd"],
            "quali_pct": pace[e.driver]["quali_pct"],
        }
    winner_team: dict[str, float] = {}
    win_idx = np.argmin(rs.positions, axis=1)
    for j, e in enumerate(entries):
        winner_team[e.team_name] = winner_team.get(e.team_name, 0.0) + float(np.mean(win_idx == j))
    sc = rs.sc_count
    m = rs.margin[np.isfinite(rs.margin)]
    finishers = (~rs.dnf).sum(axis=1)
    return {
        "extra": extra_rows,
        "winner_team": sorted(({"team": k, "p": v} for k, v in winner_team.items()), key=lambda r: -r["p"]),
        "events": {
            "sc_dist": [float(np.mean(sc == 0)), float(np.mean(sc == 1)), float(np.mean(sc == 2)), float(np.mean(sc >= 3))],
            "vsc_any": float(np.mean(rs.vsc_count > 0)),
            "neutral_any": float(np.mean((sc + rs.vsc_count) > 0)),
            "margin_q": [float(np.percentile(m, q)) for q in (10, 25, 50, 75, 90)] if m.size else None,
            "margin_lt5": float(np.mean(m < 5)) if m.size else None,
            "finishers_mean": float(np.mean(finishers)),
            "finishers_dist": {str(k): float(np.mean(finishers == k)) for k in range(max(int(finishers.min()), D - 6), D + 1)},
        },
    }


WIN_K_GRID = np.arange(1.0, 4.01, 0.25)


def _best_entry(mode: dict) -> dict:
    return next(v for k, v in mode["by_params"].items()
                if [float(x) for x in k.split("|")] == [float(x) for x in mode["best"]])


def _win_ll(rounds: list[dict], k: float) -> float:
    """Kazanma olasılıkları p^k ile yeniden ölçeklenince gerçek kazananın ortalama log loss'u."""
    out = []
    for r in rounds:
        p = np.clip(np.array(r["p_win"]), 1e-4, None) ** k
        out.append(-np.log(p[r["winner_idx"]] / p.sum()))
    return float(np.mean(out))


def _fit_win_k(rounds: list[dict], min_rounds: int = 5) -> float:
    """Kazanma kalibrasyon katsayısı k (1 = değişiklik yok, >1 = favoriler güçlenir).
    Az veride 1'e yakın kalması için hafif bir öncül: (k-1)² cezası."""
    if len(rounds) < min_rounds:
        return 1.0
    return float(min(WIN_K_GRID, key=lambda k: _win_ll(rounds, k) + 0.02 * (k - 1) ** 2))


def _walk_forward_win_calib(rounds: list[dict]) -> dict:
    """İleriye dönük değerlendirme: her yarış için k yalnızca önceki yarışlardan."""
    raw, cal, fav_p_raw, fav_p_cal = [], [], [], []
    for i, r in enumerate(rounds):
        k = _fit_win_k(rounds[:i])
        p = np.clip(np.array(r["p_win"]), 1e-4, None)
        pk = p ** k / np.sum(p ** k)
        raw.append(-np.log(p[r["winner_idx"]] / p.sum()))
        cal.append(-np.log(pk[r["winner_idx"]]))
        fav_p_raw.append(float(p.max() / p.sum()))
        fav_p_cal.append(float(pk.max()))
    fav_hit = float(np.mean([r["favourite"] == r["winner"] for r in rounds]))
    return {"logloss_raw": float(np.mean(raw)), "logloss_cal": float(np.mean(cal)),
            "fav_prob_raw": float(np.mean(fav_p_raw)), "fav_prob_cal": float(np.mean(fav_p_cal)),
            "fav_hit": fav_hit, "k_final": _fit_win_k(rounds)}


def _binary_ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(-np.sum(np.where(y, np.log(p), np.log(1 - p))))


def _calibration(pairs: list[tuple[float, int]], bins: int = 10) -> list[dict]:
    p = np.array([a for a, _ in pairs])
    y = np.array([b for _, b in pairs])
    out = []
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if m.sum() >= 5:
            out.append({"bin": f"{int(lo*100)}–{int(hi*100)}%", "pred": float(p[m].mean()),
                        "actual": float(y[m].mean()), "n": int(m.sum())})
    return out
