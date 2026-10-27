"""Tur tur Monte Carlo yarış hafta sonu simülatörü.

Her simülasyon (hepsi vektörize, aynı anda):

 0. Belirsizlik: araç/sürücü temposu Gauss sürecinin ortak (korelasyonlu) dağılımından çekilir;
    üstüne hafta sonu formu eklenir (sıralama ve yarış formu arasında korelasyon ρ).
 1. Sıralama: Q1 (22→16), Q2 (16→10), Q3; her seansta ayrı tek tur gürültüsü.
 2. Strateji: durak sayısı pistin geçmiş dağılımından, pit turları stint eşit bölünmesi + gürültü.
 3. Olaylar: güvenlik aracı / VSC sayısı Poisson; bir kısmı DNF'lerin tetiklediği olaylar (korelasyon).
 4. Start: grid aralığı + start gürültüsü (pistin geçmiş ilk tur pozisyon değişimine kalibre).
 5. Her tur:
      tur = ref + tempo + yakıt·tur + aşınma·lastik_yaşı + σ·t(5) gürültü
      pit: plan geldiyse veya nötr turda pit penceresi açıksa (ucuz pit)
      yeşil turda sıra sıra takip: öndeki araca ATTACK sn'den yakınsa sollama denemesi,
          P(geçer) = σ(a_pist + b1·tempo_farkı + b2·mesafe)   (veriden öğrenilmiş lojistik)
          geçemezse önündekinin MIN_GAP arkasında kalır; 1 sn içinde kirli hava kaybı
      SC turu: alan sıkışır (0.8 sn aralıklı tren), sollama yok
      VSC turu: herkes yavaş, aralar korunur
      DNF: önceden çekilen tur numarasında araç çekilir
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import TrackParams

MIN_GAP = 0.3
ATTACK = 1.5
DIRTY = 1.0
SC_SPACING = 0.8
SC_LAP = 1.45
VSC_LAP = 1.35
GRID_SPACING = 0.25
DNF_SC = 0.30         # bir DNF'nin güvenlik aracı getirme olasılığı (varsayım)
DNF_VSC = 0.15        # ... VSC getirme olasılığı (varsayım)
STRATEGY_DEVIATE = 0.25  # sürücünün baskın stratejiden sapma olasılığı (varsayım)
PIT_JITTER = 0.035       # bireysel pit turu sapması (yarış uzunluğunun oranı)
STRATEGY_NOISE = 2.0     # strateji uygulama farkı (sn, yarış toplamı)
SC_PIT_INDIVIDUAL = 0.15 # nötr turda pit kararında alan genelinden sapma olasılığı
COVER_GAP = 3.0          # arkadaki bu mesafede pite girerse öndeki bir sonraki tur karşılık verir (undercut savunması)
COVER_WINDOW = 8         # planlanan pite en fazla bu kadar tur varsa
COVER_PROB = 0.7
SC_PIT_WINDOW = 12   # nötr turda, planlanan pite bu kadar tur kaldıysa erken pit
SC_PIT_TAKE = 0.85
RAIN_NOISE = 2.5      # yağmurda tur gürültüsü çarpanı (varsayım)
RAIN_HAZARD = 2.5     # yağmurda DNF ve olay çarpanı (varsayım)


@dataclass
class Inputs:
    drivers: list[str]
    teams: list[str]
    race_mean: np.ndarray     # % (negatif = hızlı)
    race_cov: np.ndarray
    race_tau: float
    quali_mean: np.ndarray
    quali_cov: np.ndarray
    quali_tau: float
    quali_session_sigma: float
    rho: float
    p_dnf: np.ndarray         # yarış başına DNF olasılığı
    team_pit: np.ndarray      # sn
    p_slow_pit: float
    slow_pit_mean: float
    track: TrackParams
    b_pace: float
    b_gap: float
    known_grid: np.ndarray | None = None   # sürücü başına grid (1 = pole)
    rain_prob: float = 0.0
    pace_scale: float = 1.0
    form_scale: float = 1.0   # hafta sonu formu çarpanı: simülatörün kendi ürettiği strateji/trafik
                              # rastgeleliği GP'nin form varyansında zaten var; çift saymayı önler


@dataclass
class RaceSim:
    positions: np.ndarray     # (N, D) 1 = kazanan
    grid: np.ndarray
    dnf: np.ndarray
    laps_done: np.ndarray
    fastest_lap: np.ndarray   # (N,) sürücü indeksi
    margin: np.ndarray        # (N,) kazanma farkı (sn)
    sc_count: np.ndarray
    vsc_count: np.ndarray
    rain: np.ndarray
    lap1_leader: np.ndarray
    stops: np.ndarray
    race_pace: np.ndarray     # (N, D) sn/tur, örneklenen tempo
    leader_by_lap: np.ndarray | None = None  # (N, L) her tur sonundaki lider (teşhis için)
    status: np.ndarray | None = None         # (N, L) 0 yeşil, 1 SC, 2 VSC
    pit_by_lap: np.ndarray | None = None     # (N, D, L) teşhis modu


def _mvn(rng, mean, cov, n):
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(mean)))
    return mean[None, :] + rng.standard_normal((n, len(mean))) @ L.T


def _knockout(rng, base: np.ndarray, sigma: float) -> np.ndarray:
    n, d = base.shape
    out_n = max((d - 10) // 2, 0)
    grid = np.zeros((n, d), int)
    alive = np.ones((n, d), bool)
    rows = np.arange(n)[:, None]
    slots = [d - out_n, 10, 0]  # Q1'den sonra kalan, Q2'den sonra kalan
    for stage in range(3):
        t = np.where(alive, base + sigma * rng.standard_normal((n, d)), np.inf)
        order = np.argsort(t, axis=1)
        keep = slots[stage]
        n_alive = int(alive[0].sum())
        if stage < 2:
            # elenenler: keep+1 .. n_alive
            out = order[:, keep:n_alive]
            grid[rows, out] = np.arange(keep + 1, n_alive + 1)[None, :]
            alive[rows, out] = False
        else:
            grid[rows, order[:, :n_alive]] = np.arange(1, n_alive + 1)[None, :]
    return grid


def simulate(inp: Inputs, n: int = 20000, seed: int = 0, debug: bool = False) -> RaceSim:
    rng = np.random.default_rng(seed)
    tp = inp.track
    D, L = len(inp.drivers), tp.laps
    rows = np.arange(n)[:, None]

    # 0) Tempo örnekleri (parametre belirsizliği + hafta sonu formu)
    z1, z2 = rng.standard_normal((n, D)), rng.standard_normal((n, D))
    f_q = _mvn(rng, inp.quali_mean, inp.quali_cov, n) + inp.quali_tau * z1
    f_r = _mvn(rng, inp.race_mean, inp.race_cov, n) + inp.form_scale * inp.race_tau * (inp.rho * z1 + np.sqrt(1 - inp.rho ** 2) * z2)
    pace = inp.pace_scale * f_r * tp.ref / 100.0  # sn/tur

    # 1) Grid
    if inp.known_grid is not None:
        grid = np.broadcast_to(inp.known_grid, (n, D)).copy()
    else:
        grid = _knockout(rng, f_q, inp.quali_session_sigma)

    # Yağmur
    rain = rng.random(n) < inp.rain_prob
    sigma = tp.sigma * np.where(rain, RAIN_NOISE, 1.0)[:, None]
    haz = np.where(rain, RAIN_HAZARD, 1.0)

    # 2) Strateji: yarışın baskın stratejisi ortak (herkes aynı lastik verisini görür),
    #    sürücülerin bir kısmı alternatif stratejiye sapar
    probs = np.array(tp.stop_probs)
    draw = lambda shape: 1 + (rng.random(shape)[..., None] > np.cumsum(probs)[None, :-1]).sum(-1)
    base_stops = draw((n,))
    n_stops = np.where(rng.random((n, D)) < STRATEGY_DEVIATE, draw((n, D)), base_stops[:, None])
    # Strateji dengesi: takımlar durak sayısını toplam süre ~eşit olacak şekilde seçer. Doğrusal aşınma
    # modeli bunu bilemez (lastik hamuru verisi yok), bu yüzden n duraklının aşınma hızı, toplam
    # "aşınma + pit" maliyeti en iyi stratejiye eşit olacak şekilde ayarlanır; ±STRATEGY_NOISE sn uygulama farkı.
    def deg_cost(nn, dl):
        return dl * L * (L / (nn + 1) + 1) / 2
    best_cost = min(deg_cost(k, tp.delta) + k * tp.pit_green for k in (1, 2, 3))
    delta_n = {k: max((best_cost - k * tp.pit_green) / deg_cost(k, 1.0), 0.002) for k in (1, 2, 3)}
    delta = np.vectorize(delta_n.get)(n_stops).astype(float)
    delta = delta + rng.normal(0, STRATEGY_NOISE, (n, D)) / deg_cost(n_stops, 1.0)
    delta = np.maximum(delta, 0.0)
    k = np.arange(1, 4)[None, None, :]
    # pit penceresi tüm alan için ortak kayar (lastik verisi herkes için aynı), bireysel sapma küçük
    shift = rng.normal(0, 0.05 * L, (n, 1, 1))
    plan = L * k / (n_stops[..., None] + 1) + shift + rng.normal(0, PIT_JITTER * L, (n, D, 3))
    plan = np.where(k <= n_stops[..., None], np.clip(np.round(plan), 2, L - 2), np.inf)
    plan = np.sort(plan, axis=-1)

    # 3) DNF ve olaylar
    p = np.clip(inp.p_dnf[None, :] * haz[:, None], 0, 0.95)
    u = rng.random((n, D))
    dnf_lap = np.full((n, D), np.inf)
    lap1 = u < p * tp.dnf_lap1_share
    later = (~lap1) & (u < p)
    dnf_lap[lap1] = 1
    dnf_lap[later] = rng.integers(2, L + 1, later.sum())

    exp_dnf = float(inp.p_dnf.sum())
    status = np.zeros((n, L), np.int8)  # 0 yeşil, 1 SC, 2 VSC

    def place(mask_n, starts, lens, code):
        idx = np.nonzero(mask_n)[0]
        for i, s, ln in zip(idx, starts[mask_n], lens[mask_n]):
            s = int(s) - 1
            seg = status[i, s:s + int(ln)]
            if code == 1:
                status[i, s:s + int(ln)] = 1
            else:
                status[i, s:s + int(ln)] = np.where(seg == 1, 1, 2)

    lam_sc = max(tp.sc_rate - DNF_SC * exp_dnf, 0.15 * tp.sc_rate)
    lam_vsc = max(tp.vsc_rate - DNF_VSC * exp_dnf, 0.15 * tp.vsc_rate)
    sc_len, vsc_len = np.array(tp.sc_len), np.array(tp.vsc_len)
    for lam, lens, code in ((lam_vsc, vsc_len, 2), (lam_sc, sc_len, 1)):
        cnt = rng.poisson(lam * haz)
        for j in range(int(cnt.max()) if cnt.size else 0):
            m = cnt > j
            first = rng.random(n) < (tp.sc_lap1_share if code == 1 else 0.05)
            starts = np.where(first, 1, rng.integers(2, max(L - 3, 3), n))
            place(m, starts, rng.choice(lens, n), code)
    # DNF kaynaklı olaylar
    finite = np.isfinite(dnf_lap)
    trig = rng.random((n, D))
    for code, thr_lo, thr_hi, lens in ((1, 0, DNF_SC, sc_len), (2, DNF_SC, DNF_SC + DNF_VSC, vsc_len)):
        hit = finite & (trig >= thr_lo) & (trig < thr_hi) & (dnf_lap < L - 1)
        si, di = np.nonzero(hit)
        for i, d in zip(si, di):
            s = int(dnf_lap[i, d]) - 1
            ln = int(rng.choice(lens))
            if code == 1:
                status[i, s:s + ln] = 1
            else:
                status[i, s:s + ln] = np.where(status[i, s:s + ln] == 1, 1, 2)
    starts_of = lambda code: ((status == code) & np.concatenate([np.ones((n, 1), bool), status[:, :-1] != code], 1)).sum(1)
    sc_count, vsc_count = starts_of(1), starts_of(2)

    # 4) Start
    prof = np.array(tp.start_profile)
    start_sd = tp.start_scale * prof[np.clip(grid - 1, 0, len(prof) - 1)]
    cum = (grid - 1) * GRID_SPACING + tp.ref + pace + start_sd * rng.standard_normal((n, D))
    alive = ~(dnf_lap <= 1)
    laps_done = np.where(alive, 1, 0)
    cum = np.where(alive, cum, np.inf)
    lap1_leader = np.argmin(cum, axis=1)
    age = np.ones((n, D))
    stops_done = np.zeros((n, D), int)
    best = np.full((n, D), np.inf)
    ot_a, b1, b2 = tp.overtake_logit, inp.b_pace, inp.b_gap
    leader_by_lap = np.zeros((n, L), np.int16)
    leader_by_lap[:, 0] = lap1_leader
    pit_by_lap = np.zeros((n, D, L), bool) if debug else None
    pit_prev = np.zeros((n, D), bool)

    # 5) Turlar
    for i in range(2, L + 1):
        st = status[:, i - 1]
        green, sc, vsc = st == 0, st == 1, st == 2
        age += 1
        noise = rng.standard_t(5, (n, D)) * np.sqrt(3 / 5)
        t = tp.ref + pace + tp.beta * (i - L / 2) + delta * age + sigma * noise

        nxt = np.take_along_axis(plan, np.minimum(stops_done, 2)[..., None], -1)[..., 0]
        nxt = np.where(stops_done < n_stops, nxt, np.inf)
        due = alive & (nxt <= i)
        # nötr turda pit kararı alan genelinde ortak (herkes aynı fırsatı görür), az sayıda sürücü ters karar verir
        common = (rng.random(n) < SC_PIT_TAKE)[:, None]
        take = np.where(rng.random((n, D)) < SC_PIT_INDIVIDUAL, ~common, common)
        opp = alive & ~green[:, None] & (nxt - i <= SC_PIT_WINDOW) & take
        # undercut savunması: hemen arkadaki araç geçen tur yeşil bayrakta pite girdiyse karşılık ver
        o_prev = np.argsort(cum, axis=1)
        c_sorted = np.take_along_axis(cum, o_prev, 1)
        behind_pit = np.zeros((n, D), bool)
        with np.errstate(invalid="ignore"):  # çekilmiş araçlar: inf - inf
            close = (c_sorted[:, 1:] - c_sorted[:, :-1]) < COVER_GAP + tp.pit_green
        bp = np.take_along_axis(pit_prev, o_prev[:, 1:], 1) & close
        behind_pit[rows, o_prev[:, :-1]] = bp
        cover = alive & green[:, None] & behind_pit & (nxt - i <= COVER_WINDOW) & (rng.random((n, D)) < COVER_PROB)
        pit = due | opp | cover
        slow = rng.random((n, D)) < inp.p_slow_pit
        cost = tp.pit_green + inp.team_pit[None, :] + rng.normal(0, 0.5, (n, D)) + slow * rng.exponential(inp.slow_pit_mean, (n, D))
        cost = np.where(green[:, None], cost, cost - tp.pit_green * (1 - tp.pit_neutral_ratio))
        pit_cost = np.where(pit, cost, 0.0)
        age = np.where(pit, 1, age)
        stops_done = stops_done + pit
        pit_prev = pit & green[:, None]

        # --- yeşil tur: sıra sıra takip/sollama
        order = np.argsort(cum, axis=1)
        cp = np.take_along_axis(cum, order, 1)
        tt = np.take_along_axis(t + pit_cost, order, 1)
        ep = np.take_along_axis(pace + delta * age, order, 1)
        act = np.take_along_axis(alive & ~pit, order, 1)
        new = cp + tt
        m1 = np.full(n, -np.inf); m2 = np.full(n, -np.inf)
        m1_prev = np.full(n, -np.inf); m1_pace = np.zeros(n); m1_k = np.zeros(n, int)
        has = np.zeros(n, bool)
        for kk in range(D):
            a = act[:, kk]
            tent = new[:, kk]
            gap = cp[:, kk] - m1_prev
            tent = tent + np.where(has & (gap < DIRTY), tp.theta, 0.0)
            attack = has & ((gap < ATTACK) | (tent < m1 + MIN_GAP))
            z = ot_a + b1 * (ep[:, kk] - m1_pace) + b2 * np.minimum(gap, ATTACK)
            passed = a & attack & (rng.random(n) < 1 / (1 + np.exp(-z)))
            blocked = a & has & ~passed
            # geçemeyen araç kirli havada tipik takip mesafesine düşer (veriden), yapışık kalmaz
            follow = np.maximum(tp.follow_gap + 0.2 * rng.standard_normal(n), MIN_GAP)
            val = np.where(blocked & (tent < m1 + follow), m1 + follow, tent)
            val = np.where(passed, np.maximum(np.minimum(tent, m1 - MIN_GAP), m2 + 0.05), val)
            new[:, kk] = np.where(a, val, new[:, kk])
            # geçilen aracı yeni konumunun arkasına it
            pk = m1_k
            pushed = np.maximum(m1, val + MIN_GAP)
            new[np.arange(n)[passed], pk[passed]] = pushed[passed]
            upd = a & ~passed
            m2 = np.where(upd, m1, np.where(passed, val, m2))
            m1 = np.where(upd, val, np.where(passed, pushed, m1))
            m1_prev = np.where(upd, cp[:, kk], m1_prev)
            m1_pace = np.where(upd, ep[:, kk], m1_pace)
            m1_k = np.where(upd, kk, m1_k)
            has = has | a
        green_new = np.empty_like(cum)
        green_new[rows, order] = new

        # --- SC turu: pit yapanlar geriye düşer, alan sıkışır
        key = cum + pit_cost
        o2 = np.argsort(key, axis=1)
        rank = np.empty_like(o2); rank[rows, o2] = np.arange(D)[None, :]
        lead = np.min(np.where(alive, cum, np.inf), axis=1, keepdims=True)
        sc_new = lead + tp.ref * SC_LAP + rank * SC_SPACING
        # --- VSC turu: aralar korunur
        vsc_new = cum + tp.ref * VSC_LAP + pit_cost

        new_cum = np.where(green[:, None], green_new, np.where(sc[:, None], sc_new, vsc_new))

        with np.errstate(invalid="ignore"):
            lap_t = new_cum - cum
        clean = green[:, None] & alive & ~pit & np.isfinite(lap_t)
        best = np.where(clean, np.minimum(best, lap_t), best)

        died = alive & (dnf_lap == i)
        alive = alive & ~died
        cum = np.where(alive, new_cum, np.inf)
        laps_done = laps_done + alive
        leader_by_lap[:, i - 1] = np.argmin(cum, axis=1)
        if debug:
            pit_by_lap[:, :, i - 1] = pit

    # Sınıflandırma: tamamlanan tur, sonra toplam süre
    key = np.where(alive, cum, 1e9 - laps_done * 1e4 + rng.random((n, D)))
    order = np.argsort(key, axis=1)
    positions = np.empty_like(order); positions[rows, order] = np.arange(1, D + 1)[None, :]
    srt = np.sort(np.where(alive, cum, np.inf), axis=1)
    margin = np.where(np.isfinite(srt[:, 1]), srt[:, 1] - srt[:, 0], np.nan)
    fastest = np.argmin(np.where(np.isfinite(best), best, 1e9), axis=1)

    return RaceSim(positions, grid, ~alive, laps_done, fastest, margin, sc_count, vsc_count,
                   rain, lap1_leader, stops_done, pace, leader_by_lap, status if debug else None, pit_by_lap)
