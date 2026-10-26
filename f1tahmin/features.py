"""Bir yarışın tur tur verisinden ölçülen her şey.

Girdi: Jolpica tur süreleri + pozisyonlar + pit stoplar + sonuç tablosu.
Çıktı (hepsi saniye cinsinden):
  - nötr turlar (güvenlik aracı / VSC / kırmızı bayrak) ve olay listesi
  - sürücü yarış temposu (alpha) ve standart hatası; Huber-robust regresyonla
        tur_süresi = b + alpha_sürücü + beta*tur_no + delta*lastik_yaşı + stint_etkisi + theta*kirli_hava
  - yakıt etkisi (beta), lastik aşınması (delta), kirli hava kaybı (theta), tur gürültüsü (sigma)
  - pit kaybı (yeşil bayrakta / nötr turda), takım bazında pit süresi
  - ilk tur pozisyon değişimleri, sollama denemeleri (takip mesafesi, tempo farkı, geçti mi)
"""
from __future__ import annotations

import numpy as np

NEUTRAL = 1.15        # alan medyanı, yeşil referansın %115'ini aşarsa tur nötr sayılır
NEUTRAL_LAP1 = 1.30   # ilk tur duran start yüzünden zaten yavaş
SC_SLOW = 1.45        # güvenlik aracı arkası VSC'den belirgin yavaş
RED = 3.0             # referansın 3 katı: kırmızı bayrak (bekleme süresi tura yazılır)
DIRTY = 1.0           # öndeki araca bu kadar saniyeden yakınsa kirli hava
ATTACK = 1.5          # sollama denemesi sayılacak takip mesafesi (sn)


def _huber_lstsq(X: np.ndarray, y: np.ndarray, ridge: np.ndarray, iters: int = 6):
    w = np.ones(len(y))
    for _ in range(iters):
        Xw = X * w[:, None]
        coef = np.linalg.solve(X.T @ Xw + np.diag(ridge), Xw.T @ y)
        r = y - X @ coef
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        k = 1.345 * s
        w = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
    return coef, r, s, w


def process_race(lapdata: dict, race: dict) -> dict | None:
    drivers = [d for d in lapdata["laps"] if lapdata["laps"][d]]
    if len(drivers) < 10:
        return None
    team = {x["driver"]: x["team"] for x in race["results"]}
    grid = {x["driver"]: x["grid"] for x in race["results"]}
    L = max(len(v) for v in lapdata["laps"].values())
    D = len(drivers)
    T = np.full((D, L), np.nan)
    for j, d in enumerate(drivers):
        v = lapdata["laps"][d]
        T[j, :len(v)] = [x if x else np.nan for x in v]
    done = np.array([len(lapdata["laps"][d]) for d in drivers])

    # Pit turları
    pit_laps: dict[str, list[int]] = {}
    for d, lap, stop, dur in lapdata["pits"]:
        pit_laps.setdefault(d, []).append(lap)
    in_lap = np.zeros((D, L), bool)
    out_lap = np.zeros((D, L), bool)
    age = np.zeros((D, L))
    stint = np.zeros((D, L), int)
    for j, d in enumerate(drivers):
        last, s = 0, 0
        for i in range(L):
            age[j, i] = i + 1 - last
            stint[j, i] = s
            if (i + 1) in pit_laps.get(d, []):
                in_lap[j, i] = True
                if i + 1 < L:
                    out_lap[j, i + 1] = True
                last, s = i + 1, s + 1

    # Nötr tur tespiti: pit yapmayan araçların tur medyanı
    pitting = in_lap | out_lap
    med = np.array([np.nanmedian(np.where(pitting[:, i], np.nan, T[:, i])) if np.isfinite(T[:, i]).sum() > 3 else np.nan
                    for i in range(L)])
    ref = float(np.nanpercentile(med[1:], 25))
    neutral = np.zeros(L, bool)
    red = np.zeros(L, bool)
    for i in range(L):
        thr = NEUTRAL_LAP1 if i == 0 else NEUTRAL
        if np.isfinite(med[i]) and med[i] > thr * ref:
            neutral[i] = True
            red[i] = med[i] > RED * ref
    # Kümülatif süreler (eksik tur = medyan)
    Tf = np.where(np.isfinite(T), T, med[None, :])
    cum = np.cumsum(Tf, axis=1)
    cum[np.arange(L)[None, :] >= done[:, None]] = np.inf

    # Nötr dönemleri olaylara böl; ilk-10 arasındaki makas daralıyorsa SC, korunuyorsa VSC
    events = []
    i = 0
    while i < L:
        if neutral[i]:
            j = i
            while j + 1 < L and neutral[j + 1]:
                j += 1
            before = cum[:, i - 1] if i > 0 else None
            after = cum[:, j]
            kind = "SC"
            if red[i:j + 1].any():
                kind = "RED"
            elif before is not None:
                def spread(c):
                    f = np.sort(c[np.isfinite(c)])
                    k = min(9, len(f) - 1)
                    return f[k] - f[0] if k > 0 else 0.0
                sb, sa = spread(before), spread(after)
                slow = np.nanmax(med[i:j + 1]) / ref
                kind = "SC" if (sb > 0 and sa / sb < 0.6) or (slow >= SC_SLOW and j > i) else "VSC"
            events.append({"start": i + 1, "laps": j - i + 1, "kind": kind})
            i = j + 1
        else:
            i += 1
    after_neutral = np.zeros(L, bool)
    after_neutral[1:] = neutral[:-1] & ~neutral[1:]

    # Öndeki araca mesafe (önceki tur sonunda)
    gap_ahead = np.full((D, L), np.inf)
    for i in range(1, L):
        c = cum[:, i - 1]
        order = np.argsort(c)
        cs = c[order]
        g = np.full(D, np.inf)
        with np.errstate(invalid="ignore"):
            g[order[1:]] = cs[1:] - cs[:-1]
        g[~np.isfinite(g)] = np.inf
        gap_ahead[:, i] = g

    # Temiz turlar: ilk tur, pit giriş/çıkış, nötr ve sonrası, %107 üstü hariç
    clean = np.isfinite(T) & ~pitting & ~neutral[None, :] & ~after_neutral[None, :]
    clean[:, 0] = False
    drv_med = np.array([np.nanmedian(np.where(clean[j], T[j], np.nan)) if clean[j].any() else np.nan for j in range(D)])
    clean &= T < 1.07 * drv_med[:, None]
    rows = np.argwhere(clean)
    if len(rows) < 100:
        return None

    n_stint = int(stint.max()) + 1
    jj, ii = rows[:, 0], rows[:, 1]
    y = T[jj, ii]
    cols = [np.ones(len(y))]
    Xd = np.zeros((len(y), D)); Xd[np.arange(len(y)), jj] = 1
    lapn = (ii + 1 - L / 2) / L                     # merkezlenmiş tur ilerlemesi
    Xs = np.zeros((len(y), max(n_stint - 1, 0)))
    for s in range(1, n_stint):
        Xs[:, s - 1] = stint[jj, ii] == s
    dirty = (gap_ahead[jj, ii] < DIRTY).astype(float)
    X = np.column_stack(cols + [Xd, lapn * L, age[jj, ii], Xs, dirty])
    ridge = np.concatenate([[1e-6], np.full(D, 1e-2), [1e-6, 1e-6], np.full(Xs.shape[1], 1e-2), [1e-2]])
    coef, resid, sigma, w = _huber_lstsq(X, y, ridge)
    alpha = coef[1:1 + D]
    alpha = alpha - np.median(alpha)
    beta, delta = coef[1 + D], coef[2 + D]
    theta = coef[-1]
    n_d = np.bincount(jj, minlength=D)
    se = sigma / np.sqrt(np.maximum(n_d, 1))

    def predict(j, i):
        x = np.zeros(X.shape[1]); x[0] = 1; x[1 + j] = 1; x[1 + D] = i + 1 - L / 2; x[2 + D] = age[j, i]
        if stint[j, i] >= 1 and Xs.shape[1]:
            x[3 + D + min(stint[j, i], Xs.shape[1]) - 1] = 1
        return float(x @ coef)

    # Pit kaybı
    pit_green, pit_neutral, team_pit = [], [], []
    durs = [p[3] for p in lapdata["pits"] if p[3] and p[3] < 60]
    med_dur = float(np.median(durs)) if durs else None
    idx = {d: j for j, d in enumerate(drivers)}
    for d, lap, stop, dur in lapdata["pits"]:
        j = idx.get(d)
        i = lap - 1
        if j is None or i + 1 >= done[j] or not (np.isfinite(T[j, i]) and np.isfinite(T[j, i + 1])):
            continue
        if neutral[i] or neutral[i + 1]:
            # nötr turda referans, o turdaki pit yapmayan alanın medyanı (SC yavaşlığı kayba yazılmasın)
            loss = T[j, i] + T[j, i + 1] - med[i] - med[i + 1]
        else:
            loss = T[j, i] + T[j, i + 1] - predict(j, i) - predict(j, i + 1)
        if not np.isfinite(loss):
            continue
        if neutral[i] or neutral[i + 1]:
            if 0 < loss < 60:
                pit_neutral.append(loss)
        elif 0 < loss < 60:
            pit_green.append(loss)
        if dur and med_dur and d in team:
            team_pit.append([team[d], dur - med_dur])

    # İlk tur
    lap1 = [[grid[d], lapdata["pos"][d][0]] for d in drivers
            if grid.get(d) and lapdata["pos"].get(d) and not neutral[0]]

    # Sollama denemeleri: yeşil turda, ikisi de pit yapmıyorken 1.5 sn içinde takip
    duels = []
    for i in range(2, L):
        if neutral[i] or neutral[i - 1] or after_neutral[i]:
            continue
        c0, c1 = cum[:, i - 1], cum[:, i]
        order = np.argsort(c0)
        for a, b in zip(order[:-1], order[1:]):
            if not (np.isfinite(c0[b]) and np.isfinite(c1[a]) and np.isfinite(c1[b])):
                continue
            g = c0[b] - c0[a]
            if g >= ATTACK or pitting[a, i] or pitting[b, i] or (i + 1 < L and (in_lap[a, i] or in_lap[b, i])):
                continue
            dpace = (alpha[b] + delta * age[b, i]) - (alpha[a] + delta * age[a, i])
            duels.append([round(float(dpace), 3), round(float(g), 3), int(c1[b] < c1[a])])

    finishers = [d for d in drivers if done[idx[d]] >= L - 3]
    stops = [len(pit_laps.get(d, [])) for d in finishers]
    first_stop = [pit_laps[d][0] / L for d in finishers if pit_laps.get(d)]
    # tur verisinde hiç görünmeyen sürücü ilk turu tamamlayamamıştır
    dnf_laps = [int(done[idx[x["driver"]]]) if x["driver"] in idx else 0
                for x in race["results"] if x["kind"] == "dnf"]

    return {
        "season": race["season"], "round": race["round"], "circuit": race["circuit"],
        "laps": L, "ref": ref, "events": events,
        "pace": {d: {"alpha": float(alpha[j]), "se": float(se[j]), "n": int(n_d[j]), "team": team.get(d)}
                 for j, d in enumerate(drivers) if n_d[j] >= 5},
        "beta": float(beta), "delta": float(delta), "theta": float(theta), "sigma": float(sigma),
        "resid_tail": float(np.mean(np.abs(resid) > 3 * sigma)),
        "pit_green": float(np.median(pit_green)) if pit_green else None,
        "pit_neutral": float(np.median(pit_neutral)) if pit_neutral else None,
        "team_pit": team_pit,
        "lap1": lap1,
        "duels": duels,
        "stops": stops,
        "first_stop": first_stop,
        "dnf_laps": dnf_laps,
        "n_starters": D,
    }
