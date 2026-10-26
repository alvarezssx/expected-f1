"""Pist parametreleri: tüm yılların tur verisinden, az verili pistler genel ortalamaya büzülerek.

Her parametre için:  pist_değeri = (n·pist_ortalaması + k·genel_ortalama) / (n + k)
(k = "kaç yarışlık" öncül ağırlığı). Güvenlik aracı sayıları için Gamma-Poisson aynı formüle denk.

Sollama: tüm sezonların takip verisi üzerine hiyerarşik lojistik regresyon
    P(tur içinde geçti) = σ(a + a_pist + a_2026 + b1·tempo_farkı + b2·takip_mesafesi)
pist sabitleri sıfıra doğru (ridge) cezalandırılır; 2026 için ayrı kaydırma (yeni aero kuralları).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

GLOBAL_SC_LEN = [4, 4, 5, 5, 6, 3, 4, 5]
GLOBAL_VSC_LEN = [1, 2, 2, 2, 3]


def _shrink(values: list[float], prior: float, k: float) -> float:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float((np.sum(v) + k * prior) / (len(v) + k))


@dataclass
class Overtake:
    a: float
    b_pace: float
    b_gap: float
    a_2026: float
    circuit: dict[str, float] = field(default_factory=dict)
    follow_gap: float = 0.77  # geçemeyen hızlı takipçinin tipik mesafesi (sn), kirli hava yüzünden

    def logit(self, circuit: str, season: int) -> float:
        return self.a + self.circuit.get(circuit, 0.0) + (self.a_2026 if season >= 2026 else 0.0)


def fit_overtake(feats: list[dict]) -> Overtake:
    rows = [(f["circuit"], f["season"], *d) for f in feats for d in f["duels"]]
    if len(rows) < 200:
        return Overtake(-2.0, -1.5, -1.0, 0.0)
    circuits = sorted({r[0] for r in rows})
    ci = {c: i for i, c in enumerate(circuits)}
    c_idx = np.array([ci[r[0]] for r in rows])
    s26 = np.array([r[1] >= 2026 for r in rows], float)
    dp = np.clip(np.array([r[2] for r in rows]), -4, 4)
    gap = np.array([r[3] for r in rows])
    y = np.array([r[4] for r in rows], float)
    nC = len(circuits)

    def nll(p):
        a, b1, b2, a26 = p[:4]
        ac = p[4:]
        z = a + ac[c_idx] + a26 * s26 + b1 * dp + b2 * gap
        ll = y * z - np.logaddexp(0, z)
        pr = 1 / (1 + np.exp(-z))
        g = y - pr
        grad = np.concatenate([[g.sum(), (g * dp).sum(), (g * gap).sum(), (g * s26).sum()],
                               np.bincount(c_idx, weights=g, minlength=nC)])
        reg = 0.5 * np.sum(ac ** 2) / 0.5 ** 2
        grad[4:] -= ac / 0.5 ** 2
        return -(ll.sum() - reg), -grad

    res = minimize(nll, np.zeros(4 + nC), jac=True, method="L-BFGS-B")
    p = res.x
    stuck = gap[(y == 0) & (dp < 0)]
    return Overtake(float(p[0]), float(p[1]), float(p[2]), float(p[3]), {c: float(p[4 + ci[c]]) for c in circuits},
                    follow_gap=float(np.median(stuck)) if len(stuck) > 50 else 0.77)


def start_profile(feats: list[dict], spacing: float = 0.25, pace_per_slot: float = 0.08,
                  pace_noise: float = 0.3, slots: int = 22, iters: int = 25) -> np.ndarray:
    """Grid sırasına göre start gürültüsü (sn): her sıradaki ortalama |ilk tur Δpozisyon| veriyle eşleşene
    kadar simülasyonla iteratif ayarlanır. Pole'dan başlayan az pozisyon kaybeder, orta grup çok karışır.
    pace_per_slot / pace_noise: simülatördeki ilk tur süresine giren tempo sıralaması ve onun dağılımı
    (kalibrasyon bunları hesaba katmazsa start iki kez karıştırılmış olur)."""
    pairs = np.array([(g, p) for f in feats for g, p in f["lap1"] if 1 <= g <= slots])
    if len(pairs) < 100:
        return np.full(slots, 0.8)
    target = np.zeros(slots)
    pooled = np.mean(np.abs(pairs[:, 1] - pairs[:, 0]))
    for g in range(1, slots + 1):
        # komşu sıralarla düzleştir (pole hariç: kendine özgü)
        w = [g] if g == 1 else [max(2, g - 1), g, min(slots, g + 1)]
        m = np.isin(pairs[:, 0], w)
        target[g - 1] = (np.sum(np.abs(pairs[m, 1] - pairs[m, 0])) + 3 * pooled) / (m.sum() + 3)
    rng = np.random.default_rng(0)
    base = np.arange(slots) * (spacing + pace_per_slot)
    sig = np.full(slots, 0.8)
    noise = rng.standard_normal((4000, slots))
    pace = pace_noise * rng.standard_normal((4000, slots))
    for _ in range(iters):
        t = base[None, :] + pace + sig[None, :] * noise
        pos = np.argsort(np.argsort(t, axis=1), axis=1)
        sim = np.mean(np.abs(pos - np.arange(slots)[None, :]), axis=0)
        sig = np.clip(sig * (target / np.maximum(sim, 1e-3)) ** 0.7, 0.02, 6.0)
    return sig


@dataclass
class TrackParams:
    circuit: str
    laps: int
    ref: float               # yeşil bayrak referans tur süresi (sn)
    sigma: float             # tur tur gürültü (sn)
    delta: float             # lastik aşınması (sn / tur yaşı)
    beta: float              # yakıt etkisi (sn / tur)
    theta: float             # kirli hava kaybı (sn)
    pit_green: float
    pit_neutral_ratio: float
    sc_rate: float           # yarış başına güvenlik aracı sayısı (Poisson)
    vsc_rate: float
    sc_lap1_share: float
    sc_len: list[int]
    vsc_len: list[int]
    start_scale: float       # pistin ilk tur karışıklığı / genel ortalama
    start_profile: list[float]  # grid sırasına göre start gürültüsü (sn)
    stop_probs: list[float]  # [1, 2, 3] durak olasılıkları
    dnf_lap1_share: float
    overtake_logit: float
    follow_gap: float
    n_history: int


def build_track_params(all_feats: list[dict], circuit: str, upto: tuple[int, int],
                       overtake: Overtake, profile: np.ndarray) -> TrackParams:
    """upto=(sezon, tur): bu yarıştan ÖNCEKİ veriler kullanılır (backtest'te sızıntı yok)."""
    past = [f for f in all_feats if (f["season"], f["round"]) < upto]
    here = [f for f in past if f["circuit"] == circuit]
    cur = [f for f in past if f["season"] == upto[0]]
    pool = cur if len(cur) >= 3 else past  # güncel sezon genel ortalaması (yoksa tümü)

    def g(key, fs=None):
        vals = [f[key] for f in (fs or pool) if f.get(key) is not None]
        return float(np.median(vals)) if vals else None

    # Referans tur süresi: bu pistteki en güncel yarış; eski sezonsa sezonlar arası oranla ölçekle
    if here:
        last = max(here, key=lambda f: (f["season"], f["round"]))
        ref = last["ref"]
        if last["season"] != upto[0]:
            ratios = []
            for c in {f["circuit"] for f in cur}:
                a = [f["ref"] for f in past if f["circuit"] == c and f["season"] == upto[0]]
                b = [f["ref"] for f in past if f["circuit"] == c and f["season"] == last["season"]]
                if a and b:
                    ratios.append(a[-1] / b[-1])
            ref *= float(np.median(ratios)) if ratios else 1.0
        laps = last["laps"]
    else:
        ref, laps = g("ref") or 90.0, 57

    sc_counts = [sum(e["kind"] in ("SC", "RED") for e in f["events"]) for f in here]
    vsc_counts = [sum(e["kind"] == "VSC" for e in f["events"]) for f in here]
    all_sc = [sum(e["kind"] in ("SC", "RED") for e in f["events"]) for f in past]
    all_vsc = [sum(e["kind"] == "VSC" for e in f["events"]) for f in past]
    sc_g = float(np.mean(all_sc)) if all_sc else 0.7
    vsc_g = float(np.mean(all_vsc)) if all_vsc else 0.4
    sc_events = [e for f in past for e in f["events"] if e["kind"] in ("SC", "RED")]
    vsc_events = [e for f in past for e in f["events"] if e["kind"] == "VSC"]

    # pist karakteri (tüm yıllar) × güncel sezon / tüm yıllar oranı: yeni kurallara uyarlama
    def adapted(key, k=2.0):
        allv = g(key, past) or 0.0
        curv = g(key) if g(key) is not None else allv
        local = _shrink([f[key] for f in here], allv, k)
        return local * (curv / allv) if allv else curv

    stops = [s for f in (here or pool) for s in f["stops"]]
    stops_g = [s for f in pool for s in f["stops"]]
    probs = []
    for n in (1, 2, 3):
        local = np.mean([s == n for s in stops]) if stops else None
        glob = np.mean([s == n for s in stops_g]) if stops_g else [0.6, 0.35, 0.05][n - 1]
        probs.append(_shrink([local] if local is not None else [], glob, 1.0) if stops else glob)
    probs = list(np.array(probs) / np.sum(probs))

    lap1 = [abs(a - b) for f in here for a, b in f["lap1"]]
    lap1_g = [abs(a - b) for f in past for a, b in f["lap1"]]
    g_move = float(np.mean(lap1_g)) if lap1_g else 1.6
    start_scale = _shrink([np.mean(lap1) / g_move] if lap1 else [], 1.0, 1.0)

    ratio = [f["pit_neutral"] / f["pit_green"] for f in past
             if f["pit_neutral"] and f["pit_green"] and np.isfinite(f["pit_neutral"] / f["pit_green"])]
    dnf_laps = [l for f in past for l in f["dnf_laps"]]

    return TrackParams(
        circuit=circuit, laps=int(laps), ref=float(ref),
        sigma=adapted("sigma"), delta=max(adapted("delta"), 0.0), beta=adapted("beta"),
        theta=max(adapted("theta"), 0.0),
        pit_green=_shrink([f["pit_green"] for f in here], g("pit_green", past) or 22.0, 1.0),
        pit_neutral_ratio=float(np.clip(np.median(ratio), 0.3, 1.0)) if ratio else 0.55,
        sc_rate=_shrink(sc_counts, sc_g, 3.0),
        vsc_rate=_shrink(vsc_counts, vsc_g, 3.0),
        sc_lap1_share=float(np.mean([e["start"] <= 2 for e in sc_events])) if sc_events else 0.2,
        sc_len=[e["laps"] for e in sc_events] or GLOBAL_SC_LEN,
        vsc_len=[e["laps"] for e in vsc_events] or GLOBAL_VSC_LEN,
        start_scale=start_scale,
        start_profile=[float(x) for x in profile],
        stop_probs=[float(p) for p in probs],
        dnf_lap1_share=float(np.mean([l <= 1 for l in dnf_laps])) if dnf_laps else 0.2,
        overtake_logit=overtake.logit(circuit, upto[0]),
        follow_gap=overtake.follow_gap,
        n_history=len(here),
    )


def team_pit_effects(feats: list[dict], k: float = 6.0) -> tuple[dict[str, float], float, float]:
    """Takım pit ekibi etkisi (sn, pist medyanına göre), yavaş pit olasılığı ve ortalama ek süresi."""
    rows = [(t, d) for f in feats for t, d in f["team_pit"]]
    normal = [(t, d) for t, d in rows if abs(d) < 5]
    slow = [d for _, d in rows if 5 <= d < 40]
    eff = {}
    for t in {t for t, _ in normal}:
        v = [d for tt, d in normal if tt == t]
        eff[t] = float(np.sum(v) / (len(v) + k))
    p_slow = len(slow) / max(len(rows), 1)
    return eff, p_slow, float(np.mean(slow)) if slow else 6.0
