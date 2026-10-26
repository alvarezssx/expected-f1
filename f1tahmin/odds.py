"""Oran matematiği: ima edilen olasılık, marj (overround) ve marjı ayıklama yöntemleri.

- Çarpımsal (orantılı): p = π / B. Basit, ama marjı tüm sürücülere eşit dağıtır.
- Toplamsal: p = π - (B - K) / n. Uzun oranlara en çok zarar veren yöntem.
- Power: Σ π^k = K olacak k bulunur, p = π^k. Marjın çoğunu uzun oranlara yükler;
  ilk-K piyasaları (podyum, top 10) için de çalışır.
- Shin: bahisçinin "içeriden bilgili bahisçi" payına karşı koyduğu marj modeli. Aslen
  at yarışı için geliştirildi; çok sonuçlu kazanan piyasasında en tutarlı sonucu verir.

K = piyasada kazanan seçim sayısı (kazanan 1, podyum 3, top 10 10). Bu yüzden podyumda
olasılıkların toplamı %100 değil %300 olmalı.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

MARKET_K = {"win": 1, "podium": 3, "top6": 6, "top10": 10}


def multiplicative(pi: np.ndarray, K: int) -> np.ndarray:
    return np.clip(pi * K / pi.sum(), 0, 0.999)


def additive(pi: np.ndarray, K: int) -> np.ndarray:
    p = np.clip(pi - (pi.sum() - K) / len(pi), 1e-6, None)
    return np.clip(p * K / p.sum(), 0, 0.999)


def power(pi: np.ndarray, K: int) -> np.ndarray:
    pi = np.clip(pi, 1e-9, 0.999)
    f = lambda k: np.sum(pi ** k) - K
    try:
        k = brentq(f, 0.05, 20.0)
    except ValueError:
        return multiplicative(pi, K)
    return pi ** k


def shin(pi: np.ndarray) -> tuple[np.ndarray, float]:
    """Kazanan piyasası için Shin olasılıkları ve z (içeriden bilgi payı)."""
    B = pi.sum()

    def probs(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / B) - z) / (2 * (1 - z))

    f = lambda z: probs(z).sum() - 1
    if B <= 1 or f(0.0) * f(0.4) > 0:
        return pi / B, 0.0
    z = brentq(f, 0.0, 0.4)
    return probs(z), float(z)


def analyze(market: str, odds: dict[str, float], model: dict[str, float], labels: dict[str, str]) -> dict:
    K = MARKET_K[market]
    ids = [d for d, o in odds.items() if o and o > 1.0]
    if not ids:
        return {"error": "Geçerli oran girilmedi (ondalık oran > 1.00 olmalı)."}
    o = np.array([odds[d] for d in ids], dtype=float)
    pi = 1.0 / o
    B = float(pi.sum())

    fair_pow = power(pi, K)
    fair_mul = multiplicative(pi, K)
    fair_add = additive(pi, K)
    if K == 1:
        fair_shin, z = shin(pi)
        fair_ref, ref_name = fair_shin, "Shin"
    else:
        fair_shin, z = None, None
        fair_ref, ref_name = fair_pow, "Power"

    rows = []
    for i, d in enumerate(ids):
        p_model = model.get(d)
        rows.append({
            "driver": d,
            "label": labels.get(d, d),
            "odds": float(o[i]),
            "implied": float(pi[i]),
            "fair_mult": float(fair_mul[i]),
            "fair_add": float(fair_add[i]),
            "fair_power": float(fair_pow[i]),
            "fair_shin": float(fair_shin[i]) if fair_shin is not None else None,
            "fair_odds": float(1 / fair_ref[i]) if fair_ref[i] > 0 else None,
            "margin_in_price": float(pi[i] - fair_ref[i]),  # bu fiyata yüklenen marj payı
            "model": p_model,
            "model_odds": (1 / p_model) if p_model else None,
            "ev": (p_model * o[i] - 1) if p_model is not None else None,
            "edge_vs_market": (p_model - float(fair_ref[i])) if p_model is not None else None,
        })
    rows.sort(key=lambda r: r["odds"])
    n_expected = len(model)
    return {
        "market": market,
        "K": K,
        "book_sum": B,
        "overround": B / K - 1,
        "margin_per_selection": (B - K) / len(ids),
        "n_selections": len(ids),
        "complete": len(ids) >= n_expected,
        "reference_method": ref_name,
        "shin_z": z,
        "rows": rows,
    }
