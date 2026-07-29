"""
RCN (Reverse Convertible Note) Pricer
─────────────────────────────────────
支援三種變形（詳見 rcn_cln_zerocallable_pricing_guide.md 第 1 節）：
  1. Vanilla RCN            — Black-Scholes 歐式賣權閉式解
  2. Barrier RCN (KI put)   — Reiner-Rubinstein 閉式解
                              每日觀察用 Broadie-Glasserman-Kou 障礙修正
                              歐式觀察（只看到期）另有閉式解
  3. Worst-of RCN           — 相關 GBM Monte Carlo（每日監控障礙）

輸出：票券現值、內嵌賣權拆解、公允票息（PV=100 之票息）
自我驗證：DIP+DOP=vanilla put 對價關係、閉式解 vs Monte Carlo

把「使用者輸入」區塊改成你的條件後直接執行：python3 rcn_pricer.py
"""

import numpy as np
from scipy.stats import norm

np.random.seed(42)

N = norm.cdf

# ══════════════════════════════════════════════════════════════════════════════
# 使用者輸入（範例條件 — 請改成實際交易）
# ══════════════════════════════════════════════════════════════════════════════

SINGLE_RCN = dict(
    label       = "單一標的 Barrier RCN 範例",
    S0          = 100.0,     # 期初定價
    S           = 100.0,     # 目前股價（估值日；定價日 = S0）
    strike_pct  = 1.00,      # 履約/轉換價（% of S0）
    barrier_pct = 0.60,      # 障礙（% of S0）；None = vanilla（無障礙）
    barrier_obs = "daily",   # "daily"（美式/每日收盤）或 "expiry"（歐式/只看到期）
    T           = 0.5,       # 天期（年）
    coupon      = 0.12,      # 年化票息；None = 求公允票息
    coupon_freq = 12,        # 每年付息次數
    vol         = 0.45,      # 隱含波動率（建議用障礙價位附近的 vol）
    q           = 0.00,      # 股利率
    r           = 0.043,     # 無風險利率（連續複利近似）
    fund_spread = 0.0080,    # 發行人資金利差（折現票券現金流用）
)

WORST_OF_RCN = dict(
    label       = "三標的 Worst-of RCN 範例",
    names       = ["A", "B", "C"],
    vols        = [0.40, 0.45, 0.50],
    qs          = [0.00, 0.00, 0.01],
    corr        = 0.55,      # 兩兩相關係數（也可給完整矩陣）
    perf0       = [1.0, 1.0, 1.0],   # 目前價/期初價（定價日 = 1.0）
    strike_pct  = 1.00,
    barrier_pct = 0.60,
    barrier_obs = "daily",
    T           = 1.0,
    coupon      = None,      # None = 求公允票息
    coupon_freq = 12,
    r           = 0.043,
    fund_spread = 0.0080,
)

N_SIMS   = 200_000
STEPS_YR = 252
FACE     = 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 閉式解：BS 賣權、障礙賣權（Reiner-Rubinstein）、歐式 KI 賣權
# ══════════════════════════════════════════════════════════════════════════════

def bs_put(S, K, T, r, q, sig):
    if T <= 0:
        return max(K - S, 0.0)
    sqT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * sqT)
    d2 = d1 - sig * sqT
    return K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)


def _barrier_terms(S, K, H, T, r, q, sig, phi, eta):
    """Haug 記號之 A/B/C/D 項（無 rebate）。b = r − q。"""
    b   = r - q
    sqT = np.sqrt(T)
    mu  = (b - 0.5 * sig ** 2) / sig ** 2
    x1  = np.log(S / K) / (sig * sqT) + (1 + mu) * sig * sqT
    x2  = np.log(S / H) / (sig * sqT) + (1 + mu) * sig * sqT
    y1  = np.log(H ** 2 / (S * K)) / (sig * sqT) + (1 + mu) * sig * sqT
    y2  = np.log(H / S) / (sig * sqT) + (1 + mu) * sig * sqT
    fS  = S * np.exp((b - r) * T)
    fK  = K * np.exp(-r * T)
    hs1 = (H / S) ** (2 * (mu + 1))
    hs2 = (H / S) ** (2 * mu)
    A = phi * fS * N(phi * x1) - phi * fK * N(phi * (x1 - sig * sqT))
    B = phi * fS * N(phi * x2) - phi * fK * N(phi * (x2 - sig * sqT))
    C = phi * fS * hs1 * N(eta * y1) - phi * fK * hs2 * N(eta * (y1 - sig * sqT))
    D = phi * fS * hs1 * N(eta * y2) - phi * fK * hs2 * N(eta * (y2 - sig * sqT))
    return A, B, C, D


def down_in_put(S, K, H, T, r, q, sig):
    """連續觀察 down-and-in put（H < S）。"""
    A, B, C, D = _barrier_terms(S, K, H, T, r, q, sig, phi=-1, eta=+1)
    return (B - C + D) if K > H else A


def down_out_put(S, K, H, T, r, q, sig):
    A, B, C, D = _barrier_terms(S, K, H, T, r, q, sig, phi=-1, eta=+1)
    return (A - B + C - D) if K > H else 0.0


def bgk_adjust_down(H, sig, dt):
    """Broadie-Glasserman-Kou 離散監控修正：下障礙往下移。"""
    BETA = 0.5826
    return H * np.exp(-BETA * sig * np.sqrt(dt))


def expiry_ki_put(S, K, H, T, r, q, sig):
    """歐式觀察 KI：E[(K−S_T)·1{S_T<H}]，H ≤ K。"""
    assert H <= K
    sqT = np.sqrt(T)
    d1H = (np.log(S / H) + (r - q + 0.5 * sig ** 2) * T) / (sig * sqT)
    d2H = d1H - sig * sqT
    return K * np.exp(-r * T) * N(-d2H) - S * np.exp(-q * T) * N(-d1H)


# ══════════════════════════════════════════════════════════════════════════════
# Monte Carlo 引擎（單一標的與 worst-of 共用）
# ══════════════════════════════════════════════════════════════════════════════

def mc_put_single(S, S0, K, H, T, r, q, sig, obs, n_sims, steps_yr):
    """單一標的內嵌賣權 PV（每 1 單位股票、strike K）。H=None 為 vanilla。"""
    n_steps = max(int(round(T * steps_yr)), 1)
    dt = T / n_steps
    Z = np.random.standard_normal((n_sims // 2, n_steps))
    Z = np.vstack([Z, -Z])                                   # antithetic
    logpath = np.cumsum((r - q - 0.5 * sig ** 2) * dt
                        + sig * np.sqrt(dt) * Z, axis=1)
    S_path = S * np.exp(logpath)
    S_T = S_path[:, -1]
    if H is None:
        knocked = np.ones(len(S_T), dtype=bool)
    elif obs == "daily":
        knocked = (np.minimum(S_path.min(axis=1), S) <= H)
    else:                                                    # 只看到期
        knocked = S_T < H
    payoff = np.maximum(K - S_T, 0.0) * knocked
    return np.exp(-r * T) * payoff.mean()


def mc_worstof_put(spec, n_sims, steps_yr):
    """worst-of 內嵌賣權 PV（以 % of face 計，face=100）。
    KI 條件：任一標的觸及各自障礙（市場慣例）。"""
    m       = len(spec["vols"])
    T       = spec["T"]
    r       = spec["r"]
    k       = spec["strike_pct"]
    h       = spec["barrier_pct"]
    n_steps = max(int(round(T * steps_yr)), 1)
    dt      = T / n_steps

    corr = spec["corr"]
    if np.isscalar(corr):
        C = np.full((m, m), corr); np.fill_diagonal(C, 1.0)
    else:
        C = np.asarray(corr, dtype=float)
    L = np.linalg.cholesky(C)

    vols = np.asarray(spec["vols"]); qs = np.asarray(spec["qs"])
    perf0 = np.asarray(spec["perf0"], dtype=float)

    half = n_sims // 2
    Z = np.random.standard_normal((half, n_steps, m))
    Z = np.concatenate([Z, -Z], axis=0)
    Zc = Z @ L.T
    drift = (r - qs - 0.5 * vols ** 2) * dt
    logp  = np.cumsum(drift + vols * np.sqrt(dt) * Zc, axis=1)
    perf  = perf0 * np.exp(logp)                 # 相對期初定價之表現

    worst_T = perf[:, -1, :].min(axis=1)
    if h is None:
        knocked = np.ones(len(worst_T), dtype=bool)
    elif spec["barrier_obs"] == "daily":
        path_min = np.minimum(perf.min(axis=1), perf0).min(axis=1)
        knocked  = path_min <= h
    else:
        knocked = perf[:, -1, :].min(axis=1) < h

    payoff_pct = np.maximum(k - worst_T, 0.0) / k * knocked  # 每 1 面額
    pv = np.exp(-r * T) * payoff_pct.mean() * FACE
    se = np.exp(-r * T) * payoff_pct.std(ddof=1) / np.sqrt(len(payoff_pct)) * FACE
    ki_prob = knocked.mean()
    return pv, se, ki_prob


# ══════════════════════════════════════════════════════════════════════════════
# 票券組裝：債券腿 + 賣權腿 → 現值與公允票息
# ══════════════════════════════════════════════════════════════════════════════

def note_legs(T, freq, r_fund):
    """付息時點與折現因子（發行人資金曲線）。回傳 (annuity, DF_T)。"""
    n_cpn = max(int(round(T * freq)), 1)
    times = np.arange(1, n_cpn + 1) / freq
    times = times[times <= T + 1e-9]
    if abs(times[-1] - T) > 1e-9:
        times = np.append(times, T)
    dfs = np.exp(-r_fund * times)
    accr = np.diff(np.concatenate([[0.0], times]))
    annuity = float((accr * dfs).sum())          # Σ Δᵢ·DF(tᵢ)
    return annuity, float(np.exp(-r_fund * T))


def price_note(put_pv_pct, T, freq, coupon, r_fund):
    """給定內嵌賣權 PV（% of face）→ 票券 PV 與公允票息。"""
    annuity, df_T = note_legs(T, freq, r_fund)
    fair_c = (FACE - FACE * df_T + put_pv_pct) / (FACE * annuity)
    pv = None
    if coupon is not None:
        pv = coupon * FACE * annuity + FACE * df_T - put_pv_pct
    return pv, fair_c, annuity, df_T


# ══════════════════════════════════════════════════════════════════════════════
# 執行
# ══════════════════════════════════════════════════════════════════════════════

SEP = "═" * 72

def run_single(spec):
    print(f"\n{SEP}\n  {spec['label']}\n{SEP}")
    S0, S, T = spec["S0"], spec["S"], spec["T"]
    K  = spec["strike_pct"] * S0
    r, q, sig = spec["r"], spec["q"], spec["vol"]
    r_fund = r + spec["fund_spread"]
    obs = spec["barrier_obs"]
    dt  = 1 / STEPS_YR

    # ── 內嵌賣權（每股）與轉換成 % of face（數量 = FACE/K）───────────────
    vanilla = bs_put(S, K, T, r, q, sig)
    if spec["barrier_pct"] is None:
        put_ps, desc = vanilla, "vanilla put（BS 閉式解）"
    else:
        H = spec["barrier_pct"] * S0
        if obs == "daily":
            H_eff  = bgk_adjust_down(H, sig, dt)
            put_ps = down_in_put(S, K, H_eff, T, r, q, sig)
            desc   = f"down-and-in put（每日觀察，BGK 修正 H {H:.2f}→{H_eff:.2f}）"
            # 自我驗證 1：in-out parity（用同一 H_eff）
            parity = down_in_put(S, K, H_eff, T, r, q, sig) \
                   + down_out_put(S, K, H_eff, T, r, q, sig)
            assert abs(parity - vanilla) < 1e-8, "in-out parity 檢查失敗"
        else:
            put_ps = expiry_ki_put(S, K, H, T, r, q, sig)
            desc   = "歐式觀察 KI put（閉式解）"
        # 自我驗證 2：Monte Carlo 交叉比對
        mc = mc_put_single(S, S0, K, H, T, r, q, sig, obs, N_SIMS, STEPS_YR)
        print(f"  [驗證] 閉式解 {put_ps:.4f} vs MC {mc:.4f}"
              f"（差 {abs(put_ps-mc):.4f}，含 MC 誤差與離散監控近似）")

    put_pct = put_ps * FACE / K
    pv, fair_c, annuity, df_T = price_note(put_pct, T, spec["coupon_freq"],
                                           spec["coupon"], r_fund)

    print(f"\n  內嵌賣權：{desc}")
    print(f"    每股價值 {put_ps:.4f} × 轉換股數 {FACE/K:.4f} = {put_pct:.3f}（% of face）")
    print(f"  債券腿：面額 PV = {FACE*df_T:.3f}，年金因子 = {annuity:.4f}（折現 {r_fund:.2%}）")
    print(f"  公允票息（PV=100）：{fair_c:.3%}")
    if pv is not None:
        print(f"  報價票息 {spec['coupon']:.2%} 下之理論價值：{pv:.3f}"
              f"（發行費用 ≈ {100-pv:.2f} 點）")


def run_worstof(spec):
    print(f"\n{SEP}\n  {spec['label']}\n{SEP}")
    put_pct, se, ki_prob = mc_worstof_put(spec, N_SIMS, STEPS_YR)
    r_fund = spec["r"] + spec["fund_spread"]
    pv, fair_c, annuity, df_T = price_note(put_pct, spec["T"],
                                           spec["coupon_freq"],
                                           spec["coupon"], r_fund)
    print(f"  標的：{', '.join(spec['names'])}，vol = {spec['vols']}，"
          f"相關係數 = {spec['corr']}")
    print(f"  worst-of KI put = {put_pct:.3f} ± {2*se:.3f}（% of face，95% CI）")
    print(f"  KI 觸發機率（風險中性）= {ki_prob:.1%}")
    print(f"  債券腿：面額 PV = {FACE*df_T:.3f}，年金因子 = {annuity:.4f}")
    print(f"  公允票息（PV=100）：{fair_c:.3%}")
    if pv is not None:
        print(f"  報價票息 {spec['coupon']:.2%} 下之理論價值：{pv:.3f}")


if __name__ == "__main__":
    run_single(SINGLE_RCN)
    run_worstof(WORST_OF_RCN)
    print(f"\n{SEP}\n  註：障礙 RCN 對波動率 skew 敏感，本模型用單一平坦 vol；"
          f"\n      建議提供障礙價位附近之隱含波動率以免低估 KI put。\n{SEP}")
