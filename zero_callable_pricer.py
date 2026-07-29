"""
Zero-Callable Note Pricer（零息可贖回票據）
──────────────────────────────────────────
模型（詳見 rcn_cln_zerocallable_pricing_guide.md 第 3 節）：
  Hull-White 一因子短率模型，校準到發行人資金曲線（swap + 資金利差）：
      dr = [θ(t) − a·r] dt + σ dW，   r(t) = x(t) + α(t)，x 為 OU 過程
  - x 與 ∫x dt 以「精確聯合高斯抽樣」模擬（僅需在贖回日取節點，無離散化偏誤）
  - 路徑折現因子有精確式：E[DF(0,t)] = P(0,t) 完全成立（下方有 martingale 驗證）
  - Bermudan 贖回權用 LSMC（Longstaff-Schwartz）：發行人在各贖回日
    比較「贖回價」與「繼續持有價值（回歸估計）」，取對發行人有利者（最小化票券價值）

輸出：票券現值、不可贖回零息債 PV、贖回權價值、公允殖利率（PV=100）
自我驗證：
  - martingale：模擬 E[DF(0,T)] vs 曲線 P(0,T)
  - 單一贖回日（歐式）LSMC vs Jamshidian 零息債選擇權閉式解
  - Bermudan 贖回權價值 ≥ 各單一贖回日歐式價值之最大值

把「使用者輸入」區塊改成你的條件後直接執行：python3 zero_callable_pricer.py
"""

import numpy as np
from scipy.stats import norm

np.random.seed(42)

N = norm.cdf

# ══════════════════════════════════════════════════════════════════════════════
# 使用者輸入（範例條件 — 請改成實際交易）
# ══════════════════════════════════════════════════════════════════════════════

ZC_NOTE = dict(
    label        = "5NC1 年贖 Zero-Callable 範例",
    T            = 5.0,        # 天期（年）
    yield_note   = 0.052,      # 票面殖利率（年複利，累積用）；None = 求公允殖利率
    first_call   = 1.0,        # 第一個贖回日（年）
    call_every   = 1.0,        # 之後每隔幾年可贖
    call_prices  = "accreted", # "accreted" = 贖回價=累積價值 100·(1+y)^t；
                               # 或明確時程 [(1.0, 104.9), (2.0, 110.0), ...]
    # 無風險/swap 零利率曲線（連續複利近似）：[(年期, 利率)]；單點 = 平坦
    disc_curve   = [(1.0, 0.042), (2.0, 0.040), (3.0, 0.0395), (5.0, 0.039),
                    (7.0, 0.039)],
    fund_spread  = 0.0090,     # 發行人資金利差
    sigma_bp     = 90.0,       # 短率 normal vol（bp/年）≈ ATM swaption normal vol
    mean_rev     = 0.03,       # Hull-White 均值回歸 a
)

N_SIMS = 200_000
FACE   = 100.0


# ══════════════════════════════════════════════════════════════════════════════
# 折現曲線（log-DF 線性內插 = 分段平坦遠期）
# ══════════════════════════════════════════════════════════════════════════════

class Curve:
    def __init__(self, pillars, spread=0.0):
        pillars = sorted(pillars)
        self.t  = np.array([p[0] for p in pillars], dtype=float)
        self.zt = np.array([(p[1] + spread) * p[0] for p in pillars])
        if self.t[0] > 0:
            self.t  = np.insert(self.t, 0, 0.0)
            self.zt = np.insert(self.zt, 0, 0.0)

    def df(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        zt = np.interp(t, self.t, self.zt)
        beyond = t > self.t[-1]
        if beyond.any():
            fwd = (self.zt[-1] - self.zt[-2]) / (self.t[-1] - self.t[-2])
            zt[beyond] = self.zt[-1] + fwd * (t[beyond] - self.t[-1])
        out = np.exp(-zt)
        return out if out.size > 1 else float(out[0])


# ══════════════════════════════════════════════════════════════════════════════
# Hull-White：x 與 ∫x 的精確聯合抽樣、路徑折現因子
# ══════════════════════════════════════════════════════════════════════════════

def _B(a, tau):
    return (1.0 - np.exp(-a * tau)) / a

def _G(a, t):
    """∫₀ᵗ (1−e^{−as})² ds，出現在 ∫α 與 Var(∫x) 的閉式解中。"""
    return t - 2.0 * _B(a, t) + (1.0 - np.exp(-2.0 * a * t)) / (2.0 * a)


def simulate_paths(nodes, a, sigma, curve, n_sims, Z=None):
    """在事件節點 nodes（含 0 與 T）精確模擬。
    回傳 x[node, path] 與 df_step[i] = 路徑在 [t_i, t_{i+1}] 的折現因子。"""
    nodes = np.asarray(nodes, dtype=float)
    m = len(nodes) - 1
    if Z is None:
        Z = np.random.standard_normal((m, 2, n_sims // 2))
        Z = np.concatenate([Z, -Z], axis=2)                  # antithetic
    n = Z.shape[2]
    x = np.zeros((len(nodes), n))
    df_step = np.empty((m, n))
    for i in range(m):
        t0, t1 = nodes[i], nodes[i + 1]
        dt  = t1 - t0
        e   = np.exp(-a * dt)
        Vx  = sigma ** 2 * (1 - e ** 2) / (2 * a)
        VI  = sigma ** 2 / a ** 2 * _G(a, dt)
        Cov = sigma ** 2 / (2 * a ** 2) * (1 - e) ** 2
        eps_x = np.sqrt(Vx) * Z[i, 0]
        # I | eps_x 之條件分配（聯合高斯）
        cond_sd = np.sqrt(max(VI - Cov ** 2 / Vx, 0.0))
        I_inc = x[i] * _B(a, dt) + (Cov / Vx) * eps_x + cond_sd * Z[i, 1]
        x[i + 1] = x[i] * e + eps_x
        det = (curve.df(t1) / curve.df(t0)) \
              * np.exp(-sigma ** 2 / (2 * a ** 2) * (_G(a, t1) - _G(a, t0)))
        df_step[i] = np.exp(-I_inc) * det
    return x, df_step, Z


def lsmc_price(nodes, call_idx, call_px, redemption, x, df_step):
    """LSMC 反向遞迴。call_idx：nodes 中屬於贖回日的索引；
    call_px：對應贖回價。發行人最小化票券價值。"""
    V = np.full(x.shape[1], redemption)
    for i in range(len(nodes) - 2, 0, -1):
        V = V * df_step[i]                                   # 折回節點 i
        if i in call_idx:
            K = call_px[call_idx.index(i)]
            basis = np.vstack([np.ones_like(x[i]), x[i], x[i] ** 2]).T
            beta, *_ = np.linalg.lstsq(basis, V, rcond=None)
            C_hat = basis @ beta
            V = np.where(K < C_hat, K, V)                    # 發行人贖回
    return (V * df_step[0]).mean(), (V * df_step[0]).std(ddof=1) / np.sqrt(len(V))


def jamshidian_zbc(curve, a, sigma, T_call, T_mat, strike_per_face):
    """歐式零息債買權閉式解（Hull-White）。"""
    P_c, P_m = curve.df(T_call), curve.df(T_mat)
    sig_P = sigma * _B(a, T_mat - T_call) \
            * np.sqrt((1 - np.exp(-2 * a * T_call)) / (2 * a))
    h = np.log(P_m / (strike_per_face * P_c)) / sig_P + sig_P / 2
    return P_m * N(h) - strike_per_face * P_c * N(h - sig_P)


# ══════════════════════════════════════════════════════════════════════════════
# 票券組裝
# ══════════════════════════════════════════════════════════════════════════════

def build_schedule(spec, y):
    """回傳 (nodes, call_idx, call_px, redemption)。"""
    T = spec["T"]
    call_t = np.arange(spec["first_call"], T - 1e-9, spec["call_every"])
    if spec["call_prices"] == "accreted":
        call_px = [FACE * (1 + y) ** t for t in call_t]
    else:
        call_t  = np.array([c[0] for c in spec["call_prices"]])
        call_px = [c[1] for c in spec["call_prices"]]
    nodes = np.concatenate([[0.0], call_t, [T]])
    call_idx = list(range(1, 1 + len(call_t)))
    redemption = FACE * (1 + y) ** T
    return nodes, call_idx, call_px, redemption


def price_note(spec, curve, y, Z=None):
    nodes, call_idx, call_px, redemption = build_schedule(spec, y)
    a, sigma = spec["mean_rev"], spec["sigma_bp"] / 1e4
    x, df_step, Z = simulate_paths(nodes, a, sigma, curve, N_SIMS, Z)
    pv, se = lsmc_price(nodes, call_idx, call_px, redemption, x, df_step)
    return pv, se, nodes, call_idx, call_px, redemption, x, df_step, Z


# ══════════════════════════════════════════════════════════════════════════════
# 執行
# ══════════════════════════════════════════════════════════════════════════════

SEP = "═" * 72

if __name__ == "__main__":
    spec  = ZC_NOTE
    curve = Curve(spec["disc_curve"], spec["fund_spread"])
    a, sigma, T = spec["mean_rev"], spec["sigma_bp"] / 1e4, spec["T"]

    print(f"\n{SEP}\n  {spec['label']}\n{SEP}")

    # ── 驗證 1：martingale（模擬折現因子 vs 曲線）─────────────────────────
    y0 = spec["yield_note"] if spec["yield_note"] else 0.05
    pv, se, nodes, call_idx, call_px, redemption, x, df_step, Z = \
        price_note(spec, curve, y0)
    df_path = np.prod(df_step, axis=0)
    print(f"  [驗證] E[DF(0,{T:.0f}y)] 模擬 {df_path.mean():.6f} "
          f"vs 曲線 {curve.df(T):.6f}"
          f"（差 {abs(df_path.mean()-curve.df(T)):.2e}，僅 MC 誤差）")

    # ── 驗證 2：單一贖回日 → LSMC vs Jamshidian 閉式解 ──────────────────
    spec_1call = dict(spec, call_every=99.0)                 # 只留第一個贖回日
    n1, ci1, cp1, red1 = build_schedule(spec_1call, y0)
    x1, dfs1, _ = simulate_paths(n1, a, sigma, curve, N_SIMS)
    pv1, se1 = lsmc_price(n1, ci1, cp1, red1, x1, dfs1)
    zbc = red1 * jamshidian_zbc(curve, a, sigma, spec["first_call"], T,
                                cp1[0] / red1)
    closed_1call = red1 * curve.df(T) - zbc
    print(f"  [驗證] 單一贖回日：LSMC {pv1:.4f} ± {2*se1:.4f} "
          f"vs 閉式解 {closed_1call:.4f}")

    # ── 定價（給定票面殖利率）────────────────────────────────────────────
    noncall = redemption * curve.df(T)
    option  = noncall - pv
    euro_max = max(
        redemption * jamshidian_zbc(curve, a, sigma, tc, T, K / redemption)
        for tc, K in zip(nodes[1:-1], call_px))
    print(f"\n  票面殖利率 y = {y0:.3%}，到期還本 {redemption:.3f}，"
          f"贖回日 {[f'{t:.0f}y' for t in nodes[1:-1]]}")
    print(f"  不可贖回零息債 PV：{noncall:.3f}")
    print(f"  Bermudan 贖回權價值：{option:.3f}"
          f"（≥ 歐式最大值 {euro_max:.3f} ✓）" if option >= euro_max - 3*se
          else f"  Bermudan 贖回權 {option:.3f} < 歐式 {euro_max:.3f} ✗ 檢查失敗")
    print(f"  Zero-Callable 票券 PV：{pv:.3f} ± {2*se:.3f}（95% CI）")
    if spec["yield_note"]:
        print(f"  （發行費用 ≈ {100-pv:.2f} 點）" if pv < 100
              else f"  （票息高於公允 {pv-100:.2f} 點）")

    # ── 求公允殖利率（PV = 100，二分法，共用亂數）────────────────────────
    if spec["call_prices"] == "accreted":
        y_bullet = curve.df(T) ** (-1.0 / T) - 1.0           # 不可贖回公允殖利率
        lo, hi = y_bullet, y_bullet + 0.05
        for _ in range(40):
            mid = (lo + hi) / 2
            pv_m, *_ = price_note(spec, curve, mid, Z)
            lo, hi = (mid, hi) if pv_m < FACE else (lo, mid)
        y_fair = (lo + hi) / 2
        print(f"\n  不可贖回零息公允殖利率：{y_bullet:.3%}（資金曲線直接反推）")
        print(f"  Zero-Callable 公允殖利率：{y_fair:.3%}"
              f"（賣出贖回權補償 ≈ {(y_fair-y_bullet)*1e4:.0f} bp/年）")

    print(f"\n{SEP}\n  註：σ 以 ATM swaption normal vol 近似（a 小時誤差有限）；"
          f"\n      嚴謹作法為對 co-terminal swaptions（1y×4y、2y×3y…）校準。\n{SEP}")
