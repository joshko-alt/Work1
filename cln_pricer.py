"""
CLN (Credit Linked Note) Pricer
───────────────────────────────
Reduced-form（hazard rate）模型（詳見 rcn_cln_zerocallable_pricing_guide.md 第 2 節）：
  1. 由 CDS 利差曲線 bootstrap piecewise-constant 違約強度 λ → 存活曲線 Q(t)
  2. CLN PV = 票息腿（存活）+ 到期本金（存活）+ 違約回收腿（月頻網格）
  3. 公允票息 = 使 PV = 100 之票息（線性求解）

自我驗證：
  - 以 bootstrap 出的 λ 重新計算 CDS par spread，須還原市場報價
  - credit triangle（λ ≈ s/(1−R)）對照
  - 健全性：公允票息 ≈ swap 利率 + 發行人資金利差 + CDS 利差

把「使用者輸入」區塊改成你的條件後直接執行：python3 cln_pricer.py
"""

import numpy as np
from scipy.optimize import brentq

# ══════════════════════════════════════════════════════════════════════════════
# 使用者輸入（範例條件 — 請改成實際交易）
# ══════════════════════════════════════════════════════════════════════════════

CLN = dict(
    label        = "5 年期單名 CLN 範例",
    T            = 5.0,          # 天期（年）
    coupon       = 0.058,        # 年化固定票息；None = 求公允票息
    coupon_freq  = 4,            # 每年付息次數
    recovery     = 0.40,         # 回收率假設（senior unsecured 慣例 40%）
    fund_spread  = 0.0080,       # 發行人資金利差（加在折現曲線上）
    # 無風險/swap 零利率曲線（連續複利近似）：[(年期, 利率)]；單點 = 平坦
    disc_curve   = [(1.0, 0.042), (2.0, 0.0405), (3.0, 0.040), (5.0, 0.039),
                    (7.0, 0.039)],
    # 參考實體 CDS 利差曲線：[(年期, 利差)]；單點 = 平坦
    cds_curve    = [(1.0, 0.0080), (3.0, 0.0120), (5.0, 0.0150)],
)

CDS_PREM_FREQ = 4      # CDS 保費季付（市場慣例）
PROT_GRID_YR  = 12     # 保護腿/回收腿積分網格（月頻）


# ══════════════════════════════════════════════════════════════════════════════
# 折現曲線：log-DF 線性內插（= 分段平坦遠期），外插用最後一段遠期
# ══════════════════════════════════════════════════════════════════════════════

class Curve:
    def __init__(self, pillars, spread=0.0):
        pillars = sorted(pillars)
        self.t  = np.array([p[0] for p in pillars], dtype=float)
        self.zt = np.array([(p[1] + spread) * p[0] for p in pillars])  # z(t)·t
        if self.t[0] > 0:                                # 補 t=0 錨點
            self.t  = np.insert(self.t, 0, 0.0)
            self.zt = np.insert(self.zt, 0, 0.0)

    def df(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        zt = np.interp(t, self.t, self.zt)
        beyond = t > self.t[-1]                          # 平坦遠期外插
        if beyond.any():
            fwd = (self.zt[-1] - self.zt[-2]) / (self.t[-1] - self.t[-2])
            zt[beyond] = self.zt[-1] + fwd * (t[beyond] - self.t[-1])
        out = np.exp(-zt)
        return out if out.size > 1 else float(out[0])


# ══════════════════════════════════════════════════════════════════════════════
# 存活曲線與 CDS bootstrap
# ══════════════════════════════════════════════════════════════════════════════

class SurvivalCurve:
    """piecewise-constant hazard：pillar 年期 T_k，區間強度 λ_k。"""
    def __init__(self, pillar_t, hazards):
        self.pt = np.asarray(pillar_t, dtype=float)
        self.lam = np.asarray(hazards, dtype=float)

    def Q(self, t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        lo = np.concatenate([[0.0], self.pt[:-1]])
        cum = np.zeros_like(t)
        for a, b, l in zip(lo, self.pt, self.lam):
            cum += l * np.clip(t - a, 0.0, b - a)
        cum += self.lam[-1] * np.clip(t - self.pt[-1], 0.0, None)  # 外插
        out = np.exp(-cum)
        return out if out.size > 1 else float(out[0])


def cds_legs(T, disc, surv, recovery, prem_freq=CDS_PREM_FREQ,
             grid_yr=PROT_GRID_YR):
    """回傳 (風險年金 annuity, 保護腿 PV)（每 1 名目、每 1 利差）。"""
    tp = np.arange(1, int(round(T * prem_freq)) + 1) / prem_freq
    dt = np.diff(np.concatenate([[0.0], tp]))
    Qp = np.atleast_1d(surv.Q(tp))
    Dp = np.atleast_1d(disc.df(tp))
    Q0 = np.concatenate([[1.0], Qp[:-1]])
    annuity = float((dt * Dp * Qp).sum()
                    + (dt / 2 * Dp * (Q0 - Qp)).sum())     # 違約時應計保費
    tg = np.arange(1, int(round(T * grid_yr)) + 1) / grid_yr
    Qg = np.atleast_1d(surv.Q(tg))
    Qg0 = np.concatenate([[1.0], Qg[:-1]])
    Dmid = np.atleast_1d(disc.df(tg - 0.5 / grid_yr))
    protection = float((1 - recovery) * (Dmid * (Qg0 - Qg)).sum())
    return annuity, protection


def bootstrap_hazards(cds_curve, disc, recovery):
    """逐天期解 λ_k 使 CDS par spread = 市場報價。"""
    pillar_t = [q[0] for q in cds_curve]
    hazards = []
    for k, (Tk, sk) in enumerate(cds_curve):
        def objective(lam):
            sc = SurvivalCurve(pillar_t[:k + 1], hazards + [lam])
            ann, prot = cds_legs(Tk, disc, sc, recovery)
            return sk * ann - prot
        hazards.append(brentq(objective, 1e-8, 3.0, xtol=1e-12))
    return SurvivalCurve(pillar_t, hazards)


# ══════════════════════════════════════════════════════════════════════════════
# CLN 定價
# ══════════════════════════════════════════════════════════════════════════════

FACE = 100.0

def cln_price(spec, disc_fund, surv):
    T, freq, R = spec["T"], spec["coupon_freq"], spec["recovery"]
    tc = np.arange(1, int(round(T * freq)) + 1) / freq
    dt = np.diff(np.concatenate([[0.0], tc]))
    Qc = np.atleast_1d(surv.Q(tc))
    Dc = np.atleast_1d(disc_fund.df(tc))
    risky_annuity = float((dt * Dc * Qc).sum())            # 每 1 票息率
    Q0c = np.concatenate([[1.0], Qc[:-1]])
    accr_on_dflt = float((dt / 2 * Dc * (Q0c - Qc)).sum())

    principal = FACE * disc_fund.df(T) * surv.Q(T)

    tg = np.arange(1, int(round(T * PROT_GRID_YR)) + 1) / PROT_GRID_YR
    Qg = np.atleast_1d(surv.Q(tg))
    Qg0 = np.concatenate([[1.0], Qg[:-1]])
    Dmid = np.atleast_1d(disc_fund.df(tg - 0.5 / PROT_GRID_YR))
    recovery_leg = float(R * FACE * (Dmid * (Qg0 - Qg)).sum())

    ann_total = FACE * (risky_annuity + accr_on_dflt)
    fair_c = (FACE - principal - recovery_leg) / ann_total
    pv = None
    if spec["coupon"] is not None:
        pv = spec["coupon"] * ann_total + principal + recovery_leg
    return dict(pv=pv, fair_c=fair_c, principal=principal,
                recovery_leg=recovery_leg, risky_annuity=risky_annuity,
                ann_total=ann_total)


# ══════════════════════════════════════════════════════════════════════════════
# 執行
# ══════════════════════════════════════════════════════════════════════════════

SEP = "═" * 72

if __name__ == "__main__":
    spec = CLN
    disc_rf   = Curve(spec["disc_curve"])                    # bootstrap 用
    disc_fund = Curve(spec["disc_curve"], spec["fund_spread"])  # 票券折現用
    surv = bootstrap_hazards(spec["cds_curve"], disc_rf, spec["recovery"])

    print(f"\n{SEP}\n  {spec['label']}\n{SEP}")

    # ── 驗證 1：還原 CDS 市場報價 ────────────────────────────────────────
    print("  [驗證] bootstrap 還原 CDS par spread：")
    for Tk, sk in spec["cds_curve"]:
        ann, prot = cds_legs(Tk, disc_rf, surv, spec["recovery"])
        par = prot / ann
        tri = sk / (1 - spec["recovery"])                    # credit triangle
        lam_k = surv.lam[[q[0] for q in spec["cds_curve"]].index(Tk)]
        print(f"    {Tk:>4.0f}y：市場 {sk*1e4:>6.1f} bp → 模型 {par*1e4:>6.1f} bp"
              f"（誤差 {abs(par-sk)*1e4:.2e} bp）；"
              f"λ = {lam_k:.4f} vs 三角速算 {tri:.4f}")
    assert all(abs(cds_legs(Tk, disc_rf, surv, spec['recovery'])[1]
                   / cds_legs(Tk, disc_rf, surv, spec['recovery'])[0] - sk) < 1e-9
               for Tk, sk in spec["cds_curve"]), "bootstrap 驗證失敗"

    # ── 定價 ────────────────────────────────────────────────────────────
    res = cln_price(spec, disc_fund, surv)
    T = spec["T"]
    print(f"\n  {T:.0f} 年累積違約機率（風險中性）：{1-surv.Q(T):.2%}"
          f"，存活機率 Q(T) = {surv.Q(T):.4f}")
    print(f"  票息腿風險年金：{res['ann_total']:.3f}（每 1 票息率，含違約應計）")
    print(f"  到期本金 PV（存活）：{res['principal']:.3f}")
    print(f"  違約回收腿 PV（R={spec['recovery']:.0%}）：{res['recovery_leg']:.3f}")
    print(f"  公允票息（PV=100）：{res['fair_c']:.3%}")

    z5 = np.interp(T, [p[0] for p in spec["disc_curve"]],
                   [p[1] for p in spec["disc_curve"]])
    s5 = np.interp(T, [q[0] for q in spec["cds_curve"]],
                   [q[1] for q in spec["cds_curve"]])
    approx = z5 + spec["fund_spread"] + s5
    print(f"  [健全性] swap {z5:.2%} + 資金利差 {spec['fund_spread']:.2%} "
          f"+ CDS {s5:.2%} = {approx:.2%}（應接近公允票息）")

    if res["pv"] is not None:
        print(f"\n  報價票息 {spec['coupon']:.2%} 下之理論價值：{res['pv']:.3f}"
              f"（發行費用 ≈ {100-res['pv']:.2f} 點）")
    print(f"\n{SEP}\n  註：投資人同時承擔參考實體與發行人信用（雙重暴險）；"
          f"\n      發行人信用在此以折現利差近似。一籃子/FTD 需另用 copula 模型。\n{SEP}")
