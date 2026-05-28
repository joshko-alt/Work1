"""
CMT10 Range Accrual Note — Fair Coupon Analysis
Solves for the theoretical fair coupon (PV = 100) under four structures:
  1. Plain vanilla bullet     (no RA, no call)   — baseline
  2. Plain vanilla callable   (no RA, with call)  — call cost only
  3. Range accrual bullet     (RA,  no call)      — RA cost only
  4. Full structure           (RA + call)         — as quoted

Bisection on coupon scale until simulated PV = par (100).
"""

import numpy as np

np.random.seed(42)

# ── Parameters (same as base analysis) ────────────────────────────────────────
r0     = 0.0430
kappa  = 0.20
theta  = 0.0430
sigma  = 0.0110
disc_r = 0.0430

T_HIGH = 0.0500
T_MID  = 0.0525

PRODUCTS = {
    "2yNC1y": dict(maturity=2, no_call=1, cpn_high=0.0510, cpn_mid=0.0445),
    "3yNC1y": dict(maturity=3, no_call=1, cpn_high=0.0665, cpn_mid=0.0555),
}

N_SIMS   = 300_000
STEPS_YR = 252
FACE     = 100.0


def simulate_vasicek(r0, kappa, theta, sigma, T_years, steps_yr, n_sims):
    n  = int(T_years * steps_yr)
    dt = 1 / steps_yr
    r  = np.empty((n_sims, n + 1))
    r[:, 0] = r0
    e       = np.exp(-kappa * dt)
    mu      = theta * (1 - e)
    vol     = sigma * np.sqrt((1 - e ** 2) / (2 * kappa))
    Z = np.random.standard_normal((n_sims, n))
    for t in range(n):
        r[:, t + 1] = r[:, t] * e + mu + vol * Z[:, t]
    return r


def compute_pv(cpn_high, spec, rate_matrix, disc_r, steps_yr,
               cpn_ratio=1.0, use_ra=True, use_call=True):
    """
    cpn_high  : high-tier annual coupon rate
    cpn_ratio : cpn_mid / cpn_high  (ignored when use_ra=False)
    use_ra    : apply range-accrual barriers
    use_call  : allow issuer to call after no-call period
    Returns simulated mean PV across all paths.
    """
    n_sims   = rate_matrix.shape[0]
    T_years  = spec["maturity"]
    no_call  = spec["no_call"]
    dt_q     = 0.25
    n_qtrs   = int(T_years * 4)
    no_call_q= int(no_call * 4)
    q_steps  = [int(round(q * steps_yr / 4)) for q in range(1, n_qtrs + 1)]
    n_cols   = rate_matrix.shape[1] - 1

    alive = np.ones(n_sims, dtype=bool)
    pv    = np.zeros(n_sims)

    for qi, step in enumerate(q_steps):
        step = min(step, n_cols)
        t_yr = (qi + 1) * dt_q
        r_q  = rate_matrix[:, step]
        df   = np.exp(-disc_r * t_yr)

        # ── Coupon rate for this period ──
        if use_ra:
            cpn_mid = cpn_high * cpn_ratio
            rate = np.where(r_q <= T_HIGH, cpn_high,
                   np.where(r_q <= T_MID,  cpn_mid, 0.0))
        else:
            rate = np.full(n_sims, cpn_high)   # always pays, flat coupon

        cpn_cf = rate * FACE * dt_q
        pv    += np.where(alive, cpn_cf * df, 0.0)

        # ── Issuer call ──
        # Call when fixed coupon > stochastic funding rate (issuer can reissue cheaper)
        if use_call and qi >= no_call_q:
            remaining = n_qtrs - qi - 1
            if remaining > 0:
                if use_ra:
                    next_rate = np.where(r_q <= T_HIGH, cpn_high,
                                np.where(r_q <= T_MID,  cpn_high * cpn_ratio, 0.0))
                else:
                    next_rate = np.full(n_sims, cpn_high)

                next_cost    = next_rate * FACE * dt_q
                funding_cost = r_q * FACE * dt_q   # stochastic: call when rate has fallen
                will_call    = alive & (next_cost > funding_cost)
                pv          += np.where(will_call, FACE * df, 0.0)
                alive        = alive & ~will_call

    # Principal at maturity
    df_mat = np.exp(-disc_r * T_years)
    pv    += np.where(alive, FACE * df_mat, 0.0)
    return float(np.mean(pv))


def bisect_fair_coupon(spec, rate_matrix, disc_r, steps_yr,
                       cpn_ratio=1.0, use_ra=True, use_call=True,
                       lo=0.0001, hi=0.40, tol=1e-7):
    """Binary search for coupon_high s.t. PV(note) = 100."""
    for _ in range(80):
        mid = (lo + hi) / 2
        pv  = compute_pv(mid, spec, rate_matrix, disc_r, steps_yr,
                         cpn_ratio, use_ra, use_call)
        if pv > FACE:
            hi = mid   # coupon too high → PV > par → reduce it
        else:
            lo = mid   # coupon too low  → PV < par → raise it
    return (lo + hi) / 2


def run():
    print("=" * 72)
    print("  CMT10 Range Accrual Note — Fair Coupon Decomposition")
    print(f"  {N_SIMS:,} simulations | r0={r0:.2%} | κ={kappa} | θ={theta:.2%} | σ={sigma:.2%}")
    print("=" * 72)

    print(f"\nSimulating {N_SIMS:,} rate paths …")
    rm3 = simulate_vasicek(r0, kappa, theta, sigma, 3, STEPS_YR, N_SIMS)
    rm2 = rm3[:, : int(2 * STEPS_YR) + 1]
    rate_mats = {"2yNC1y": rm2, "3yNC1y": rm3}

    # Pre-compute coupon ratios from quoted product
    ratios = {
        name: spec["cpn_mid"] / spec["cpn_high"]
        for name, spec in PRODUCTS.items()
    }

    print(f"\nQuoted coupon ratios (mid/high):")
    for name, ratio in ratios.items():
        print(f"  {name}: {PRODUCTS[name]['cpn_mid']:.2%} / "
              f"{PRODUCTS[name]['cpn_high']:.2%} = {ratio:.4f}")

    # ── Four structures × two products ────────────────────────────────────────
    structures = [
        ("① Plain vanilla bullet",   False, False),
        ("② Callable bullet",        False, True ),
        ("③ Range accrual bullet",   True,  False),
        ("④ Full (RA + callable)",   True,  True ),
    ]

    results = {}
    print("\n" + "─" * 72)
    print(f"  {'Structure':<30} {'2yNC1y Fair Cpn':>18} {'3yNC1y Fair Cpn':>18}")
    print("─" * 72)

    for label, use_ra, use_call in structures:
        row = {}
        for name, spec in PRODUCTS.items():
            rm   = rate_mats[name]
            r    = ratios[name] if use_ra else 1.0
            fair = bisect_fair_coupon(spec, rm, disc_r, STEPS_YR,
                                      cpn_ratio=r, use_ra=use_ra, use_call=use_call)
            row[name] = fair
        results[label] = row
        print(f"  {label:<30} {row['2yNC1y']:>17.4%}  {row['3yNC1y']:>17.4%}")

    # ── Premium decomposition ─────────────────────────────────────────────────
    base_2y = results["① Plain vanilla bullet"]["2yNC1y"]
    base_3y = results["① Plain vanilla bullet"]["3yNC1y"]

    call_prem_2y = results["② Callable bullet"]["2yNC1y"] - base_2y
    call_prem_3y = results["② Callable bullet"]["3yNC1y"] - base_3y

    ra_prem_2y   = results["③ Range accrual bullet"]["2yNC1y"] - base_2y
    ra_prem_3y   = results["③ Range accrual bullet"]["3yNC1y"] - base_3y

    full_2y = results["④ Full (RA + callable)"]["2yNC1y"]
    full_3y = results["④ Full (RA + callable)"]["3yNC1y"]

    quoted_2y = PRODUCTS["2yNC1y"]["cpn_high"]
    quoted_3y = PRODUCTS["3yNC1y"]["cpn_high"]

    surplus_2y = quoted_2y - full_2y
    surplus_3y = quoted_3y - full_3y

    print("\n── Coupon Premium Decomposition ─────────────────────────────────────")
    print(f"  {'Component':<35} {'2yNC1y':>10} {'3yNC1y':>10}")
    print(f"  {'─'*35} {'─'*10} {'─'*10}")
    print(f"  {'Baseline (plain vanilla bullet)':<35} {base_2y:>9.4%}  {base_3y:>9.4%}")
    print(f"  {'+ Callable premium (issuer option)':<35} {call_prem_2y:>+9.4%}  {call_prem_3y:>+9.4%}")
    print(f"  {'+ Range accrual premium (RA risk)':<35} {ra_prem_2y:>+9.4%}  {ra_prem_3y:>+9.4%}")
    print(f"  {'─'*35} {'─'*10} {'─'*10}")
    print(f"  {'= Theoretical fair coupon (full)':<35} {full_2y:>9.4%}  {full_3y:>9.4%}")
    print(f"  {'  Quoted coupon (high tier)':<35} {quoted_2y:>9.4%}  {quoted_3y:>9.4%}")
    print(f"  {'─'*35} {'─'*10} {'─'*10}")
    surplus_label_2y = "excess" if surplus_2y > 0 else "deficit"
    surplus_label_3y = "excess" if surplus_3y > 0 else "deficit"
    print(f"  {'  Investor surplus (+) / deficit (−)':<35} {surplus_2y:>+9.4%}  {surplus_3y:>+9.4%}")

    # ── PV verification ───────────────────────────────────────────────────────
    pv_fair_2y  = compute_pv(full_2y,    PRODUCTS["2yNC1y"], rm2, disc_r, STEPS_YR,
                              cpn_ratio=ratios["2yNC1y"], use_ra=True, use_call=True)
    pv_quoted_2y= compute_pv(quoted_2y,  PRODUCTS["2yNC1y"], rm2, disc_r, STEPS_YR,
                              cpn_ratio=ratios["2yNC1y"], use_ra=True, use_call=True)
    pv_fair_3y  = compute_pv(full_3y,    PRODUCTS["3yNC1y"], rm3, disc_r, STEPS_YR,
                              cpn_ratio=ratios["3yNC1y"], use_ra=True, use_call=True)
    pv_quoted_3y= compute_pv(quoted_3y,  PRODUCTS["3yNC1y"], rm3, disc_r, STEPS_YR,
                              cpn_ratio=ratios["3yNC1y"], use_ra=True, use_call=True)

    print("\n── PV Verification (should be ≈100 for fair; > 100 = investor gains) ─")
    print(f"  {'Product':<15} {'Fair PV':>12} {'Quoted PV':>12} {'Investor NPV':>14}")
    print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*14}")
    print(f"  {'2yNC1y':<15} {pv_fair_2y:>11.4f}  {pv_quoted_2y:>11.4f}  {pv_quoted_2y-100:>+13.4f}")
    print(f"  {'3yNC1y':<15} {pv_fair_3y:>11.4f}  {pv_quoted_3y:>11.4f}  {pv_quoted_3y-100:>+13.4f}")

    # ── Sensitivity: fair coupon at different r0 levels ───────────────────────
    print("\n── Fair Coupon Sensitivity to Starting Rate (r0) ───────────────────")
    print(f"  {'r0':<8} {'2y fair':>10} {'2y quoted':>10} {'surplus':>10}"
          f"  {'3y fair':>10} {'3y quoted':>10} {'surplus':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10}  {'─'*10} {'─'*10} {'─'*10}")

    for r0_test in [0.030, 0.035, 0.040, 0.043, 0.045, 0.050, 0.055, 0.060]:
        rm_t3 = simulate_vasicek(r0_test, kappa, r0_test, sigma, 3, STEPS_YR, 80_000)
        rm_t2 = rm_t3[:, : int(2 * STEPS_YR) + 1]
        f2 = bisect_fair_coupon(PRODUCTS["2yNC1y"], rm_t2, disc_r, STEPS_YR,
                                cpn_ratio=ratios["2yNC1y"], use_ra=True, use_call=True)
        f3 = bisect_fair_coupon(PRODUCTS["3yNC1y"], rm_t3, disc_r, STEPS_YR,
                                cpn_ratio=ratios["3yNC1y"], use_ra=True, use_call=True)
        mark = " ← current" if abs(r0_test - 0.043) < 0.001 else ""
        print(f"  {r0_test:.2%}    {f2:>9.3%}  {quoted_2y:>9.3%}  {quoted_2y-f2:>+9.3%}"
              f"   {f3:>9.3%}  {quoted_3y:>9.3%}  {quoted_3y-f3:>+9.3%}{mark}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  INTERPRETATION")
    print("=" * 72)
    print(f"""
  理論公允票息分解（以當前 r0 = {r0:.2%} 為基準）：

  基準票息（plain vanilla bullet）：
    2yNC1y ≈ {base_2y:.3%}  |  3yNC1y ≈ {base_3y:.3%}
    → 這是在平坦 {disc_r:.2%} 折現率下，無任何結構特徵的公允票息。
      因 PV(本金) 已貼現，票息需略高於 disc_r 才能使 PV = 100。

  Callable 溢價（發行人選擇權成本）：
    2yNC1y +{call_prem_2y:.3%}  |  3yNC1y +{call_prem_3y:.3%}
    → 投資人出售了一個 Bermudan Call Option 給發行人，
      必須要求更高票息作為補償。3y 期限較長，溢價略高。

  Range Accrual 溢價（零息障礙風險成本）：
    2yNC1y +{ra_prem_2y:.3%}  |  3yNC1y +{ra_prem_3y:.3%}
    → 當 CMT > 5.25% 時票息歸零，投資人承擔這個跳躍風險，
      需要更高票息補償。
    → 3y 產品暴露時間更長，RA 溢價也應更高（但模型因
      Callable 提前贖回使暴露期縮短，兩者差距不大）。

  全結構公允票息 = 以上合計：
    2yNC1y 理論公允 {full_2y:.4%}  vs 報價 {quoted_2y:.2%}  → 投資人{("超額收益" if surplus_2y > 0 else "讓利")} {abs(surplus_2y):.3%}
    3yNC1y 理論公允 {full_3y:.4%}  vs 報價 {quoted_3y:.2%}  → 投資人{("超額收益" if surplus_3y > 0 else "讓利")} {abs(surplus_3y):.3%}
""")

    if surplus_2y > 0 and surplus_3y > 0:
        winner_idx = "3yNC1y" if surplus_3y > surplus_2y else "2yNC1y"
        print(f"  ★ 兩個產品的報價票息均高於理論公允票息（對投資人有利）。")
        print(f"    絕對超額收益較大者：{winner_idx}（+{max(surplus_2y,surplus_3y):.3%}）")
    elif surplus_2y < 0 and surplus_3y < 0:
        print(f"  ⚠ 兩個產品的報價票息均低於理論公允票息（投資人讓利給發行人）。")
    else:
        winner = "2yNC1y" if surplus_2y > surplus_3y else "3yNC1y"
        print(f"  ★ {winner} 的報價對投資人更有利。")

    print("\n  重要限制：公允票息高度依賴模型假設（尤其是 θ 和 σ）。")
    print("  敏感度表格顯示隨 r0 上升，公允票息急速升高（Barrier 越來越近），")
    print("  而報價票息固定，因此在高利率環境下產品快速轉為對發行人有利。")
    print("=" * 72)


if __name__ == "__main__":
    run()
