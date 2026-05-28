"""
CMT10 Range Accrual Note — Monte Carlo Analysis
Products:
  - 2yNC1y: 2-year maturity, non-callable for 1 year
  - 3yNC1y: 3-year maturity, non-callable for 1 year

Coupon structure (quarterly, 30/360):
  10yCMT <= 5.00%  => 2y: 5.10%,  3y: 6.65%
  10yCMT <= 5.25%  => 2y: 4.45%,  3y: 5.55%
  10yCMT >  5.25%  => 0% (range accrual out-of-range)

Issuer CGMHI can call at any quarterly date after the no-call period.
"""

import numpy as np

np.random.seed(42)

# ── Market Assumptions (May 2026) ──────────────────────────────────────────────
r0     = 0.0430   # current 10y CMT ≈ 4.30%
kappa  = 0.20     # Vasicek mean-reversion speed
theta  = 0.0430   # long-run mean (neutral / current level)
sigma  = 0.0110   # annual vol of 10y CMT (~110 bps)
disc_r = 0.0430   # flat risk-free discount rate

T_HIGH = 0.0500   # 5.00% threshold
T_MID  = 0.0525   # 5.25% threshold

PRODUCTS = {
    "2yNC1y": dict(maturity=2, no_call=1, cpn_high=0.0510, cpn_mid=0.0445),
    "3yNC1y": dict(maturity=3, no_call=1, cpn_high=0.0665, cpn_mid=0.0555),
}

N_SIMS   = 200_000
STEPS_YR = 252
FACE     = 100.0


def simulate_vasicek(r0, kappa, theta, sigma, T_years, steps_yr, n_sims):
    n = int(T_years * steps_yr)
    dt = 1 / steps_yr
    r = np.empty((n_sims, n + 1))
    r[:, 0] = r0
    e        = np.exp(-kappa * dt)
    mu_inc   = theta * (1 - e)
    vol_inc  = sigma * np.sqrt((1 - e**2) / (2 * kappa))
    Z = np.random.standard_normal((n_sims, n))
    for t in range(n):
        r[:, t + 1] = r[:, t] * e + mu_inc + vol_inc * Z[:, t]
    return r


def price_product(spec, rate_matrix, disc_r, steps_yr):
    n_sims      = rate_matrix.shape[0]
    T_years     = spec["maturity"]
    no_call     = spec["no_call"]
    cpn_high    = spec["cpn_high"]
    cpn_mid     = spec["cpn_mid"]
    dt_q        = 0.25                          # quarter (30/360)
    n_qtrs      = int(T_years * 4)
    no_call_q   = int(no_call * 4)              # quarters in no-call window

    # quarter-end day indices
    q_steps = [int(round(q * steps_yr / 4)) for q in range(1, n_qtrs + 1)]
    n_cols  = rate_matrix.shape[1] - 1         # last valid index

    alive        = np.ones(n_sims, dtype=bool)
    pv_coupons   = np.zeros(n_sims)            # PV of coupon cash flows only
    total_cpn    = np.zeros(n_sims)            # undiscounted total coupon received
    eff_tenor    = np.full(n_sims, T_years)    # realised tenor in years

    for qi, step in enumerate(q_steps):
        step = min(step, n_cols)
        t_yr = (qi + 1) * dt_q
        r_q  = rate_matrix[:, step]

        # coupon rate for each path
        cpn_rate = np.where(r_q <= T_HIGH, cpn_high,
                   np.where(r_q <= T_MID,  cpn_mid, 0.0))
        cpn_cf   = cpn_rate * FACE * dt_q          # quarterly cash flow

        df = np.exp(-disc_r * t_yr)
        pv_coupons += np.where(alive, cpn_cf * df, 0.0)
        total_cpn  += np.where(alive, cpn_cf,      0.0)

        # ── Issuer call decision (after no-call window) ──
        if qi >= no_call_q:
            remaining = n_qtrs - qi - 1
            if remaining > 0:
                # Issuer's next-quarter coupon cost vs funding cost
                next_cpn_rate = np.where(r_q <= T_HIGH, cpn_high,
                                np.where(r_q <= T_MID,  cpn_mid, 0.0))
                next_cpn_cost = next_cpn_rate * FACE * dt_q
                funding_cost  = disc_r * FACE * dt_q
                # Call if it saves the issuer money (coupon > funding rate)
                will_call = alive & (next_cpn_cost > funding_cost)

                # Investor receives principal on call
                pv_coupons += np.where(will_call, FACE * df, 0.0)
                eff_tenor   = np.where(will_call, t_yr, eff_tenor)
                alive       = alive & ~will_call

    # ── Maturity: return principal for surviving paths ──
    t_mat = T_years
    df_mat = np.exp(-disc_r * t_mat)
    pv_coupons += np.where(alive, FACE * df_mat, 0.0)

    # ── Metrics ────────────────────────────────────────────────────────────────
    # Annualized coupon yield = total coupons received / (face * effective tenor)
    annual_yield = total_cpn / (FACE * eff_tenor)
    call_prob    = np.mean(~alive | (eff_tenor < T_years))

    return dict(
        price_mean         = np.mean(pv_coupons),
        price_5pct         = np.percentile(pv_coupons, 5),
        price_95pct        = np.percentile(pv_coupons, 95),
        annual_yield_mean  = np.mean(annual_yield),
        annual_yield_std   = np.std(annual_yield),
        annual_yield_5pct  = np.percentile(annual_yield, 5),
        annual_yield_95pct = np.percentile(annual_yield, 95),
        call_probability   = np.mean(eff_tenor < T_years),
        avg_tenor_yrs      = np.mean(eff_tenor),
    )


def stress_yield(r0_s, theta_s, label, n_sims=50_000):
    rm = simulate_vasicek(r0_s, kappa, theta_s, sigma * 0.7, 3, STEPS_YR, n_sims)
    row = {"Scenario": label}
    for name, spec in PRODUCTS.items():
        cut = rm[:, : int(spec["maturity"] * STEPS_YR) + 1]
        res = price_product(spec, cut, disc_r, STEPS_YR)
        row[name] = res["annual_yield_mean"]
    return row


def run():
    print("=" * 70)
    print("  CMT10 Range Accrual Note — Monte Carlo Analysis")
    print(f"  {N_SIMS:,} simulations | Vasicek model")
    print(f"  r0 = {r0:.2%}  |  κ = {kappa}  |  θ = {theta:.2%}  |  σ = {sigma:.2%}")
    print("=" * 70)

    print(f"\nSimulating {N_SIMS:,} rate paths...")
    rm3 = simulate_vasicek(r0, kappa, theta, sigma, 3, STEPS_YR, N_SIMS)
    rm2 = rm3[:, : int(2 * STEPS_YR) + 1]
    rate_mats = {"2yNC1y": rm2, "3yNC1y": rm3}

    # ── Rate distribution ─────────────────────────────────────────────────────
    print("\n── 10y CMT Rate Distribution (simulated) ──────────────────────────")
    for T_yr in [1, 2, 3]:
        r_T = rm3[:, int(T_yr * STEPS_YR)]
        print(f"  {T_yr}y:  mean {np.mean(r_T):.2%}  "
              f"| 5th {np.percentile(r_T,5):.2%}  "
              f"| 95th {np.percentile(r_T,95):.2%}  "
              f"| P(>5.25%) {np.mean(r_T > T_MID):.1%}")

    # ── Coupon scenario probs at first coupon (3 months out) ──────────────────
    r1q = rm3[:, int(STEPS_YR / 4)]
    p_hi = np.mean(r1q <= T_HIGH)
    p_md = np.mean((r1q > T_HIGH) & (r1q <= T_MID))
    p_zr = np.mean(r1q > T_MID)
    print("\n── Coupon Scenario Probabilities (first quarter) ───────────────────")
    print(f"  10yCMT ≤ 5.00%  → high coupon tier  :  {p_hi:.1%}")
    print(f"  5.00% < CMT ≤ 5.25% → mid tier      :  {p_md:.1%}")
    print(f"  10yCMT > 5.25%  → zero coupon       :  {p_zr:.1%}")

    # ── Product-level results ─────────────────────────────────────────────────
    results = {}
    print("\n── Product Results ──────────────────────────────────────────────────")
    for name, spec in PRODUCTS.items():
        res = price_product(spec, rate_mats[name], disc_r, STEPS_YR)
        results[name] = res
        print(f"\n  [{name}]  Maturity {spec['maturity']}y | No-call {spec['no_call']}y")
        print(f"  Coupons: {spec['cpn_high']:.2%} (hi) / {spec['cpn_mid']:.2%} (mid) / 0% (out)")
        print(f"  Simulated Price (PV/100):  {res['price_mean']:.3f}"
              f"  [5th: {res['price_5pct']:.2f}  95th: {res['price_95pct']:.2f}]")
        print(f"  Expected Annual Yield:     {res['annual_yield_mean']:.2%}"
              f"  ± {res['annual_yield_std']:.2%}"
              f"  [5th: {res['annual_yield_5pct']:.2%}  95th: {res['annual_yield_95pct']:.2%}]")
        print(f"  Issuer Call Probability:   {res['call_probability']:.1%}")
        print(f"  Avg Effective Tenor:       {res['avg_tenor_yrs']:.2f} yrs")

    # ── Head-to-head comparison ───────────────────────────────────────────────
    r2, r3 = results["2yNC1y"], results["3yNC1y"]
    yield_pickup = r3["annual_yield_mean"] - r2["annual_yield_mean"]
    sr2 = r2["annual_yield_mean"] / r2["annual_yield_std"]
    sr3 = r3["annual_yield_mean"] / r3["annual_yield_std"]

    print("\n── Head-to-Head Comparison ──────────────────────────────────────────")
    print(f"  Yield pickup  3y over 2y   :  +{yield_pickup:.2%} p.a.")
    print(f"  Yield/Risk (Sharpe proxy)  :  2yNC1y {sr2:.2f}  |  3yNC1y {sr3:.2f}")

    # ── Stress scenarios ──────────────────────────────────────────────────────
    print("\n── Stress Scenario — Expected Annual Yield ──────────────────────────")
    print(f"  {'Scenario':<35} {'2yNC1y':>9} {'3yNC1y':>9}")
    print(f"  {'─'*35} {'─'*9} {'─'*9}")
    scenarios = [
        ("Bull: rates fall -100bps",     r0 - 0.01, theta - 0.01),
        ("Base: rates unchanged",        r0,        theta),
        ("Bear: rates rise +50bps",      r0 + 0.005, theta + 0.005),
        ("Shock: rates rise +100bps",    r0 + 0.01, theta + 0.01),
        ("Severe: rates rise +150bps",   r0 + 0.015, theta + 0.015),
    ]
    for label, r0_s, th_s in scenarios:
        row = stress_yield(r0_s, th_s, label)
        print(f"  {label:<35} {row['2yNC1y']:>8.2%}  {row['3yNC1y']:>8.2%}")

    # ── Recommendation ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RECOMMENDATION")
    print("=" * 70)

    if sr3 >= sr2:
        winner = "3yNC1y"
        print(f"\n  ★  3yNC1y is the MORE ATTRACTIVE product")
        print(f"\n  • Yield pickup:  +{yield_pickup:.2%} p.a. over 2yNC1y")
        print(f"  • Better Yield/Risk ratio: {sr3:.2f} vs {sr2:.2f}")
        print(f"  • High coupon {PRODUCTS['3yNC1y']['cpn_high']:.2%} is earned {p_hi:.0%} of the time in base case")
    else:
        winner = "2yNC1y"
        print(f"\n  ★  2yNC1y is the MORE ATTRACTIVE product (better risk-adjusted return)")
        print(f"  • Yield/Risk ratio: {sr2:.2f} vs {sr3:.2f}")

    print(f"\n  Supporting factors (base case, r0 = {r0:.2%}):")
    print(f"  • 10yCMT needs to rise {T_MID - r0:.0%} to reach 5.25% zero-coupon barrier")
    print(f"  • {p_hi:.0%} probability of earning the highest coupon tier next quarter")
    print(f"  • Both notes trade ~par; 3y priced slightly richer (PV {r3['price_mean']:.2f})")

    print(f"\n  Key risks:")
    print(f"  • Rate shock > +{T_MID - r0:.0%}: coupon drops to zero (worst case)")
    print(f"  • Issuer call risk: CGMHI likely calls early if rates fall")
    print(f"    (limits upside; avg effective tenor only {r3['avg_tenor_yrs']:.1f}y for 3y note)")
    print(f"  • Credit risk: CGMHI (Citigroup) — investment grade but not govt")
    print("=" * 70)


if __name__ == "__main__":
    run()
