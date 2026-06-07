"""
MBS Convexity Hedging Backtest
==============================
Context: "Awakening the MBS Convexity Beast" — Harley Bassman (MOVE Index creator)
  - Goldman Sachs: ~USD 40B 10Y-TSY-equivalent hedging needed post-selloff
  - Barclays: 5%+ coupon MBS > USD 2T (4x vs 3 years ago)
  - Bassman: ~1/3 of outstanding MBS near par → peak convexity sensitivity zone

Model:
  - Pool WAC = 7.0% (post-2022 high-coupon vintage; typical 2023-2024 origination)
  - Mortgage rate = 10Y CMT + 250bp spread (historical mortgage basis)
  - At current 10Y = 4.3%, mortgage = 6.8%; WAC 7.0% > 6.8% → bond ~101 (near par)
  - CPR: smooth logistic; inflection at +100bp refinancing incentive; floor 3.5%/yr
  - Effective duration: accounts for option-embedded prepayment asymmetry
  - Hedge DV01 expressed as 10Y TSY face equivalent
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

WAC        = 0.070   # 7.0% WAC — high-coupon 2022-2024 pool
SPREAD     = 0.025   # 250bp mortgage-over-10Y-CMT spread
FACE       = 100.0
POOL_B     = 2_000.0 # pool size in USD billions (Barclays $2T estimate)

# Vasicek params (consistent with cmt10_monte_carlo.py)
KAPPA  = 0.20
THETA  = 0.043
SIGMA  = 0.011   # 110bp annualised vol
R0     = 0.043   # starting 10Y CMT
DT     = 1/252
N_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# 1. SMOOTH PREPAYMENT MODEL
# ─────────────────────────────────────────────────────────────────────────────

# Logistic parameters (all in basis-points space)
_CPR_BASE  = 0.035   # 3.5% floor (housing turnover; borrowers who don't refi)
_CPR_RANGE = 0.345   # range up to 38% at deep in-the-money
_CPR_K     = 100     # inflection incentive (bps) — refi break-even cost
_CPR_SIGMA = 60      # logistic width (bps)


def cpr_model(wac: float, tenY: float) -> float:
    """
    Annual CPR as smooth logistic of refinancing incentive.
    incentive_bps = (WAC − mortgage_rate) in bps.
    Positive incentive → borrow can save → CPR rises.
    """
    incentive_bps = (wac - (tenY + SPREAD)) * 10_000
    cpr = (_CPR_BASE
           + _CPR_RANGE / (1 + np.exp(-(incentive_bps - _CPR_K) / _CPR_SIGMA)))
    return float(np.clip(cpr, 0.005, 0.45))


def smm(cpr: float) -> float:
    """Single Monthly Mortality."""
    return 1 - (1 - cpr) ** (1 / 12)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MBS PRICING
# ─────────────────────────────────────────────────────────────────────────────

def _pv_mbs(wac: float, face: float, tenY: float, n_months: int = 360) -> float:
    """Discounted cash flows; discount rate = mortgage rate = tenY + SPREAD."""
    mort     = tenY + SPREAD
    m_wac    = wac / 12
    m_disc   = mort / 12
    c        = cpr_model(wac, tenY)
    s        = smm(c)
    balance  = face
    pv       = 0.0
    for t in range(1, n_months + 1):
        if balance < 1e-8:
            break
        n_rem = n_months - t + 1
        intr  = balance * m_wac
        if m_wac > 0 and n_rem > 0:
            sched = balance * m_wac / (1 - (1 + m_wac) ** (-n_rem))
        else:
            sched = balance
        sp    = max(sched - intr, 0.0)
        prep  = (balance - sp) * s
        cf    = intr + sp + prep
        pv   += cf / (1 + m_disc) ** t
        balance -= sp + prep
    return pv


def mbs_px(wac: float, face: float, tenY: float) -> float:
    return _pv_mbs(wac, face, tenY) / face * 100.0


def eff_dur_conv(wac: float, face: float, tenY: float, dr: float = 0.0025):
    """Effective duration/convexity via symmetric finite difference."""
    p0 = mbs_px(wac, face, tenY)
    pu = mbs_px(wac, face, tenY + dr)
    pd = mbs_px(wac, face, tenY - dr)
    d  = (pd - pu) / (2 * p0 * dr)
    c  = (pu + pd - 2 * p0) / (p0 * dr ** 2)
    return d, c, p0


# ─────────────────────────────────────────────────────────────────────────────
# 3. BENCHMARK: OPTION-FREE 10Y TREASURY
# ─────────────────────────────────────────────────────────────────────────────

def tsy_px(coupon: float, face: float, ytm: float, n: int = 20) -> float:
    c  = coupon / 2 * face
    pv = sum(c / (1 + ytm / 2) ** t for t in range(1, n + 1))
    pv += face / (1 + ytm / 2) ** n
    return pv / face * 100.0


def tsy_mod_dur(coupon: float, face: float, ytm: float, n: int = 20) -> float:
    p   = tsy_px(coupon, face, ytm, n) / 100 * face
    c   = coupon / 2 * face
    mac = sum((t / 2) * c / (1 + ytm / 2) ** t for t in range(1, n + 1))
    mac += (n / 2) * face / (1 + ytm / 2) ** n
    mac /= p
    return mac / (1 + ytm / 2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROFILE CURVES
# ─────────────────────────────────────────────────────────────────────────────

tenY_range = np.arange(0.015, 0.085, 0.002)

mbs_durs, mbs_convs, mbs_pxs = [], [], []
tsy_durs, tsy_pxs             = [], []

for r in tenY_range:
    d, c, p = eff_dur_conv(WAC, FACE, r)
    mbs_durs.append(d);  mbs_convs.append(c);  mbs_pxs.append(p)
    mbs_par_coupon = r + SPREAD          # par coupon ≈ mortgage rate at this 10Y
    td  = tsy_mod_dur(r, FACE, r)
    tp  = tsy_px(r, FACE, r)
    tsy_durs.append(td);  tsy_pxs.append(tp)

mbs_durs  = np.array(mbs_durs)
mbs_convs = np.array(mbs_convs)
mbs_pxs   = np.array(mbs_pxs)
tsy_durs  = np.array(tsy_durs)
tsy_pxs   = np.array(tsy_pxs)

near_par = (mbs_pxs >= 98.0) & (mbs_pxs <= 102.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. VASICEK SIMULATION + DAILY HEDGING FLOWS
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)

scenarios = {
    'Baseline (Mean-Reverting)':  {'extra_drift': 0.000,  'col': '#2196F3'},
    'Rate Spike (+100bps)':       {'extra_drift': +0.010, 'col': '#F44336'},
    'Rate Rally (-80bps)':        {'extra_drift': -0.008, 'col': '#4CAF50'},
}

sim = {}
for name, cfg in scenarios.items():
    path = np.zeros(N_DAYS); path[0] = R0
    for i in range(1, N_DAYS):
        dW       = np.random.normal(0, np.sqrt(DT))
        path[i]  = (path[i-1]
                    + KAPPA * (THETA - path[i-1]) * DT
                    + cfg['extra_drift'] * DT
                    + SIGMA * dW)
        path[i]  = max(path[i], 0.005)

    dur_arr = np.array([eff_dur_conv(WAC, FACE, r)[0] for r in path])
    px_arr  = np.array([mbs_px(WAC, FACE, r)         for r in path])

    # DV01-based hedge flow in USD billions of 10Y TSY face
    # When duration extends by Δd, extra MBS DV01 = Δd × pool_B × 0.0001
    # To offset: short face_tsy of 10Y TSY where DV01 = tsy_dur × face_tsy × 0.0001
    tsy_dur_base = tsy_mod_dur(R0, FACE, R0)
    delta_dur    = np.diff(dur_arr)
    delta_r      = np.diff(path)
    # Hedge in 10Y TSY face (USD billions): positive = need to short more TSY
    hedge_flow   = delta_dur * POOL_B / tsy_dur_base
    cum_hedge    = np.cumsum(hedge_flow)

    # Mark-to-market P&L
    unhedged = px_arr - px_arr[0]                 # pure MBS price change (% pts)

    # Delta-hedged P&L (holds MBS + short 10Y TSY each day)
    hedged = np.zeros(N_DAYS)
    for i in range(1, N_DAYS):
        td         = tsy_mod_dur(path[i-1], FACE, path[i-1])
        dr         = path[i] - path[i-1]
        mbs_chg    = -dur_arr[i-1] * dr * 100       # % pts
        tsy_chg    = +dur_arr[i-1] * dr * 100       # offsetting short
        hedged[i]  = hedged[i-1] + mbs_chg + tsy_chg

    sim[name] = dict(path=path, dur=dur_arr, dr=delta_r, hf=hedge_flow,
                     cum=cum_hedge, ux=unhedged, hx=hedged, col=cfg['col'])


# ─────────────────────────────────────────────────────────────────────────────
# 6. PRO-CYCLICALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

bl       = sim['Baseline (Mean-Reverting)']
corr_val = np.corrcoef(bl['dr'] * 10000, bl['hf'])[0, 1]
slope    = np.polyfit(bl['dr'] * 10000, bl['hf'], 1)[0]   # USD bn / bp
daily_v  = SIGMA * np.sqrt(DT) * 10000                    # 1σ daily rate move, bps


# ─────────────────────────────────────────────────────────────────────────────
# 7. GOLDMAN USD 40B VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

r_sh       = R0 + 0.005   # +50bp shock
d0, _, p0  = eff_dur_conv(WAC, FACE, R0)
ds, _, ps  = eff_dur_conv(WAC, FACE, r_sh)
delta_d    = ds - d0
tsy_d_base = tsy_mod_dur(R0, FACE, R0)
hedge_B    = delta_d * POOL_B / tsy_d_base    # USD bn face of 10Y TSY


# ─────────────────────────────────────────────────────────────────────────────
# 8. DURATION SHOCK SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────

shocks   = np.arange(-150, 175, 25)
d_shocks = np.array([eff_dur_conv(WAC, FACE, R0 + s/10000)[0] for s in shocks])


# ─────────────────────────────────────────────────────────────────────────────
# 9. VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor('#0D1117')
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

TC='#E6EDF3'; LC='#8B949E'; GC='#21262D'
MC='#FF6B35'; BC='#58A6FF'; GR='#3FB950'; RD='#F85149'; YL='#F0C419'


def _ax(ax, title, xl='', yl=''):
    ax.set_facecolor('#161B22')
    for sp in ax.spines.values(): sp.set_color(GC)
    ax.tick_params(colors=LC, labelsize=9)
    ax.set_title(title, color=TC, fontsize=11, fontweight='bold', pad=8)
    if xl: ax.set_xlabel(xl, color=LC, fontsize=9)
    if yl: ax.set_ylabel(yl, color=LC, fontsize=9)
    ax.grid(True, color=GC, linewidth=0.7, alpha=0.8)


# Panel 1 — Duration Profile
ax1 = fig.add_subplot(gs[0, 0])
_ax(ax1, 'Effective Duration vs 10Y CMT Rate',
    '10Y CMT (%)', 'Effective Duration (years)')
ax1.plot(tenY_range*100, mbs_durs, color=MC, lw=2.5, label='MBS WAC 7%')
ax1.plot(tenY_range*100, tsy_durs, color=BC, lw=2.5, ls='--', label='10Y TSY (par)')
if near_par.any():
    ax1.axvspan(tenY_range[near_par][0]*100, tenY_range[near_par][-1]*100,
                alpha=0.18, color=YL, label='Near-par (Bassman alert)')
ax1.axvline(R0*100, color=LC, lw=1, ls=':', alpha=0.6)
idx0 = np.argmin(np.abs(tenY_range - R0))
ax1.annotate(f'Current\n{R0*100:.1f}%', xy=(R0*100, mbs_durs[idx0]),
             xytext=(R0*100+0.5, mbs_durs[idx0]+0.5), color=LC, fontsize=8,
             arrowprops=dict(arrowstyle='->', color=LC, lw=0.8))
ax1.legend(facecolor=GC, edgecolor=GC, labelcolor=TC, fontsize=8)
ax1.set_xlim(tenY_range[0]*100, tenY_range[-1]*100)
# Annotate asymmetry arrows
ax1.annotate('', xy=(2.0, 2.5), xytext=(4.3, 6.0),
             arrowprops=dict(arrowstyle='->', color=GR, lw=1.5))
ax1.text(2.2, 2.0, 'Short in rally\n(prepayments)', color=GR, fontsize=8)
ax1.annotate('', xy=(7.5, 10.5), xytext=(5.5, 7.5),
             arrowprops=dict(arrowstyle='->', color=RD, lw=1.5))
ax1.text(6.5, 10.8, 'Long in selloff\n(extension)', color=RD, fontsize=8)


# Panel 2 — Price-Yield Curve
ax2 = fig.add_subplot(gs[0, 1])
_ax(ax2, 'Price-Yield: MBS Negative Convexity vs Treasury',
    '10Y CMT (%)', 'Clean Price (% of face)')
ax2.plot(tenY_range*100, mbs_pxs, color=MC, lw=2.5, label='MBS (WAC 7%)')
ax2.plot(tenY_range*100, tsy_pxs, color=BC, lw=2.5, ls='--', label='10Y TSY')
ax2.axhline(100, color=LC, lw=0.8, ls=':', alpha=0.5)
ax2.axvline(R0*100, color=LC, lw=1, ls=':', alpha=0.5)
neg_zone = mbs_convs < 0
if neg_zone.any():
    ax2.fill_between(tenY_range[neg_zone]*100,
                     mbs_pxs[neg_zone], tsy_pxs[neg_zone],
                     alpha=0.2, color=RD, label='Negative convexity drag')
par_10y = WAC - SPREAD  # 10Y rate where MBS prices at par
ax2.axvline(par_10y*100, color=MC, lw=1, ls=':', alpha=0.6)
ax2.text(par_10y*100+0.1, 78, f'Par 10Y\n({par_10y*100:.1f}%)', color=MC, fontsize=7.5)
ax2.legend(facecolor=GC, edgecolor=GC, labelcolor=TC, fontsize=8)
ax2.set_xlim(tenY_range[0]*100, tenY_range[-1]*100)


# Panel 3 — Effective Convexity
ax3 = fig.add_subplot(gs[1, 0])
_ax(ax3, 'MBS Effective Convexity (Negative = Investor Must Sell Duration)',
    '10Y CMT (%)', 'Effective Convexity (yrs^2)')
ax3.plot(tenY_range*100, mbs_convs, color=MC, lw=2.5, label='MBS Convexity')
ax3.axhline(0, color=LC, lw=1, ls='-', alpha=0.5)
ax3.fill_between(tenY_range*100, mbs_convs, 0, where=mbs_convs<0,
                 alpha=0.3, color=RD, label='Negative convexity zone')
ax3.fill_between(tenY_range*100, mbs_convs, 0, where=mbs_convs>0,
                 alpha=0.15, color=GR, label='Positive convexity')
if near_par.any():
    ax3.axvspan(tenY_range[near_par][0]*100, tenY_range[near_par][-1]*100,
                alpha=0.18, color=YL, label='Near-par zone')
ax3.axvline(R0*100, color=LC, lw=1, ls=':', alpha=0.5)
ax3.legend(facecolor=GC, edgecolor=GC, labelcolor=TC, fontsize=8)
ax3.set_xlim(tenY_range[0]*100, tenY_range[-1]*100)


# Panel 4 — Duration Shock Sensitivity
ax4 = fig.add_subplot(gs[1, 1])
_ax(ax4, f'Duration Response to Rate Shocks  (base {R0*100:.1f}%)',
    'Rate Shock (bps)', 'Effective Duration (years)')
bcolors = [GR if s < 0 else RD for s in shocks]
ax4.bar(shocks, d_shocks, color=bcolors, width=20, alpha=0.75)
ax4.axhline(d0, color=LC, lw=1, ls='--', alpha=0.6,
            label=f'Base {d0:.1f}y')
ax4.axvline(0, color=LC, lw=0.8, ls='-', alpha=0.4)
mxi = np.argmax(d_shocks)
ax4.annotate(f'{d_shocks[mxi]:.1f}y',
             xy=(shocks[mxi], d_shocks[mxi]),
             xytext=(shocks[mxi]-35, d_shocks[mxi]+0.3),
             color=RD, fontsize=8,
             arrowprops=dict(arrowstyle='->', color=RD, lw=0.8))
mni = np.argmin(d_shocks)
ax4.annotate(f'{d_shocks[mni]:.1f}y',
             xy=(shocks[mni], d_shocks[mni]),
             xytext=(shocks[mni]+10, d_shocks[mni]-0.5),
             color=GR, fontsize=8,
             arrowprops=dict(arrowstyle='->', color=GR, lw=0.8))
ax4.legend(facecolor=GC, edgecolor=GC, labelcolor=TC, fontsize=8)


# Panel 5 — Simulated Rate Paths
ax5 = fig.add_subplot(gs[2, 0])
_ax(ax5, 'Simulated 10Y CMT Paths (Vasicek, 252 days)',
    'Trading Day', '10Y CMT (%)')
for name, res in sim.items():
    ax5.plot(res['path']*100, color=res['col'], lw=1.8, label=name, alpha=0.85)
ax5.axhline(par_10y*100, color=MC, lw=1, ls=':', alpha=0.7,
            label=f'Near-par rate ({par_10y*100:.1f}%)')
ax5.legend(facecolor=GC, edgecolor=GC, labelcolor=TC, fontsize=7.5)


# Panel 6 — Cumulative Hedge Flows
ax6 = fig.add_subplot(gs[2, 1])
_ax(ax6, 'Cumulative Hedging Flows (10Y TSY face equiv, USDbn)',
    'Trading Day', 'Cumulative Flow (USDbn, + = Short/Sell)')
for name, res in sim.items():
    ax6.plot(res['cum'], color=res['col'], lw=1.8, label=name, alpha=0.85)
ax6.axhline(0, color=LC, lw=0.8, ls='-', alpha=0.4)
ax6.axhline(+40, color=YL, lw=1.2, ls='--', alpha=0.8, label='GS USD40B estimate')
ax6.axhline(-40, color=YL, lw=1.2, ls='--', alpha=0.8)
ax6.legend(facecolor=GC, edgecolor=GC, labelcolor=TC, fontsize=7.5)


# Panel 7 — Pro-Cyclical Scatter
ax7 = fig.add_subplot(gs[3, 0])
_ax(ax7, f'Pro-Cyclical Mechanism: dRate vs Hedge Flow  (rho={corr_val:.3f})',
    'dRate (bps/day)', 'Hedge Flow (USDbn, + = Sell TSY)')
dr_bps = bl['dr'] * 10000
hf_arr = bl['hf']
cx = [RD if x > 0 else GR for x in dr_bps]
ax7.scatter(dr_bps, hf_arr, c=cx, alpha=0.35, s=14)
xfit = np.linspace(dr_bps.min(), dr_bps.max(), 100)
ax7.plot(xfit, slope*xfit, color='white', lw=1.5, alpha=0.8,
         label=f'Slope {slope:.2f}B/bp')
ax7.axhline(0, color=LC, lw=0.8, ls='-', alpha=0.4)
ax7.axvline(0, color=LC, lw=0.8, ls='-', alpha=0.4)
ax7.text(0.73, 0.88, 'Rates up\nSell TSY (pro-cyclical)',
         transform=ax7.transAxes, color=RD, fontsize=7.5, ha='center')
ax7.text(0.27, 0.12, 'Rates down\nCover TSY (pro-cyclical)',
         transform=ax7.transAxes, color=GR, fontsize=7.5, ha='center')
ax7.legend(facecolor=GC, edgecolor=GC, labelcolor=TC, fontsize=8)


# Panel 8 — Duration Strategy Zones
ax8 = fig.add_subplot(gs[3, 1])
_ax(ax8, 'Duration Strategy Zones for Bond Portfolio',
    '10Y CMT (%)', 'Effective Duration (years)')
ax8.plot(tenY_range*100, mbs_durs, color=MC, lw=2.5, label='MBS Duration')
ax8.plot(tenY_range*100, tsy_durs, color=BC, lw=2.5, ls='--', label='10Y TSY Duration')
# Zone C: extension trap (rates well above par)
trap_lo = par_10y + 0.005
ax8.axvspan(trap_lo*100, tenY_range[-1]*100, alpha=0.12, color='red',
            label='Zone C: Extension trap — cut MBS, short duration')
# Zone B: near par
if near_par.any():
    ax8.axvspan(tenY_range[near_par][0]*100, tenY_range[near_par][-1]*100,
                alpha=0.22, color=YL, label='Zone B: Near-par — max hedge volatility')
# Zone A: deep rally, prepayment compression
comp_hi = par_10y - 0.015
ax8.axvspan(tenY_range[0]*100, comp_hi*100, alpha=0.12, color='cyan',
            label='Zone A: Prepayment cap — prefer TSY duration over MBS')
ax8.axvline(R0*100, color=LC, lw=1.2, ls=':', alpha=0.7)
ax8.text(R0*100+0.1, 1.0, f'Now\n{R0*100:.1f}%', color=LC, fontsize=7.5)
ax8.set_xlim(tenY_range[0]*100, tenY_range[-1]*100)
ax8.legend(facecolor=GC, edgecolor=GC, labelcolor=TC, fontsize=7.5)


fig.suptitle(
    'MBS Convexity Beast — Backtest Analysis\n'
    'Pool: WAC 7% / USD2T (5%+ coupon universe, Barclays) | Hedge: 10Y UST',
    color=TC, fontsize=13, fontweight='bold', y=0.99
)

out = '/home/user/Work1/mbs_convexity_backtest.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 10. SUMMARY OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

W = 70
print("=" * W)
print("MBS CONVEXITY BACKTEST — KEY RESULTS")
print(f"WAC={WAC*100:.0f}%  |  MBS spread={SPREAD*10000:.0f}bp  |  Pool USD{POOL_B/1000:.0f}T")
print("=" * W)

print("\n── Duration Profile at Key 10Y CMT Levels ──")
for r in [0.020, 0.030, 0.043, 0.050, 0.065, 0.080]:
    idx = np.argmin(np.abs(tenY_range - r))
    c   = cpr_model(WAC, r)
    m   = (r + SPREAD) * 100
    print(f"  10Y={r*100:.1f}%  mort={m:.1f}%"
          f"  Dur={mbs_durs[idx]:.2f}y"
          f"  Conv={mbs_convs[idx]:+.1f}y2"
          f"  Px={mbs_pxs[idx]:.2f}"
          f"  CPR={c*100:.1f}%")

print("\n── Goldman USD40B Validation (+50bp shock) ──")
print(f"  Base: 10Y={R0*100:.2f}%  Dur={d0:.2f}y  Px={p0:.2f}")
print(f"  Shock: 10Y={r_sh*100:.2f}%  Dur={ds:.2f}y  Px={ps:.2f}")
print(f"  Duration extension: {delta_d:+.3f}y")
print(f"  10Y TSY face equiv needed: USD{hedge_B:.1f}B")
print(f"  Goldman reference:         USD40B")
print(f"  Ratio: {hedge_B/40:.2f}x")

print("\n── Pro-Cyclicality ──")
print(f"  Corr(dRate, HedgeFlow):    {corr_val:.4f}  (positive = pro-cyclical)")
print(f"  Amplification:             {slope:.2f} USDbn per 1bp rate move")
print(f"  Daily 1-sigma rate vol:    {daily_v:.2f}bp")
print(f"  Expected daily hedge:      {abs(slope)*daily_v:.1f} USDbn (1-sigma move)")

print("\n── Near-Par Zone ──")
if near_par.any():
    nr = tenY_range[near_par]
    nd = mbs_durs[near_par]
    nc = mbs_convs[near_par]
    print(f"  10Y rate range:  {nr[0]*100:.2f}% – {nr[-1]*100:.2f}%")
    print(f"  Mortgage range:  {(nr[0]+SPREAD)*100:.2f}% – {(nr[-1]+SPREAD)*100:.2f}%")
    print(f"  Duration range:  {nd.min():.2f}y – {nd.max():.2f}y")
    print(f"  Convexity range: {nc.min():.1f} – {nc.max():.1f} y2")
else:
    print("  No near-par region in current rate range.")

print("\n── Scenario Hedge Flow Summary ──")
for name, res in sim.items():
    cf = res['cum']
    print(f"  {name[:35]:35s}  "
          f"final={cf[-1]:+7.1f}B  max={cf.max():+7.1f}B  min={cf.min():+7.1f}B")

print(f"\nChart saved: {out}")
print("=" * W)
