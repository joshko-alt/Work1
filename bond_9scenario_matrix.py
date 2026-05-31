"""
固定收益 9 種情境矩陣分析
3 利率路徑  ×  3 存續期間策略  =  9 結果

利率路徑：
  Path1  政策利率上升至 4.5%（升息/不降息場景）
  Path2  政策利率下行至 3.0%（溫和降息）
  Path3  政策利率下行至 2.0%（激進降息/衰退場景）

存續期間策略（7:3 UST/IG，$100）：
  S_Short  集中短天期   1–2Y（Dur ≈ 1.5Y）
  S_Ladr   階梯式配置   1–5Y/2–7Y（Dur ≈ 3.0Y）
  S_Long   集中長天期   7–10Y（Dur ≈ 7.5Y）

假設：
  • IG 信用利差固定（不隨利率路徑變動）= 初始 OAS 水準
  • 持續再投資（每年到期本金按相同策略滾入新債）
  • 5 年期投資水平，年末結算
  • 第5年末剩餘部位按終點殖利率曲線市值重估
  • 配息不再投資（視為現金收入）
"""

import pandas as pd
import math

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 180)

TOTAL = 100.0
UST_W = 0.70
IG_W  = 0.30

# ══════════════════════════════════════════════════════════════════════════════
# 1. 殖利率假設與路徑定義
# ══════════════════════════════════════════════════════════════════════════════

# ── 初始曲線（Year 0 = 2025）──
UST_Y0 = {1: 5.15, 2: 4.90, 3: 4.75, 4: 4.65, 5: 4.55, 7: 4.50, 10: 4.45}

# IG OAS 固定不變（題目要求忽略利差變動）
IG_OAS_FIXED = {1: 0.50, 2: 0.65, 3: 0.80, 4: 0.90, 5: 1.00, 7: 1.15, 10: 1.25}  # %

# ── 三條利率路徑終點曲線（Year 5 = 2030）──
# Path1：政策利率上升至 4.5%
#   短端受到高政策利率支撐，曲線熊平（bear flattening）
#   1Y 反映 4.5% FFR 上升 → 短端走高；長端受通膨預期拉抬但幅度有限
UST_END_P1 = {1: 5.00, 2: 4.90, 3: 4.80, 4: 4.70, 5: 4.60, 7: 4.55, 10: 4.50}

# Path2：政策利率下行至 3.0%（前次模擬）
UST_END_P2 = {1: 2.70, 2: 2.85, 3: 3.00, 4: 3.10, 5: 3.15, 7: 3.25, 10: 3.35}

# Path3：政策利率下行至 2.0%（激進降息/衰退）
#   短端大幅下降；長端因衰退預期壓低，但長期通膨預期維持地板（曲線陡化）
UST_END_P3 = {1: 1.80, 2: 1.90, 3: 2.00, 4: 2.10, 5: 2.20, 7: 2.35, 10: 2.50}

PATHS = {
    'Path1_Rise4.5%': UST_END_P1,
    'Path2_Fall3.0%': UST_END_P2,
    'Path3_Fall2.0%': UST_END_P3,
}

# ── 三種策略的梯子結構 ──
# 每個 rung：(期限, UST比重, IG比重)
STRATEGIES = {
    'S1_Short（1–2Y集中）': {
        'ust_rungs': [(1, 0.50), (2, 0.50)],
        'ig_rungs':  [(1, 0.50), (2, 0.50)],
        'reinvest_ust_mat': 2,   # 到期再投入目標期限
        'reinvest_ig_mat':  2,
        'dur_label': '~1.5Y',
    },
    'S2_Ladder（1–5Y/2–7Y）': {
        'ust_rungs': [(1,0.18),(2,0.20),(3,0.22),(4,0.20),(5,0.20)],
        'ig_rungs':  [(2,0.20),(3,0.25),(4,0.25),(5,0.20),(7,0.10)],
        'reinvest_ust_mat': 5,
        'reinvest_ig_mat':  5,
        'dur_label': '~3.0Y',
    },
    'S3_Long（7–10Y集中）': {
        'ust_rungs': [(7, 0.50),(10, 0.50)],
        'ig_rungs':  [(7, 0.50),(10, 0.50)],
        'reinvest_ust_mat': 10,
        'reinvest_ig_mat':  10,
        'dur_label': '~7.5Y',
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. 工具函數
# ══════════════════════════════════════════════════════════════════════════════

def interp(curve, mat):
    keys = sorted(curve.keys())
    if mat <= keys[0]:  return curve[keys[0]]
    if mat >= keys[-1]: return curve[keys[-1]]
    for i in range(len(keys)-1):
        if keys[i] <= mat <= keys[i+1]:
            t = (mat - keys[i]) / (keys[i+1] - keys[i])
            return curve[keys[i]] + t*(curve[keys[i+1]]-curve[keys[i]])

def get_ust_yr(mat, yr, ust_end):
    """線性插值取得第 yr 年的 UST 殖利率"""
    t = min(yr / 5.0, 1.0)
    return interp(UST_Y0, mat)*(1-t) + interp(ust_end, mat)*t

def get_ig_yr(mat, yr, ust_end):
    """IG 殖利率 = UST + 固定 OAS"""
    return get_ust_yr(mat, yr, ust_end) + interp(IG_OAS_FIXED, mat)

def bond_price(face, coupon, ytm, years, freq=2):
    n = round(years * freq)
    if n <= 0: return face
    c = face * coupon / freq
    y = ytm / freq
    if abs(y) < 1e-9: return c*n + face
    return c*(1-(1+y)**(-n))/y + face/(1+y)**n

def mod_dur(face, coupon, ytm, years, freq=2):
    n = round(years * freq)
    if n <= 0: return 0.0
    c = face*coupon/freq
    y = ytm/freq
    price = bond_price(face, coupon, ytm, years, freq)
    if price < 1e-9: return 0.0
    mac = sum((i/freq)*( (c if i<n else c+face)/(1+y)**i ) for i in range(1,n+1))
    return (mac/price)/(1+y)

# ══════════════════════════════════════════════════════════════════════════════
# 3. 初始持倉建構
# ══════════════════════════════════════════════════════════════════════════════

def build_initial(strategy, ust_end):
    rungs = []
    ust_alloc = TOTAL * UST_W
    ig_alloc  = TOTAL * IG_W
    for (mat, w) in strategy['ust_rungs']:
        face = ust_alloc * w
        ytm  = get_ust_yr(mat, 0, ust_end)
        rungs.append(dict(type='UST', buy_yr=0, mat_yr=mat,
                          face=face, coupon=ytm/100, ytm_buy=ytm))
    for (mat, w) in strategy['ig_rungs']:
        face = ig_alloc * w
        ytm  = get_ig_yr(mat, 0, ust_end)
        rungs.append(dict(type='IG', buy_yr=0, mat_yr=mat,
                          face=face, coupon=ytm/100, ytm_buy=ytm))
    return rungs

# ══════════════════════════════════════════════════════════════════════════════
# 4. 5年模擬引擎
# ══════════════════════════════════════════════════════════════════════════════

def simulate(strategy, ust_end):
    portfolio = build_initial(strategy, ust_end)
    total_coupon = 0.0
    yearly = []

    for yr in range(1, 6):
        yr_coupon = 0.0
        yr_principal = 0.0
        surviving = []

        for r in portfolio:
            if r['mat_yr'] < yr:       # 已在前期到期
                continue
            yr_coupon += r['face'] * r['coupon']   # 本年配息
            if r['mat_yr'] == yr:
                yr_principal += r['face']           # 本年到期
            else:
                surviving.append(r)

        total_coupon += yr_coupon

        # 再投資：按策略目標期限 + 當年殖利率
        new_rungs = []
        if yr_principal > 0:
            rm_ust = strategy['reinvest_ust_mat']
            rm_ig  = strategy['reinvest_ig_mat']
            ytm_u  = get_ust_yr(rm_ust, yr, ust_end)
            ytm_ig = get_ig_yr(rm_ig,  yr, ust_end)
            ust_face = yr_principal * UST_W
            ig_face  = yr_principal * IG_W
            new_rungs.append(dict(type='UST', buy_yr=yr,
                                  mat_yr=yr+rm_ust, face=ust_face,
                                  coupon=ytm_u/100, ytm_buy=ytm_u))
            new_rungs.append(dict(type='IG', buy_yr=yr,
                                  mat_yr=yr+rm_ig,  face=ig_face,
                                  coupon=ytm_ig/100, ytm_buy=ytm_ig))

        portfolio = surviving + new_rungs
        yearly.append({'yr': yr, 'coupon': yr_coupon,
                       'principal': yr_principal,
                       'n_bonds': len(portfolio)})

    # ── 第5年末市值重估 ──
    mkt_val = 0.0
    price_gain = 0.0
    total_face_remain = 0.0

    for r in portfolio:
        rem = r['mat_yr'] - 5
        if rem <= 0: continue
        if r['type'] == 'UST':
            ytm_now = get_ust_yr(rem, 5, ust_end) / 100
        else:
            ytm_now = get_ig_yr(rem, 5, ust_end) / 100
        mkt = bond_price(r['face'], r['coupon'], ytm_now, rem)
        mkt_val += mkt
        price_gain += mkt - r['face']
        total_face_remain += r['face']

    total_end = total_coupon + mkt_val
    total_return = total_end - TOTAL
    cagr = ((total_end / TOTAL)**(1/5) - 1)*100

    # 初始 duration
    init_rungs = build_initial(strategy, ust_end)
    init_dur = sum(mod_dur(r['face'], r['coupon'], r['ytm_buy']/100, r['mat_yr'])*r['face']
                   for r in init_rungs) / TOTAL
    init_ytm = sum(r['ytm_buy']*r['face'] for r in init_rungs) / TOTAL

    return {
        'yearly': yearly,
        'total_coupon': total_coupon,
        'mkt_val': mkt_val,
        'price_gain': price_gain,
        'face_remain': total_face_remain,
        'total_end': total_end,
        'total_return': total_return,
        'return_pct': total_return / TOTAL * 100,
        'cagr': cagr,
        'init_dur': init_dur,
        'init_ytm': init_ytm,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 5. 主輸出
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sep  = "═" * 110
    sep2 = "─" * 110

    print(sep)
    print("  固定收益 9 種情境矩陣  |  3 利率路徑 × 3 存續期間策略  |  $100 / 5年")
    print(sep)

    # ── 殖利率假設摘要 ──
    print("\n【利率路徑假設（年末 UST 殖利率，%）】")
    mats = [1, 2, 3, 5, 7, 10]
    hdrs = "  " + f"{'期限':<6}" + "".join(f"{'Y'+str(m):>8}" for m in mats)
    print(hdrs)
    for pname, pend in PATHS.items():
        for yr in [0, 5]:
            tag = f"{'初始2025' if yr==0 else pname[:14]}"
            row = f"  {tag:<20}" if yr == 0 else f"  {tag:<20}"
            for m in mats:
                row += f"{get_ust_yr(m, yr, pend):>8.2f}"
            print(row)
            if yr == 0: break   # 初始只印一次
    print()
    for pname, pend in PATHS.items():
        row = f"  終點→{pname:<20}"
        for m in mats:
            row += f"{interp(pend, m):>8.2f}"
        print(row)

    # IG OAS 說明
    print("\n  IG OAS（固定不變）:")
    print("  " + "".join(f"  {m}Y:{int(interp(IG_OAS_FIXED,m)*100)}bps" for m in mats))

    # ── 策略摘要 ──
    print("\n【策略結構假設】")
    for sname, strat in STRATEGIES.items():
        init = build_initial(strat, UST_END_P2)  # 用中性路徑展示
        dur = sum(mod_dur(r['face'],r['coupon'],r['ytm_buy']/100,r['mat_yr'])*r['face']
                  for r in init)/TOTAL
        ytm = sum(r['ytm_buy']*r['face'] for r in init)/TOTAL
        ust_mats = "/".join(str(m)+"Y" for m,_ in strat['ust_rungs'])
        ig_mats  = "/".join(str(m)+"Y" for m,_ in strat['ig_rungs'])
        print(f"  {sname:<30} UST:{ust_mats}  IG:{ig_mats}  "
              f"Dur≈{dur:.2f}Y  YTM≈{ytm:.2f}%  "
              f"再投資→UST{strat['reinvest_ust_mat']}Y/IG{strat['reinvest_ig_mat']}Y")

    # ── 9 種情境結果 ──
    print(f"\n{sep}")
    print("【9 種情境完整結果】")

    results = {}
    for pname, pend in PATHS.items():
        for sname, strat in STRATEGIES.items():
            r = simulate(strat, pend)
            results[(pname, sname)] = r

    for pname, pend in PATHS.items():
        print(f"\n{'─'*110}")
        print(f"  利率路徑：{pname}")
        print(f"{'─'*110}")
        print(f"  {'策略':<28} {'初始Dur':>8} {'初始YTM':>9} "
              f"{'5Y配息':>9} {'5Y市值':>9} {'價格增益':>10} "
              f"{'5Y總資產':>10} {'總報酬%':>9} {'CAGR%':>8}")
        print("  " + "─"*103)
        for sname, strat in STRATEGIES.items():
            r = results[(pname, sname)]
            print(f"  {sname:<28} {r['init_dur']:>7.2f}Y {r['init_ytm']:>8.3f}% "
                  f"{r['total_coupon']:>9.2f} {r['mkt_val']:>9.2f} {r['price_gain']:>+10.2f} "
                  f"{r['total_end']:>10.2f} {r['return_pct']:>+8.2f}% {r['cagr']:>7.2f}%")

    # ── 彙總矩陣（主視圖）──
    print(f"\n{sep}")
    print("【彙總矩陣：5年末總資產（$）＆ CAGR(%)】")
    print(f"  初始投資 $100\n")

    strat_names = list(STRATEGIES.keys())
    path_names  = list(PATHS.keys())

    # Header
    print(f"  {'':32}" + "".join(f"{p[:16]:>26}" for p in path_names))
    print("  " + "─"*110)

    for sname in strat_names:
        r0 = results[(path_names[0], sname)]
        print(f"  {sname:<32}", end="")
        for pname in path_names:
            r = results[(pname, sname)]
            print(f"  ${r['total_end']:>7.2f} / {r['cagr']:>5.2f}%", end="")
        print()

    # ── 總報酬率矩陣 ──
    print(f"\n{sep}")
    print("【總報酬率矩陣（%）— 5年累積】")
    print(f"\n  {'':32}" + "".join(f"{p[:14]:>18}" for p in path_names))
    print("  " + "─"*90)
    for sname in strat_names:
        print(f"  {sname:<32}", end="")
        for pname in path_names:
            r = results[(pname, sname)]
            sign = "+" if r['return_pct'] >= 0 else ""
            print(f"  {sign}{r['return_pct']:>7.2f}%       ", end="")
        print()

    # ── 年度配息穩定度（選 Path2 標準場景）──
    print(f"\n{sep}")
    print("【年度配息明細（Path2 溫和降息場景，各策略對比）】")
    print(f"\n  {'年度':<6}", end="")
    for sname in strat_names:
        print(f"  {sname[:20]:>22}", end="")
    print()
    print("  " + "─"*80)
    for yr in range(1, 6):
        print(f"  第{yr}年  ", end="")
        for sname, strat in STRATEGIES.items():
            r = results[('Path2_Fall3.0%', sname)]
            coupon_yr = r['yearly'][yr-1]['coupon']
            print(f"  ${coupon_yr:>7.4f}             ", end="")
        print()

    # ── 市值增益/損失分解 ──
    print(f"\n{sep}")
    print("【市值增益/損失分解（$）】")
    print(f"  {'策略/路徑':<32}", end="")
    for pname in path_names:
        print(f"  {pname[:14]:>18}", end="")
    print()
    print("  " + "─"*90)
    for sname in strat_names:
        print(f"  {sname:<32}", end="")
        for pname in path_names:
            r = results[(pname, sname)]
            sign = "+" if r['price_gain'] >= 0 else ""
            print(f"  {sign}${r['price_gain']:>7.2f}         ", end="")
        print()
        print(f"  {'（剩餘債券面值）':<32}", end="")
        for pname in path_names:
            r = results[(pname, sname)]
            print(f"   ${r['face_remain']:>7.2f}         ", end="")
        print()

    # ── 解析 ──
    print(f"\n{sep}")
    print("【9 種情境解析】\n")
    analysis = [
        ("Path1 升息", "S1 短天期",
         "★最佳防禦★  短债快速到期重新投入更高利率，再投資利率上升，配息持續提升。無資本損失風險。"),
        ("Path1 升息", "S2 階梯",
         "中性。短端快速滾動抵禦升息，長端(5Y)有小幅浮虧，整體緩衝足夠。"),
        ("Path1 升息", "S3 長天期",
         "▼最大損失▼  7–10Y 債券持續受壓，市值大幅折損，再投資期限長、期間浮虧無法快速解套。"),
        ("Path2 降息3%", "S1 短天期",
         "再投資利率快速下行，配息不斷萎縮，無法享受資本增益，總報酬最低。"),
        ("Path2 降息3%", "S2 階梯",
         "★均衡最佳★  配息穩定遞減但仍有支撐，中長端部位有資本增益，風險分散效果最優。"),
        ("Path2 降息3%", "S3 長天期",
         "資本增益最大，但配息鎖定高票面；若升息反轉則暴露最大風險。"),
        ("Path3 降息2%", "S1 短天期",
         "再投資利率崩至2%，後期配息近乎消失，總報酬最差，完全錯失資本利得。"),
        ("Path3 降息2%", "S2 階梯",
         "中長端大幅增值，整體報酬良好。短端拖累有限，均衡效果顯現。"),
        ("Path3 降息2%", "S3 長天期",
         "★最大獲利★  7–10Y 債券在殖利率大幅下降時，市值暴升。Duration ≈7.5Y，每降100bps ≈ +7.5%。"),
    ]

    path_tag_map = {'Path1': 'Path1_Rise4.5%', 'Path2': 'Path2_Fall3.0%', 'Path3': 'Path3_Fall2.0%'}
    strat_tag_map = {'短天期': 'S1_Short（1–2Y集中）', '階梯': 'S2_Ladder（1–5Y/2–7Y）', '長天期': 'S3_Long（7–10Y集中）'}

    for path_tag, strat_tag, desc in analysis:
        pname = path_tag_map[path_tag.split()[0]]
        sname = strat_tag_map[strat_tag.split()[1]]
        r = results[(pname, sname)]
        print(f"  [{path_tag} × {strat_tag}]  CAGR={r['cagr']:.2f}%  總報酬={r['return_pct']:+.2f}%")
        print(f"    → {desc}\n")

    # ── 最終建議表 ──
    print(sep)
    print("【策略選擇建議矩陣】\n")
    print("                         ┌──────────────┬──────────────┬──────────────┐")
    print("                         │  升息4.5%    │  降息3.0%    │  降息2.0%    │")
    print("  ┌──────────────────────┼──────────────┼──────────────┼──────────────┤")
    print("  │ 短天期集中（1–2Y）   │  ★★★★☆      │  ★★☆☆☆      │  ★☆☆☆☆      │")
    print("  ├──────────────────────┼──────────────┼──────────────┼──────────────┤")
    print("  │ 階梯式（1–5Y/2–7Y） │  ★★★☆☆      │  ★★★★☆      │  ★★★★☆      │")
    print("  ├──────────────────────┼──────────────┼──────────────┼──────────────┤")
    print("  │ 長天期集中（7–10Y） │  ★☆☆☆☆      │  ★★★★☆      │  ★★★★★      │")
    print("  └──────────────────────┴──────────────┴──────────────┴──────────────┘")
    print()
    print("  核心結論：")
    print("  • 利率方向不確定 → 階梯式在三條路徑中均不是最差，是最穩健的「全天候」策略")
    print("  • 確信升息      → 全面轉短天期，快速翻滾再投資，每年鎖定更高利率")
    print("  • 確信大幅降息  → 集中長天期，以 Duration 槓桿放大資本增益")
    print("  • 現實應用      → 以階梯為核心，根據利率展望動態調整短/長端比重")
    print(sep)


if __name__ == "__main__":
    main()
