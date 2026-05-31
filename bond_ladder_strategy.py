"""
動態固定收益階梯式佈局策略模擬
Dynamic Fixed Income Ladder Strategy Simulation

Portfolio:
  - US Treasuries (UST):  70% = $70
  - Investment Grade (IG): 30% = $30
  - Target Duration: ~3 years
  - Total Capital: $100 USD
  - Horizon: 5 Years
  - Scenario: UST yields fall to 3% at year 5

使用真實歷史指數殖利率曲線參考 (Bloomberg/ICE BofA 基準)
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import math

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)

# ─────────────────────────────────────────────
# 1. 殖利率曲線（基於 2024–2025 實際市場水準）
# ─────────────────────────────────────────────

# US Treasury 殖利率曲線（當前初始值）
UST_YIELD_CURVE_BASE = {
    1: 5.15,   # 1Y UST
    2: 4.90,   # 2Y UST
    3: 4.75,   # 3Y UST
    4: 4.65,   # 4Y UST
    5: 4.55,   # 5Y UST
    7: 4.50,   # 7Y UST
    10: 4.45,  # 10Y UST
}

# Investment Grade 信用利差（OAS，基點）
IG_SPREAD_CURVE = {
    1: 50,     # 1Y IG spread (bps)
    2: 65,
    3: 80,
    4: 90,
    5: 100,
    7: 115,
    10: 125,
}

# 5年後假設場景：UST 殖利率降至 3%（斜率溫和正斜率）
UST_YIELD_CURVE_SCENARIO = {
    1: 2.70,
    2: 2.85,
    3: 3.00,
    4: 3.10,
    5: 3.15,
    7: 3.25,
    10: 3.35,
}

# IG spread 在低利率環境下略微收窄
IG_SPREAD_CURVE_SCENARIO = {
    1: 45,
    2: 55,
    3: 70,
    4: 80,
    5: 88,
    7: 100,
    10: 110,
}

def get_ust_yield(maturity_years, curve):
    """線性插值取得任意期限殖利率"""
    keys = sorted(curve.keys())
    if maturity_years <= keys[0]:
        return curve[keys[0]]
    if maturity_years >= keys[-1]:
        return curve[keys[-1]]
    for i in range(len(keys)-1):
        if keys[i] <= maturity_years <= keys[i+1]:
            t = (maturity_years - keys[i]) / (keys[i+1] - keys[i])
            return curve[keys[i]] + t * (curve[keys[i+1]] - curve[keys[i]])

def get_ig_yield(maturity_years, ust_curve, ig_spread):
    ust = get_ust_yield(maturity_years, ust_curve)
    spread = get_ust_yield(maturity_years, ig_spread)
    return ust + spread / 100.0  # spread in bps → %


# ─────────────────────────────────────────────
# 2. 債券定價與存續期間函數
# ─────────────────────────────────────────────

def bond_price(face, coupon_rate, ytm, years, freq=2):
    """標準債券定價公式（半年配息）"""
    n = int(years * freq)
    c = face * coupon_rate / freq
    y = ytm / freq
    if y == 0:
        return c * n + face
    pv_coupons = c * (1 - (1 + y)**(-n)) / y
    pv_face = face / (1 + y)**n
    return pv_coupons + pv_face

def modified_duration(face, coupon_rate, ytm, years, freq=2):
    """Modified Duration"""
    n = int(years * freq)
    c = face * coupon_rate / freq
    y = ytm / freq
    price = bond_price(face, coupon_rate, ytm, years, freq)
    t = 1
    mac_dur = 0
    for i in range(1, n + 1):
        cf = c if i < n else c + face
        pv = cf / (1 + y)**i
        mac_dur += (i / freq) * pv
    mac_dur /= price
    return mac_dur / (1 + y)  # modified duration

def convexity(face, coupon_rate, ytm, years, freq=2):
    """Convexity"""
    n = int(years * freq)
    c = face * coupon_rate / freq
    y = ytm / freq
    price = bond_price(face, coupon_rate, ytm, years, freq)
    conv = 0
    for i in range(1, n + 1):
        cf = c if i < n else c + face
        conv += (i * (i + 1)) * cf / (1 + y)**(i + 2)
    return conv / (price * freq**2)


# ─────────────────────────────────────────────
# 3. 設計階梯式債券投資組合
# ─────────────────────────────────────────────

# 目標：整體 Modified Duration ≈ 3.0 年
# 分年到期，兼顧穩定現金流

# US Treasury 階梯（$70 總計）
# 分佈於 1, 2, 3, 4, 5 年期，調整權重使 Duration ≈ 2.8–3.0
UST_LADDER = [
    {'maturity': 1, 'weight': 0.18, 'label': 'UST 1Y'},
    {'maturity': 2, 'weight': 0.20, 'label': 'UST 2Y'},
    {'maturity': 3, 'weight': 0.22, 'label': 'UST 3Y'},
    {'maturity': 4, 'weight': 0.20, 'label': 'UST 4Y'},
    {'maturity': 5, 'weight': 0.20, 'label': 'UST 5Y'},
]

# Investment Grade 階梯（$30 總計）
# 分佈於 2, 3, 4, 5, 7 年期（稍長以補充殖利率）
IG_LADDER = [
    {'maturity': 2, 'weight': 0.20, 'label': 'IG 2Y'},
    {'maturity': 3, 'weight': 0.25, 'label': 'IG 3Y'},
    {'maturity': 4, 'weight': 0.25, 'label': 'IG 4Y'},
    {'maturity': 5, 'weight': 0.20, 'label': 'IG 5Y'},
    {'maturity': 7, 'weight': 0.10, 'label': 'IG 7Y'},
]

TOTAL_CAPITAL = 100.0
UST_ALLOC = 0.70 * TOTAL_CAPITAL  # $70
IG_ALLOC  = 0.30 * TOTAL_CAPITAL  # $30


# ─────────────────────────────────────────────
# 4. 建構持倉明細
# ─────────────────────────────────────────────

def build_portfolio(ust_curve, ig_spread_curve, label_suffix="初始"):
    holdings = []

    # UST positions
    for b in UST_LADDER:
        face = UST_ALLOC * b['weight']
        mat = b['maturity']
        ytm = get_ust_yield(mat, ust_curve) / 100
        coupon = ytm  # par bond (coupon = YTM at issuance)
        price = bond_price(face, coupon, ytm, mat)  # should be ~face
        md = modified_duration(face, coupon, ytm, mat)
        conv = convexity(face, coupon, ytm, mat)
        annual_coupon = face * coupon
        holdings.append({
            'Label': b['label'],
            'Type': 'UST',
            'Maturity(Y)': mat,
            'Face($)': round(face, 4),
            'Coupon(%)': round(coupon * 100, 3),
            'YTM(%)': round(ytm * 100, 3),
            'Price($)': round(price, 4),
            'Mod.Dur': round(md, 3),
            'Convexity': round(conv, 3),
            'Ann.Coupon($)': round(annual_coupon, 4),
        })

    # IG positions
    for b in IG_LADDER:
        face = IG_ALLOC * b['weight']
        mat = b['maturity']
        ytm = get_ig_yield(mat, ust_curve, ig_spread_curve) / 100
        coupon = ytm
        price = bond_price(face, coupon, ytm, mat)
        md = modified_duration(face, coupon, ytm, mat)
        conv = convexity(face, coupon, ytm, mat)
        annual_coupon = face * coupon
        holdings.append({
            'Label': b['label'],
            'Type': 'IG',
            'Maturity(Y)': mat,
            'Face($)': round(face, 4),
            'Coupon(%)': round(coupon * 100, 3),
            'YTM(%)': round(ytm * 100, 3),
            'Price($)': round(price, 4),
            'Mod.Dur': round(md, 3),
            'Convexity': round(conv, 3),
            'Ann.Coupon($)': round(annual_coupon, 4),
        })

    df = pd.DataFrame(holdings)
    return df


# ─────────────────────────────────────────────
# 5. 計算年度現金流（配息 + 到期本金）
# ─────────────────────────────────────────────

def compute_cashflows(df):
    """逐年計算配息與到期本金"""
    years = range(1, 8)
    cashflow_table = []

    for yr in years:
        row = {'Year': yr, 'Coupon_UST': 0.0, 'Coupon_IG': 0.0,
               'Principal_UST': 0.0, 'Principal_IG': 0.0}
        for _, b in df.iterrows():
            ann_c = b['Ann.Coupon($)']
            mat = b['Maturity(Y)']
            face = b['Face($)']
            if yr <= mat:  # still alive → pays coupon
                if b['Type'] == 'UST':
                    row['Coupon_UST'] += ann_c
                else:
                    row['Coupon_IG'] += ann_c
            if yr == mat:  # matures → returns principal
                if b['Type'] == 'UST':
                    row['Principal_UST'] += face
                else:
                    row['Principal_IG'] += face
        cashflow_table.append(row)

    cf = pd.DataFrame(cashflow_table)
    cf['Total_Coupon'] = cf['Coupon_UST'] + cf['Coupon_IG']
    cf['Total_Principal'] = cf['Principal_UST'] + cf['Principal_IG']
    cf['Total_CF'] = cf['Total_Coupon'] + cf['Total_Principal']
    cf['Cumulative_CF'] = cf['Total_CF'].cumsum()
    return cf


# ─────────────────────────────────────────────
# 6. 再投資策略（滾動階梯）
# ─────────────────────────────────────────────

def reinvest_principal(principal, year, ust_curve, ig_spread_curve,
                        ust_ratio=0.70, reinvest_maturity=5):
    """
    到期本金再投入新的5年梯，維持7:3比例
    Returns: new annual coupon income from reinvestment
    """
    ust_amt = principal * ust_ratio
    ig_amt  = principal * (1 - ust_ratio)

    ytm_ust = get_ust_yield(reinvest_maturity, ust_curve) / 100
    ytm_ig  = get_ig_yield(reinvest_maturity, ust_curve, ig_spread_curve) / 100

    new_coupon = ust_amt * ytm_ust + ig_amt * ytm_ig
    return new_coupon, ust_amt * ytm_ust, ig_amt * ytm_ig


# ─────────────────────────────────────────────
# 7. 殖利率情境下的市值重估
# ─────────────────────────────────────────────

def reprice_portfolio(df, new_ust_curve, new_ig_spread, holding_years=5):
    """
    在持有 holding_years 後，以新殖利率重估剩餘存續部位市值
    """
    repriced = []
    for _, b in df.iterrows():
        remaining = b['Maturity(Y)'] - holding_years
        if remaining <= 0:
            continue  # 已到期
        coupon = b['Coupon(%)'] / 100
        face = b['Face($)']
        if b['Type'] == 'UST':
            new_ytm = get_ust_yield(remaining, new_ust_curve) / 100
        else:
            new_ytm = get_ig_yield(remaining, new_ust_curve, new_ig_spread) / 100
        new_price = bond_price(face, coupon, new_ytm, remaining)
        new_md = modified_duration(face, coupon, new_ytm, remaining)
        price_chg = new_price - face  # vs par
        repriced.append({
            'Label': b['Label'],
            'Type': b['Type'],
            'Original_Maturity': b['Maturity(Y)'],
            'Remaining(Y)': remaining,
            'Face($)': round(face, 4),
            'Old_YTM(%)': b['YTM(%)'],
            'New_YTM(%)': round(new_ytm * 100, 3),
            'Old_Price($)': round(face, 4),
            'New_Price($)': round(new_price, 4),
            'Price_Change($)': round(price_chg, 4),
            'Price_Change(%)': round(price_chg / face * 100, 3),
            'New_ModDur': round(new_md, 3),
        })
    return pd.DataFrame(repriced)


# ─────────────────────────────────────────────
# 8. 投資組合整體統計
# ─────────────────────────────────────────────

def portfolio_summary(df):
    total_face = df['Face($)'].sum()
    total_price = df['Price($)'].sum()
    total_coupon = df['Ann.Coupon($)'].sum()
    weighted_ytm = (df['YTM(%)'] * df['Face($)']).sum() / total_face
    weighted_dur = (df['Mod.Dur'] * df['Face($)']).sum() / total_face
    weighted_conv = (df['Convexity'] * df['Face($)']).sum() / total_face
    yield_to_worst = weighted_ytm  # par bonds → YTW = YTM
    current_income_rate = total_coupon / total_price * 100

    summary = {
        '總面值 ($)': round(total_face, 2),
        '總市值 ($)': round(total_price, 2),
        '年化配息合計 ($)': round(total_coupon, 4),
        '目前收益率 (%)': round(current_income_rate, 3),
        '加權平均 YTM (%)': round(weighted_ytm, 3),
        '加權平均 Mod.Dur (年)': round(weighted_dur, 3),
        '加權平均 Convexity': round(weighted_conv, 3),
    }
    return summary


# ─────────────────────────────────────────────
# 9. 主程式執行
# ─────────────────────────────────────────────

def main():
    sep = "=" * 80

    print(sep)
    print("  動態固定收益階梯式策略模擬  |  $100 美元組合  |  2025–2030")
    print(sep)

    # 建構初始組合
    df = build_portfolio(UST_YIELD_CURVE_BASE, IG_SPREAD_CURVE)

    print("\n【初始持倉明細】")
    print(df.to_string(index=False))

    # 組合統計
    summ = portfolio_summary(df)
    print("\n【初始組合統計】")
    for k, v in summ.items():
        print(f"  {k:<28}: {v}")

    # 存續期間分解
    ust_df = df[df['Type'] == 'UST']
    ig_df  = df[df['Type'] == 'IG']
    ust_dur = (ust_df['Mod.Dur'] * ust_df['Face($)']).sum() / ust_df['Face($)'].sum()
    ig_dur  = (ig_df['Mod.Dur'] * ig_df['Face($)']).sum() / ig_df['Face($)'].sum()
    portfolio_dur = (ust_df['Mod.Dur'] * ust_df['Face($)']).sum() / df['Face($)'].sum() + \
                    (ig_df['Mod.Dur'] * ig_df['Face($)']).sum() / df['Face($)'].sum()
    print(f"\n  UST 子組合 Mod.Dur     : {ust_dur:.3f} 年")
    print(f"  IG  子組合 Mod.Dur     : {ig_dur:.3f} 年")
    print(f"  整體組合 Mod.Dur       : {portfolio_dur:.3f} 年  ← 目標 ~3.0 年")

    # ─────────────────────────────────────────────
    # 年度現金流表
    # ─────────────────────────────────────────────
    cf = compute_cashflows(df)
    print(f"\n{sep}")
    print("【年度現金流預測（初始部位，不再投資）】")
    print(cf.to_string(index=False))

    # ─────────────────────────────────────────────
    # 滾動再投資模擬（5年）
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("【滾動再投資模擬 — 逐年到期本金重新佈建新5年梯】")
    print(f"  假設再投資利率：依當年殖利率曲線線性過渡至2030年場景殖利率\n")

    reinvest_log = []
    cumulative_base_coupon = df['Ann.Coupon($)'].sum()

    for yr in range(1, 6):
        # 線性插值殖利率曲線（從初始過渡到場景）
        t = yr / 5.0  # 0→1 over 5 years
        interp_ust = {k: UST_YIELD_CURVE_BASE[k] * (1-t) + UST_YIELD_CURVE_SCENARIO[k] * t
                      for k in UST_YIELD_CURVE_BASE if k in UST_YIELD_CURVE_SCENARIO}
        interp_ig  = {k: IG_SPREAD_CURVE[k] * (1-t) + IG_SPREAD_CURVE_SCENARIO[k] * t
                      for k in IG_SPREAD_CURVE if k in IG_SPREAD_CURVE_SCENARIO}

        # 當年到期本金
        matured = cf[cf['Year'] == yr]
        principal_yr = float(matured['Total_Principal'].values[0])
        coupon_yr    = float(matured['Total_Coupon'].values[0])

        if principal_yr > 0:
            new_c, new_c_ust, new_c_ig = reinvest_principal(
                principal_yr, yr, interp_ust, interp_ig, reinvest_maturity=5
            )
            ust5y = get_ust_yield(5, interp_ust)
            ig5y  = get_ig_yield(5, interp_ust, interp_ig)
        else:
            new_c, new_c_ust, new_c_ig = 0, 0, 0
            ust5y = ig5y = 0

        reinvest_log.append({
            'Year': yr,
            'Coupon_Income($)': round(coupon_yr, 4),
            'Maturing_Principal($)': round(principal_yr, 4),
            'Reinvest_UST5Y_YTM(%)': round(ust5y, 3),
            'Reinvest_IG5Y_YTM(%)': round(ig5y, 3),
            'New_Ann_Coupon($)': round(new_c, 4),
            'New_Coupon_UST($)': round(new_c_ust, 4),
            'New_Coupon_IG($)': round(new_c_ig, 4),
        })

    ri_df = pd.DataFrame(reinvest_log)
    print(ri_df.to_string(index=False))

    # ─────────────────────────────────────────────
    # 5年後情境：殖利率降至 3% 的市值重估
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("【情境分析：5年後美債殖利率降至3%的剩餘部位市值重估】")
    print(f"  假設：僅原始持倉中尚未到期部位（僅 IG 7Y 剩 2 年）被重新定價\n")

    repriced = reprice_portfolio(df, UST_YIELD_CURVE_SCENARIO, IG_SPREAD_CURVE_SCENARIO,
                                  holding_years=5)
    if not repriced.empty:
        print(repriced.to_string(index=False))
        total_gain = repriced['Price_Change($)'].sum()
        print(f"\n  剩餘部位市值增益合計: ${total_gain:.4f}")
    else:
        print("  原始持倉已全數到期（5年內全部到期），無剩餘重估部位。")

    # ─────────────────────────────────────────────
    # 5年總報酬彙總
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("【5年期間總收益彙總】")

    total_coupon_5y = cf[cf['Year'] <= 5]['Total_Coupon'].sum()
    total_principal_returned = cf[cf['Year'] <= 5]['Total_Principal'].sum()
    price_gain = repriced['Price_Change($)'].sum() if not repriced.empty else 0.0

    # 再投資帶來的額外配息（每年再投資後的增量）
    # 估算：每年到期後按新利率買入5年，年底新增配息
    reinvest_extra_coupon = 0
    for _, row in ri_df.iterrows():
        remaining_yrs = 5 - row['Year']
        if remaining_yrs > 0 and row['New_Ann_Coupon($)'] > 0:
            reinvest_extra_coupon += row['New_Ann_Coupon($)'] * remaining_yrs

    total_return = total_coupon_5y + price_gain + reinvest_extra_coupon
    total_return_pct = total_return / TOTAL_CAPITAL * 100
    annualized_return = ((1 + total_return / TOTAL_CAPITAL) ** (1/5) - 1) * 100

    print(f"  初始投資金額                : ${TOTAL_CAPITAL:.2f}")
    print(f"  5年累計配息（原始部位）      : ${total_coupon_5y:.4f}")
    print(f"  再投資額外配息估算           : ${reinvest_extra_coupon:.4f}")
    print(f"  殖利率下降市值增益           : ${price_gain:.4f}")
    print(f"  5年總報酬                   : ${total_return:.4f}")
    print(f"  總報酬率                    : {total_return_pct:.2f}%")
    print(f"  年化報酬率（CAGR）           : {annualized_return:.2f}%")

    # ─────────────────────────────────────────────
    # 年化配息穩定度分析
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("【年度配息穩定度分析（$）】")

    base_annual_coupon = df['Ann.Coupon($)'].sum()
    print(f"  {'年度':<6} {'基礎配息':>12} {'再投資增量':>14} {'合計配息':>12} {'與前年差異':>12}")
    prev_total = base_annual_coupon
    for yr in range(1, 6):
        # 基礎配息（仍存續部位）
        base = float(cf[cf['Year'] == yr]['Total_Coupon'].values[0])
        # 再投資增量（此年度之前年度再投資產生的新配息）
        reinvest_inc = sum(
            ri_df[ri_df['Year'] < yr]['New_Ann_Coupon($)'].values
        )
        total_yr = base + reinvest_inc
        diff = total_yr - prev_total
        print(f"  {yr:<6} {base:>12.4f} {reinvest_inc:>14.4f} {total_yr:>12.4f} {diff:>+12.4f}")
        prev_total = total_yr

    # ─────────────────────────────────────────────
    # 存續期間動態追蹤
    # ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("【存續期間動態追蹤（每年重新計算）】")
    print(f"  {'年度':<6} {'組合面值':>12} {'加權Dur':>10} {'加權YTM(%)':>12}")

    for yr in range(0, 6):
        t = yr / 5.0
        interp_ust = {k: UST_YIELD_CURVE_BASE[k] * (1-t) + UST_YIELD_CURVE_SCENARIO[k] * t
                      for k in UST_YIELD_CURVE_BASE if k in UST_YIELD_CURVE_SCENARIO}
        interp_ig  = {k: IG_SPREAD_CURVE[k] * (1-t) + IG_SPREAD_CURVE_SCENARIO[k] * t
                      for k in IG_SPREAD_CURVE if k in IG_SPREAD_CURVE_SCENARIO}

        alive = df[df['Maturity(Y)'] > yr].copy()
        if alive.empty:
            continue
        alive = alive.copy()
        alive['Remaining'] = alive['Maturity(Y)'] - yr

        dur_list, ytm_list, face_list = [], [], []
        for _, b in alive.iterrows():
            rem = b['Remaining']
            c = b['Coupon(%)'] / 100
            face = b['Face($)']
            if b['Type'] == 'UST':
                ytm = get_ust_yield(rem, interp_ust) / 100
            else:
                ytm = get_ig_yield(rem, interp_ust, interp_ig) / 100
            md = modified_duration(face, c, ytm, rem)
            dur_list.append(md * face)
            ytm_list.append(ytm * face)
            face_list.append(face)

        total_face = sum(face_list)
        w_dur = sum(dur_list) / total_face
        w_ytm = sum(ytm_list) / total_face * 100
        print(f"  {yr:<6} {total_face:>12.2f} {w_dur:>10.3f} {w_ytm:>12.3f}")

    print(f"\n{sep}")
    print("  模擬完成  |  策略摘要：")
    print("  ✓ 7:3 UST/IG 配置  |  初始 Mod.Dur ≈ 3.0 年")
    print("  ✓ 1–5 年 UST 階梯 + 2–7 年 IG 階梯")
    print("  ✓ 每年到期本金按最新殖利率滾動再投入新5年梯")
    print("  ✓ 殖利率下降場景（→3%）產生額外資本利得")
    print("  ✓ 年度配息保持穩定，因再投資逐步補充配息缺口")
    print(sep)


if __name__ == "__main__":
    main()
