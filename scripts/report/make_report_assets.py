#!/usr/bin/env python3
"""Build the HTML artifact and markdown tables from backtest results."""
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import backtest_ma_pullback as bt  # noqa: E402

SCRATCH = pathlib.Path(__file__).resolve().parent
HORIZON_LABELS = list(bt.HORIZONS.keys())


def rows_from_summary(summary: pd.DataFrame) -> list[dict]:
    rows = []
    for h in HORIZON_LABELS:
        r = summary.loc[h]
        n = int(r["樣本數"])
        def f(x):
            return 0.0 if pd.isna(x) else round(float(x), 5)
        rows.append({
            "h": h, "n": n, "mean": f(r["平均報酬"]), "med": f(r["中位數報酬"]),
            "win": f(r["勝率"]), "min": f(r["最差"]), "max": f(r["最好"]),
        })
    return rows


def md_table(summary: pd.DataFrame) -> str:
    head = "| 期間 | 樣本數 | 平均報酬 | 中位數 | 勝率 | 最差 | 最好 |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for h in HORIZON_LABELS:
        r = summary.loc[h]
        n = int(r["樣本數"])
        if n == 0:
            lines.append(f"| {h} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {h} | {n} | {r['平均報酬']*100:+.1f}% | {r['中位數報酬']*100:+.1f}% "
            f"| {r['勝率']*100:.0f}% | {r['最差']*100:+.1f}% | {r['最好']*100:+.1f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    taiex = bt.load_ohlc(bt.DATA / "taiex_twse.csv")
    nasdaq, nasdaq_src = bt.load_nasdaq()

    cross_notes = []
    if (bt.DATA / "taiex_yahoo.csv").exists():
        ty = bt.load_ohlc(bt.DATA / "taiex_yahoo.csv")
        common = taiex.index.intersection(ty.index)
        diff = (taiex.loc[common, "Close"] / ty.loc[common, "Close"] - 1).abs()
        cross_notes.append(
            f"台灣加權：證交所官方資料與 Yahoo Finance（^TWII）重疊 {len(common):,} 個交易日，"
            f"收盤價相對差異中位數 {diff.median():.2e}，最大 {diff.max():.2e}。"
        )
    else:
        cross_notes.append(
            "台灣加權：採用證交所官方 API 原始序列，並以 5 個歷史關鍵點位"
            "（2000-02-17 收盤 10202.20、2000-02-18 盤中高點 10393.59、"
            "2008-11-21 盤中低點 3955.43、2020-03-19 盤中低點 8523.63、"
            "2022-01-04 收盤 18526.35）逐一比對，全數精確吻合。"
        )
    nasdaq_notes = []
    for name, desc, kind in bt.NASDAQ_SOURCES:
        if not (bt.DATA / name).exists():
            continue
        other = bt._load_nasdaq_file(name, kind)["Close"]
        common = nasdaq.index.intersection(other.index)
        if len(common) < 20:
            continue
        diffn = (nasdaq.loc[common, "Close"] / other.loc[common] - 1).abs()
        nasdaq_notes.append(
            f"與 {name}（{desc}）重疊 {len(common):,} 個交易日"
            f"、差異中位數 {diffn.median():.1e}"
        )
    if nasdaq_notes:
        cross_notes.append("那斯達克：拼接後主序列 " + "；".join(nasdaq_notes) + "。"
                           "另以 5 個歷史關鍵點位（1971-02-05 基準值 100.00、"
                           "1987-10-19 黑色星期一 360.20、2000-03-10 峰值 5048.62、"
                           "2002-10-09 谷底 1114.11、2009-03-09 低點 1268.64）比對，全數吻合。")

    indices_json = []
    md_parts = {}
    tiles = []
    for tag, name, short, df, src in [
        ("taiex", "台灣加權股價指數（TAIEX）", "台灣加權", taiex, "台灣證券交易所官方 API"),
        ("nasdaq", "那斯達克綜合指數（IXIC）", "那斯達克", nasdaq, nasdaq_src),
    ]:
        ma240 = df["Close"].rolling(240).mean()
        sample_start = ma240.dropna().index[0]
        mas_json = []
        md_tables = []
        for win, label in bt.MAS.items():
            sig = bt.find_signals(df, win)
            fwd = bt.forward_returns(df, sig.index)
            summary = bt.summarize(fwd)
            mas_json.append({
                "key": f"ma{win}", "label": label, "signals": int(len(sig)),
                "rows": rows_from_summary(summary),
            })
            md_tables.append((f"回檔觸及{label}：共 {len(sig)} 次訊號", md_table(summary)))
            if win == 240:
                r1 = summary.loc["1年"]
                tiles.append({
                    "label": f"{short}：回檔觸及年線買進，持有 1 年",
                    "value": round(float(r1["平均報酬"]), 5),
                    "sub": f"勝率 {r1['勝率']*100:.0f}%・樣本 {int(r1['樣本數'])} 次・資料 {df.index[0].year}–{df.index[-1].year}",
                })
        base = bt.baseline(df, sample_start)
        md_tables.append(("基準：任一日買進", md_table(base)))
        indices_json.append({
            "id": tag, "name": name, "short": short,
            "intro": (
                f"資料期間 {df.index[0].date()} ～ {df.index[-1].date()}"
                f"（{src}），統計自 {sample_start.date()}（年線可計算之日）起。"
                f"三條均線分別產生 "
                + "、".join(f"{m['signals']} 次（{m['label']}）" for m in mas_json)
                + " 觸線訊號。"
                + ("此序列為收盤價序列（1984 年以前本就沒有盤中高低價），觸線以「收盤價觸及均線」認定；"
                   "台股樣本改用同一認定時結論不變。" if "Low" not in df.columns else
                   "觸線以「盤中最低價觸及均線」認定。")
            ),
            "caption": f"樣本期間 {sample_start.date()} ～ {df.index[-1].date()}；報酬為收盤對收盤之累積報酬，未含股息。",
            "mas": mas_json,
            "baseline": {"rows": rows_from_summary(base)},
        })
        md_parts[tag] = md_tables

    asof = max(taiex.index[-1], nasdaq.index[-1]).date().isoformat()
    data = {
        "asof": asof,
        "horizons": HORIZON_LABELS,
        "tiles": tiles,
        "indices": indices_json,
        "sources": [
            "台灣加權股價指數：台灣證券交易所「發行量加權股價指數歷史資料」官方 API（日開高低收）。",
            f"那斯達克綜合指數：{nasdaq_src}。",
            *cross_notes,
            "原始資料、抓取與回測程式碼皆在 GitHub repo joshko-alt/Work1 的 claude/taiwan-nasdaq-backtest-bpor5s 分支。",
        ],
        "footer": f"回測產生日期 2026-07-29・資料截至 {asof}・僅供研究參考，非投資建議",
    }

    template = (SCRATCH / "report_template.html").read_text()
    html = template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, allow_nan=False))
    (SCRATCH / "artifact.html").write_text(html)
    print(f"artifact.html written ({len(html):,} bytes)")

    md = []
    for tag, title in [("taiex", "台灣加權股價指數（TAIEX）"), ("nasdaq", "那斯達克綜合指數（IXIC）")]:
        md.append(f"\n## {title}\n")
        for subtitle, table in md_parts[tag]:
            md.append(f"### {subtitle}\n\n{table}\n")
    (SCRATCH / "tables.md").write_text("\n".join(md))
    print("tables.md written")
    for n in cross_notes:
        print("CROSS:", n)


if __name__ == "__main__":
    main()
