#!/usr/bin/env python3
"""MA-pullback backtest for TAIEX and Nasdaq Composite.

Strategy question: historically, when the index pulls back from above and
touches its quarterly (60-day), half-year (120-day) or yearly (240-day)
moving average, and you buy that touch, what is the average performance
3 months / 6 months / 1 / 2 / 3 years later?

Signal definition (per MA line):
- Setup:   the index traded fully above the MA (daily Low > MA) for the
           previous ABOVE_DAYS consecutive sessions, i.e. it is pulling
           back from above, not chopping around the line.
- Trigger: today's Low touches or breaks the MA (Low <= MA).
- Entry:   today's Close.
The setup requirement also de-duplicates signals: after a touch, the MA
line has to be "re-armed" by ABOVE_DAYS clean sessions above it.

Forward performance is Close-to-Close over fixed trading-day horizons and
is compared against the unconditional baseline (buying on any day).

Data: raw CSVs under data/ fetched by scripts/fetch_market_data.py
(Stooq primary, Yahoo Finance / FRED for cross-validation).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

ABOVE_DAYS = 20
MAS = {60: "季線(60日)", 120: "半年線(120日)", 240: "年線(240日)"}
HORIZONS = {"3個月": 63, "6個月": 126, "1年": 252, "2年": 504, "3年": 756}


# ---------------------------------------------------------------- loading
def load_ohlc(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().capitalize() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df = df[cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["Close"])


def load_fred(path: pathlib.Path) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = ["Date", "Close"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna().set_index("Date")["Close"]


def validate(name: str, df: pd.DataFrame) -> None:
    ret = df["Close"].pct_change().abs()
    big = ret[ret > 0.16]
    if not big.empty:
        print(f"[{name}] WARNING daily moves >16%:")
        print(big.to_string())
    if {"High", "Low", "Open"} <= set(df.columns):
        bad = df[(df["Low"] > df[["Open", "Close"]].min(axis=1) + 1e-6)
                 | (df["High"] < df[["Open", "Close"]].max(axis=1) - 1e-6)]
        if not bad.empty:
            print(f"[{name}] WARNING {len(bad)} rows with inconsistent OHLC")
    print(f"[{name}] {len(df)} rows  {df.index[0].date()} -> {df.index[-1].date()}")


def cross_check(name: str, a: pd.Series, b: pd.Series, tol: float = 0.005) -> None:
    common = a.index.intersection(b.index)
    if len(common) == 0:
        print(f"[{name}] no overlap to cross-check")
        return
    diff = (a.loc[common] / b.loc[common] - 1).abs()
    print(f"[{name}] overlap {len(common)} days, median rel diff "
          f"{diff.median():.2e}, >{tol:.1%} on {(diff > tol).sum()} days "
          f"({(diff > tol).mean():.2%})")


# ---------------------------------------------------------------- backtest
def find_signals(df: pd.DataFrame, ma_win: int) -> pd.DataFrame:
    ma = df["Close"].rolling(ma_win).mean()
    low = df["Low"] if "Low" in df.columns else df["Close"]
    above = (low > ma)                       # whole bar above the line
    armed = above.rolling(ABOVE_DAYS).sum().shift(1) == ABOVE_DAYS
    touch = low <= ma
    sig = armed & touch & ma.notna()
    out = df.loc[sig, ["Close"]].copy()
    out["MA"] = ma[sig]
    return out


def forward_returns(df: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    close = df["Close"]
    pos = df.index.get_indexer(dates)
    res = {}
    for label, h in HORIZONS.items():
        idx = pos + h
        vals = np.full(len(pos), np.nan)
        ok = idx < len(close)
        vals[ok] = close.to_numpy()[idx[ok]] / close.to_numpy()[pos[ok]] - 1
        res[label] = vals
    return pd.DataFrame(res, index=dates)


def summarize(fwd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in HORIZONS:
        r = fwd[label].dropna()
        rows.append({
            "期間": label,
            "樣本數": len(r),
            "平均報酬": r.mean(),
            "中位數報酬": r.median(),
            "勝率": (r > 0).mean() if len(r) else np.nan,
            "最差": r.min() if len(r) else np.nan,
            "最好": r.max() if len(r) else np.nan,
        })
    return pd.DataFrame(rows).set_index("期間")


def baseline(df: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    dates = df.index[df.index >= start]
    return summarize(forward_returns(df, dates))


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["平均報酬", "中位數報酬", "最差", "最好"]:
        out[c] = (out[c] * 100).map(lambda v: f"{v:+.1f}%" if pd.notna(v) else "-")
    out["勝率"] = (out["勝率"] * 100).map(lambda v: f"{v:.0f}%" if pd.notna(v) else "-")
    return out


def run_index(name: str, df: pd.DataFrame) -> dict:
    print(f"\n================ {name} ================")
    results = {}
    ma240 = df["Close"].rolling(240).mean()
    sample_start = ma240.dropna().index[0]
    for win, label in MAS.items():
        sig = find_signals(df, win)
        fwd = forward_returns(df, sig.index)
        results[label] = {"signals": sig, "fwd": fwd, "summary": summarize(fwd)}
        print(f"\n--- 回檔觸及{label}: {len(sig)} 次訊號")
        print(fmt(results[label]["summary"]).to_string())
    results["baseline"] = baseline(df, sample_start)
    print(f"\n--- 基準(任一日買進, 自 {sample_start.date()}):")
    print(fmt(results["baseline"]).to_string())
    return results


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    taiex = load_ohlc(DATA / "taiex_stooq.csv")
    taiex_y = load_ohlc(DATA / "taiex_yahoo.csv")
    nasdaq = load_ohlc(DATA / "nasdaq_stooq.csv")
    nasdaq_y = load_ohlc(DATA / "nasdaq_yahoo.csv")
    nasdaq_f = load_fred(DATA / "nasdaq_fred.csv")

    validate("TAIEX/stooq", taiex)
    validate("TAIEX/yahoo", taiex_y)
    validate("NASDAQ/stooq", nasdaq)
    validate("NASDAQ/yahoo", nasdaq_y)
    cross_check("TAIEX stooq vs yahoo", taiex["Close"], taiex_y["Close"])
    cross_check("NASDAQ stooq vs yahoo", nasdaq["Close"], nasdaq_y["Close"])
    cross_check("NASDAQ stooq vs FRED", nasdaq["Close"], nasdaq_f)

    all_results = {}
    for name, df in [("台灣加權指數 TAIEX", taiex), ("那斯達克綜合指數 IXIC", nasdaq)]:
        all_results[name] = run_index(name, df)

    # persist machine-readable outputs
    for name, res in all_results.items():
        tag = "taiex" if "TAIEX" in name else "nasdaq"
        frames = []
        for win, label in MAS.items():
            s = res[label]["summary"].copy()
            s.insert(0, "均線", label)
            frames.append(s.reset_index())
            sig = res[label]["signals"].join(res[label]["fwd"])
            sig.to_csv(RESULTS / f"signals_{tag}_ma{win}.csv",
                       float_format="%.4f", encoding="utf-8-sig")
        b = res["baseline"].copy()
        b.insert(0, "均線", "基準(任一日)")
        frames.append(b.reset_index())
        pd.concat(frames, ignore_index=True).to_csv(
            RESULTS / f"summary_{tag}.csv", index=False,
            float_format="%.4f", encoding="utf-8-sig")
    print(f"\nresults written to {RESULTS}/")


if __name__ == "__main__":
    main()
