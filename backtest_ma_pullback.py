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

import json
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


# Nasdaq file candidates: live fetches first, then byte-exact Internet
# Archive snapshots collected when live sources block CI egress IPs.
NASDAQ_SOURCES = [
    ("nasdaq_yahoo.csv", "Yahoo Finance ^IXIC（日開高低收）", "ohlc"),
    ("nasdaq_yahoo_wayback.csv",
     "Yahoo Finance ^IXIC 日開高低收（Internet Archive 快照）", "ohlc"),
    ("nasdaq_stooq_wayback.csv",
     "Stooq ^ndq 日開高低收（Internet Archive 快照）", "ohlc"),
    ("nasdaq_stooq_wayback_tail.csv",
     "Stooq ^ndq 日開高低收（Internet Archive 較新快照）", "ohlc"),
    ("nasdaq_fred.csv", "FRED NASDAQCOM（日收盤）", "fred"),
    ("nasdaq_fred_wayback.csv",
     "FRED NASDAQCOM（Internet Archive 快照，日收盤）", "fred"),
    ("nasdaq_api.json", "那斯達克官方 api.nasdaq.com COMP（日收盤）", "api"),
]

NASDAQ_INCEPTION = "1971-02-05"  # composite base date; earlier rows in the
                                 # archived Stooq file are a predecessor index


def load_nasdaq_api(path: pathlib.Path) -> pd.DataFrame:
    """Close-only frame from an api.nasdaq.com chart payload."""
    chart = json.loads(path.read_text())["data"]["chart"]
    rows = []
    for pt in chart:
        m, d, y = pt["z"]["dateTime"].split("/")
        rows.append((f"{int(y):04d}-{int(m):02d}-{int(d):02d}", float(pt["y"])))
    df = pd.DataFrame(rows, columns=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.drop_duplicates("Date").set_index("Date").sort_index()


def _load_nasdaq_file(name: str, kind: str) -> pd.DataFrame:
    if kind == "ohlc":
        return load_ohlc(DATA / name)
    if kind == "api":
        return load_nasdaq_api(DATA / name)
    return load_fred(DATA / name).to_frame("Close")


def repair_bad_prints(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Fix single-day spike-and-revert bad prints (e.g. the archived Stooq
    series shows -13.5%/+16.2% on 1972-04-13/14; no such move happened).
    A day is repaired only when the move exceeds 12%, reverses the next
    day, and the two-day round trip nets out to under 3%."""
    close = df["Close"]
    ret = close.pct_change()
    round_trip = close.shift(-1) / close.shift(1) - 1
    bad = (ret.abs() > 0.12) & (ret.shift(-1).abs() > 0.12) & \
          (np.sign(ret) != np.sign(ret.shift(-1))) & (round_trip.abs() < 0.03)
    for ts in df.index[bad.fillna(False)]:
        i = df.index.get_loc(ts)
        patched = (close.iloc[i - 1] + close.iloc[i + 1]) / 2
        print(f"[{name}] repaired bad print {ts.date()}: "
              f"{close.iloc[i]:.2f} -> {patched:.2f}")
        for col in df.columns:
            df.loc[ts, col] = patched
    return df


def splice(base: pd.DataFrame, tail: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extend base with tail rows after base's end; the overlap must agree.
    Only columns common to both survive, so a close-only tail deliberately
    downgrades the whole series to close-only (consistent touch rule)."""
    overlap = base.index.intersection(tail.index)
    if len(overlap) < 20:
        raise ValueError(f"insufficient overlap ({len(overlap)} days)")
    diff = (base.loc[overlap, "Close"] / tail.loc[overlap, "Close"] - 1).abs()
    if diff.median() > 0.005:
        raise ValueError(f"overlap disagrees (median {diff.median():.2%})")
    cols = [c for c in base.columns if c in tail.columns]
    comp = pd.concat([base[cols], tail.loc[tail.index > base.index.max(), cols]])
    return comp, diff


def load_nasdaq() -> tuple[pd.DataFrame, str]:
    """Best single Nasdaq series, or a validated splice when no single
    file covers 1971->present. Returns (daily frame, source description)."""
    available = []
    for name, desc, kind in NASDAQ_SOURCES:
        if (DATA / name).exists():
            df = _load_nasdaq_file(name, kind)
            available.append((df, desc, kind))
    if not available:
        raise FileNotFoundError("no Nasdaq data file found under data/")

    latest_end = max(df.index.max() for df, _, _ in available)
    # A single file that starts by 1972 and reaches the latest end wins.
    for df, desc, kind in available:
        if df.index.min().year <= 1972 and df.index.max() >= latest_end - pd.Timedelta(days=7):
            return _finalize_nasdaq(df), desc

    # Otherwise splice: longest history as base (prefer OHLC), then keep
    # extending with whichever remaining series reaches furthest.
    available.sort(key=lambda t: (t[0].index.min(), t[2] != "ohlc"))
    base, base_desc, _ = available[0]
    desc = base_desc + f"（至 {base.index.max().date()}）"
    for df, d_desc, _ in sorted(available[1:], key=lambda t: t[0].index.max()):
        if df.index.max() <= base.index.max():
            continue
        switch = base.index.max().date()
        base, diff = splice(base, df)
        print(f"[splice] + {d_desc} after {switch}: "
              f"overlap median rel diff {diff.median():.2e}")
        desc += f"，{switch} 之後接 {d_desc}"
    return _finalize_nasdaq(base), desc


def _finalize_nasdaq(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[NASDAQ_INCEPTION:].copy()
    return repair_bad_prints(df, "NASDAQ")


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

    # TAIEX: exchange's official series is the primary; Yahoo cross-checks it.
    taiex = load_ohlc(DATA / "taiex_twse.csv")
    validate("TAIEX/twse", taiex)
    if (DATA / "taiex_yahoo.csv").exists():
        taiex_y = load_ohlc(DATA / "taiex_yahoo.csv")
        validate("TAIEX/yahoo", taiex_y)
        cross_check("TAIEX twse vs yahoo", taiex["Close"], taiex_y["Close"])

    # Nasdaq: best available source (see NASDAQ_SOURCES preference order),
    # cross-checked against any other Nasdaq series that is also present.
    nasdaq, nasdaq_src = load_nasdaq()
    print(f"NASDAQ source: {nasdaq_src}")
    validate("NASDAQ", nasdaq)
    if "Low" not in nasdaq.columns:
        print("NASDAQ series is close-only: MA touches use Close, not intraday Low")
    for name, desc, kind in NASDAQ_SOURCES:
        if desc in nasdaq_src or not (DATA / name).exists():
            continue
        other = _load_nasdaq_file(name, kind)["Close"]
        cross_check(f"NASDAQ vs {name}", nasdaq["Close"], other)

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
