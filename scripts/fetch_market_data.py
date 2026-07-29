#!/usr/bin/env python3
"""Fetch TAIEX and Nasdaq Composite daily history from public sources.

This script is meant to run inside GitHub Actions (the analysis container
has no direct internet egress). It saves daily CSVs under data/ so the
backtest can run offline and results stay reproducible.

Sources:
- TWSE (official exchange API, monthly pages) -> taiex_twse.csv   OHLC, primary
- FRED NASDAQCOM                              -> nasdaq_fred.csv  close, primary
- Yahoo Finance ^TWII / ^IXIC (best effort)   -> *_yahoo.csv      OHLC, cross-check

Exit code is non-zero if either primary source failed; Yahoo failures
only produce a warning. Politeness: TWSE is throttled to one request
per ~3.5s per their rate-limit guidance.
"""
import datetime
import http.cookiejar
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.8,zh-TW;q=0.6",
}

_jar = http.cookiejar.CookieJar()
_opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_jar))


def fetch(url: str, tries: int = 4, base_wait: float = 5.0) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with _opener.open(req, timeout=90) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last = exc
            wait = base_wait * (2 ** i)
            print(f"  retry {i + 1} for {url.split('?')[0]} in {wait:.0f}s ({exc})")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def save(name: str, content: bytes) -> None:
    path = OUT / name
    path.write_bytes(content)
    lines = content.decode("utf-8", "replace").strip().splitlines()
    print(f"--- {name}: {len(lines)} lines")
    for line in lines[:2]:
        print("  head:", line[:120])
    for line in lines[-1:]:
        print("  tail:", line[:120])


# ------------------------------------------------------------------ TWSE
def roc_to_iso(s: str) -> str:
    y, m, d = s.strip().split("/")
    return f"{int(y) + 1911:04d}-{int(m):02d}-{int(d):02d}"


def twse_month(yyyymm01: str) -> list[list[str]]:
    """Return normalized rows [date, open, high, low, close] for one month."""
    urls = [
        f"https://www.twse.com.tw/rwd/zh/TAIEX/MI_5MINS_HIST?date={yyyymm01}&response=json",
        f"https://www.twse.com.tw/indicesReport/MI_5MINS_HIST?response=json&date={yyyymm01}",
    ]
    last = None
    for url in urls:
        try:
            payload = json.loads(fetch(url, tries=3, base_wait=8.0))
        except Exception as exc:  # noqa: BLE001
            last = exc
            continue
        if payload.get("stat") != "OK":
            return []  # month not available (e.g. before the series starts)
        rows = []
        for row in payload.get("data", []):
            date = roc_to_iso(row[0])
            nums = [v.replace(",", "").strip() for v in row[1:5]]
            rows.append([date] + nums)
        return rows
    raise RuntimeError(f"TWSE month {yyyymm01} failed: {last}")


def fetch_twse() -> None:
    today = datetime.date.today()
    months = []
    y, m = 1999, 1
    while (y, m) <= (today.year, today.month):
        months.append(f"{y:04d}{m:02d}01")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    all_rows: list[list[str]] = []
    missing: list[str] = []
    started = False
    for i, month in enumerate(months):
        rows = twse_month(month)
        if rows:
            started = True
            all_rows.extend(rows)
        elif started:
            missing.append(month)
        if i % 24 == 0 or rows == []:
            print(f"  TWSE {month}: {len(rows)} rows (total {len(all_rows)})")
        time.sleep(3.5)

    if missing:
        print(f"  TWSE WARNING missing months after series start: {missing}")
    csv = "Date,Open,High,Low,Close\n" + "\n".join(",".join(r) for r in all_rows) + "\n"
    save("taiex_twse.csv", csv.encode())


# ----------------------------------------------------------------- Yahoo
def yahoo_daily_csv(symbol: str) -> bytes:
    try:  # warm up the cookie jar; yahoo rate-limits bare API hits harder
        fetch("https://finance.yahoo.com/", tries=1)
    except Exception:  # noqa: BLE001
        pass
    url = (
        "https://query2.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}"
        "?period1=0&period2=9999999999&interval=1d"
    )
    payload = json.loads(fetch(url, tries=5, base_wait=15.0))
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    gmtoffset = result["meta"].get("gmtoffset", 0)

    rows = ["Date,Open,High,Low,Close,Volume"]
    for i, ts in enumerate(timestamps):
        close = quote["close"][i]
        if close is None:
            continue
        day = datetime.datetime.fromtimestamp(
            ts + gmtoffset, datetime.timezone.utc
        ).strftime("%Y-%m-%d")

        def num(x):
            return "" if x is None else f"{x:.4f}"

        vol = quote["volume"][i]
        rows.append(
            f"{day},{num(quote['open'][i])},{num(quote['high'][i])},"
            f"{num(quote['low'][i])},{num(close)},{'' if vol is None else vol}"
        )
    return ("\n".join(rows) + "\n").encode()


def main() -> None:
    failures = []

    try:
        save(
            "nasdaq_fred.csv",
            fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM"),
        )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"FRED NASDAQCOM: {exc}")

    for symbol, name in [("^TWII", "taiex_yahoo.csv"), ("^IXIC", "nasdaq_yahoo.csv")]:
        try:
            save(name, yahoo_daily_csv(symbol))
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING optional source Yahoo {symbol} failed: {exc}")

    try:
        fetch_twse()
    except Exception as exc:  # noqa: BLE001
        failures.append(f"TWSE: {exc}")

    if failures:
        print("PRIMARY SOURCE FAILURES:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("primary sources fetched")


if __name__ == "__main__":
    main()
