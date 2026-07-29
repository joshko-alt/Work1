#!/usr/bin/env python3
"""Fetch TAIEX and Nasdaq Composite daily history from public sources.

This script is meant to run inside GitHub Actions (the analysis container
has no direct internet egress). It saves raw daily CSVs under data/ so the
backtest can run offline and results stay reproducible.

Sources (multiple, so the series can be cross-validated):
- Stooq          ^twse (TAIEX), ^ndq (Nasdaq Composite) - daily OHLCV
- Yahoo Finance  ^TWII, ^IXIC via the v8 chart API      - daily OHLCV
- FRED           NASDAQCOM                              - daily close only
"""
import datetime
import json
import pathlib
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def fetch(url: str, tries: int = 4) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=90) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last = exc
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def save(name: str, content: bytes) -> None:
    path = OUT / name
    path.write_bytes(content)
    lines = content.decode("utf-8", "replace").strip().splitlines()
    print(f"--- {name}: {len(lines)} lines")
    for line in lines[:3]:
        print("  head:", line)
    for line in lines[-2:]:
        print("  tail:", line)


def yahoo_daily_csv(symbol: str) -> bytes:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}"
        "?period1=0&period2=9999999999&interval=1d"
    )
    payload = json.loads(fetch(url))
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
    save("taiex_stooq.csv", fetch("https://stooq.com/q/d/l/?s=%5Etwse&i=d"))
    save("nasdaq_stooq.csv", fetch("https://stooq.com/q/d/l/?s=%5Endq&i=d"))
    save("taiex_yahoo.csv", yahoo_daily_csv("^TWII"))
    save("nasdaq_yahoo.csv", yahoo_daily_csv("^IXIC"))
    save(
        "nasdaq_fred.csv",
        fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM"),
    )
    print("all sources fetched")


if __name__ == "__main__":
    main()
