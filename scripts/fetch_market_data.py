#!/usr/bin/env python3
"""Fetch TAIEX and Nasdaq Composite daily history from public sources.

This script is meant to run inside GitHub Actions (the analysis container
has no direct internet egress). It saves daily CSVs under data/ so the
backtest can run offline and results stay reproducible.

Sources:
- TWSE (official exchange API, monthly pages) -> taiex_twse.csv   OHLC, primary
- FRED NASDAQCOM                              -> nasdaq_fred.csv  close, primary
- Yahoo Finance ^TWII / ^IXIC (best effort)   -> *_yahoo.csv      OHLC, cross-check

The TWSE monthly loop is throttled to one request per ~3.2s per their
rate-limit guidance and takes ~20 minutes for 1999->today, so FRED and
Yahoo run in a background thread alongside it. The TWSE CSV is written
incrementally so a partial run still leaves usable data behind.

Exit code is non-zero if either primary source failed; Yahoo failures
only produce a warning.
"""
import datetime
import http.cookiejar
import json
import pathlib
import sys
import threading
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
_print_lock = threading.Lock()


def log(*args) -> None:
    with _print_lock:
        print(*args, flush=True)


def fetch(url: str, tries: int = 3, base_wait: float = 6.0, timeout: int = 45) -> bytes:
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with _opener.open(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last = exc
            if i < tries - 1:
                wait = base_wait * (2 ** i)
                log(f"  retry {i + 1} for {url.split('?')[0]} in {wait:.0f}s ({exc})")
                time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def is_fresh(name: str, max_age_days: int = 5) -> bool:
    """True if the CSV already exists and its last row is recent enough."""
    path = OUT / name
    if not path.exists():
        return False
    try:
        last_date = path.read_text().strip().splitlines()[-1].split(",")[0]
        day = datetime.date.fromisoformat(last_date)
    except Exception:  # noqa: BLE001 - malformed file -> refetch
        return False
    age = (datetime.date.today() - day).days
    if age <= max_age_days:
        log(f"--- {name}: already fresh (last row {day}), skipping fetch")
        return True
    return False


def save(name: str, content: bytes) -> None:
    path = OUT / name
    path.write_bytes(content)
    lines = content.decode("utf-8", "replace").strip().splitlines()
    log(f"--- {name}: {len(lines)} lines")
    for line in lines[:2]:
        log("  head: " + line[:120])
    for line in lines[-1:]:
        log("  tail: " + line[:120])


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

    path = OUT / "taiex_twse.csv"
    total = 0
    missing: list[str] = []
    started = False
    with path.open("w") as fh:
        fh.write("Date,Open,High,Low,Close\n")
        for i, month in enumerate(months):
            rows = twse_month(month)
            if rows:
                started = True
                total += len(rows)
                fh.write("\n".join(",".join(r) for r in rows) + "\n")
                fh.flush()
            elif started:
                missing.append(month)
            log(f"  TWSE {month}: {len(rows):2d} rows (total {total})")
            time.sleep(3.2)

    if missing:
        log(f"  TWSE WARNING missing months after series start: {missing}")
    log(f"--- taiex_twse.csv: {total} data rows written")


# ------------------------------------------------------------------ FRED
def fred_nasdaq() -> bytes:
    """NASDAQCOM daily closes. The one-shot fredgraph endpoint tar-pits CI
    runners, so try the static download endpoint first, then fredgraph in
    four 15-year windows whose smaller responses stream before the timeout."""
    try:
        content = fetch(
            "https://fred.stlouisfed.org/series/NASDAQCOM/downloaddata/NASDAQCOM.csv",
            tries=2, base_wait=8.0, timeout=120,
        )
        first = content[:80].decode("utf-8", "replace").split("\n")[0].lower()
        if first.startswith(("date", "observation_date")):
            return content
        log("  FRED downloaddata returned unexpected payload, falling back")
    except Exception as exc:  # noqa: BLE001
        log(f"  FRED downloaddata failed ({exc}), trying chunked fredgraph")

    today = datetime.date.today().isoformat()
    windows = [
        ("1971-01-01", "1985-12-31"),
        ("1986-01-01", "2000-12-31"),
        ("2001-01-01", "2015-12-31"),
        ("2016-01-01", today),
    ]
    header = None
    rows: list[str] = []
    for start, end in windows:
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id=NASDAQCOM&cosd={start}&coed={end}"
        )
        lines = fetch(url, tries=3, base_wait=10.0, timeout=150).decode(
            "utf-8", "replace").strip().splitlines()
        if header is None:
            header = lines[0]
        rows.extend(lines[1:])
        log(f"  FRED chunk {start}..{end}: {len(lines) - 1} rows")
        time.sleep(2)
    return (header + "\n" + "\n".join(rows) + "\n").encode()


# ------------------------------------------------------------------- WSJ
def wsj_nasdaq() -> bytes:
    """Nasdaq Composite daily OHLC from WSJ's historical-prices download."""
    end = datetime.date.today().strftime("%m/%d/%Y")
    url = (
        "https://www.wsj.com/market-data/quotes/index/US/COMP/"
        f"historical-prices/download?MOD=mw_quote&startDate=02/05/1971&endDate={end}"
    )
    content = fetch(url, tries=2, base_wait=10.0, timeout=120)
    first = content[:200].decode("utf-8", "replace").split("\n")[0]
    if "Date" not in first:
        raise RuntimeError(f"unexpected WSJ payload: {first[:80]!r}")
    return content


# ----------------------------------------------------------------- Yahoo
def yahoo_daily_csv(symbol: str) -> bytes:
    url = (
        "https://query2.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol)}"
        "?period1=0&period2=9999999999&interval=1d"
    )
    payload = json.loads(fetch(url, tries=3, base_wait=12.0))
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
    failures: list[str] = []

    def secondary_sources() -> None:
        if not is_fresh("nasdaq_fred.csv"):
            try:
                save("nasdaq_fred.csv", fred_nasdaq())
            except Exception as exc:  # noqa: BLE001
                failures.append(f"FRED NASDAQCOM: {exc}")

        if not is_fresh("nasdaq_wsj.csv"):
            try:
                save("nasdaq_wsj.csv", wsj_nasdaq())
            except Exception as exc:  # noqa: BLE001
                log(f"WARNING optional source WSJ failed: {exc}")

        try:  # warm up the cookie jar; yahoo rate-limits bare API hits harder
            fetch("https://finance.yahoo.com/", tries=1)
        except Exception:  # noqa: BLE001
            pass
        for symbol, name in [("^TWII", "taiex_yahoo.csv"), ("^IXIC", "nasdaq_yahoo.csv")]:
            if is_fresh(name):
                continue
            try:
                save(name, yahoo_daily_csv(symbol))
            except Exception as exc:  # noqa: BLE001
                log(f"WARNING optional source Yahoo {symbol} failed: {exc}")

    worker = threading.Thread(target=secondary_sources)
    worker.start()

    if not is_fresh("taiex_twse.csv"):
        try:
            fetch_twse()
        except Exception as exc:  # noqa: BLE001
            failures.append(f"TWSE: {exc}")

    worker.join()

    if failures:
        log("PRIMARY SOURCE FAILURES:")
        for f in failures:
            log(" - " + f)
        sys.exit(1)
    log("primary sources fetched")


if __name__ == "__main__":
    main()
