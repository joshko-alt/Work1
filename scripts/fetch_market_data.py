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


def fetch(url: str, tries: int = 3, base_wait: float = 6.0, timeout: int = 45,
          extra_headers: dict | None = None) -> bytes:
    last = None
    headers = {**HEADERS, **(extra_headers or {})}
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
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
    """NASDAQCOM daily closes. FRED tar-pits CI runner IPs (reads stall
    even for tiny ranges), so this is a single quick probe in case this
    particular runner's egress IP is not affected."""
    for url in (
        "https://fred.stlouisfed.org/series/NASDAQCOM/downloaddata/NASDAQCOM.csv",
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM",
    ):
        try:
            content = fetch(url, tries=1, timeout=45)
        except Exception as exc:  # noqa: BLE001
            log(f"  FRED probe failed ({exc})")
            continue
        first = content[:80].decode("utf-8", "replace").split("\n")[0].lower()
        if first.startswith(("date", "observation_date")):
            return content
        log("  FRED probe returned unexpected payload")
    raise RuntimeError("FRED unreachable from this runner")


# -------------------------------------------------------------- Wayback
def cdx_captures(query: str) -> list[list[str]]:
    """Query the Wayback CDX API; rows are oldest->newest capture tuples."""
    url = (
        "https://web.archive.org/cdx/search/cdx?" + query
        + "&output=json&filter=statuscode:200&collapse=digest"
    )
    try:
        rows = json.loads(fetch(url, tries=3, base_wait=8.0, timeout=60))
    except Exception as exc:  # noqa: BLE001
        log(f"  cdx query failed ({exc})")
        return []
    return rows[1:] if isinstance(rows, list) and len(rows) > 1 else []


def csv_span(text: str):
    """(first_date, last_date, n_rows) of a Date-first CSV, else None."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    try:
        d0 = datetime.date.fromisoformat(lines[1].split(",")[0].strip())
        d1 = datetime.date.fromisoformat(lines[-1].split(",")[0].strip())
    except ValueError:
        return None
    return d0, d1, len(lines) - 1


def wayback_hunt_nasdaq_tail() -> None:
    """The committed Stooq snapshot ends 2015-03-20; hunt newer captures of
    full-history Nasdaq CSVs to cover 2015->present. Each hunt saves the
    newest capture that starts before 2015-03 (so series can be spliced
    with overlap validation) and ends after 2016."""
    quote = lambda s: urllib.parse.quote(s, safe="")  # noqa: E731
    hunts = [
        ("url=" + quote("fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQCOM")
         + "&limit=-25", "nasdaq_fred_wayback.csv"),
        ("url=" + quote("fred.stlouisfed.org/graph/fredgraph.csv")
         + "&matchType=prefix&filter=original:.*NASDAQCOM.*&limit=-50",
         "nasdaq_fred_wayback.csv"),
        ("url=" + quote("query1.finance.yahoo.com/v7/finance/download/^IXIC")
         + "&matchType=prefix&limit=-50", "nasdaq_yahoo_wayback.csv"),
        ("url=" + quote("query2.finance.yahoo.com/v7/finance/download/^IXIC")
         + "&matchType=prefix&limit=-50", "nasdaq_yahoo_wayback.csv"),
        ("url=" + quote("stooq.com/q/d/l/")
         + "&matchType=prefix&filter=original:.*ndq.*&limit=-50",
         "nasdaq_stooq_wayback_tail.csv"),
    ]
    for query, name in hunts:
        if (OUT / name).exists():
            continue
        for cap in reversed(cdx_captures(query)):  # newest capture first
            ts, original = cap[1], cap[2]
            if int(ts[:4]) < 2016:
                break  # older captures cannot extend the tail
            try:
                content = fetch(f"https://web.archive.org/web/{ts}id_/{original}",
                                tries=2, base_wait=6.0, timeout=120)
            except Exception as exc:  # noqa: BLE001
                log(f"  wayback fetch @{ts} failed: {exc}")
                continue
            span = csv_span(content.decode("utf-8", "replace"))
            if not span:
                continue
            d0, d1, n = span
            if d0 <= datetime.date(2015, 3, 1) and d1 >= datetime.date(2016, 1, 1) and n >= 5000:
                save(name, content)
                with (OUT / "nasdaq_wayback_provenance.txt").open("a") as fh:
                    fh.write(f"{name}: {original} snapshot {ts} covering {d0}..{d1} ({n} rows)\n")
                break
            log(f"  capture @{ts} spans {d0}..{d1} ({n} rows), not useful")


def nasdaq_official_api_probe() -> None:
    """One shot at api.nasdaq.com for the 2015->today tail (close only)."""
    if (OUT / "nasdaq_api.json").exists():
        return
    url = (
        "https://api.nasdaq.com/api/quote/COMP/chart?assetclass=index"
        f"&fromdate=2015-01-01&todate={datetime.date.today().isoformat()}"
    )
    try:
        content = fetch(url, tries=1, timeout=45, extra_headers={
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        })
        points = (json.loads(content).get("data") or {}).get("chart") or []
        if len(points) > 1000:
            save("nasdaq_api.json", content)
        else:
            log(f"  api.nasdaq.com returned {len(points)} points, ignoring")
    except Exception as exc:  # noqa: BLE001
        log(f"  api.nasdaq.com probe failed: {exc}")


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
        # Yahoo first: it fails fast (immediate 429) when blocked.
        try:  # warm up the cookie jar; yahoo rate-limits bare API hits harder
            fetch("https://finance.yahoo.com/", tries=1, timeout=30)
        except Exception:  # noqa: BLE001
            pass
        for symbol, name in [("^TWII", "taiex_yahoo.csv"), ("^IXIC", "nasdaq_yahoo.csv")]:
            if is_fresh(name):
                continue
            try:
                save(name, yahoo_daily_csv(symbol))
            except Exception as exc:  # noqa: BLE001
                log(f"WARNING optional source Yahoo {symbol} failed: {exc}")

        def nasdaq_tail_end():
            """Latest date across all Nasdaq CSVs on disk (None if none)."""
            best = None
            for name in ("nasdaq_yahoo.csv", "nasdaq_fred.csv",
                         "nasdaq_fred_wayback.csv", "nasdaq_yahoo_wayback.csv",
                         "nasdaq_stooq_wayback_tail.csv", "nasdaq_stooq_wayback.csv"):
                path = OUT / name
                if not path.exists():
                    continue
                try:
                    last = path.read_text().strip().splitlines()[-1].split(",")[0]
                    day = datetime.date.fromisoformat(last.strip())
                except Exception:  # noqa: BLE001
                    continue
                best = day if best is None or day > best else best
            return best

        today = datetime.date.today()
        end = nasdaq_tail_end()
        if end is None or (today - end).days > 30:
            if not (OUT / "nasdaq_fred.csv").exists():
                try:
                    save("nasdaq_fred.csv", fred_nasdaq())
                except Exception as exc:  # noqa: BLE001
                    log(f"WARNING FRED direct failed: {exc}")
            end = nasdaq_tail_end()
            if end is None or (today - end).days > 30:
                wayback_hunt_nasdaq_tail()
                nasdaq_official_api_probe()

        end = nasdaq_tail_end()
        if end is None and not (OUT / "nasdaq_api.json").exists():
            failures.append("no Nasdaq source succeeded (Yahoo, FRED, Wayback, API)")
        else:
            log(f"Nasdaq coverage on disk ends {end}"
                + (" + api.json tail" if (OUT / 'nasdaq_api.json').exists() else ""))

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
