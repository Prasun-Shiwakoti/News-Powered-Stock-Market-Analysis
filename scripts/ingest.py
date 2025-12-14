# Ingests news articles and stock price data
import argparse, re, hashlib
from pathlib import Path
import pandas as pd
import yaml
from dateutil import parser as dtparser
import pytz

def norm(s): 
    # Normalize whitespace in text strings
    return re.sub(r"\s+", " ", str(s)).strip() if pd.notna(s) else s  

def sha1(s):  
    # Generate SHA1 hash for deduplication
    import hashlib, pandas as pd
    return hashlib.sha1(str(s).lower().encode("utf-8")).hexdigest() if pd.notna(s) else None

def pick(cols, *cands):  
    # Find first matching column name from candidates (case-insensitive)
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low: return low[c]
    return None

def parse_utc(series):  
    # Parse datetime strings to UTC timezone
    return pd.to_datetime([dtparser.parse(str(x)) if str(x).lower()!="nan" else None],
                          utc=True, errors="coerce")

def read_news_chunked(path, tickers, ex_tz, chunksize=200_000):  
    # Process large news CSV files in chunks with deduplication
    use_all_tickers = tickers[0].lower() == "all"
    keep = set(tickers) if not use_all_tickers else None
    
    out = []
    chunkCount = 0
    for ch in pd.read_csv(path, low_memory=False, chunksize=chunksize, encoding='latin-1'):
        cols = ch.columns
        date  = pick(cols, "date","datetime","timestamp","time","news_dt","published_at")
        title = pick(cols, "article_title","title","headline")
        body  = pick(cols, "article_content","content","text","body")
        tick  = pick(cols, "stock_symbol","ticker","symbol")
        src   = pick(cols, "publisher","source")
        url   = pick(cols, "url","link")
        if not (date and tick and (title or body)): 
            continue

        ch["__ticker"] = ch[tick].astype(str).str.upper().str.strip()
        
        if not use_all_tickers:
            ch = ch[ch["__ticker"].isin(keep)]
            if ch.empty: continue

        chunkCount += 1
        print(f"Processing chunk {chunkCount}...")
        sub = pd.DataFrame()
        sub["ticker"] = ch["__ticker"]
        if title: sub["title"] = ch[title].map(norm)
        if body:  sub["body"]  = ch[body].map(norm)
        if src:   sub["source"]= ch[src].astype(str)
        if url:   sub["url"]   = ch[url].astype(str)
        sub["news_dt"] = pd.to_datetime([dtparser.parse(str(x)) if str(x).lower()!="nan" else None
                                         for x in ch[date]], utc=True, errors="coerce")
        sub["title_hash"] = sub.get("title","").map(sha1)
        sub["url_hash"]   = sub.get("url","").map(sha1)
        sub["local_day"]  = sub["news_dt"].dt.tz_convert(ex_tz).dt.date
        sub = sub.drop_duplicates(subset=["ticker","local_day","url_hash","title_hash"]).drop(columns=["local_day"])
        out.append(sub.dropna(subset=["news_dt"]))
        print(f"Processed chunk {chunkCount}, total rows so far: {sum(len(df) for df in out)}")
    
    print("All chunks processed! Merging all the chunks...")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["ticker","title","body","source","url","news_dt","title_hash","url_hash"]
    )

def read_price_one(price_root: Path, ticker: str) -> pd.DataFrame:  
    # Load and standardize price data for a single ticker
    candidates = [
        price_root / f"{ticker}.csv",
        price_root / "full_history" / f"{ticker}.csv",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        hits = list(price_root.rglob(f"{ticker}.csv"))
        if hits: found = hits[0]
    if found is None:
        print(f"[warn] price file not found for {ticker}"); return pd.DataFrame()
    d = pd.read_csv(found, low_memory=False)
    colmap = {c: c.lower().replace(" ","_") for c in d.columns}
    d = d.rename(columns=colmap)
    if "date" not in d.columns: 
        print(f"[warn] no Date in {found}"); return pd.DataFrame()

    sub = pd.DataFrame()
    sub["date"] = pd.to_datetime(d["date"]).dt.date
    sub["ticker"]= ticker
    sub["open"]  = d.get("open")
    sub["high"]  = d.get("high")
    sub["low"]   = d.get("low")
    sub["close"] = d["close"]
    sub["volume"]= d.get("volume")
    sub = (sub.dropna(subset=["date","close"]).sort_values(["ticker","date"])
             .drop_duplicates(subset=["ticker","date"], keep="last"))
    return sub

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True, help="List of tickers (e.g. --tickers AAPL MSFT GOOGL META) or 'all' to process all available tickers")
    ap.add_argument("--process", choices=["news","price","both"], default="both")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml"))
    ex_tz = pytz.timezone(cfg["prediction"]["exchange_tz"])
    proc  = Path(cfg["data"]["processed_dir"]); proc.mkdir(parents=True, exist_ok=True)
    base  = Path(cfg["data"]["raw_dir"])

    # Process News data
    if args.process in ("news","both"):
        news = pd.DataFrame()
        for nf in [base/"Stock_news"/"All_external.csv", base/"Stock_news"/"nasdaq_exteral_data.csv"]:
            if nf.exists():
                print(f"\n[news] {nf.name} Processing...\n")
                n = read_news_chunked(nf, args.tickers, ex_tz, chunksize=args.chunksize)
                print(f"\n[news] {nf.name} Completed!\n")
                print("Merging...")
                news = pd.concat([news, n], ignore_index=True)
        if news.empty:
            print("[warn] no news rows found")
        else:
            print("Writing", proc/"news_processed.csv", "...")
            news.to_csv(proc/"news_processed.csv", index=False)
            print("Wrote", proc/"news_processed.csv", news.shape)

    #Process price data
    if args.process not in ("price","both"):
        return 
    
    price_root = base / "Stock_price"
    parts = []

    if args.tickers[0].lower() == "all":
        print("Scanning for all available price files...")
        available_tickers = []
        
        full_history_dir = price_root / "full_history"
        if full_history_dir.exists():
            for csv_file in full_history_dir.glob("*.csv"):
                ticker = csv_file.stem.upper()
                if ticker not in available_tickers:  
                    available_tickers.append(ticker)
        
        if not available_tickers:
            raise SystemExit("No price CSV files found in the price directory")
        
        available_tickers = sorted(available_tickers)
        print(f"Found {len(available_tickers)} price files: {', '.join(available_tickers[:10])}{'...' if len(available_tickers) > 10 else ''}")
        process_tickers = available_tickers
    else:
        process_tickers = [t.upper() for t in args.tickers]
    
    for t in process_tickers:
        p = read_price_one(price_root, t)
        if not p.empty:
            print(f"[price] {t} Processed!")
            parts.append(p)

    prices = pd.concat([p for p in parts], ignore_index=True)
    if prices.empty:
        raise SystemExit("No price rows found; check your tickers exist in full_history/")
    prices.to_csv(proc/"prices_processed.csv", index=False)
    print("Wrote", proc/"prices_processed.csv", prices.shape)

if __name__ == "__main__":
    main()
