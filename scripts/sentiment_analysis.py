# Performs sentiment analysis on news articles using the FinBERT model
import argparse, numpy as np, pandas as pd, yaml, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=96)
    ap.add_argument("--cutoff-hour", type=int, default=None, help="Override cutoff (default from config)")
    ap.add_argument("--late-mode", choices=["drop-late","shift-late"], default="shift-late",
                    help="drop late news or assign after-close to next day (default)")
    args = ap.parse_args()

    cfg  = yaml.safe_load(open("config.yaml"))
    proc = Path(cfg["data"]["processed_dir"])
    feat = Path(cfg["data"]["features_dir"]); feat.mkdir(parents=True, exist_ok=True)

    news = pd.read_csv(proc / "news_processed.csv")
    if news.empty: raise SystemExit("No news to score.")
    text = (news.get("title", pd.Series([""]*len(news))).fillna("") + " " +
            news.get("body",  pd.Series([""]*len(news))).fillna("").str.slice(0, 500)).str.strip().tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "yiyanghkust/finbert-tone"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    mdl.eval()

    def score(arr):
        out_s, out_pos, out_neg = [], [], []
        for i in tqdm(range(0, len(arr), args.batch_size), desc="FinBERT"):
            enc = tok(arr[i:i+args.batch_size], padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = mdl(**enc).logits
            p = torch.softmax(logits, dim=-1)  # [neutral, positive, negative]
            pos = p[:,1].cpu().numpy()
            neg = p[:,2].cpu().numpy()
            out_pos.extend(pos.tolist())
            out_neg.extend(neg.tolist())
            out_s.extend((pos - neg).tolist())
        return np.array(out_s), np.array(out_pos), np.array(out_neg)

    sent, p_pos, p_neg = score(text)
    news["sent_finbert"] = sent
    news["p_pos"] = p_pos
    news["p_neg"] = p_neg

    # Time logic
    exchange_tz = cfg["prediction"]["exchange_tz"]
    cutoff_hour = int(args.cutoff_hour if args.cutoff_hour is not None else cfg["prediction"]["cutoff_hour"])
    news["news_dt"] = pd.to_datetime(news["news_dt"], errors="coerce", utc=True)
    news["local_dt"]   = news["news_dt"].dt.tz_convert(exchange_tz)
    news["date_local"] = news["local_dt"].dt.date
    news["hour_local"] = news["local_dt"].dt.hour
    news["minute_local"] = news["local_dt"].dt.minute
    news["second_local"] = news["local_dt"].dt.second

    missing_time = (news["hour_local"]==0) & (news["minute_local"]==0) & (news["second_local"]==0)

    if args.late_mode == "drop-late":
        use = (news["hour_local"] <= cutoff_hour) | missing_time
        used = news[use].copy()
        used["eff_date"] = used["date_local"]
    else:
        used = news.copy()
        shift_one = (used["hour_local"] > cutoff_hour) & (~missing_time)
        used["eff_date"] = pd.to_datetime(used["date_local"]) + pd.to_timedelta(shift_one.astype(int), unit="D")
        used["eff_date"] = used["eff_date"].dt.date

    # Daily aggregates
    daily = (used.groupby(["ticker","eff_date"])
             .agg(mean_sent=("sent_finbert","mean"),
                  std_sent=("sent_finbert","std"),
                  max_abs_sent=("sent_finbert", lambda x: float(np.max(np.abs(x)))),
                  n_articles=("sent_finbert","size"),
                  pos_rate=("p_pos", lambda x: float((np.array(x) > 0.5).mean())),
                  neg_rate=("p_neg", lambda x: float((np.array(x) > 0.5).mean())))
             .reset_index().rename(columns={"eff_date":"date_local"}))

    daily = daily.sort_values(["ticker","date_local"])
    for col in ["mean_sent", "n_articles"]:
        daily[f"{col}_r3"] = daily.groupby("ticker")[col].transform(lambda s: s.fillna(0).rolling(3, min_periods=1).mean())
        daily[f"{col}_r5"] = daily.groupby("ticker")[col].transform(lambda s: s.fillna(0).rolling(5, min_periods=1).mean())

    outp = feat / "daily_sentiment_features.csv"
    daily.to_csv(outp, index=False)
    print("Wrote", outp, daily.shape)

if __name__ == "__main__":
    main()
