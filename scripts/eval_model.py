# Evaluate trained models on the test dataset
import argparse
import pandas as pd
from pathlib import Path
import sys
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_model import LSTMPredictor
from models.xgb_model import XGBPredictor
from models.ridge_model import RidgePredictor
from models.chronos_model import ChronosPredictor
from add_features import prepare_data

def load_config():
    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found. Please ensure it exists in the root directory.")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(cfg, use_sentiment=True,tickers=None):
    proc_dir = Path(cfg["data"]["processed_dir"])
    feat_dir = Path(cfg["data"]["features_dir"])
    prices_path = proc_dir / "prices_processed.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices data not found at {prices_path}")
    prices = pd.read_csv(prices_path)
    feats = None
    if use_sentiment:
        feats_path = feat_dir / "daily_sentiment_features.csv"
        if feats_path.exists():
            feats = pd.read_csv(feats_path).rename(columns={"date_local": "date"})
    if feats is not None:
        df = prices.merge(feats, on=["ticker", "date"], how="left")
        df = df.sort_values(["ticker", "date"])
    else:
        df = prices.sort_values(["ticker", "date"])
    # Filter by tickers if provided
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)].copy()
        print(f"Filtered data shape by tickers {tickers}: {df.shape}")

    return df

def split_data(df, cfg):
    df["date"] = pd.to_datetime(df["date"])
    years = df["date"].dt.year
    train_end_year = cfg["cv"]["train_end_year"]
    val_end_year = cfg["cv"]["val_end_year"]
    test_start_year = cfg["cv"]["test_start_year"]
    test_mask = years >= test_start_year
    test_df = df[test_mask].copy()
    return test_df

def main():
    parser = argparse.ArgumentParser(description="Load and evaluate a trained model.")
    parser.add_argument("--model", choices=["lstm", "xgb", "ridge", "chronos"], required=True, help="Model type to load")
    parser.add_argument("--target-type", choices=["return", "direction"], default="return",
                       help=" Target type (default: return)")
    
    parser.add_argument("--use-sentiment", action="store_true", default=True,
                       help="Include sentiment features (default: True)")
    parser.add_argument("--no-sentiment", action="store_false", dest="use_sentiment",
                       help="Exclude sentiment features")
    parser.add_argument("--save-predictions", action="store_true", default=False, help="Save predictions to CSV")
    args = parser.parse_args()

    if args.model != "xgb" and args.target_type == "direction":
        print("Warning: Target type 'direction' is only supported by XGBoost model. Overriding to 'return'.")
        args.target_type = "return"

    print(f"Loading config and data...")
    cfg = load_config()
    df = load_data(cfg,use_sentiment=args.use_sentiment)
    df = prepare_data(df)
    test_df = split_data(df, cfg)

    if args.model == "lstm":
        model = LSTMPredictor(use_sentiment=args.use_sentiment)
        model.load()
        print("\nLoaded LSTM Model Config:")
        print(f"  seq_len: {model.seq_len}")
        print(f"  prediction_days: {model.prediction_days}")
        print(f"  lstm_units: {model.lstm_units}")
        print(f"  dropout_rate: {model.dropout_rate}")
        print(f"  dense_units: {model.dense_units}")
        print(f"  learning_rate: {model.learning_rate}")
        print(f"  batch_size: {model.batch_size}")
        print(f"  feature_cols: {model.feature_cols}")
    elif args.model == "xgb":
        model = XGBPredictor(target_type=args.target_type,use_sentiment=args.use_sentiment)
        model.load()
        print("\nLoaded XGBoost Model Config:")
        print(f"  n_estimators: {model.n_estimators}")
        print(f"  max_depth: {model.max_depth}")
        print(f"  learning_rate: {model.learning_rate}")
        print(f"  subsample: {model.subsample}")
        print(f"  colsample_bytree: {model.colsample_bytree}")
        print(f"  reg_lambda: {model.reg_lambda}")
        print(f"  n_jobs: {model.n_jobs}")
        print(f"  tree_method: {model.tree_method}")
        print(f"  feature_cols: {model.feature_cols}")
    elif args.model == "ridge":
        model = RidgePredictor(use_sentiment=args.use_sentiment)
        model.load()
        print("\nLoaded Ridge Model Config:")
        print(f"  alpha: {model.alpha}")
        print(f"  fit_intercept: {model.fit_intercept}")
        print(f"  solver: {model.solver}")
        print(f"  max_iter: {model.max_iter}")
        print(f"  feature_cols: {model.feature_cols}")
    elif args.model == "chronos":
        model = ChronosPredictor(use_sentiment=args.use_sentiment)
        model.load()
        print("\nLoaded Chronos Model Config:")
        print(f"  Context length: {model.context_length}")
        print(f"  Prediction length: {model.prediction_length}")
        print(f"  Num samples: {model.num_samples}")

    print("\nEVALUATING MODEL\n" + "="*40)
    results = model.evaluate(test_df)
    print("Evaluation complete.")

    if args.save_predictions:
        sentiment_enabled = "_noSentiment" if not args.use_sentiment else ""
        results_dir = Path(cfg["data"]["results_dir"]) / f"{args.model}"
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = f"test_predictions_{args.model}{sentiment_enabled}.csv"
        results.to_csv(results_dir / filename, index=False)
        print(f"Predictions saved to {results_dir / filename}")

if __name__ == "__main__":
    main()
