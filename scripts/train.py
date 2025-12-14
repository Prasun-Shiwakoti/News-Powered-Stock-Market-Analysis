# Trains multiple stock prediction models (LSTM, XGBoost, Ridge)
import argparse
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_model import LSTMPredictor
from models.xgb_model import XGBPredictor
from models.ridge_model import RidgePredictor
from add_features import prepare_data

def load_config():
    # Load configuration from config.yaml

    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found. Please ensure it exists in the root directory.")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(cfg,use_sentiment=True, tickers=None):  
    # Load and prepare data following the original train.py pattern

    proc_dir = Path(cfg["data"]["processed_dir"])
    feat_dir = Path(cfg["data"]["features_dir"])
    
    # Load price data
    prices_path = proc_dir / "prices_processed.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"Prices data not found at {prices_path}")
    
    prices = pd.read_csv(prices_path)
    print(f"Loaded prices: {prices.shape}")
    
    # Load sentiment features
    feats = None
    if use_sentiment == True:
        feats_path = feat_dir / "daily_sentiment_features.csv"
        if feats_path.exists():
            feats = pd.read_csv(feats_path).rename(columns={"date_local": "date"})
            print(f"Loaded sentiment features: {feats.shape}")
        else:
            print("Warning: Sentiment features not found. Will run without sentiment features.")
    
    # Merge data if sentiment features are available
    if feats is not None:
        df = prices.merge(feats, on=["ticker", "date"], how="left")
        df = df.sort_values(["ticker", "date"])
        print(f"Merged data shape: {df.shape}")
    else:
        df = prices.sort_values(["ticker", "date"])

    # Filter by tickers if provided
    if tickers is not None:
        df = df[df["ticker"].isin(tickers)].copy()
        print(f"Filtered data shape by tickers {tickers}: {df.shape}")

    return df, feats is not None


def split_data(df, cfg):
    # Split data into train/val/test sets following the original train.py pattern

    df["date"] = pd.to_datetime(df["date"])
    years = df["date"].dt.year
    
    # Get split years from config
    train_end_year = cfg["cv"]["train_end_year"]
    val_end_year = cfg["cv"]["val_end_year"]
    test_start_year = cfg["cv"]["test_start_year"]
    
    # Create splits
    train_mask = years <= train_end_year
    val_mask = (years > train_end_year) & (years <= val_end_year)
    test_mask = years >= test_start_year
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Data splits:")
    print(f"  Train: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Val:   {len(val_df)} samples ({val_df['date'].min().date()} to {val_df['date'].max().date()})" if len(val_df) > 0 else "  Val:   0 samples")
    print(f"  Test:  {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    if len(test_df) == 0:
        raise ValueError("Empty test set. Please adjust CV years in config.yaml")
    
    return train_df, val_df, test_df

def main():

    parser = argparse.ArgumentParser(description="Training script for stock prediction models")
    
    # Model selection
    parser.add_argument("--model", choices=["lstm", "xgb", "ridge"], required=True,
                       help="Model type to train")
    parser.add_argument("--target-type", choices=["return", "direction"], default="return",
                       help=" Target type (default: return)")
    
    # Common parameters
    parser.add_argument("--use-sentiment", action="store_true", default=True,
                       help="Include sentiment features (default: True)")
    parser.add_argument("--no-sentiment", action="store_false", dest="use_sentiment",
                       help="Exclude sentiment features")
    parser.add_argument("--save-model", action="store_true", default=False,
                       help="Flag to save the trained model to the default location (models/saved/{model}_model_saved)")
    parser.add_argument("--save-predictions", action="store_true", default=False,
                       help="Save predictions to CSV/Parquet files")
    
    # LSTM-specific parameters
    parser.add_argument("--seq-len", type=int, default=60,
                       help="LSTM sequence length (default: 60)")
    parser.add_argument("--prediction-days", type=int, default=5,
                       help="LSTM prediction days (default: 5)")
    parser.add_argument("--lstm-units", nargs="+", type=int, default=[256, 128, 64],
                       help="LSTM layer units (default: 256 128 64)")
    parser.add_argument("--dropout-rate", type=float, default=0.2,
                       help="LSTM dropout rate (default: 0.2)")
    parser.add_argument("--dense-units", type=int, default=64,
                       help="LSTM dense layer units (default: 64)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="LSTM learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="LSTM batch size (default: 128)")
    parser.add_argument("--epochs", type=int, default=25,
                       help="LSTM training epochs (default: 25)")
    parser.add_argument("--patience", type=int, default=5,
                       help="LSTM early stopping patience (default: 5)")
    
    # XGBoost-specific parameters
    parser.add_argument("--n-estimators", type=int, default=800,
                       help="XGBoost n_estimators (default: 800)")
    parser.add_argument("--max-depth", type=int, default=8,
                       help="XGBoost max_depth (default: 8)")
    parser.add_argument("--lr", type=float, default=0.05,
                       help="XGBoost learning_rate (default: 0.05)")
    parser.add_argument("--subsample", type=float, default=0.8,
                       help="XGBoost subsample (default: 0.8)")
    parser.add_argument("--colsample", type=float, default=0.8,
                       help="XGBoost colsample_bytree (default: 0.8)")
    
    # Ridge-specific parameters
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Ridge alpha (default: 1.0)")
    parser.add_argument("--ridge-solver", choices=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], 
                       default="auto", help="Ridge solver (default: auto)")
    parser.add_argument("--ridge-max-iter", type=int, default=None,
                       help="Ridge max iterations for iterative solvers")
    parser.add_argument("--ridge-no-intercept", action="store_true", default=False,
                       help="Disable Ridge intercept fitting")
    parser.add_argument("--ridge-target", choices=["return", "price"], default="return",
                       help="Ridge target type (default: return)")
    
    args = parser.parse_args()

    if args.model != "xgb" and args.target_type == "direction":
        print("Warning: Target type 'direction' is only supported by XGBoost model. Overriding to 'return'.")
        args.target_type = "return"
    
    print(f"Training {args.model.upper()} model...")
    print(f"Sentiment features: {'Enabled' if args.use_sentiment else 'Disabled'}")
    

    # Load configuration and data
    cfg = load_config()
    df, has_sentiment = load_data(cfg,use_sentiment=args.use_sentiment)

    # Check sentiment availability
    if args.use_sentiment and not has_sentiment:
        print("Warning: Sentiment features requested but not available. Running without sentiment.")
        args.use_sentiment = False

    # Split data
    print("Preparing data...")
    df = prepare_data(df)

    print("Splitting data...")
    train_df, val_df, test_df = split_data(df, cfg)
    feature_cols = [col for col in df.columns if col not in ["ticker", "date", "ret_t1_cc"]]

    # Initialize model based on type
    if args.model == "lstm":
        model = LSTMPredictor(
            seq_len=args.seq_len,
            prediction_days=args.prediction_days,
            lstm_units=args.lstm_units,
            dropout_rate=args.dropout_rate,
            dense_units=args.dense_units,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            feature_cols=feature_cols,
            use_sentiment = args.use_sentiment
        )
        
        print(f"LSTM Configuration:")
        print(f"  Sequence length: {args.seq_len}")
        print(f"  Prediction days: {args.prediction_days}")
        print(f"  LSTM units: {args.lstm_units}")
        print(f"  Dropout rate: {args.dropout_rate}")
        print(f"  Dense units: {args.dense_units}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        print(f"  Patience: {args.patience}")
        
    elif args.model == "xgb":
        model = XGBPredictor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.lr,
            subsample=args.subsample,
            colsample_bytree=args.colsample,
            feature_cols=feature_cols,
            target_type=args.target_type,
            use_sentiment = args.use_sentiment
        )
        
        print(f"XGBoost Configuration:")
        print(f"  N estimators: {args.n_estimators}")
        print(f"  Max depth: {args.max_depth}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Subsample: {args.subsample}")
        print(f"  Colsample bytree: {args.colsample}")
        
    elif args.model == "ridge":
        model = RidgePredictor(
            alpha=args.alpha,
            fit_intercept=not args.ridge_no_intercept,
            solver=args.ridge_solver,
            max_iter=args.ridge_max_iter,
            feature_cols=feature_cols,
            use_sentiment = args.use_sentiment
        )
        
        print(f"Ridge Configuration:")
        print(f"  Alpha: {args.alpha}")
        print(f"  Solver: {args.ridge_solver}")
        print(f"  Max iterations: {args.ridge_max_iter}")
        print(f"  Fit intercept: {not args.ridge_no_intercept}")
        print(f"  Target type: {args.ridge_target}")
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    try:
        if args.model == "lstm":
            model.fit(
                train_df=train_df, 
                val_df=val_df if len(val_df) > 0 else None,
                epochs=args.epochs,
                patience=args.patience
            )
        elif args.model == "xgb":
            model.fit(
                train_df,
                val_df if len(val_df) > 0 else None
            )
        elif args.model == "ridge":
            model.fit(
                train_df,
                val_df if len(val_df) > 0 else None
            )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return 1
    
    # Evaluate model
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    try:
        if args.model == "lstm":
            results = model.evaluate(test_df)
        elif args.model == "xgb":
            results = model.evaluate(test_df)
        elif args.model == "ridge":
            results = model.evaluate(test_df)
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return 1
    
    # Save model to default location if flag is set
    if args.save_model:
        print(f"\nSaving model to default location...")
        try:
            model.save()
            print("Model saved successfully!")
        except Exception as e:
            print(f"Failed to save model: {str(e)}")
    
    # Save predictions if requested
    if args.save_predictions:
        print("\nSaving predictions...")
        try:
            feat_dir = Path(cfg["data"]["results_dir"])
            feat_dir.mkdir(parents=True, exist_ok=True)
            
            # Create filename based on model type and settings
            direction_as_result = "_direction" if args.target_type == "direction" else ""
            sentiment_enabled = "_noSentiment" if not args.use_sentiment else ""
            
            if args.model == "lstm":
                filename = f"lstm/test_predictions_lstm{sentiment_enabled}"
            elif args.model == "xgb":
                filename = f"xgb/test_predictions_xgb{direction_as_result}{sentiment_enabled}"
            elif args.model == "ridge":
                filename = f"ridge/test_predictions_ridge{sentiment_enabled}"
            
            # Save CSV
            csv_path = feat_dir / f"{filename}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(csv_path, index=False)

            print(f"Predictions saved to: {csv_path}")

        except Exception as e:
            print(f"Failed to save predictions: {str(e)}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    main()