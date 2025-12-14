# Add technical indicators for stock price prediction
import pandas as pd
import numpy as np

def prepare_data(df):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    
    # Calculate technical indicators for each ticker group
    df = df.groupby("ticker", group_keys=False).apply(lambda g: (
        g.assign(
            high_low=g['high']-g['low'],
            open_close=g['open']-g['close'],
            open_vs_prev_close_pct=g['open']/(g["close"].shift(1) + 1e-9) - 1,

            # Core Returns
            ret_cc=g["close"].pct_change(),
            ret_t1_cc=g["close"].pct_change().shift(-1),
            log_ret=np.log(g["close"]).diff(),

            # Lagged returns
            ret_lag1=g["close"].pct_change().shift(1),
            ret_lag2=g["close"].pct_change().shift(2),
            ret_lag3=g["close"].pct_change().shift(3),
            ret_lag5=g["close"].pct_change().shift(5),

            # Moving averages
            ma5=g["close"].rolling(5).mean(),
            ma10=g["close"].rolling(10).mean(),
            ma20=g["close"].rolling(20).mean(),
            ma50=g["close"].rolling(50).mean(),

            # Price ratios
            close_ma20_ratio=g["close"] / g["close"].rolling(20).mean(),

            # Volatility
            volatility5=g["close"].rolling(5).std(),
            volatility10=g["close"].rolling(10).std(),
            atr14=(g["high"] - g["low"]).rolling(14).mean(),

            # RSI
            rsi14=(lambda delta: 100.0 - (100.0 / (1.0 + 
                (delta.clip(lower=0).rolling(14).mean() / 
                 (-delta.clip(upper=0).rolling(14).mean() + 1e-9))
            )))(g["close"].diff()),

            # MACD
            macd=g["close"].ewm(span=12).mean() - g["close"].ewm(span=26).mean(),
            macd_signal=(g["close"].ewm(span=12).mean() - g["close"].ewm(span=26).mean()).ewm(span=9).mean(),
            
            # Momentum
            momentum5=g["close"] / g["close"].shift(5) - 1,

            # Volume features  
            vol_ma20=g["volume"].rolling(20).mean(),
            vol_ratio=g["volume"] / (g["volume"].rolling(20).mean() + 1e-9),
            obv=(np.sign(g["close"].diff()) * g["volume"]).cumsum()

        ).bfill().ffill()
    )).reset_index(drop=True)
    
    # Clean up infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df
