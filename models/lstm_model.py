import argparse
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class LSTMPredictor:
    def __init__(self, seq_len=60, prediction_days=5,
                 lstm_units=[256, 128, 64], dropout_rate=0.2, dense_units=64,
                 learning_rate=0.001, batch_size=128,feature_cols=None,use_sentiment=True):
        """
        Initialize LSTM predictor with configuration.
        
        Args:
            seq_len (int): Number of past days to use for prediction
            prediction_days (int): Number of future days to predict
            lstm_units (list): List of LSTM layer units
            dropout_rate (float): Dropout rate for regularization
            dense_units (int): Number of units in dense layer
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Training batch size
        """
        self.seq_len = seq_len
        self.prediction_days = prediction_days
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.feature_cols = feature_cols
        self.scaler_X = None
        self.is_fitted = False
        self.use_sentiment = use_sentiment
        
    def safe_spearman(self, a, b):
        """Safe spearman correlation that falls back to pearson if spearman fails"""
        rho, _ = spearmanr(a, b)
        if np.isnan(rho):
            rho, _ = pearsonr(a, b)
        return float(rho)

    def build_sequences_for_price_prediction(
        self, df, feat_cols, price_col='close', return_metadata=False
    ):
        X_list, y_list, last_prices = [], [], []
        metadata_list = []  # Store ticker, date, current_price

        df = df.sort_values(['ticker','date']).reset_index(drop=True)
        grouped = df.groupby('ticker')

        for ticker, g in grouped:
            feats = g[feat_cols].values.astype(np.float32)
            prices = g[price_col].values.astype(np.float64)
            dates = g['date'].values
            n = len(g)
            if n < self.seq_len + self.prediction_days:
                continue

            for i in range(n - self.seq_len - self.prediction_days + 1):
                X_list.append(feats[i:i+self.seq_len])

                last_idx = i + self.seq_len - 1
                future_idx_start = last_idx + 1
                future_idx_end = last_idx + self.prediction_days + 1
                future_prices = prices[future_idx_start:future_idx_end]

                last_prices.append(prices[last_idx])

                y_window = (future_prices / prices[last_idx] - 1.0).astype(np.float32)
                y_list.append(y_window)

                if return_metadata:
                    # Store metadata for each prediction
                    metadata_list.append({
                        'ticker': ticker,
                        'date': dates[future_idx_start],  # Date of the first predicted day
                        'current_price': prices[last_idx]
                    })

        X = np.stack(X_list) if X_list else np.zeros((0, self.seq_len, len(feat_cols)), dtype=np.float32)
        y = np.stack(y_list) if y_list else np.zeros((0, self.prediction_days), dtype=np.float32)
        last_prices = np.array(last_prices, dtype=np.float64) if last_prices else np.zeros((0,), dtype=np.float64)

        if return_metadata:
            return X, y, last_prices, metadata_list
        return X, y, last_prices

    def fit_scaler_on_sequences(self, X_train, scaler=None, scaler_type=StandardScaler):
        # X_train: (B, S, F)
        B, S, F = X_train.shape
        X_flat = X_train.reshape(-1, F)          # (B*S, F)

        if scaler is None:
            scaler = scaler_type()
            X_flat_scaled = scaler.fit_transform(X_flat)
        else:
            X_flat_scaled = scaler.transform(X_flat)
    
        X_scaled = X_flat_scaled.reshape(B, S, F)
        return X_scaled, scaler
    

    def fit(self, train_df, val_df=None, epochs=25, patience=5):
        """Train LSTM model following the same interface as other models."""
        print(f"Feature columns: {self.feature_cols}")
        print(f"Train set: {len(train_df)} samples")
        if val_df is not None:
            print(f"Val set: {len(val_df)} samples")

        # Build sequences
        X_train, y_train, _ = self.build_sequences_for_price_prediction(
            train_df, self.feature_cols, price_col='close'
        )
        
        if val_df is not None:
            X_val, y_val, _ = self.build_sequences_for_price_prediction(
                val_df, self.feature_cols, price_col='close'
            )
        else:
            X_val, y_val = None, None

        if len(X_train) == 0:
            raise ValueError("No training sequences generated. Try reducing seq_len or check data density per ticker.")

        # Scale features
        X_train_scaled, self.scaler_X = self.fit_scaler_on_sequences(X_train)
        if X_val is not None:
            X_val_scaled, _ = self.fit_scaler_on_sequences(X_val, scaler=self.scaler_X)
        else:
            X_val_scaled = None

        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train)).shuffle(10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        if X_val_scaled is not None:
            val_ds = tf.data.Dataset.from_tensor_slices((X_val_scaled, y_val)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            val_ds = None

        print(f"Training sequences: {X_train.shape}")
        print(f"Training targets: {y_train.shape}")
        if X_val is not None:
            print(f"Validation sequences: {X_val.shape}")

        # Create callbacks
        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]

        # Build and compile model
        self.model = models.Sequential()
        self.model.add(layers.Input(shape=(self.seq_len, X_train.shape[2])))
        
        # Add LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            self.model.add(layers.LSTM(units, return_sequences=return_sequences))
            self.model.add(layers.Dropout(self.dropout_rate))
        
        # Add dense layers
        self.model.add(layers.Dense(self.dense_units, activation='relu'))
        self.model.add(layers.Dense(self.prediction_days))
        
        self.model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])

        print("Training LSTM model...")
        # Train model
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=1,
            callbacks=cb if val_ds is not None else []
        )

        self.is_fitted = True
        print("Training completed!")
        
        return history

    def predict(self, test_df):
        """Make predictions using the trained LSTM model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.feature_cols is None:
            raise ValueError("feature_cols is not set. Please load a model or fit one first.")

        X_test, _, _ = self.build_sequences_for_price_prediction(
            test_df, self.feature_cols, price_col='close'
        )
        
        if len(X_test) == 0:
            raise ValueError("No test sequences generated")
            
        X_test_scaled, _ = self.fit_scaler_on_sequences(X_test, scaler=self.scaler_X)
        predictions = self.model.predict(X_test_scaled)
        
        return predictions

    def evaluate(self, test_df):
        """
        Evaluate LSTM model performance.
        Returns a dictionary with metrics and a predictions DataFrame.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        X_test, y_test, last_prices, metadata = self.build_sequences_for_price_prediction(
            test_df, self.feature_cols, price_col='close', return_metadata=True
        )
        
        if len(X_test) == 0:
            raise ValueError("No test sequences generated")
            
        X_test_scaled, _ = self.fit_scaler_on_sequences(X_test, scaler=self.scaler_X)
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test)).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # predict
        preds = self.model.predict(test_ds)
        # handle predict returning list of arrays or one array
        if isinstance(preds, list):
            preds = np.vstack(preds)
        preds = np.asarray(preds)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
            
        # collect ground-truth y from test_ds (same ordering)
        y_true = np.vstack([y for _, y in test_ds])
        y_true = np.asarray(y_true)

        n_samples = y_true.shape[0]
        assert last_prices.shape[0] == n_samples, "last_prices length must match number of samples"

        true_prices = last_prices[:, None] * (1.0 + y_true)
        pred_prices = last_prices[:, None] * (1.0 + preds)
        true_returns = y_true
        pred_returns = preds

        FUTURE_DAYS = true_prices.shape[1]
        metrics = {}

        # Build predictions DataFrame: only store day 1 prediction in predictions_list
        predictions_list = []
        for i in range(n_samples):
            for day in range(FUTURE_DAYS):
                y_true_price = true_prices[i, day]
                y_pred_price = pred_prices[i, day]
                current_price = last_prices[i]
                # Direction: -1 for down, 1 for up
                true_direction = np.sign(true_returns[i, day])
                pred_direction = np.sign(pred_returns[i, day])
                directional_correct = (true_direction == pred_direction)
                # Only store day 1 prediction in predictions_list
                if day == 0:
                    predictions_list.append({
                        'ticker': metadata[i]['ticker'],
                        'date': metadata[i]['date'],
                        'y_true': y_true_price,
                        'y_pred': y_pred_price,
                        'current_price': current_price,
                        'true_direction': true_direction,
                        'pred_direction': pred_direction,
                        'directional_correct': directional_correct
                    })
        
        results = pd.DataFrame(predictions_list)

        # Calculate metrics per day
        for i in range(FUTURE_DAYS):
            t = true_prices[:, i]
            p = pred_prices[:, i]
            t_ret = true_returns[:, i]
            p_ret = pred_returns[:, i]

            mae = mean_absolute_error(t, p)

            # MAPE with safe division
            mask = t != 0
            if mask.sum() == 0:
                mape = np.nan
            else:
                mape = np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100.0

            # Directional Accuracy
            correct_direction = np.sum(np.sign(t_ret) == np.sign(p_ret))
            directional_accuracy = correct_direction / n_samples * 100.0

            metrics[f"day_{i+1}"] = {
                "mae": float(mae),
                "mape_pct": float(mape),
                "directional_accuracy_pct": float(directional_accuracy)
            }

            print(f"Day {i+1}: MAE={mae:.4f}, MAPE={mape:.4f}%, Directional Accuracy={directional_accuracy:.4f}%")

        return results


    def save(self):
        """Save the trained model, scaler, and config to models/saved/lstm_model_saved"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        sentiment_enabled = "_noSentiment" if not self.use_sentiment else ""
        filenameModel = f"lstm_model_saved{sentiment_enabled}.keras"
        filenameConfig = f"lstm_model_saved{sentiment_enabled}.joblib"
        save_dir = Path(__file__).parent / "saved" /"lstm"
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = save_dir / filenameModel
        joblib_path = save_dir / filenameConfig
        # Save Keras model
        self.model.save(model_path)
        # Save scaler and config
        config_to_save = {
            'seq_len': self.seq_len,
            'prediction_days': self.prediction_days,
            'feature_cols': self.feature_cols,
            'scaler_X': self.scaler_X,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'dense_units': self.dense_units,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        joblib.dump(config_to_save, joblib_path)
        print(f"Model saved to {model_path}")
        print(f"Configuration and scaler saved to {joblib_path}")


    def load(self):
        """Load a trained model, scaler, and config from models/saved/lstm_model_saved"""
        sentiment_enabled = "_noSentiment" if not self.use_sentiment else ""
        filenameModel = f"lstm_model_saved{sentiment_enabled}.keras"
        filenameConfig = f"lstm_model_saved{sentiment_enabled}.joblib"
        save_dir = Path(__file__).parent / "saved" / "lstm"
        model_path = save_dir / filenameModel
        joblib_path = save_dir / filenameConfig

        if not model_path.exists():
            raise FileNotFoundError(f"Keras model file not found at {model_path}")
        if not joblib_path.exists():
            raise FileNotFoundError(f"Config/scaler file not found at {joblib_path}")
        
        # Load Keras model
        self.model = tf.keras.models.load_model(model_path)
        # Load scaler and config
        config_loaded = joblib.load(joblib_path)
        self.seq_len = config_loaded['seq_len']
        self.prediction_days = config_loaded['prediction_days']
        self.feature_cols = config_loaded['feature_cols']
        self.scaler_X = config_loaded['scaler_X']
        self.lstm_units = config_loaded.get('lstm_units', self.lstm_units)
        self.dropout_rate = config_loaded.get('dropout_rate', self.dropout_rate)
        self.dense_units = config_loaded.get('dense_units', self.dense_units)
        self.learning_rate = config_loaded.get('learning_rate', self.learning_rate)
        self.batch_size = config_loaded.get('batch_size', self.batch_size)
        self.is_fitted = True
        print(f"Model loaded from {model_path}")
        print(f"Configuration and scaler loaded from {joblib_path}")