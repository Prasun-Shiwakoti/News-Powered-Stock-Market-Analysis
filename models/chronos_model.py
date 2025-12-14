"""
Chronos-T5 Predictor for Stock Price Forecasting
Wraps the fine-tuned Chronos model to match the project's model interface.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import torch
import joblib

# Import Chronos pipeline (requires: pip install git+https://github.com/amazon-science/chronos-forecasting.git)
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    print("Warning: chronos library not installed. Install with:")
    print("  pip install git+https://github.com/amazon-science/chronos-forecasting.git")


class ChronosPredictor:
    """
    Chronos-T5 time series forecasting model wrapper.
    
    This model uses a fine-tuned T5-based architecture for probabilistic 
    time series forecasting. Unlike LSTM/XGBoost, it generates multiple 
    forecast trajectories (samples) and returns quantile-based predictions.
    """
    
    def __init__(self, 
                 context_length=512,
                 prediction_length=30,
                 num_samples=20,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 dtype=torch.bfloat16,
                 feature_cols=None,
                 use_sentiment= True):
        """
        Initialize Chronos predictor.
        
        Args:
            context_length (int): Number of historical time steps to use as context
            prediction_length (int): Number of future steps to predict
            num_samples (int): Number of forecast trajectories to sample (for probabilistic forecasting)
            device (str): Device to run inference on ('cuda' or 'cpu')
            dtype: PyTorch dtype for model weights (bfloat16 for efficiency, float32 for CPU)
            feature_cols (list): Feature columns (for compatibility; Chronos uses only close prices)
        """
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "Chronos library not installed. Please run:\n"
                "  pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.device = device
        self.dtype = dtype
        self.feature_cols = feature_cols  # For interface compatibility
        
        self.pipeline = None
        self.is_fitted = False  # For Chronos, "fitted" means "loaded from saved model"
        self.use_sentiment = use_sentiment  # For interface compatibility
        
    def safe_spearman(self, a, b):
        """Safe spearman correlation that falls back to pearson if spearman fails"""
        try:
            rho, _ = spearmanr(a, b)
            if np.isnan(rho):
                rho, _ = pearsonr(a, b)
        except Exception:
            rho, _ = pearsonr(a, b)
        return float(rho)
    
    def fit(self, train_df, val_df=None):
        """
        For Chronos, 'fit' means loading a pre-trained/fine-tuned model.
        
        NOTE: Chronos models are typically fine-tuned externally using the 
        chronos training scripts. This method loads an existing checkpoint.
        
        Args:
            train_df: Not used (kept for interface compatibility)
            val_df: Not used (kept for interface compatibility)
        """
        print("Note: Chronos models are loaded from pre-trained checkpoints.")
        print("To train/fine-tune Chronos, use the official chronos-forecasting scripts.")
        print("This method will load the model from the default save location.")
        self.load()
        
    def load(self, model_path=None):
        """
        Load a fine-tuned Chronos model from disk.
        
        Args:
            model_path (str or Path): Path to the model directory.
                If None, loads from models/saved/chronos/
        """
        if model_path is None:
            model_path = Path(__file__).parent / "saved" / "chronos"
        else:
            model_path = Path(model_path)
            
        if not model_path.exists():
            raise FileNotFoundError(
                f"Chronos model not found at {model_path}\n"
                f"Expected files: config.json, model.safetensors, generation_config.json"
            )
        
        print(f"Loading Chronos model from {model_path}...")
        
        # Load the pipeline from local path
        self.pipeline = ChronosPipeline.from_pretrained(
            str(model_path),
            device_map=self.device,
            dtype=self.dtype,
        )
        
        # Load training config if available
        config_path = model_path / "training_info.json"
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                training_info = json.load(f)
                training_config = training_info.get('training_config', {})
                
                # Override with saved config if available
                self.context_length = training_config.get('context_length', self.context_length)
                self.prediction_length = training_config.get('prediction_length', self.prediction_length)
                self.num_samples = training_config.get('num_samples', self.num_samples)
            
        self.is_fitted = True
        print("Chronos model loaded successfully!")
        
    def save(self, save_path=None):
        """
        Save method for interface compatibility.
        
        Note: Chronos models are typically saved during fine-tuning using
        the HuggingFace Trainer. This method is provided for compatibility
        but doesn't perform any action.
        """
        print("Note: Chronos models are saved during training via HuggingFace Trainer.")
        print("The model is already saved at models/saved/chronos/")
        
    def predict(self, test_df):
        """
        Generate probabilistic forecasts for each ticker in test_df and output returns instead of prices.
        
        Args:
            test_df (pd.DataFrame): Test data with columns ['ticker', 'date', 'close', ...]
        
        Returns:
            np.ndarray: Predicted returns (shape: [n_samples,])
                Each value is (predicted_price - current_price) / current_price
        """
        if not self.is_fitted:
            raise ValueError("Model must be loaded before making predictions. Call .load() first.")
        
        if 'ticker' not in test_df.columns or 'close' not in test_df.columns:
            raise ValueError("test_df must contain 'ticker' and 'close' columns")
        
        # Sort by ticker and date
        test_df = test_df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        returns = []
        
        # Generate forecasts per ticker
        for ticker, group in test_df.groupby('ticker'):
            historical_prices = group['close'].values
            # For each point, predict the next value using available history
            for i in range(len(historical_prices)):
                context_end = i
                context_start = max(0, context_end - self.context_length)
                context = historical_prices[context_start:context_end]
                if len(context) < 2 or i == 0:
                    # For first point(s), return zero return (no change)
                    returns.append(0.0)
                    continue
                context_tensor = torch.tensor(context, dtype=torch.float32)
                with torch.no_grad():
                    forecast = self.pipeline.predict(
                        inputs=context_tensor,
                        prediction_length=1,
                        num_samples=self.num_samples,
                    )
                forecast_samples = forecast[0].cpu().numpy()  # shape: [num_samples, 1]
                median_forecast = np.median(forecast_samples[:, 0])
                prev_price = historical_prices[i-1]
                predicted_return = (median_forecast - prev_price) / prev_price if prev_price != 0 else 0.0
                returns.append(predicted_return)
        return np.array(returns)
    
    def evaluate(self, test_df):
        """
        Evaluate model performance on test data.
        
        Args:
            test_df (pd.DataFrame): Test data with columns ['ticker', 'date', 'close', ...]
        
        Returns:
            pd.DataFrame: Results with columns ['ticker', 'date', 'y_true', 'y_pred', 
                         'current_price', 'true_direction', 'pred_direction', 'directional_correct']
        """
        if not self.is_fitted:
            raise ValueError("Model must be loaded before evaluation. Call .load() first.")
        
        print("Evaluating Chronos model...")
        print(f"Test samples: {len(test_df)} | Tickers: {test_df['ticker'].nunique()}")
        
        # Sort by ticker and date
        test_df = test_df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
        predictions_list = []
        
        # Generate predictions per ticker
        for ticker, group in test_df.groupby('ticker'):
            group = group.reset_index(drop=True)
            historical_prices = group['close'].values
            dates = group['date'].values
            
            for i in range(len(historical_prices) - 1):  # Predict i+1 from 0:i
                # Build context
                context_end = i + 1
                context_start = max(0, context_end - self.context_length)
                context = historical_prices[context_start:context_end]
                
                if len(context) < 2:
                    continue  # Skip insufficient context
                
                # True next price
                y_true = historical_prices[i + 1]
                current_price = historical_prices[i]
                
                # Predict
                context_tensor = torch.tensor(context, dtype=torch.float32)
                with torch.no_grad():
                    forecast = self.pipeline.predict(
                        inputs=context_tensor,
                        prediction_length=1,
                        num_samples=self.num_samples,
                    )
                
                # Median forecast
                forecast_samples = forecast[0].cpu().numpy()
                y_pred = np.median(forecast_samples[:, 0])
                
                # Directional accuracy
                true_direction = np.sign(y_true - current_price)
                pred_direction = np.sign(y_pred - current_price)
                directional_correct = (true_direction == pred_direction)
                
                predictions_list.append({
                    'ticker': ticker,
                    'date': dates[i + 1],
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'current_price': current_price,
                    'true_direction': true_direction,
                    'pred_direction': pred_direction,
                    'directional_correct': directional_correct,
                })
        
        results = pd.DataFrame(predictions_list)
        
        if len(results) == 0:
            print("No predictions generated (insufficient context).")
            return results
        
        # Compute metrics
        y_true = results['y_true'].values
        y_pred = results['y_pred'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE with safe division
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
        else:
            mape = np.nan
        
        # Directional accuracy
        directional_accuracy = results['directional_correct'].mean() * 100.0
        
        print(f"Chronos Model Evaluation:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  MAPE: {mape:.4f}%")
        print(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        return results


# Example usage (commented out):
"""
# Load and use the model
predictor = ChronosPredictor(
    context_length=512,
    prediction_length=30,
    num_samples=20,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Load fine-tuned model
predictor.load()  # loads from models/saved/chronos/

# Make predictions
predictions = predictor.predict(test_df)

# Evaluate
results = predictor.evaluate(test_df)
"""