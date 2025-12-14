import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib

class RidgePredictor:    
    def __init__(self, alpha=1.0, fit_intercept=True, solver='auto', max_iter=None, feature_cols=None,use_sentiment=True):
        """
        Initialize Ridge predictor with configuration.
        
        Args:
            alpha (float): Regularization strength
            fit_intercept (bool): Whether to calculate the intercept for this model
            solver (str): Solver to use in the computational routines
            max_iter (int): Maximum number of iterations for iterative solvers
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iter = max_iter
        
        self.model = None
        self.feature_cols = feature_cols
        self.is_fitted = False
        self.use_sentiment = use_sentiment
        
    def safe_spearman(self, a, b):
        """Safe spearman correlation that falls back to pearson if spearman fails"""
        rho, _ = spearmanr(a, b)
        if np.isnan(rho):
            rho, _ = pearsonr(a, b)
        return float(rho)
    
    
    def fit(self, train_df, val_df=None):
        """
        Train Ridge model matching XGBoost training approach.
        """   
        X_train = train_df[self.feature_cols]
        y_train = train_df["ret_t1_cc"]
        
        if val_df is not None and not val_df.empty:
            X_val = val_df[self.feature_cols]
            y_val = val_df["ret_t1_cc"]
            X_fit = pd.concat([X_train, X_val], axis=0)
            y_fit = pd.concat([y_train, y_val], axis=0)
        else:
            X_fit = X_train
            y_fit = y_train

        print(f"Training with {len(X_fit)} samples and {len(self.feature_cols)} features")
        print(f"Target: Next day closing price (ret_t1_cc)")
        
        # Initialize and train model
        self.model = Ridge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            max_iter=self.max_iter
        )
        
        print("Training Ridge model...")
        print(f"NaN columns in training data (X_fit): {X_fit.columns[X_fit.isna().any()].tolist()}")
        self.model.fit(X_fit, y_fit)
        self.is_fitted = True
        
        print("Training completed!")
    
    def predict(self, test_df):
        """
        Make predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_test = test_df[self.feature_cols]
        
        predictions = self.model.predict(X_test)
        return predictions
    
    def evaluate(self, test_df):
        """
        Evaluate model
        """
        y_pred_return = self.predict(test_df)
        
        # Get true values
        y_true_return = test_df["ret_t1_cc"].values

        # Convert returns to prices for evaluation in prices
        current_prices = test_df["close"].values
        y_true_prices = current_prices * (1 + y_true_return)
        y_pred_prices = current_prices * (1 + y_pred_return)
        
        # Overall metrics
        mae = mean_absolute_error(y_true_prices, y_pred_prices)
        mape = mean_absolute_percentage_error(y_true_prices, y_pred_prices) * 100
    
        # Calculate directional accuracy
        true_direction = np.sign(y_true_return)
        pred_direction = np.sign(y_pred_return)
        directional_correct = (true_direction == pred_direction)
        directional_accuracy = (true_direction == pred_direction).mean()

        
        # Model type description
        model_type = "Sentiment Included" if self.use_sentiment else "Sentiment Excluded"
        
        # Print results
        print(f"Model: {model_type} RIDGE")
        print(f"Feature list: {self.feature_cols}")
        print(f"Test samples: {len(test_df)} | Unique prediction days: {test_df['date'].nunique()}")
        print(f"Target: Next day closing price (actual $)")
        print(f"Test Metrics - MAE: ${mae:.2f} | MAPE: {mape:.4f}% | Directional Acc: {directional_accuracy:.3f}")

        tickers = test_df["ticker"].values
        dates = test_df["date"].values
        results = pd.DataFrame({
            'ticker': tickers,
            'date': dates,
            'y_true': y_true_prices,
            'y_pred': y_pred_prices,
            'current_price': current_prices,
            'true_direction': true_direction,
            'pred_direction': pred_direction,
            'directional_correct': directional_correct
        })
        return results
    
    
    def save(self):
        """Save the trained model and config using joblib to models/saved/ridge_model_saved.joblib"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        sentiment_enabled = "_noSentiment" if not self.use_sentiment else ""
        filename = f"ridge_model_saved{sentiment_enabled}.joblib"
        save_path = Path(__file__).parent / "saved" / "ridge"/ filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'solver': self.solver,
            'max_iter': self.max_iter
        }, save_path)
        print(f"Model and config saved to {save_path}")

    def load(self):
        """Load a trained model and config using joblib from models/saved/ridge_model_saved.joblib"""
        sentiment_enabled = "_noSentiment" if not self.use_sentiment else ""
        filename = f"ridge/ridge_model_saved{sentiment_enabled}.joblib"
        load_path = Path(__file__).parent / "saved" / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at {load_path}")
        
        data = joblib.load(load_path)
        self.model = data['model']
        self.feature_cols = data['feature_cols']
        self.alpha = data.get('alpha', self.alpha)
        self.fit_intercept = data.get('fit_intercept', self.fit_intercept)
        self.solver = data.get('solver', self.solver)
        self.max_iter = data.get('max_iter', self.max_iter)
        self.is_fitted = True
        print(f"Model and config loaded from {load_path}")
