import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor, XGBClassifier
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch

class XGBPredictor:
    
    def __init__(self, n_estimators=800, max_depth=8, learning_rate=0.05, 
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, 
                 n_jobs=4, feature_cols=None, target_type="return",use_sentiment=True):
        """
        Initialize XGBoost predictor with configuration.
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Boosting learning rate
            subsample (float): Subsample ratio of training instances
            colsample_bytree (float): Subsample ratio of columns when constructing each tree
            reg_lambda (float): L2 regularization term
            n_jobs (int): Number of parallel threads
            tree_method (str): Tree construction algorithm
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.tree_method = "hist"
        self.target_type = target_type
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
        Train the XGBoost model.
        """

        # Prepare features and targets
        X_train = train_df[self.feature_cols]
        y_train = train_df["ret_t1_cc"] if self.target_type == "return" else train_df["ret_t1_cc"].apply(lambda x: 1 if x > 0 else 0)

        if val_df is not None and not val_df.empty:
            X_val = val_df[self.feature_cols]
            y_val = val_df["ret_t1_cc"] if self.target_type == "return" else val_df["ret_t1_cc"].apply(lambda x: 1 if x > 0 else 0)
            X_fit = pd.concat([X_train, X_val], axis=0)
            y_fit = pd.concat([y_train, y_val], axis=0)
        else:
            X_fit = X_train
            y_fit = y_train
        
        print(f"Training with {len(X_fit)} samples and {len(self.feature_cols)} features")
        print(f"Features: {self.feature_cols}")
        
        # Initialize and train model
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ) if self.target_type == "return" else XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            n_jobs=self.n_jobs,
            tree_method=self.tree_method,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )

        
        print("Training XGBoost model...")
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

        if self.target_type == "direction":
            if "direction" in test_df.columns:
                y_true_class = test_df["direction"].values.astype(int)
            else:
                y_true_class = (test_df["ret_t1_cc"].values > 0).astype(int)

            # If model provides probabilities, use them; otherwise infer from predictions
            if hasattr(self.model, "predict_proba"):
                y_pred_prob = self.model.predict_proba(test_df[self.feature_cols])[:, 1]
                y_pred_class = (y_pred_prob > 0.5).astype(int)
            else:
                # model.predict may already have returned probs or classes
                y_pred_raw = y_pred_return
                if np.issubdtype(y_pred_raw.dtype, np.floating) and y_pred_raw.max() <= 1.0:
                    y_pred_prob = y_pred_raw
                    y_pred_class = (y_pred_prob > 0.5).astype(int)
                else:
                    y_pred_prob = None
                    y_pred_class = y_pred_raw.astype(int)

            # Keep y_pred_return as numeric class labels so downstream price code runs
            y_pred_return = y_pred_class.astype(float)

            # Classification metrics

            accuracy = accuracy_score(y_true_class, y_pred_class)
            precision = precision_score(y_true_class, y_pred_class, zero_division=0)
            recall = recall_score(y_true_class, y_pred_class, zero_division=0)
            f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
            cm = confusion_matrix(y_true_class, y_pred_class)
            roc_auc = None
            if y_pred_prob is not None and len(np.unique(y_true_class)) == 2:
                try:
                    roc_auc = roc_auc_score(y_true_class, y_pred_prob)
                except Exception:
                    roc_auc = None

            print(f"Classification metrics - Acc: {accuracy:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
            if roc_auc is not None:
                print(f"ROC AUC: {roc_auc:.4f}")
            print("Confusion matrix:")
            print(cm)

            tickers = test_df["ticker"].values
            dates = test_df["date"].values
            results = pd.DataFrame({
                'ticker': tickers,
                'date': dates,
                'y_true': y_true_class,
                'y_pred': y_pred_class
            })
            return results

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
        directional_accuracy = directional_correct.mean()

        
        # Model type description
        model_type = "Sentiment Included" if self.use_sentiment else "Sentiment Excluded"
        
        print(f"Model: {model_type} XGB")
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
        """Save the trained model and config using joblib to models/saved/xgb_model_saved.joblib"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        direction_as_result = "_direction" if self.target_type == "direction" else ""
        sentiment_enabled = "_noSentiment" if not self.use_sentiment else ""
        filename = f"xgb_model_saved{direction_as_result}{sentiment_enabled}.joblib"
        
        save_path = Path(__file__).parent / "saved" / "xgb" / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_lambda': self.reg_lambda,
            'n_jobs': self.n_jobs,
            'tree_method': self.tree_method
        }, save_path)
        print(f"Model and config saved to {save_path}")

    def load(self):
        """Load a trained model and config using joblib from models/saved/xgb_model_saved.joblib"""
        direction_as_result = "_direction" if self.target_type == "direction" else ""
        sentiment_enabled = "_noSentiment" if not self.use_sentiment else ""
        filename = f"xgb/xgb_model_saved{direction_as_result}{sentiment_enabled}.joblib"
        
        load_path = Path(__file__).parent / "saved" / filename

        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found at {load_path}")
        
        data = joblib.load(load_path)
        self.model = data['model']
        self.feature_cols = data['feature_cols']
        self.n_estimators = data.get('n_estimators', self.n_estimators)
        self.max_depth = data.get('max_depth', self.max_depth)
        self.learning_rate = data.get('learning_rate', self.learning_rate)
        self.subsample = data.get('subsample', self.subsample)
        self.colsample_bytree = data.get('colsample_bytree', self.colsample_bytree)
        self.reg_lambda = data.get('reg_lambda', self.reg_lambda)
        self.n_jobs = data.get('n_jobs', self.n_jobs)
        self.tree_method = data.get('tree_method', self.tree_method)
        self.is_fitted = True
        print(f"Model and config loaded from {load_path}")