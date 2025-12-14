"""
Performance Metrics Generator for Stock Prediction Models
Generates comprehensive evaluation metrics for all models across sentiment variants
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from scipy.stats import spearmanr, pearsonr
from datetime import datetime
import json


class ModelEvaluator:
    """Evaluates model predictions and generates comprehensive metrics"""
    
    def __init__(self, results_dir="./"):
        self.results_dir = Path(results_dir)
        self.metrics = {}
        
    def safe_correlation(self, a, b, corr_type="spearman"):
        """Calculate correlation safely, handling NaN and inf values"""
        # Remove NaN and inf values
        mask = ~(np.isnan(a) | np.isnan(b) | np.isinf(a) | np.isinf(b))
        a_clean = a[mask]
        b_clean = b[mask]
        
        if len(a_clean) < 2:
            return 0.0
        
        try:
            if corr_type == "spearman":
                corr, _ = spearmanr(a_clean, b_clean)
            else:  # pearson
                corr, _ = pearsonr(a_clean, b_clean)
            
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception as e:
            print(f"Warning: Correlation calculation failed - {e}")
            return 0.0
    
    def calculate_regression_metrics(self, y_true, y_pred):
        """Calculate regression performance metrics"""
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {}
        
        metrics = {
            'mae': float(mean_absolute_error(y_true_clean, y_pred_clean)),
            'rmse': float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))),
            'mape': float(mean_absolute_percentage_error(y_true_clean, y_pred_clean)),
            'spearman_corr': self.safe_correlation(y_true_clean, y_pred_clean, "spearman"),
            'pearson_corr': self.safe_correlation(y_true_clean, y_pred_clean, "pearson"),
        }
        
        # Calculate directional accuracy if available
        true_direction = np.sign(y_true_clean)
        pred_direction = np.sign(y_pred_clean)
        metrics['directional_accuracy'] = float(
            np.mean(true_direction == pred_direction)
        )
        
        return metrics
    
    def calculate_classification_metrics(self, y_true, y_pred):
        """Calculate classification performance metrics (for direction prediction)"""
        
        # Ensure classification
        y_true_clean = np.array(y_true, dtype=int)
        y_pred_clean = np.array(y_pred, dtype=int)
        
        if len(np.unique(y_true_clean)) < 2:
            return {}
        
        # Determine if binary or multiclass
        num_classes = len(np.unique(y_true_clean))
        average_method = 'binary' if num_classes == 2 else 'weighted'
        
        try:
            metrics = {
                'accuracy': float(accuracy_score(y_true_clean, y_pred_clean)),
                'precision': float(precision_score(y_true_clean, y_pred_clean, average=average_method, zero_division=0)),
                'recall': float(recall_score(y_true_clean, y_pred_clean, average=average_method, zero_division=0)),
                'f1_score': float(f1_score(y_true_clean, y_pred_clean, average=average_method, zero_division=0)),
            }
            
            # Add confusion matrix only for binary classification
            if num_classes == 2:
                try:
                    cm = confusion_matrix(y_true_clean, y_pred_clean)
                    if cm.size == 4:
                        tn, fp, fn, tp = cm.ravel()
                        metrics['confusion_matrix'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
                except Exception as e:
                    print(f"Warning: Could not calculate confusion matrix - {e}")
        except Exception as e:
            print(f"Warning: Classification metrics calculation failed - {e}")
            return {}
        
        return metrics
    
    def evaluate_csv(self, csv_path):
        """Evaluate a single prediction CSV file"""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            return None
        
        result = {
            'file': csv_path.name,
            'num_samples': len(df),
            'tickers': df['ticker'].nunique() if 'ticker' in df.columns else 'N/A',
            'date_range': f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else 'N/A'
        }
        
        # Determine task type based on filename (direction = classification, others = regression)
        is_classification = 'direction' in csv_path.name.lower()
        
        if 'y_true' in df.columns and 'y_pred' in df.columns:
            if is_classification:
                result['task'] = 'classification'
                result['metrics'] = self.calculate_classification_metrics(
                    df['y_true'].values, df['y_pred'].values
                )
            else:
                result['task'] = 'regression'
                result['metrics'] = self.calculate_regression_metrics(
                    df['y_true'].values, df['y_pred'].values
                )
        
        # Per-ticker metrics if available
        if 'ticker' in df.columns and 'y_true' in df.columns and 'y_pred' in df.columns:
            result['per_ticker'] = {}
            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker]
                
                if is_classification:
                    ticker_metrics = self.calculate_classification_metrics(
                        ticker_df['y_true'].values, ticker_df['y_pred'].values
                    )
                else:
                    ticker_metrics = self.calculate_regression_metrics(
                        ticker_df['y_true'].values, ticker_df['y_pred'].values
                    )
                
                if ticker_metrics:
                    result['per_ticker'][ticker] = ticker_metrics
        
        return result
    
    def evaluate_all_models(self):
        """Evaluate all models in the results directory"""
        
        models = ['lstm', 'xgb', 'ridge', 'chronos']
        all_results = {}
        
        for model in models:
            model_dir = self.results_dir / model
            if not model_dir.exists():
                print(f"Warning: {model_dir} does not exist")
                continue
            
            all_results[model] = {}
            
            # Get all CSV files in the model directory
            csv_files = sorted(model_dir.glob('*.csv'))
            
            for csv_file in csv_files:
                print(f"Evaluating {model}/{csv_file.name}...")
                result = self.evaluate_csv(csv_file)
                if result:
                    all_results[model][csv_file.name] = result
        
        self.metrics = all_results
        return all_results
    
    def generate_summary_report(self):
        """Generate a concise summary report of all metrics"""
        
        report = []
        report.append("=" * 120)
        report.append("STOCK PRICE PREDICTION MODELS - PERFORMANCE METRICS REPORT")
        report.append("=" * 120)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for model, files in self.metrics.items():
            report.append("\n" + "█" * 120)
            report.append(f"MODEL: {model.upper()}")
            report.append("█" * 120)
            
            for filename, result in files.items():
                report.append(f"\n📄 {filename}")
                report.append(f"   Samples: {result['num_samples']} | Tickers: {result['tickers']} | Task: {result.get('task', 'N/A')}")
                report.append(f"   Date Range: {result['date_range']}")
                
                # Overall metrics
                if result.get('metrics'):
                    report.append("")
                    metrics = result['metrics']
                    metric_items = []
                    
                    # Format metrics in two columns for better readability
                    for metric_name, metric_value in metrics.items():
                        if metric_name == 'confusion_matrix':
                            cm = metric_value
                            metric_items.append(f"   CM: TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")
                        elif isinstance(metric_value, float):
                            metric_items.append(f"   {metric_name}: {metric_value:.4f}")
                    
                    for item in metric_items:
                        report.append(item)
        
        return "\n".join(report)
    
    def generate_comparison_table(self):
        """Generate a comparison table across all models"""
        
        comparison_data = []
        
        for model, files in self.metrics.items():
            for filename, result in files.items():
                if result.get('metrics'):
                    row = {
                        'Model': model,
                        'File': filename,
                        'Samples': result['num_samples'],
                        'Task': result.get('task', 'N/A')
                    }
                    
                    # Add key metrics
                    metrics = result['metrics']
                    if 'mae' in metrics:
                        row['MAE'] = f"{metrics['mae']:.4f}"
                    if 'rmse' in metrics:
                        row['RMSE'] = f"{metrics['rmse']:.4f}"
                    if 'mape' in metrics:
                        row['MAPE'] = f"{metrics['mape']:.4f}"
                    if 'accuracy' in metrics:
                        row['Accuracy'] = f"{metrics['accuracy']:.4f}"
                    if 'f1_score' in metrics:
                        row['F1-Score'] = f"{metrics['f1_score']:.4f}"
                    if 'spearman_corr' in metrics:
                        row['Spearman'] = f"{metrics['spearman_corr']:.4f}"
                    if 'directional_accuracy' in metrics:
                        row['Directional_Acc'] = f"{metrics['directional_accuracy']:.4f}"
                    
                    comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        comparison_report = []
        comparison_report.append("\n" + "=" * 100)
        comparison_report.append("MODEL COMPARISON TABLE")
        comparison_report.append("=" * 100)
        comparison_report.append("")
        comparison_report.append(comparison_df.to_string(index=False))
        
        return "\n".join(comparison_report)
    
    def generate_sentiment_analysis(self):
        """Compare metrics with and without sentiment features"""
        
        sentiment_report = []
        sentiment_report.append("\n" + "=" * 100)
        sentiment_report.append("SENTIMENT ANALYSIS: WITH vs WITHOUT SENTIMENT FEATURES")
        sentiment_report.append("=" * 100)
        sentiment_report.append("")
        
        models_with_sentiment = {}
        
        for model, files in self.metrics.items():
            with_sent = None
            without_sent = None
            
            for filename, result in files.items():
                if 'noSentiment' in filename:
                    without_sent = result
                else:
                    with_sent = result
            
            if with_sent and without_sent:
                sentiment_report.append(f"\n{model.upper()}:")
                sentiment_report.append("─" * 50)
                
                with_metrics = with_sent.get('metrics', {})
                without_metrics = without_sent.get('metrics', {})
                
                # Compare metrics
                comparison_rows = []
                for metric_key in with_metrics.keys():
                    if metric_key != 'confusion_matrix' and metric_key in without_metrics:
                        with_val = with_metrics[metric_key]
                        without_val = without_metrics[metric_key]
                        
                        if isinstance(with_val, (int, float)) and isinstance(without_val, (int, float)):
                            diff = with_val - without_val
                            pct_diff = (diff / abs(without_val) * 100) if without_val != 0 else 0
                            
                            comparison_rows.append({
                                'Metric': metric_key,
                                'With Sentiment': f"{with_val:.6f}",
                                'Without Sentiment': f"{without_val:.6f}",
                                'Difference': f"{diff:.6f}",
                                '% Change': f"{pct_diff:.2f}%"
                            })
                
                if comparison_rows:
                    comp_df = pd.DataFrame(comparison_rows)
                    sentiment_report.append(comp_df.to_string(index=False))
        
        return "\n".join(sentiment_report)
    
    def save_report(self, output_file='performance_report.txt'):
        """Save all reports to a file"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Main report
            f.write(self.generate_summary_report())
            
            # Comparison table
            f.write(self.generate_comparison_table())
            
            # Sentiment analysis
            f.write(self.generate_sentiment_analysis())
        
        print(f"\nReport saved to: {output_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj


def main():
    """Main execution function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate performance metrics for stock prediction models'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='.',
        help='Path to results directory (default: current directory)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='performance_report.txt',
        help='Output file path for the report (default: performance_report.txt)'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    print("Initializing evaluator...")
    evaluator = ModelEvaluator(results_dir=args.results_dir)
    
    # Evaluate all models
    print("Evaluating all models...")
    evaluator.evaluate_all_models()
    
    # Generate and save report
    print("Generating reports...")
    evaluator.save_report(output_file=args.output)
    
    # Print summary to console
    print("\n" + "=" * 120)
    print("METRICS SUMMARY")
    print("=" * 120)
    print(evaluator.generate_summary_report())


if __name__ == '__main__':
    main()
