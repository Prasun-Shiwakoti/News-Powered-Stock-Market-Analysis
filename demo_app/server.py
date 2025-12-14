"""
Flask backend for LSTM Stock Prediction Demo
Provides REST API for the vanilla JS frontend
"""
from flask import Flask, jsonify, request, send_from_directory
from pyngrok import ngrok
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import joblib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_model import LSTMPredictor
from models.xgb_model import XGBPredictor
from models.ridge_model import RidgePredictor
from models.chronos_model import ChronosPredictor
from scripts.add_features import prepare_data
import yaml

app = Flask(__name__)

# Global state
model = None
model_type_loaded = None
test_data = None
has_sentiment = False


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("../config.yaml")
    if not config_path.exists():
        config_path = Path("config.yaml")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data_internal(cfg):
    """Load and prepare data"""
    proc_dir = Path(cfg["data"]["processed_dir"])
    feat_dir = Path(cfg["data"]["features_dir"])
    
    # Load price data 
    prices_path = proc_dir / "prices_processed.csv"
    prices = pd.read_csv(prices_path)
    
    # Load sentiment features
    feats_path = feat_dir / "daily_sentiment_features.csv"
    
    feats = None
    if feats_path.exists():
        feats = pd.read_csv(feats_path)
        feats = feats.rename(columns={"date_local": "date"})
    
    # Merge data
    if feats is not None:
        df = prices.merge(feats, on=["ticker", "date"], how="left")
        df = df.sort_values(["ticker", "date"])
        has_sent = True
    else:
        df = prices.sort_values(["ticker", "date"])
        has_sent = False
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= cfg["cv"]["test_start_year"]]
    return df, has_sent

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    """Serve static files"""
    return send_from_directory('.', path)

@app.route('/api/data/test', methods=['GET'])
def get_test_data():
    """Get test dataset"""
    global test_data, has_sentiment
    
    try:
        if test_data is None:
            # Load data
            cfg = load_config()
            df, has_sentiment = load_data_internal(cfg)
            test_df = prepare_data(df)
            
            # Convert to JSON-serializable format
            test_df['date'] = test_df['date'].dt.strftime('%Y-%m-%d')
            test_data = test_df.to_dict('records')
        
        return jsonify({
            'success': True,
            'data': test_data,
            'has_sentiment': has_sentiment
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model/config', methods=['GET'])
def get_lstm_config():
    try:
        joblib_path = Path(__file__).parent.parent / 'models' / 'saved' / 'lstm' / 'lstm_model_saved.joblib'
        if not joblib_path.exists():
            return jsonify({'success': False, 'error': 'Config file not found!'}), 404
        config = joblib.load(joblib_path)
        seq_len = config.get('seq_len')
        prediction_days = config.get('prediction_days')
        return jsonify({'success': True, 'minDemoWindow': seq_len + prediction_days})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    

@app.route('/api/model/load', methods=['POST'])
def load_model_endpoint():
    """Load selected model (LSTM, XGB, Ridge)"""
    global model, model_type_loaded
    try:
        req = request.get_json(force=True)
        model_type = req.get('model_type', 'xgb')
        if model_type == 'xgb':
            model = XGBPredictor()
            model.load()
        elif model_type == 'lstm':
            model = LSTMPredictor()
            model.load()
        elif model_type == 'ridge':
            model = RidgePredictor()
            model.load()
        elif model_type == 'chronos':
            model = ChronosPredictor()
            model.load()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        model_type_loaded = model_type
        return jsonify({
            'success': True,
            'message': f'{model_type.upper()} model loaded successfully'
        })
    except Exception as e:
        print(e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on provided data using selected model"""
    global model, model_type_loaded
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 400
        data = request.json
        demo_data = pd.DataFrame(data['data'])
        demo_data['date'] = pd.to_datetime(demo_data['date'])
        # Use model_type from request if provided, else fallback to loaded
        model_type = data.get('model_type', model_type_loaded or 'xgb')
        # If model_type changed, reload model
        if model_type != model_type_loaded:
            if model_type == 'xgb':
                model = XGBPredictor()
                model.load()
            elif model_type == 'lstm':
                model = LSTMPredictor()
                model.load()
            elif model_type == 'ridge':
                model = RidgePredictor()
                model.load()
            elif model_type == 'chronos':
                model = ChronosPredictor()
                model.load()
            else:
                return jsonify({'success': False, 'error': f'Unknown model_type: {model_type}'}), 400
            model_type_loaded = model_type
        # Make prediction
        predictions = model.predict(demo_data)
        if len(predictions) > 0:
            last_prediction = predictions[-1]
            if hasattr(last_prediction, '__len__') and len(last_prediction) > 0:
                prediction = float(last_prediction[0])
            else:
                prediction = float(last_prediction)
        else:
            return jsonify({
                'success': False,
                'error': 'No predictions generated'
            }), 500
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def main():
    parser = argparse.ArgumentParser(description="Run a Flask server in Google Colab with an ngrok tunnel.")
    parser.add_argument("--ngrok-token", required=False,
                        help="The ngrok authentication token from your ngrok dashboard. Required if not using local mode.")
    parser.add_argument("--use-local", type=lambda x: (str(x).lower() == 'true'), default=True,
                        choices=[True, False],
                        help="Whether to use locally or not. Accepts true/false.")
    args = parser.parse_args()

    port = 5000
    if not args.use_local:
        if not args.ngrok_token:
            print("Error: --ngrok-token is required when using remotely")
            sys.exit(1)
        ngrok.set_auth_token(args.ngrok_token)
        public_url = ngrok.connect(port).public_url
        print(f"Running app remotely at: {public_url}")
    else:
        print(f"Running app locally at: http://localhost:{port}")

    try:
        app.run(port=port)  
    except Exception as e:
        print(f"Error running the server: {e}")


if __name__ == '__main__':
    main()
        

