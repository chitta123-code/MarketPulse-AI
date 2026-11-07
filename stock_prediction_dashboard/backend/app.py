import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "stocks"
FRONTEND_DIR = BASE_DIR / "frontend"
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "EMA_20", "RSI", "MACD", "BB_Upper", "BB_Lower"
]
LOOKBACK_DEFAULT = 60
DEVICE = torch.device("cpu")

STOCKS = {
    "^NSEI": "NIFTY 50",
    "TCS.NS": "TCS",
    "RELIANCE.NS": "Reliance Industries",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "TATAMOTORS.NS": "Tata Motors",
    "ADANIENT.NS": "Adani Enterprises",
}

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static")
)


class CNNTemporalTransformer(nn.Module):
    def __init__(self, n_features, conv_channels=64, num_heads=4,
                 ff_dim=128, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1)
        self.conv_activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=conv_channels, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(conv_channels)
        self.ffn = nn.Sequential(
            nn.Linear(conv_channels, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, conv_channels)
        )
        self.norm2 = nn.LayerNorm(conv_channels)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(self.conv_activation(self.conv(x)))
        x = x.transpose(1, 2)
        attn, _ = self.attention(x, x, x)
        x = self.norm1(x + attn)
        x = self.norm2(x + self.ffn(x))
        x = x.mean(dim=1)
        return self.fc(x)


def safe_symbol(symbol):
    if symbol not in STOCKS:
        raise ValueError("Unsupported stock symbol.")
    return symbol.replace("^", "INDEX_").replace(".", "_")


def create_features(raw):
    df = raw.copy()
    if df is None or df.empty:
        raise ValueError("Yahoo Finance returned no market data.")
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return either (Price, Ticker) or (Ticker, Price).
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing market columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["MACD"] = MACD(
        close=df["Close"], window_fast=12, window_slow=26, window_sign=9
    ).macd()
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)
    if len(df) < LOOKBACK_DEFAULT:
        raise ValueError(f"Not enough rows after feature calculation: {len(df)}")
    return df


def download_data(symbol, period="1y"):
    data = yf.download(
        symbol, period=period, interval="1d",
        auto_adjust=False, progress=False, threads=False
    )
    if data is None or data.empty:
        raise ValueError(f"No data returned for {symbol}. Check ticker/network.")
    return data


def load_symbol_model(symbol):
    folder = MODELS_DIR / safe_symbol(symbol)
    required = [
        folder / "best_hybrid.pt",
        folder / "feature_scaler.pkl",
        folder / "target_scaler.pkl",
        folder / "model_config.json"
    ]
    if not all(p.exists() for p in required):
        return None

    with open(required[3], "r", encoding="utf-8") as f:
        config = json.load(f)
    params = config["model_parameters"]
    model = CNNTemporalTransformer(
        n_features=params["n_features"],
        conv_channels=params["conv_channels"],
        num_heads=params["num_heads"],
        ff_dim=params["ff_dim"],
        dropout=params["dropout"]
    ).to(DEVICE)

    state = torch.load(required[0], map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    return {
        "model": model,
        "feature_scaler": joblib.load(required[1]),
        "target_scaler": joblib.load(required[2]),
        "config": config
    }


def make_dashboard_payload(symbol):
    if symbol not in STOCKS:
        raise ValueError("Unsupported stock symbol.")

    raw = download_data(symbol, period="1y")
    df = create_features(raw)
    chart_df = df.tail(90)
    current_price = float(df["Close"].iloc[-1])
    latest_date = pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d")

    payload = {
        "symbol": symbol,
        "company": STOCKS[symbol],
        "currency": "₹",
        "current_price": round(current_price, 2),
        "data_date": latest_date,
        "labels": [pd.Timestamp(x).strftime("%d %b %y") for x in chart_df.index],
        "prices": [round(float(x), 2) for x in chart_df["Close"]],
        "model_available": False,
        "prediction": None,
        "model_note": "A trained model for this stock has not been installed yet."
    }

    artifacts = load_symbol_model(symbol)
    if artifacts is None:
        return payload

    model = artifacts["model"]
    feature_scaler = artifacts["feature_scaler"]
    target_scaler = artifacts["target_scaler"]
    config = artifacts["config"]
    feature_columns = config.get("features", FEATURE_COLUMNS)
    lookback = int(config.get("lookback", LOOKBACK_DEFAULT))

    feature_values = df[feature_columns].to_numpy(dtype=np.float64)
    scaled = feature_scaler.transform(feature_values)
    if len(scaled) < lookback:
        raise ValueError(f"Need {lookback} rows; found {len(scaled)}.")

    x = torch.tensor(
        scaled[-lookback:], dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        scaled_return = model(x).cpu().numpy()

    log_return = float(target_scaler.inverse_transform(scaled_return).reshape(-1)[0])
    # Protect the UI from numerical overflow; this is not a confidence interval.
    log_return = float(np.clip(log_return, -0.20, 0.20))
    predicted_price = current_price * float(np.exp(log_return))
    threshold = float(config.get("direction_threshold", 0.001))
    direction = "UP" if log_return > threshold else (
        "DOWN" if log_return < -threshold else "FLAT"
    )

    payload.update({
        "model_available": True,
        "prediction": {
            "predicted_price": round(predicted_price, 2),
            "predicted_log_return": round(log_return, 6),
            "predicted_return_percent": round(log_return * 100, 4),
            "direction": direction
        },
        "model_note": "Experimental model output; not a guaranteed forecast."
    })

    # Append one forecast point to the chart.
    payload["forecast_label"] = "Next session*"
    payload["forecast_price"] = round(predicted_price, 2)
    return payload


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/stocks")
def api_stocks():
    return jsonify([
        {"symbol": symbol, "name": name}
        for symbol, name in STOCKS.items()
    ])


@app.route("/api/dashboard")
def api_dashboard():
    symbol = request.args.get("symbol", "^NSEI")
    try:
        return jsonify({"success": True, "data": make_dashboard_payload(symbol)})
    except Exception as exc:
        app.logger.exception("Dashboard request failed")
        return jsonify({"success": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "online",
        "model_family": "CNN + Temporal Transformer",
        "device": str(DEVICE),
        "trained_symbols": [
            s for s in STOCKS if (MODELS_DIR / safe_symbol(s) / "best_hybrid.pt").exists()
        ]
    })


if __name__ == "__main__":
    print("Stock Prediction Dashboard: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
