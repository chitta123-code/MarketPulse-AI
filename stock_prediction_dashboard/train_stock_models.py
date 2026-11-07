"""
Train one stock-specific CNN + Temporal Transformer regression model per ticker.

Run from the project root:
  python train_stock_models.py --symbols ^NSEI TCS.NS RELIANCE.NS INFY.NS

This is CPU-intensive. Start with only ^NSEI, then train one company at a time.
Each ticker gets its OWN model and feature/target scalers.
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models" / "stocks"
DEVICE = torch.device("cpu")
FEATURES = ["Open", "High", "Low", "Close", "Volume", "SMA_20", "EMA_20", "RSI", "MACD", "BB_Upper", "BB_Lower"]
LOOKBACK = 60
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 7
PARAMS = {"conv_channels": 64, "num_heads": 4, "ff_dim": 128, "dropout": 0.2}

NAMES = {
    "^NSEI": "NIFTY 50", "TCS.NS": "TCS",
    "RELIANCE.NS": "Reliance Industries", "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank", "ICICIBANK.NS": "ICICI Bank",
    "TATAMOTORS.NS": "Tata Motors", "ADANIENT.NS": "Adani Enterprises"
}

class CNNTemporalTransformer(nn.Module):
    def __init__(self, n_features, conv_channels=64, num_heads=4, ff_dim=128, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1)
        self.conv_activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(conv_channels, num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(conv_channels)
        self.ffn = nn.Sequential(nn.Linear(conv_channels, ff_dim), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(ff_dim, conv_channels))
        self.norm2 = nn.LayerNorm(conv_channels)
        self.fc = nn.Sequential(nn.Linear(conv_channels, 32), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = self.dropout(self.conv_activation(self.conv(x.transpose(1, 2)))).transpose(1, 2)
        attn, _ = self.attention(x, x, x)
        x = self.norm1(x + attn)
        x = self.norm2(x + self.ffn(x))
        return self.fc(x.mean(dim=1))

def slug(symbol):
    return symbol.replace("^", "INDEX_").replace(".", "_")

def build_frame(symbol):
    raw = yf.download(symbol, start="2015-01-01", interval="1d",
                      auto_adjust=False, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No Yahoo Finance data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            raw.columns = raw.columns.get_level_values(0)
        else:
            raw.columns = raw.columns.get_level_values(-1)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in raw.columns:
            raise RuntimeError(f"{symbol}: missing {col}")
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw["SMA_20"] = raw["Close"].rolling(20).mean()
    raw["EMA_20"] = EMAIndicator(raw["Close"], window=20).ema_indicator()
    raw["RSI"] = RSIIndicator(raw["Close"], window=14).rsi()
    raw["MACD"] = MACD(raw["Close"], window_fast=12, window_slow=26, window_sign=9).macd()
    bb = BollingerBands(raw["Close"], window=20, window_dev=2)
    raw["BB_Upper"] = bb.bollinger_hband()
    raw["BB_Lower"] = bb.bollinger_lband()
    raw["Target_Log_Return"] = np.log(raw["Close"].shift(-1) / raw["Close"])
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["Target_Log_Return"])
    if len(raw) < 500:
        raise RuntimeError(f"{symbol}: only {len(raw)} usable rows; need at least 500.")
    return raw

def make_sequences(X, y, start, end):
    xs, ys = [], []
    for i in range(max(LOOKBACK - 1, start), end):
        xs.append(X[i - LOOKBACK + 1:i + 1])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32).reshape(-1, 1)

def train_one(symbol):
    if symbol not in NAMES:
        raise ValueError(f"Ticker not in supported list: {symbol}")
    print(f"\n===== TRAINING {NAMES[symbol]} ({symbol}) =====")
    df = build_frame(symbol)
    X_raw = df[FEATURES].to_numpy(dtype=np.float64)
    y_raw = df[["Target_Log_Return"]].to_numpy(dtype=np.float64)
    n = len(df)
    train_end, val_end = int(n * .70), int(n * .85)

    x_scaler, y_scaler = StandardScaler(), StandardScaler()
    x_scaler.fit(X_raw[:train_end])
    y_scaler.fit(y_raw[:train_end])
    X = x_scaler.transform(X_raw)
    y = y_scaler.transform(y_raw)

    X_train, y_train = make_sequences(X, y, LOOKBACK - 1, train_end)
    X_val, y_val = make_sequences(X, y, train_end, val_end)
    X_test, y_test = make_sequences(X, y, val_end, n)
    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise RuntimeError("Empty train/validation/test sequence split.")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_x = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    model = CNNTemporalTransformer(len(FEATURES), **PARAMS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        print(f"Epoch {epoch:02d}/{EPOCHS} | train={np.mean(batch_losses):.5f} | val={val_loss:.5f}")

        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    model.eval()
    test_x = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(test_x).cpu().numpy()

    pred_return = y_scaler.inverse_transform(pred_scaled).reshape(-1)
    actual_return = y_scaler.inverse_transform(y_test).reshape(-1)
    test_close = df["Close"].to_numpy()[val_end: n]
    # Each test target corresponds to the close at its same row.
    model_price = test_close * np.exp(pred_return)
    actual_next_close = test_close * np.exp(actual_return)
    baseline_price = test_close

    metrics = {
        "mae": float(mean_absolute_error(actual_next_close, model_price)),
        "rmse": float(np.sqrt(mean_squared_error(actual_next_close, model_price))),
        "r2": float(r2_score(actual_next_close, model_price)),
        "baseline_mae": float(mean_absolute_error(actual_next_close, baseline_price)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(actual_next_close, baseline_price))),
        "directional_accuracy_percent": float(np.mean(np.sign(pred_return) == np.sign(actual_return)) * 100),
        "test_samples": int(len(X_test))
    }
    print("TEST METRICS:", json.dumps(metrics, indent=2))
    print("Note: compare model MAE/RMSE with baseline; do not claim improvement unless the model beats it.")

    folder = MODEL_ROOT / slug(symbol)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "best_hybrid.pt")
    joblib.dump(x_scaler, folder / "feature_scaler.pkl")
    joblib.dump(y_scaler, folder / "target_scaler.pkl")
    config = {
        "model_name": "CNNTemporalTransformer",
        "symbol": symbol,
        "company": NAMES[symbol],
        "target": "Target_Log_Return",
        "features": FEATURES,
        "lookback": LOOKBACK,
        "direction_threshold": 0.001,
        "model_parameters": {"n_features": len(FEATURES), **PARAMS},
        "metrics": metrics
    }
    with open(folder / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print("Saved model artifacts to:", folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["^NSEI"])
    args = parser.parse_args()
    for ticker in args.symbols:
        try:
            train_one(ticker)
        except Exception as exc:
            print(f"FAILED {ticker}: {type(exc).__name__}: {exc}")
