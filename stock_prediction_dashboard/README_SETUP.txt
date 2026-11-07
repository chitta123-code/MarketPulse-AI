MARKETPULSE DASHBOARD PATCH — SETUP

IMPORTANT
- This patch uses a separate model and separate scalers for every stock.
- Your existing NIFTY-only model is NOT copied into the other stock folders.
- Until a stock-specific model has been trained, its historical chart will show, but its AI prediction will say "Not trained".
- This prevents the dashboard from presenting invalid predictions as if they came from a properly trained model.

1) BACK UP YOUR PROJECT FIRST.

2) Copy these files into your existing project:
   backend/app.py
   frontend/templates/index.html
   frontend/static/style.css
   frontend/static/script.js
   train_stock_models.py

   Your project should look like:
   stock_prediction/
     backend/app.py
     frontend/templates/index.html
     frontend/static/style.css
     frontend/static/script.js
     models/stocks/
     train_stock_models.py

3) Activate your existing virtual environment in PowerShell:
   cd path\to\stock_prediction
   .\.venv\Scripts\Activate.ps1

   If your .venv is inside backend, activate that environment instead:
   .\backend\.venv\Scripts\Activate.ps1

4) Install required packages:
   python -m pip install -r requirements-dashboard.txt

5) Train NIFTY 50 first:
   python train_stock_models.py --symbols "^NSEI"

   Training may take several minutes on CPU. Wait for "Saved model artifacts".

6) Train company models separately or in small batches:
   python train_stock_models.py --symbols "TCS.NS" "RELIANCE.NS"
   python train_stock_models.py --symbols "INFY.NS" "HDFCBANK.NS"
   python train_stock_models.py --symbols "ICICIBANK.NS" "TATAMOTORS.NS" "ADANIENT.NS"

   Do not skip this step if you want an AI prediction for that company.

7) Start Flask from the project root:
   python backend/app.py

   Open:
   http://127.0.0.1:5000

8) Test:
   http://127.0.0.1:5000/api/health
   http://127.0.0.1:5000/api/stocks
   http://127.0.0.1:5000/api/dashboard?symbol=%5ENSEI

TROUBLESHOOTING
- If chart says Chart library did not load, check internet/CDN access. The chart uses Chart.js from jsdelivr.
- If Yahoo Finance returns no data, check internet, ticker symbol, and whether Yahoo is temporarily rate-limiting requests.
- If a company says "Not trained", run train_stock_models.py for that ticker.
- Do not use the old NIFTY scalers/model for individual company shares.
- Model predictions are experimental. Compare each stock model's test MAE/RMSE against its persistence baseline before describing it as better.
