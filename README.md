# GoldBot Pro

Autonomous AI-powered XAUUSD scalping bot combining XGBoost with MetaTrader 5.

## Architecture
```
MT5 (XAUUSD M5) → signal_server.py → XGBoost model → signal.csv → GoldScalperAI EA → trades
```

## Features
- XGBoost model with Optuna hyperparameter tuning
- 60+ technical features (EMA, RSI, MACD, BB, ATR, VWAP, volume, candle patterns)
- H1 + H4 higher timeframe context
- Market regime detection (TRENDING / RANGING / HIGH_VOL / LOW_VOL)
- Live news blackout filter (Forex Factory)
- Session filter (London + New York)
- Kelly-inspired position sizing (scales with win/loss streak)
- Equity curve filter + max daily trades
- Auto-retraining every Sunday with model versioning
- Telegram alerts + remote control (/pause /resume /status)
- Hourly heartbeat
- Web dashboard (localhost:5000)
- Python backtester

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost joblib optuna requests beautifulsoup4 flask schedule
cp config.example.py config.py
# Edit config.py with your MT5 credentials and Telegram token
```

## Train
```bash
python train.py
```

## Run
```bash
# Terminal 1
python signal_server.py

# Terminal 2 (optional)
python dashboard.py
```

## Backtest
```bash
python backtest.py
```

## File Structure
```
GoldBot/
├── signal_server.py      # main signal engine
├── bot.py                # paper trading loop
├── train.py              # model training + Optuna
├── backtest.py           # Python backtester
├── features.py           # feature engineering
├── indicators.py         # technical indicators
├── risk_engine.py        # position sizing + risk limits
├── retrain_scheduler.py  # weekly auto-retrain
├── news_filter.py        # Forex Factory live feed
├── notifier.py           # Telegram alerts + commands
├── dashboard.py          # web monitor
├── config.example.py     # configuration template
└── GoldScalperAI.mq5     # MT5 Expert Advisor
```

## Requirements
- Python 3.10+
- MetaTrader 5 (Wine on Linux)
- XAUUSD historical data CSVs in `data/`
