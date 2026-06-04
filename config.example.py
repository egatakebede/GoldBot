# GoldBot Pro — Configuration Template
# Copy this to config.py and fill in your values

# MT5 Connection
MT5_LOGIN    = 0
MT5_PASSWORD = "your_password"
MT5_SERVER   = "YourBroker-Server"
MT5_PATH     = r"C:/Program Files/MetaTrader 5/terminal64.exe"

# Symbol & Timeframes
SYMBOL    = "XAUUSD"
TF_SIGNAL = "M5"
TF_ENTRY  = "M1"

# Telegram
TELEGRAM_TOKEN   = "your_bot_token_here"
TELEGRAM_CHAT_ID = ["your_chat_id_here"]  # must be a list

# Risk Management
RISK_PCT         = 0.005
DAILY_LOSS_LIMIT = 0.02
MAX_DRAWDOWN     = 0.15
MAX_CONSEC_LOSS  = 4
MAX_LOTS         = 5.0
MIN_CONFIDENCE   = 0.62
MAX_DAILY_TRADES = 2

# SL / TP
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.5

# Sessions (UTC)
SESSION_LONDON = (7, 12)
SESSION_NY     = (12, 17)

# News blackout window (minutes)
NEWS_BLACKOUT_MINUTES = 30

# Model versioning
MODEL_PATH     = "models/xgb_model.json"
MODEL_PATH_UBJ = "models/xgb_model.ubj"
MODEL_VERSIONS = 5
RETRAIN_DAY    = "sunday"
RETRAIN_HOUR   = 2

# Trade log
TRADE_LOG_PATH = "data/trade_log.csv"

# Spread simulation
SPREAD = 0.30

# Max simultaneous open positions
MAX_POSITIONS = 3

# Walk-forward health check
WIN_RATE_MIN = 0.60
