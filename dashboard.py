from flask import Flask, jsonify, render_template_string
import json, os, csv
from datetime import datetime, timezone
def utcnow(): return datetime.now(timezone.utc)
import config

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>GoldBot Pro</title>
  <meta http-equiv="refresh" content="10">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: monospace; background: #0a0a0a; color: #eee; padding: 24px; }
    h1   { color: #f0a500; font-size: 22px; margin-bottom: 4px; }
    .sub { color: #555; font-size: 12px; margin-bottom: 24px; }
    .cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }
    .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 10px;
            padding: 16px 20px; min-width: 150px; }
    .card .label { color: #666; font-size: 11px; text-transform: uppercase;
                   letter-spacing: 1px; margin-bottom: 6px; }
    .card .value { font-size: 22px; font-weight: bold; }
    .green { color: #4caf50; }
    .red   { color: #f44336; }
    .gold  { color: #f0a500; }
    h2 { color: #f0a500; font-size: 15px; margin-bottom: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { color: #f0a500; padding: 8px 12px; text-align: left;
         border-bottom: 1px solid #2a2a2a; font-weight: normal; }
    td { padding: 8px 12px; border-bottom: 1px solid #1e1e1e; }
    tr:hover td { background: #1a1a1a; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 11px; }
    .badge-buy  { background: #1b3a1f; color: #4caf50; }
    .badge-sell { background: #3a1b1b; color: #f44336; }
    .badge-tp   { background: #1b2a3a; color: #64b5f6; }
    .badge-sl   { background: #3a1b1b; color: #f44336; }
  </style>
</head>
<body>
  <h1>⚡ GoldBot Pro</h1>
  <div class="sub">Paper Mode | Auto-refresh 10s | {{ time }}</div>

  <div class="cards">
    <div class="card">
      <div class="label">Balance</div>
      <div class="value gold">${{ stats.get("balance", 0) }}</div>
    </div>
    <div class="card">
      <div class="label">Daily P&L</div>
      <div class="value {{ "green" if stats.get("daily_pnl", 0) >= 0 else "red" }}">
        ${{ stats.get("daily_pnl", 0) }}
      </div>
    </div>
    <div class="card">
      <div class="label">Win Rate</div>
      <div class="value">{{ stats.get("win_rate", 0) }}%</div>
    </div>
    <div class="card">
      <div class="label">Drawdown</div>
      <div class="value {{ "red" if stats.get("drawdown", 0) > 5 else "green" }}">
        {{ stats.get("drawdown", 0) }}%
      </div>
    </div>
    <div class="card">
      <div class="label">Total Trades</div>
      <div class="value">{{ stats.get("total_trades", 0) }}</div>
    </div>
    <div class="card">
      <div class="label">Consec Loss</div>
      <div class="value {{ "red" if stats.get("consec_loss", 0) >= 3 else "green" }}">
        {{ stats.get("consec_loss", 0) }}
      </div>
    </div>
    <div class="card">
      <div class="label">Status</div>
      <div class="value {{ "green" if stats.get("is_active", True) else "red" }}">
        {{ "ACTIVE" if stats.get("is_active", True) else "PAUSED" }}
      </div>
    </div>
  </div>

  <h2>Recent Trades</h2>
  {% if trades %}
  <table>
    <tr>
      <th>Time</th><th>Dir</th><th>Entry</th><th>Exit</th>
      <th>P&L</th><th>Reason</th><th>Conf</th><th>Regime</th>
    </tr>
    {% for t in trades %}
    <tr>
      <td>{{ t.get("time","")[:19] }}</td>
      <td><span class="badge {{ "badge-buy" if t.get("direction")=="BUY" else "badge-sell" }}">
        {{ t.get("direction","") }}</span></td>
      <td>{{ t.get("entry","") }}</td>
      <td>{{ t.get("exit","") }}</td>
      <td class="{{ "green" if float(t.get("pnl",0)) >= 0 else "red" }}">
        ${{ t.get("pnl","") }}</td>
      <td><span class="badge {{ "badge-tp" if t.get("reason")=="TP" else "badge-sl" }}">
        {{ t.get("reason","") }}</span></td>
      <td>{{ t.get("confidence","") }}</td>
      <td>{{ t.get("regime","") }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <div style="color:#555; padding:20px 0">No trades yet — bot is running in paper mode.</div>
  {% endif %}
</body>
</html>
"""

def load_stats():
    path = "data/risk_state.json"
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        print(f"[Dashboard] Could not load stats: {e}")
    return {"balance": 10000, "daily_pnl": 0, "win_rate": 0,
            "drawdown": 0, "total_trades": 0, "is_active": True,
            "consec_loss": 0}

def load_trades(n=30):
    path = config.TRADE_LOG_PATH
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"[Dashboard] Could not load trades: {e}")
        return []
    # Sanitise all string values to prevent XSS via crafted CSV entries
    safe = []
    for row in rows[-n:]:
        safe.append({k: str(v).replace("<", "&lt;").replace(">", "&gt;") for k, v in row.items()})
    return safe[::-1]

@app.route("/")
def index():
    return render_template_string(
        HTML,
        stats=load_stats(),
        trades=load_trades(),
        time=utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    )

@app.route("/api/stats")
def api_stats():
    return jsonify(load_stats())

@app.route("/api/trades")
def api_trades():
    return jsonify(load_trades())

if __name__ == "__main__":
    print("Dashboard running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)