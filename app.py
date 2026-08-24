import threading
import os
from flask import Flask  # type: ignore[import-not-found]
from scoring_bot import run_engine

app = Flask(__name__)

@app.route('/')
def home():
    return "🚀 Binance Futures ETH Signal Engine is active 24/7!"

@app.route('/health')
def health():
    return {"status": "healthy"}, 200

# Start trading bot in background thread
bot_thread = threading.Thread(target=run_engine, daemon=True)
bot_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
