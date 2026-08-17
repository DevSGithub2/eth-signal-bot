# Binance 6-Factor Confluence Signal Bot

## Key Highlights & Features
* **Zero API Key Dependency:** Uses public Binance Futures REST endpoints to fetch live market order flow and Open Interest data.
* **6-Factor Quantitative Scoring:** Filters out chop and sideways noise by scoring setups from `0/6` to `6/6` (alerts trigger only at ≥ 5/6).
* **Order Flow & CVD Tracking:** Computes real-time Cumulative Volume Delta (CVD) to track aggressive taker buyers vs. sellers.
* **Concurrent Web Server:** Uses a lightweight Flask & Gunicorn WSGI architecture with a `/health` endpoint to satisfy cloud health checks.
* **Asynchronous Background Processing:** Employs Python daemon threads (`threading.Thread`) to run continuous market evaluations independently.
* **24/7 Cloud Architecture:** Hosted on Render with continuous Git deployment and external keep-alive monitoring via UptimeRobot.
* **Rich HTML Telegram Alerts:** Formats and dispatches high-confluence cards with precise technical breakdowns.

## Strategy & 6-Factor Confluence Model
The engine monitors closed 5-minute candles (`df.iloc[-2]`) and scores the market against 6 distinct criteria:
| Factor | Indicator / Metric | Long Condition (Bullish) | Short Condition (Bearish) |
| :--- | :--- | :--- | :--- |
| **1. Macro Trend** | 200 EMA | Price > 200 EMA | Price < 200 EMA |
| **2. Trend Alignment** | 20 EMA vs 200 EMA | 20 EMA > 200 EMA | 20 EMA < 200 EMA |
| **3. Momentum** | 20 EMA | Candle Close > 20 EMA | Candle Close < 20 EMA |
| **4. Volume Surge** | 20 SMA Volume | Volume > 20 SMA Volume | Volume > 20 SMA Volume |
| **5. Open Interest (OI)** | Aggregate Contracts | Long Buildup (Price ↑ + OI ↑) | Short Buildup (Price ↓ + OI ↑) |
| **6. Cumulative Delta** | Taker Buy vs Total | Delta > 0 (Taker Buy Dominance) | Delta < 0 (Taker Sell Dominance) |

## Technology Stack
* **Programming Language:** Python 3.14+
* **Data Processing & Analytics:** Pandas, NumPy
* **Networking & HTTP:** Requests, Binance Futures REST API, Telegram Bot API
* **Web Server & Concurrency:** Flask, Gunicorn, Python `threading`
* **Cloud Infrastructure & DevOps:** GitHub, Render, UptimeRobot

## 📂 Repository Structure

```text
eth-signal-bot/
├── app.py              # Flask web server & background thread orchestrator
├── scoring_bot.py      # Data fetching, 6-factor calculation & Telegram dispatcher
├── requirements.txt    # Production dependencies (Flask, Gunicorn, Pandas, NumPy, Requests)
└── README.md           # Technical documentation
```
## Local Installation & Setup
### 1. Clone the Repository
```bash
git clone [https://github.com/DevSGithub2/eth-signal-bot.git](https://github.com/DevSGithub2/eth-signal-bot.git)
cd eth-signal-bot
```
### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
###3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure Environment Variables
```bash
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHAT_ID="your_telegram_chat_id"
```
### 5. Start the Application
```bash
python3 app.py
```
## Cloud Deployment (Render & UptimeRobot)
1. **Create Web Service:** Connect your GitHub repository to [Render](https://render.com).
2. **Build Settings:**
* **Runtime:** `Python 3`
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `gunicorn app:app`
* **Instance Type:** `Free`
3. **Environment Variables:** Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` under the **Environment** tab.
4. **24/7 Keep-Alive:** Create a free HTTP monitor on [UptimeRobot](https://uptimerobot.com) pinging your Render URL every 5 minutes to prevent instance idling.

## Sample Telegram Alert Format
```text
🟢 STRONG LONG WATCH (Score: 6/6)
━━━━━━━━━━━━━━━━━━━━
🪙 Pair: ETHUSDT | ⏱ Interval: 5m
💵 Price: $1,881.00
📈 20 EMA: $1,878.50 | 200 EMA: $1,865.20
📊 Delta: +324.0 ETH (Buy Dominance)
⚡ OI State: Long Buildup (Price ↑ + OI ↑)
━━━━━━━━━━━━━━━━━━━━
📋 Confluences (6/6 Met):
✅ Price > 200 EMA (Macro Bullish)
✅ 20 EMA > 200 EMA (Bullish Alignment)
✅ Candle Close > 20 EMA (Momentum)
✅ Volume > 20 SMA Volume
✅ OI: Long Buildup (Price ↑ + OI ↑)
✅ CVD Delta Positive (+324.0 ETH)
━━━━━━━━━━━━━━━━━━━━
⚠️ Human confirmation & level retest required.
```
## Disclaimer
This bot is designed for technical analysis, screening, and educational purposes only. It does not constitute financial or investment advice. Always perform independent chart analysis and manage risk accordingly.
