# ⚡ Binance 6-Factor Confluence Signal Bot

An automated, cloud-hosted cryptocurrency market scanner and alerting engine. The bot continuously monitors the Binance USDⓈ-M Futures market (ETHUSDT 5-minute timeframe), evaluating order flow dynamics and trend metrics to dispatch real-time, high-probability signal alerts directly to Telegram.

---

## 📌 Features

* **Zero API Key Requirement:** Ingests live data directly using Binance Futures public REST endpoints.
* **6-Factor Quantitative Scoring:** Filters out chop and sideways noise by scoring setups from `0/6` to `6/6`.
* **Real-Time Order Flow & CVD:** Computes Cumulative Volume Delta (CVD) and tracks Open Interest (OI) buildup dynamics.
* **Concurrent Web Server:** Uses a lightweight Flask/Gunicorn HTTP server to handle health checks alongside a background worker thread.
* **24/7 Cloud Architecture:** Hosted on Render with continuous deployment via GitHub and automated keep-alive monitoring via UptimeRobot.
* **Rich HTML Alerts:** Delivers actionable Telegram alerts complete with confluences and directional bias.

---

## 📊 Strategy & 6-Factor Confluence Engine

The bot inspects the most recently closed candle and evaluates six core conditions. An alert is triggered **only** when a setup meets high confluence ($\ge 5/6$ score):

| Factor | Metric | Long Condition | Short Condition |
| :--- | :--- | :--- | :--- |
| **1. Macro Trend** | 200 EMA | Price > 200 EMA | Price < 200 EMA |
| **2. Trend Alignment** | 20 & 200 EMA | 20 EMA > 200 EMA | 20 EMA < 200 EMA |
| **3. Short-Term Momentum** | 20 EMA | Candle Close > 20 EMA | Candle Close < 20 EMA |
| **4. Volume Surge** | 20 SMA Volume | Volume > 20 SMA Volume | Volume > 20 SMA Volume |
| **5. Open Interest State** | Aggregate OI | Long Buildup (Price ↑ + OI ↑) | Short Buildup (Price ↓ + OI ↑) |
| **6. Cumulative Volume Delta** | Taker Buy vs Total | Delta > 0 (Buy Dominance) | Delta < 0 (Sell Dominance) |

---

## 🛠️ Tech Stack

* **Language:** Python 3.14+
* **Data Processing & Analysis:** Pandas, NumPy
* **Networking & APIs:** Requests, Telegram Bot API, Binance Futures REST API
* **Web Server & Concurrency:** Flask, Gunicorn, Python `threading`
* **Cloud Infrastructure & CI/CD:** GitHub, Render, UptimeRobot

---

## 📂 Project Structure

```text
eth-signal-bot/
├── app.py              # Flask WSGI web server & background daemon runner
├── scoring_bot.py      # Core data fetching, calculation & Telegram engine
├── requirements.txt    # Application dependencies
└── README.md           # Documentation
