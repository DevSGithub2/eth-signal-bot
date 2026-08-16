import requests
import pandas as pd
import numpy as np
import time
import os

TELEGRAM_BOT_TOKEN = "8758980368:AAEM-duL86rM3NdvO3tRPzAkfGBmcG9lU_M"
TELEGRAM_CHAT_ID = "-4379193701"

SYMBOL = "ETHUSDT"
INTERVAL = "5m"
LIMIT = 250

def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        # If standard group ID fails, try supergroup prefix
        if r.status_code != 200 and not str(TELEGRAM_CHAT_ID).startswith("-100"):
            alt_chat_id = f"-100{abs(int(TELEGRAM_CHAT_ID))}"
            payload["chat_id"] = alt_chat_id
            requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")

def fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT):
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10).json()
        if not isinstance(r, list) or len(r) < 205:
            return None
        
        df = pd.DataFrame(r, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        numeric_cols = ["open", "high", "low", "close", "volume", "taker_buy_base"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        return df
    except Exception:
        return None

def fetch_open_interest_hist(symbol=SYMBOL, period=INTERVAL, limit=30):
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": period, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10).json()
        if isinstance(r, list) and len(r) >= 2:
            return float(r[-1]["sumOpenInterest"]), float(r[-2]["sumOpenInterest"])
    except Exception:
        pass
    return None, None

def evaluate_market():
    df = fetch_klines()
    if df is None or len(df) < 205:
        return None

    # Technical Indicators
    df["20_EMA"] = df["close"].ewm(span=20, adjust=False).mean()
    df["200_EMA"] = df["close"].ewm(span=200, adjust=False).mean()
    df["vol_sma"] = df["volume"].rolling(window=20).mean()

    # CVD Taker Delta
    df["delta"] = (2 * df["taker_buy_base"]) - df["volume"]
    
    # Safe Candle Access (-2 = last closed candle)
    closed_candle = df.iloc[-2]
    prev_closed = df.iloc[-3]

    price = float(closed_candle["close"])
    ema20 = float(closed_candle["20_EMA"])
    ema200 = float(closed_candle["200_EMA"])
    vol = float(closed_candle["volume"])
    vol_sma = float(closed_candle["vol_sma"])
    delta = float(closed_candle["delta"])

    curr_oi, prev_oi = fetch_open_interest_hist()
    if curr_oi is None:
        curr_oi, prev_oi = 0, 0

    # 1. Macro Trend
    bullish_trend = price > ema200
    bearish_trend = price < ema200

    # 2. EMA Alignment
    bullish_align = ema20 > ema200
    bearish_align = ema20 < ema200

    # 3. Momentum
    bullish_mom = price > ema20
    bearish_mom = price < ema20

    # 4. Volume Surge
    volume_surge = vol > vol_sma

    # 5. Open Interest State
    price_up = price > float(prev_closed["close"])
    oi_up = curr_oi > prev_oi if prev_oi else True

    long_buildup = price_up and oi_up
    short_buildup = (not price_up) and oi_up

    # 6. CVD Delta Dominance
    cvd_bullish = delta > 0
    cvd_bearish = delta < 0

    long_score = sum([bullish_trend, bullish_align, bullish_mom, volume_surge, long_buildup, cvd_bullish])
    short_score = sum([bearish_trend, bearish_align, bearish_mom, volume_surge, short_buildup, cvd_bearish])

    return {
        "price": price,
        "ema20": ema20,
        "ema200": ema200,
        "delta": delta,
        "long_score": long_score,
        "short_score": short_score,
        "long_buildup": long_buildup,
        "short_buildup": short_buildup,
        "vol_surge": volume_surge,
        "bullish_trend": bullish_trend,
        "bearish_trend": bearish_trend,
        "bullish_align": bullish_align,
        "bearish_align": bearish_align,
        "bullish_mom": bullish_mom,
        "bearish_mom": bearish_mom,
        "cvd_bullish": cvd_bullish,
        "cvd_bearish": cvd_bearish,
        "candle_time": closed_candle["close_time"]
    }

def format_signal_message(sig, side):
    score = sig["long_score"] if side == "LONG" else sig["short_score"]
    emoji = "🟢" if side == "LONG" else "🔴"
    
    confluences = []
    if side == "LONG":
        if sig["bullish_trend"]: confluences.append("✅ Price > 200 EMA (Macro Bullish)")
        if sig["bullish_align"]: confluences.append("✅ 20 EMA > 200 EMA (Bullish Alignment)")
        if sig["bullish_mom"]: confluences.append("✅ Candle Close > 20 EMA (Momentum)")
        if sig["vol_surge"]: confluences.append("✅ Volume > 20 SMA Volume")
        if sig["long_buildup"]: confluences.append("✅ OI: Long Buildup (Price ↑ + OI ↑)")
        if sig["cvd_bullish"]: confluences.append(f"✅ CVD Delta Positive (+{sig['delta']:.1f} ETH)")
        state_str = "Long Buildup (Price ↑ + OI ↑)"
    else:
        if sig["bearish_trend"]: confluences.append("✅ Price < 200 EMA (Macro Bearish)")
        if sig["bearish_align"]: confluences.append("✅ 20 EMA < 200 EMA (Bearish Alignment)")
        if sig["bearish_mom"]: confluences.append("✅ Candle Close < 20 EMA (Momentum)")
        if sig["vol_surge"]: confluences.append("✅ Volume > 20 SMA Volume")
        if sig["short_buildup"]: confluences.append("✅ OI: Short Buildup (Price ↓ + OI ↑)")
        if sig["cvd_bearish"]: confluences.append(f"✅ CVD Delta Negative ({sig['delta']:.1f} ETH)")
        state_str = "Short Buildup (Price ↓ + OI ↑)"

    confluence_text = "\n".join(confluences)

    return f"""{emoji} <b>STRONG {side} WATCH (Score: {score}/6)</b>
━━━━━━━━━━━━━━━━━━━━
🪙 <b>Pair:</b> ETHUSDT | ⏱ <b>Interval:</b> 5m
💵 <b>Price:</b> ${sig['price']:,.2f}
📈 <b>20 EMA:</b> ${sig['ema20']:,.2f} | <b>200 EMA:</b> ${sig['ema200']:,.2f}
📊 <b>Delta:</b> {sig['delta']:+,.1f} ETH
⚡ <b>OI State:</b> {state_str}
━━━━━━━━━━━━━━━━━━━━
📋 <b>Confluences:</b>
{confluence_text}
━━━━━━━━━━━━━━━━━━━━
⚠️ <i>Human confirmation & level retest required.</i>"""

def run_engine():
    print("🚀 6-Factor Engine (EMA + Vol + OI + CVD) live for ETHUSDT (5m)...")
    last_processed_candle = None
    
    while True:
        try:
            sig = evaluate_market()
            if sig and sig["candle_time"] != last_processed_candle:
                last_processed_candle = sig["candle_time"]
                
                print(f"[5m Check] Price: ${sig['price']:.2f} | Long Score: {sig['long_score']}/6 | Short Score: {sig['short_score']}/6")

                if sig["long_score"] >= 5:
                    msg = format_signal_message(sig, "LONG")
                    send_telegram_alert(msg)
                elif sig["short_score"] >= 5:
                    msg = format_signal_message(sig, "SHORT")
                    send_telegram_alert(msg)

        except Exception as e:
            print(f"Engine Loop Error: {e}")

        time.sleep(10)

if __name__ == "__main__":
    run_engine()
