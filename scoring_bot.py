import time
import requests
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = "8758980368:AAEM-duL86rM3NdvO3tRPzAkfGBmcG9lU_M"
TELEGRAM_CHANNEL_ID = "-1004379193701"
SYMBOL = "ETHUSDT"
INTERVAL = "5m"

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"

def send_telegram_alert(message: str):
    """Sends scored signal to your Telegram Channel."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")

def fetch_klines(symbol=SYMBOL, interval=INTERVAL, limit=250):
    """Fetches public historical candles and taker volumes from Binance Futures."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        df[col] = df[col].astype(float)

    return df

def fetch_open_interest(symbol=SYMBOL, period="5m", limit=30):
    """Fetches Open Interest history from Binance Futures."""
    params = {"symbol": symbol, "period": period, "limit": limit}
    try:
        response = requests.get(BINANCE_OI_URL, params=params, timeout=10)
        data = response.json()
        df_oi = pd.DataFrame(data)
        df_oi["sumOpenInterest"] = df_oi["sumOpenInterest"].astype(float)
        df_oi["sumOpenInterestValue"] = df_oi["sumOpenInterestValue"].astype(float)
        return df_oi
    except Exception as e:
        print(f"Error fetching OI: {e}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Computes EMAs, Volume SMA, and CVD (Cumulative Volume Delta)."""
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["vol_sma_20"] = df["volume"].rolling(window=20).mean()

    # CVD & Delta Calculation
    df["taker_sell_base"] = df["volume"] - df["taker_buy_base"]
    df["delta"] = df["taker_buy_base"] - df["taker_sell_base"]
    df["cvd"] = df["delta"].cumsum()
    return df

def evaluate_scoring(df, df_oi):
    """
    Evaluates 6-Factor Model:
    1. Macro Trend: Price vs EMA 200
    2. Trend Alignment: EMA 20 vs EMA 200
    3. Momentum: Close vs EMA 20
    4. Volume Surge: Volume vs Volume SMA 20
    5. Open Interest: Buildup / Unwinding
    6. Volume Delta (CVD): Aggressive Buyer / Seller Pressure
    """
    latest = df.iloc[-2]
    prev = df.iloc[-3]

    close = latest["close"]
    prev_close = prev["close"]
    ema_20 = latest["ema_20"]
    ema_200 = latest["ema_200"]
    vol = latest["volume"]
    vol_avg = latest["vol_sma_20"]
    delta = latest["delta"]

    long_score = 0
    short_score = 0
    long_reasons = []
    short_reasons = []

    # 1. Macro Trend (EMA 200)
    if close > ema_200:
        long_score += 1
        long_reasons.append("✅ Price > 200 EMA (Macro Bullish)")
    else:
        short_score += 1
        short_reasons.append("✅ Price < 200 EMA (Macro Bearish)")

    # 2. Trend Alignment (EMA 20 vs EMA 200)
    if ema_20 > ema_200:
        long_score += 1
        long_reasons.append("✅ 20 EMA > 200 EMA (Bullish Alignment)")
    else:
        short_score += 1
        short_reasons.append("✅ 20 EMA < 200 EMA (Bearish Alignment)")

    # 3. Momentum (Close vs EMA 20)
    if close > ema_20:
        long_score += 1
        long_reasons.append("✅ Candle Close > 20 EMA (Momentum)")
    else:
        short_score += 1
        short_reasons.append("✅ Candle Close < 20 EMA (Momentum)")

    # 4. Volume Surge
    if vol > vol_avg:
        long_score += 1
        short_score += 1
        long_reasons.append("✅ Volume > 20 SMA Volume")
        short_reasons.append("✅ Volume > 20 SMA Volume")

    # 5. Open Interest (OI)
    oi_status = "Neutral"
    if not df_oi.empty and len(df_oi) >= 2:
        latest_oi = df_oi.iloc[-1]["sumOpenInterest"]
        prev_oi = df_oi.iloc[-2]["sumOpenInterest"]

        price_up = close > prev_close
        oi_up = latest_oi > prev_oi

        if price_up and oi_up:
            oi_status = "Long Buildup (Price ↑ + OI ↑)"
            long_score += 1
            long_reasons.append(f"✅ OI: {oi_status}")
        elif not price_up and oi_up:
            oi_status = "Short Buildup (Price ↓ + OI ↑)"
            short_score += 1
            short_reasons.append(f"✅ OI: {oi_status}")
        elif price_up and not oi_up:
            oi_status = "Short Covering (Price ↑ + OI ↓)"
        else:
            oi_status = "Long Unwinding (Price ↓ + OI ↓)"

    # 6. Cumulative Volume Delta (CVD)
    if delta > 0:
        long_score += 1
        long_reasons.append(f"✅ CVD Delta Positive (+{delta:,.1f} ETH Market Buy Dominance)")
    else:
        short_score += 1
        short_reasons.append(f"✅ CVD Delta Negative ({delta:,.1f} ETH Market Sell Dominance)")

    return {
        "close": close,
        "ema_20": ema_20,
        "ema_200": ema_200,
        "delta": delta,
        "long_score": long_score,
        "short_score": short_score,
        "long_reasons": long_reasons,
        "short_reasons": short_reasons,
        "oi_status": oi_status
    }

def run_engine():
    print(f"🚀 6-Factor Engine (EMA + Vol + OI + CVD) live for {SYMBOL} ({INTERVAL})...")
    last_processed_time = None

    while True:
        try:
            df = fetch_klines()
            latest_closed_time = df.iloc[-2]["open_time"]

            if latest_closed_time != last_processed_time:
                last_processed_time = latest_closed_time
                df = calculate_indicators(df)
                df_oi = fetch_open_interest()
                signal = evaluate_scoring(df, df_oi)

                print(
                    f"[{SYMBOL} 5M] Close: ${signal['close']:,.2f} | "
                    f"Long Score: {signal['long_score']}/6 | Short Score: {signal['short_score']}/6 | "
                    f"Delta: {signal['delta']:+,.1f} ETH | OI: {signal['oi_status']}"
                )

                # Send Alert when confluence score is 5/6 or 6/6
                if signal["long_score"] >= 5:
                    tag = "🟢 STRONG LONG WATCH" if signal["long_score"] == 6 else "🟡 LONG WATCH"
                    msg = (
                        f"{tag} *(Score: {signal['long_score']}/6)*\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"🪙 *Pair:* `{SYMBOL}` | ⏱ *Interval:* `{INTERVAL}`\n"
                        f"💵 *Price:* `${signal['close']:,.2f}`\n"
                        f"📈 *20 EMA:* `${signal['ema_20']:,.2f}` | *200 EMA:* `${signal['ema_200']:,.2f}`\n"
                        f"📊 *Delta:* `{signal['delta']:+,.1f} ETH`\n"
                        f"⚡ *OI State:* `{signal['oi_status']}`\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📋 *Confluences:*\n" + "\n".join(signal["long_reasons"]) + "\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"⚠️ *Human chart verification required.*"
                    )
                    send_telegram_alert(msg)

                elif signal["short_score"] >= 5:
                    tag = "🔴 STRONG SHORT WATCH" if signal["short_score"] == 6 else "🟠 SHORT WATCH"
                    msg = (
                        f"{tag} *(Score: {signal['short_score']}/6)*\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"🪙 *Pair:* `{SYMBOL}` | ⏱ *Interval:* `{INTERVAL}`\n"
                        f"💵 *Price:* `${signal['close']:,.2f}`\n"
                        f"📈 *20 EMA:* `${signal['ema_20']:,.2f}` | *200 EMA:* `${signal['ema_200']:,.2f}`\n"
                        f"📊 *Delta:* `{signal['delta']:+,.1f} ETH`\n"
                        f"⚡ *OI State:* `{signal['oi_status']}`\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📋 *Confluences:*\n" + "\n".join(signal["short_reasons"]) + "\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"⚠️ *Human chart verification required.*"
                    )
                    send_telegram_alert(msg)

            time.sleep(10)

        except Exception as e:
            print(f"Engine Loop Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run_engine()
