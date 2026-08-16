import os
import time
import requests
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = os.getenv("SYMBOL", "ETHUSDT")
INTERVAL = os.getenv("INTERVAL", "5m")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID")

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"

KLINE_LIMIT = 250
OI_LIMIT = 100

POLL_SECONDS = 10

# Alert threshold
MIN_SCORE = 5


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_alert(message: str):

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHANNEL_ID:
        print("Telegram credentials are missing.")
        return False

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHANNEL_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        result = response.json()

        if not result.get("ok", False):
            print(f"Telegram API error: {result}")
            return False

        return True

    except requests.RequestException as e:
        print(f"Telegram request error: {e}")
        return False


# ============================================================
# BINANCE KLINES
# ============================================================

def fetch_klines(
    symbol=SYMBOL,
    interval=INTERVAL,
    limit=KLINE_LIMIT
):

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(
        BINANCE_KLINES_URL,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    if not isinstance(data, list) or len(data) < 220:
        raise ValueError("Insufficient kline data received.")

    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )

    return df


# ============================================================
# BINANCE OPEN INTEREST
# ============================================================

def fetch_open_interest(
    symbol=SYMBOL,
    period="5m",
    limit=OI_LIMIT
):

    params = {
        "symbol": symbol,
        "period": period,
        "limit": limit
    }

    try:

        response = requests.get(
            BINANCE_OI_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, list) or len(data) < 3:
            print("Insufficient OI data.")
            return pd.DataFrame()

        df_oi = pd.DataFrame(data)

        required_columns = [
            "timestamp",
            "sumOpenInterest",
            "sumOpenInterestValue"
        ]

        for col in required_columns:
            if col not in df_oi.columns:
                print(f"Missing OI column: {col}")
                return pd.DataFrame()

        df_oi["timestamp"] = pd.to_datetime(
            df_oi["timestamp"],
            unit="ms",
            utc=True
        )

        df_oi["sumOpenInterest"] = pd.to_numeric(
            df_oi["sumOpenInterest"],
            errors="coerce"
        )

        df_oi["sumOpenInterestValue"] = pd.to_numeric(
            df_oi["sumOpenInterestValue"],
            errors="coerce"
        )

        df_oi = (
            df_oi
            .dropna(
                subset=[
                    "timestamp",
                    "sumOpenInterest"
                ]
            )
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        return df_oi

    except Exception as e:

        print(f"OI fetch error: {e}")

        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA 20
    df["ema_20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # EMA 200
    df["ema_200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    # 20-period volume average
    df["vol_sma_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    # Volume ratio
    df["volume_ratio"] = (
        df["volume"] /
        df["vol_sma_20"]
    )

    # Taker sell volume
    df["taker_sell_base"] = (
        df["volume"] -
        df["taker_buy_base"]
    )

    # Taker Delta
    df["delta"] = (
        df["taker_buy_base"] -
        df["taker_sell_base"]
    )

    # Delta percentage of total volume
    df["delta_pct"] = (
        df["delta"] /
        df["volume"].replace(0, np.nan)
    ) * 100

    # Cumulative Volume Delta
    df["cvd"] = df["delta"].cumsum()

    return df


# ============================================================
# OI ALIGNMENT
# ============================================================

def get_aligned_oi(
    candle_time,
    df_oi
):

    if df_oi.empty:
        return None

    oi = df_oi.copy()

    oi = oi.sort_values("timestamp")

    target = pd.DataFrame({
        "candle_time": [candle_time]
    })

    aligned = pd.merge_asof(
        target.sort_values("candle_time"),
        oi,
        left_on="candle_time",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=2)
    )

    if aligned.empty:
        return None

    row = aligned.iloc[0]

    if pd.isna(row["timestamp"]):
        return None

    return row


# ============================================================
# OI ANALYSIS
# ============================================================

def calculate_oi_state(
    close,
    previous_close,
    candle_time,
    df_oi
):

    result = {
        "status": "Unavailable",
        "change_pct": np.nan,
        "oi_time": None
    }

    if df_oi.empty:
        return result

    current = get_aligned_oi(
        candle_time,
        df_oi
    )

    if current is None:
        return result

    current_time = current["timestamp"]

    current_oi = float(
        current["sumOpenInterest"]
    )

    # Previous OI observation
    previous_rows = df_oi[
        df_oi["timestamp"] < current_time
    ]

    if previous_rows.empty:
        return result

    previous = previous_rows.iloc[-1]

    previous_oi = float(
        previous["sumOpenInterest"]
    )

    if previous_oi == 0:
        return result

    oi_change_pct = (
        (current_oi - previous_oi)
        / previous_oi
    ) * 100

    price_change_pct = (
        (close - previous_close)
        / previous_close
    ) * 100

    result["change_pct"] = oi_change_pct
    result["oi_time"] = current_time

    # Price + OI matrix

    if price_change_pct > 0 and oi_change_pct > 0:

        result["status"] = "Long Buildup"

    elif price_change_pct < 0 and oi_change_pct > 0:

        result["status"] = "Short Buildup"

    elif price_change_pct > 0 and oi_change_pct < 0:

        result["status"] = "Short Covering"

    elif price_change_pct < 0 and oi_change_pct < 0:

        result["status"] = "Long Unwinding"

    else:

        result["status"] = "Neutral"

    return result


# ============================================================
# SCORING
# ============================================================

def evaluate_scoring(
    df,
    df_oi
):

    # Latest CLOSED candle
    latest = df.iloc[-2]

    # Previous CLOSED candle
    previous = df.iloc[-3]

    candle_time = latest["open_time"]

    close = float(latest["close"])
    previous_close = float(previous["close"])

    ema_20 = float(latest["ema_20"])
    ema_200 = float(latest["ema_200"])

    volume = float(latest["volume"])
    volume_avg = float(latest["vol_sma_20"])
    volume_ratio = float(latest["volume_ratio"])

    delta = float(latest["delta"])
    delta_pct = float(latest["delta_pct"])

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    # ========================================================
    # FACTOR 1 — PRICE VS EMA 200
    # ========================================================

    if close > ema_200:

        long_score += 1

        long_reasons.append(
            "✅ Price > 200 EMA (Macro Bullish)"
        )

    elif close < ema_200:

        short_score += 1

        short_reasons.append(
            "✅ Price < 200 EMA (Macro Bearish)"
        )

    # ========================================================
    # FACTOR 2 — EMA 20 VS EMA 200
    # ========================================================

    if ema_20 > ema_200:

        long_score += 1

        long_reasons.append(
            "✅ 20 EMA > 200 EMA (Bullish Alignment)"
        )

    elif ema_20 < ema_200:

        short_score += 1

        short_reasons.append(
            "✅ 20 EMA < 200 EMA (Bearish Alignment)"
        )

    # ========================================================
    # FACTOR 3 — CLOSE VS EMA 20
    # ========================================================

    if close > ema_20:

        long_score += 1

        long_reasons.append(
            "✅ Closed Candle > 20 EMA (Momentum)"
        )

    elif close < ema_20:

        short_score += 1

        short_reasons.append(
            "✅ Closed Candle < 20 EMA (Momentum)"
        )

    # ========================================================
    # FACTOR 4 — VOLUME
    # ========================================================

    volume_confirmed = (
        not np.isnan(volume_ratio)
        and volume_ratio > 1.0
    )

    if volume_confirmed:

        volume_text = (
            f"📊 Volume {volume_ratio:.2f}x "
            f"20-period average"
        )

        long_reasons.append(volume_text)
        short_reasons.append(volume_text)

    # ========================================================
    # FACTOR 5 — OI
    # ========================================================

    oi = calculate_oi_state(
        close=close,
        previous_close=previous_close,
        candle_time=candle_time,
        df_oi=df_oi
    )

    oi_status = oi["status"]
    oi_change_pct = oi["change_pct"]

    if oi_status == "Long Buildup":

        long_score += 1

        long_reasons.append(
            f"✅ OI Long Buildup "
            f"(OI {oi_change_pct:+.3f}%)"
        )

    elif oi_status == "Short Buildup":

        short_score += 1

        short_reasons.append(
            f"✅ OI Short Buildup "
            f"(OI {oi_change_pct:+.3f}%)"
        )

    elif oi_status == "Short Covering":

        long_reasons.append(
            f"⚠️ OI Short Covering "
            f"(OI {oi_change_pct:+.3f}%)"
        )

    elif oi_status == "Long Unwinding":

        short_reasons.append(
            f"⚠️ OI Long Unwinding "
            f"(OI {oi_change_pct:+.3f}%)"
        )

    else:

        long_reasons.append(
            "⚪ OI Neutral / Unavailable"
        )

        short_reasons.append(
            "⚪ OI Neutral / Unavailable"
        )

    # ========================================================
    # FACTOR 6 — TAKER DELTA
    # ========================================================

    if delta > 0:

        long_score += 1

        long_reasons.append(
            f"✅ Taker Delta "
            f"{delta:+,.1f} ETH "
            f"({delta_pct:+.2f}% volume)"
        )

    elif delta < 0:

        short_score += 1

        short_reasons.append(
            f"✅ Taker Delta "
            f"{delta:+,.1f} ETH "
            f"({delta_pct:+.2f}% volume)"
        )

    else:

        long_reasons.append(
            "⚪ Taker Delta Neutral"
        )

        short_reasons.append(
            "⚪ Taker Delta Neutral"
        )

    # ========================================================
    # VOLUME CONFIRMATION
    #
    # Volume is not independently directional.
    # It confirms whichever side already has stronger
    # directional confluence.
    # ========================================================

    if volume_confirmed:

        if long_score > short_score:

            long_score += 1

            long_reasons.append(
                "📈 Volume confirms bullish momentum"
            )

        elif short_score > long_score:

            short_score += 1

            short_reasons.append(
                "📉 Volume confirms bearish momentum"
            )

    return {

        "candle_time": candle_time,

        "close": close,

        "ema_20": ema_20,

        "ema_200": ema_200,

        "volume": volume,

        "volume_avg": volume_avg,

        "volume_ratio": volume_ratio,

        "delta": delta,

        "delta_pct": delta_pct,

        "oi_status": oi_status,

        "oi_change_pct": oi_change_pct,

        "long_score": long_score,

        "short_score": short_score,

        "long_reasons": long_reasons,

        "short_reasons": short_reasons
    }


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def build_message(
    signal,
    direction
):

    if direction == "LONG":

        score = signal["long_score"]

        reasons = signal["long_reasons"]

        if score == 6:
            tag = "🟢 STRONG LONG WATCH"
        else:
            tag = "🟡 LONG WATCH"

    else:

        score = signal["short_score"]

        reasons = signal["short_reasons"]

        if score == 6:
            tag = "🔴 STRONG SHORT WATCH"
        else:
            tag = "🟠 SHORT WATCH"

    if np.isnan(signal["oi_change_pct"]):

        oi_change = "N/A"

    else:

        oi_change = (
            f"{signal['oi_change_pct']:+.3f}%"
        )

    message = (

        f"{tag} *(Score: {score}/6)*\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"🪙 *Pair:* `{SYMBOL}`\n"

        f"⏱ *Interval:* `{INTERVAL}`\n"

        f"🕐 *Closed Candle:* "
        f"`{signal['candle_time'].strftime('%H:%M UTC')}`\n"

        f"💵 *Price:* "
        f"`${signal['close']:,.2f}`\n"

        f"📈 *20 EMA:* "
        f"`${signal['ema_20']:,.2f}`\n"

        f"📉 *200 EMA:* "
        f"`${signal['ema_200']:,.2f}`\n"

        f"📊 *Volume Ratio:* "
        f"`{signal['volume_ratio']:.2f}x`\n"

        f"📊 *Taker Delta:* "
        f"`{signal['delta']:+,.1f} ETH`\n"

        f"📊 *Delta %:* "
        f"`{signal['delta_pct']:+.2f}%`\n"

        f"⚡ *OI State:* "
        f"`{signal['oi_status']}`\n"

        f"⚡ *OI Change:* "
        f"`{oi_change}`\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"📋 *Confluences:*\n"

        + "\n".join(reasons)

        + "\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"⚠️ *WATCH ONLY — Human chart "
        f"verification & level/retest confirmation required.*"
    )

    return message


# ============================================================
# MAIN ENGINE
# ============================================================

def run_engine():

    print(
        f"🚀 ETH Signal Engine started "
        f"for {SYMBOL} {INTERVAL}"
    )

    last_processed_time = None

    last_alert_direction = None

    last_alert_score = None

    while True:

        try:

            # Fetch candles
            df = fetch_klines()

            # Latest CLOSED candle
            latest_closed_time = (
                df.iloc[-2]["open_time"]
            )

            # Process only once per candle
            if latest_closed_time != last_processed_time:

                last_processed_time = latest_closed_time

                # Calculate indicators
                df = calculate_indicators(df)

                # Fetch OI
                df_oi = fetch_open_interest()

                # Evaluate
                signal = evaluate_scoring(
                    df,
                    df_oi
                )

                long_score = signal["long_score"]
                short_score = signal["short_score"]

                print(
                    "\n"
                    f"[{SYMBOL} {INTERVAL}] "
                    f"{latest_closed_time}\n"
                    f"Close: "
                    f"${signal['close']:,.2f}\n"
                    f"20 EMA: "
                    f"${signal['ema_20']:,.2f}\n"
                    f"200 EMA: "
                    f"${signal['ema_200']:,.2f}\n"
                    f"Volume Ratio: "
                    f"{signal['volume_ratio']:.2f}x\n"
                    f"Taker Delta: "
                    f"{signal['delta']:+,.1f} ETH "
                    f"({signal['delta_pct']:+.2f}%)\n"
                    f"OI: "
                    f"{signal['oi_status']} "
                    f"({signal['oi_change_pct'] if not np.isnan(signal['oi_change_pct']) else 'N/A'}%)\n"
                    f"Long Score: "
                    f"{long_score}/6\n"
                    f"Short Score: "
                    f"{short_score}/6"
                )

                # =================================================
                # LONG
                # =================================================

                if (
                    long_score >= MIN_SCORE
                    and long_score > short_score
                ):

                    new_signal = (
                        last_alert_direction != "LONG"
                        or last_alert_score is None
                        or long_score > last_alert_score
                    )

                    if new_signal:

                        message = build_message(
                            signal,
                            "LONG"
                        )

                        if send_telegram_alert(message):

                            last_alert_direction = "LONG"

                            last_alert_score = (
                                long_score
                            )

                            print(
                                "🟢 LONG alert sent."
                            )

                # =================================================
                # SHORT
                # =================================================

                elif (
                    short_score >= MIN_SCORE
                    and short_score > long_score
                ):

                    new_signal = (
                        last_alert_direction != "SHORT"
                        or last_alert_score is None
                        or short_score > last_alert_score
                    )

                    if new_signal:

                        message = build_message(
                            signal,
                            "SHORT"
                        )

                        if send_telegram_alert(message):

                            last_alert_direction = "SHORT"

                            last_alert_score = (
                                short_score
                            )

                            print(
                                "🔴 SHORT alert sent."
                            )

                # =================================================
                # NO HIGH-CONFLUENCE SETUP
                # =================================================

                else:

                    print(
                        "⚪ No high-confluence setup."
                    )

                    # Reset state when setup disappears
                    last_alert_direction = None
                    last_alert_score = None

            time.sleep(POLL_SECONDS)

        except Exception as e:

            print(
                f"❌ Engine Loop Error: {e}"
            )

            time.sleep(5)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    run_engine()
