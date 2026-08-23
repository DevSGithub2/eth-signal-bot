import os
import time
from datetime import datetime, timezone

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

POLL_SECONDS = 3

# Breakout / rejection detection
# A breakout must hold above the detected resistance.
# If price breaks above resistance but the CLOSED candle returns
# below that level, the bot treats it as a failed hold / rejection.
BREAKOUT_HOLD_TOLERANCE_PCT = 0.15

# Phase 2 = 8 factor system
MIN_SCORE = 6
MAX_SCORE = 8


# ============================================================
# UTC TIME
# ============================================================

def utc_now():
    return datetime.now(timezone.utc)


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

    if (
        not isinstance(data, list)
        or len(data) < 220
    ):
        raise ValueError(
            "Insufficient kline data received."
        )

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

        if (
            not isinstance(data, list)
            or len(data) < 3
        ):
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
# INDICATORS — PHASE 2
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA 20 / 50 / 200
    # --------------------------------------------------------

    df["ema_20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["ema_50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["ema_200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["vol_sma_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        /
        df["vol_sma_20"]
    )

    # --------------------------------------------------------
    # Taker Delta
    # --------------------------------------------------------

    df["taker_sell_base"] = (
        df["volume"]
        -
        df["taker_buy_base"]
    )

    df["delta"] = (
        df["taker_buy_base"]
        -
        df["taker_sell_base"]
    )

    df["delta_pct"] = (
        df["delta"]
        /
        df["volume"].replace(0, np.nan)
    ) * 100

    # --------------------------------------------------------
    # CVD
    # --------------------------------------------------------

    df["cvd"] = df["delta"].cumsum()

    df["cvd_change"] = (
        df["cvd"]
        -
        df["cvd"].shift(1)
    )

    # --------------------------------------------------------
    # MACD 12 / 26 / 9
    # --------------------------------------------------------

    ema_12 = (
        df["close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema_26 = (
        df["close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    df["macd"] = (
        ema_12 - ema_26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    df["macd_hist"] = (
        df["macd"]
        -
        df["macd_signal"]
    )

    # --------------------------------------------------------
    # Candle structure
    # --------------------------------------------------------

    df["body"] = (
        (df["close"] - df["open"])
        .abs()
    )

    df["range"] = (
        df["high"] - df["low"]
    )

    df["upper_wick"] = (
        df["high"]
        -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1)
        -
        df["low"]
    )

    # --------------------------------------------------------
    # Recent highs / lows
    # Excludes current candle
    # --------------------------------------------------------

    df["recent_high"] = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["recent_low"] = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

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

    oi = (
        df_oi
        .sort_values("timestamp")
        .copy()
    )

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

    previous_rows = df_oi[
        df_oi["timestamp"] < current_time
    ]

    if previous_rows.empty:
        return result

    previous_oi = float(
        previous_rows.iloc[-1]["sumOpenInterest"]
    )

    if previous_oi == 0:
        return result

    oi_change_pct = (
        (current_oi - previous_oi)
        /
        previous_oi
    ) * 100

    price_change_pct = (
        (close - previous_close)
        /
        previous_close
    ) * 100

    result["change_pct"] = oi_change_pct
    result["oi_time"] = current_time

    if (
        price_change_pct > 0
        and
        oi_change_pct > 0
    ):
        result["status"] = "Long Buildup"

    elif (
        price_change_pct < 0
        and
        oi_change_pct > 0
    ):
        result["status"] = "Short Buildup"

    elif (
        price_change_pct > 0
        and
        oi_change_pct < 0
    ):
        result["status"] = "Short Covering"

    elif (
        price_change_pct < 0
        and
        oi_change_pct < 0
    ):
        result["status"] = "Long Unwinding"

    else:
        result["status"] = "Neutral"

    return result


# ============================================================
# LIQUIDITY / SWEEP DETECTION
# ============================================================

def detect_liquidity(df):

    latest = df.iloc[-2]

    recent_high = latest["recent_high"]
    recent_low = latest["recent_low"]

    high = float(latest["high"])
    low = float(latest["low"])
    close = float(latest["close"])

    result = {
        "status": "Neutral",
        "long": False,
        "short": False,
        "level": np.nan
    }

    if pd.isna(recent_high) or pd.isna(recent_low):
        return result

    # --------------------------------------------------------
    # Bearish liquidity sweep
    # Price takes previous high but closes back below it
    # --------------------------------------------------------

    if (
        high > recent_high
        and
        close < recent_high
    ):

        result["status"] = "Bearish Liquidity Sweep"
        result["short"] = True
        result["level"] = recent_high

        return result

    # --------------------------------------------------------
    # Bullish liquidity sweep
    # Price takes previous low but closes back above it
    # --------------------------------------------------------

    if (
        low < recent_low
        and
        close > recent_low
    ):

        result["status"] = "Bullish Liquidity Sweep"
        result["long"] = True
        result["level"] = recent_low

        return result

    # --------------------------------------------------------
    # Breakout context
    # --------------------------------------------------------

    if close > recent_high:

        result["status"] = "High Liquidity Broken"
        result["long"] = True
        result["level"] = recent_high

    elif close < recent_low:

        result["status"] = "Low Liquidity Broken"
        result["short"] = True
        result["level"] = recent_low

    return result


# ============================================================
# BREAKOUT HOLD / REJECTION DETECTION
# ============================================================

def detect_breakout_rejection(df):
    """
    Detect two important 5M price-action events:

    1) Failed breakout / failed hold:
       A previous closed candle closes above its prior resistance,
       but the latest closed candle closes back below that level.

    2) Rejection at resistance:
       Price trades above the latest resistance, leaves an upper wick,
       and closes back below the resistance.

    This is intentionally a price-action flag, not a separate scoring
    factor, so the existing 8-factor score remains 0-8.
    """

    latest = df.iloc[-2]
    previous = df.iloc[-3]

    result = {
        "status": "No clear breakout rejection",
        "short": False,
        "long_blocked": False,
        "level": np.nan,
        "failed_hold": False
    }

    latest_level = latest["recent_high"]
    previous_level = previous["recent_high"]

    if pd.isna(latest_level):
        return result

    level = float(latest_level)
    high = float(latest["high"])
    close = float(latest["close"])
    body = float(latest["body"])
    candle_range = float(latest["range"])
    upper_wick = float(latest["upper_wick"])

    if candle_range <= 0:
        return result

    # --------------------------------------------------------
    # Strongest case: breakout happened, but the next candle
    # could not hold above the breakout level.
    # --------------------------------------------------------
    if not pd.isna(previous_level):
        previous_level = float(previous_level)
        previous_close = float(previous["close"])

        previous_broke_out = previous_close > previous_level
        latest_failed_hold = close < previous_level

        if previous_broke_out and latest_failed_hold:
            result["status"] = "Bearish Breakout Failure / Hold Lost"
            result["short"] = True
            result["long_blocked"] = True
            result["level"] = previous_level
            result["failed_hold"] = True
            return result

    # --------------------------------------------------------
    # Rejection at resistance even when there was no prior
    # candle close above the level.
    # --------------------------------------------------------
    rejection_close = close < level
    took_level = high > level
    wick_rejection = (
        upper_wick >= max(body * 1.2, candle_range * 0.20)
    )

    if took_level and rejection_close and wick_rejection:
        result["status"] = "Bearish Resistance Rejection / Hold Failed"
        result["short"] = True
        result["level"] = level

        # A meaningful close back below resistance means a LONG
        # should not be generated from this candle.
        if close < level * (1 - BREAKOUT_HOLD_TOLERANCE_PCT / 100):
            result["long_blocked"] = True

    return result


# ============================================================
# ABSORPTION / TRAP DETECTION
# ============================================================

def detect_absorption(df):

    latest = df.iloc[-2]

    body = float(latest["body"])
    candle_range = float(latest["range"])
    volume_ratio = float(latest["volume_ratio"])
    delta_pct = float(latest["delta_pct"])

    upper_wick = float(latest["upper_wick"])
    lower_wick = float(latest["lower_wick"])

    result = {
        "status": "Neutral",
        "long": False,
        "short": False
    }

    if candle_range <= 0:
        return result

    body_ratio = body / candle_range

    # --------------------------------------------------------
    # Bullish absorption
    #
    # Strong negative delta / selling pressure
    # but price rejects lows and closes relatively strongly
    # --------------------------------------------------------

    if (
        delta_pct < -5
        and
        volume_ratio >= 1.3
        and
        lower_wick >= body * 1.2
        and
        body_ratio < 0.60
    ):

        result["status"] = "Bullish Absorption"
        result["long"] = True

    # --------------------------------------------------------
    # Bearish absorption
    #
    # Strong positive delta / buying pressure
    # but price rejects highs
    # --------------------------------------------------------

    elif (
        delta_pct > 5
        and
        volume_ratio >= 1.3
        and
        upper_wick >= body * 1.2
        and
        body_ratio < 0.60
    ):

        result["status"] = "Bearish Absorption"
        result["short"] = True

    return result


# ============================================================
# SCORING — PHASE 2
# ============================================================

def evaluate_scoring(
    df,
    df_oi
):

    latest = df.iloc[-2]
    previous = df.iloc[-3]

    candle_time = latest["open_time"]
    candle_close_time = latest["close_time"]

    close = float(latest["close"])
    previous_close = float(previous["close"])

    ema_20 = float(latest["ema_20"])
    ema_50 = float(latest["ema_50"])
    ema_200 = float(latest["ema_200"])

    volume = float(latest["volume"])
    volume_avg = float(latest["vol_sma_20"])
    volume_ratio = float(latest["volume_ratio"])

    delta = float(latest["delta"])
    delta_pct = float(latest["delta_pct"])

    cvd_change = float(
        latest["cvd_change"]
    ) if not pd.isna(latest["cvd_change"]) else 0

    macd = float(latest["macd"])
    macd_signal = float(latest["macd_signal"])
    macd_hist = float(latest["macd_hist"])

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    # ========================================================
    # FACTOR 1 — EMA 20 / 50 / 200
    # ========================================================

    bullish_ema = (
        ema_20 > ema_50
        and
        ema_50 > ema_200
    )

    bearish_ema = (
        ema_20 < ema_50
        and
        ema_50 < ema_200
    )

    if bullish_ema:

        long_score += 1

        long_reasons.append(
            "✅ EMA structure bullish "
            "(20 > 50 > 200)"
        )

    elif bearish_ema:

        short_score += 1

        short_reasons.append(
            "✅ EMA structure bearish "
            "(20 < 50 < 200)"
        )

    else:

        long_reasons.append(
            "⚪ EMA structure mixed"
        )

        short_reasons.append(
            "⚪ EMA structure mixed"
        )

    # ========================================================
    # FACTOR 2 — VOLUME
    # ========================================================

    volume_confirmed = (
        not np.isnan(volume_ratio)
        and
        volume_ratio >= 1.0
    )

    if volume_confirmed:

        if close > latest["open"]:

            long_score += 1

            long_reasons.append(
                f"✅ Bullish volume "
                f"({volume_ratio:.2f}x average)"
            )

        elif close < latest["open"]:

            short_score += 1

            short_reasons.append(
                f"✅ Bearish volume "
                f"({volume_ratio:.2f}x average)"
            )

    else:

        long_reasons.append(
            f"⚪ Volume weak "
            f"({volume_ratio:.2f}x)"
        )

        short_reasons.append(
            f"⚪ Volume weak "
            f"({volume_ratio:.2f}x)"
        )

    # ========================================================
    # FACTOR 3 — OI
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
            f"({oi_change_pct:+.3f}%)"
        )

    elif oi_status == "Short Buildup":

        short_score += 1

        short_reasons.append(
            f"✅ OI Short Buildup "
            f"({oi_change_pct:+.3f}%)"
        )

    elif oi_status == "Short Covering":

        long_reasons.append(
            f"⚠️ Short Covering "
            f"({oi_change_pct:+.3f}%)"
        )

    elif oi_status == "Long Unwinding":

        short_reasons.append(
            f"⚠️ Long Unwinding "
            f"({oi_change_pct:+.3f}%)"
        )

    else:

        long_reasons.append(
            "⚪ OI Neutral / Unavailable"
        )

        short_reasons.append(
            "⚪ OI Neutral / Unavailable"
        )

    # ========================================================
    # FACTOR 4 — DELTA / CVD
    # ========================================================

    if (
        delta > 0
        and
        cvd_change > 0
    ):

        long_score += 1

        long_reasons.append(
            f"✅ Positive Delta + CVD "
            f"({delta_pct:+.2f}%)"
        )

    elif (
        delta < 0
        and
        cvd_change < 0
    ):

        short_score += 1

        short_reasons.append(
            f"✅ Negative Delta + CVD "
            f"({delta_pct:+.2f}%)"
        )

    else:

        long_reasons.append(
            "⚪ Delta/CVD mixed"
        )

        short_reasons.append(
            "⚪ Delta/CVD mixed"
        )

    # ========================================================
    # FACTOR 5 — LIQUIDITY
    # ========================================================

    liquidity = detect_liquidity(df)

    liquidity_status = liquidity["status"]

    if liquidity["long"]:

        long_score += 1

        long_reasons.append(
            f"💧 {liquidity_status}"
        )

    elif liquidity["short"]:

        short_score += 1

        short_reasons.append(
            f"💧 {liquidity_status}"
        )

    else:

        long_reasons.append(
            "⚪ No clear liquidity event"
        )

        short_reasons.append(
            "⚪ No clear liquidity event"
        )

    # ========================================================
    # FACTOR 6 — ABSORPTION / TRAP
    # ========================================================

    absorption = detect_absorption(df)

    absorption_status = absorption["status"]

    if absorption["long"]:

        long_score += 1

        long_reasons.append(
            f"🧲 {absorption_status}"
        )

    elif absorption["short"]:

        short_score += 1

        short_reasons.append(
            f"🧲 {absorption_status}"
        )

    else:

        long_reasons.append(
            "⚪ No clear absorption"
        )

        short_reasons.append(
            "⚪ No clear absorption"
        )

    # ========================================================
    # FACTOR 7 — MACD 12 / 26 / 9
    # ========================================================

    if (
        macd > macd_signal
        and
        macd_hist > 0
    ):

        long_score += 1

        long_reasons.append(
            "✅ MACD bullish "
            "(MACD > Signal)"
        )

    elif (
        macd < macd_signal
        and
        macd_hist < 0
    ):

        short_score += 1

        short_reasons.append(
            "✅ MACD bearish "
            "(MACD < Signal)"
        )

    else:

        long_reasons.append(
            "⚪ MACD mixed"
        )

        short_reasons.append(
            "⚪ MACD mixed"
        )

    # ========================================================
    # FACTOR 8 — PRICE ACTION + BREAKOUT HOLD
    # ========================================================

    candle_open = float(latest["open"])

    breakout = detect_breakout_rejection(df)
    breakout_status = breakout["status"]
    breakout_level = breakout["level"]
    long_blocked = breakout["long_blocked"]

    price_bullish = (
        close > candle_open
        and
        close > ema_20
    )

    price_bearish = (
        close < candle_open
        and
        close < ema_20
    )

    # Breakout failure / rejection has priority over ordinary
    # bullish price confirmation. This is exactly the situation
    # we want the bot to recognize at a resistance such as 2432.
    if breakout["short"]:

        short_score += 1

        if pd.isna(breakout_level):
            level_text = "dynamic resistance"
        else:
            level_text = f"${float(breakout_level):,.2f}"

        short_reasons.append(
            f"⚠️ {breakout_status} at {level_text}"
        )

        long_reasons.append(
            f"🚫 Long blocked: resistance not held "
            f"({level_text})"
        )

    elif price_bullish:

        long_score += 1

        long_reasons.append(
            "✅ Bullish price confirmation "
            "(close > 20 EMA)"
        )

    elif price_bearish:

        short_score += 1

        short_reasons.append(
            "✅ Bearish price confirmation "
            "(close < 20 EMA)"
        )

    else:

        long_reasons.append(
            "⚪ Price confirmation mixed"
        )

        short_reasons.append(
            "⚪ Price confirmation mixed"
        )

    # ========================================================
    # RESULT
    # ========================================================

    return {

        "candle_time":
            candle_time,

        "candle_close_time":
            candle_close_time,

        "close":
            close,

        "ema_20":
            ema_20,

        "ema_50":
            ema_50,

        "ema_200":
            ema_200,

        "volume":
            volume,

        "volume_avg":
            volume_avg,

        "volume_ratio":
            volume_ratio,

        "delta":
            delta,

        "delta_pct":
            delta_pct,

        "cvd_change":
            cvd_change,

        "macd":
            macd,

        "macd_signal":
            macd_signal,

        "macd_hist":
            macd_hist,

        "oi_status":
            oi_status,

        "oi_change_pct":
            oi_change_pct,

        "liquidity_status":
            liquidity_status,

        "absorption_status":
            absorption_status,

        "breakout_status":
            breakout_status,

        "breakout_level":
            breakout_level,

        "long_blocked":
            long_blocked,

        "long_score":
            long_score,

        "short_score":
            short_score,

        "long_reasons":
            long_reasons,

        "short_reasons":
            short_reasons
    }


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def build_message(
    signal,
    direction,
    generated_at
):

    if direction == "LONG":

        score = signal["long_score"]
        reasons = signal["long_reasons"]

        if score >= 7:
            tag = "🟢 STRONG LONG WATCH"
        else:
            tag = "🟡 LONG WATCH"

    else:

        score = signal["short_score"]
        reasons = signal["short_reasons"]

        if score >= 7:
            tag = "🔴 STRONG SHORT WATCH"
        else:
            tag = "🟠 SHORT WATCH"

    if np.isnan(signal["oi_change_pct"]):

        oi_change = "N/A"

    else:

        oi_change = (
            f"{signal['oi_change_pct']:+.3f}%"
        )

    candle_time = (
        signal["candle_time"]
        .strftime("%H:%M:%S UTC")
    )

    candle_close_time = (
        signal["candle_close_time"]
        .strftime("%H:%M:%S UTC")
    )

    generated_time = (
        generated_at
        .strftime("%H:%M:%S UTC")
    )

    engine_delay = (
        generated_at
        -
        signal["candle_close_time"]
    ).total_seconds()

    if engine_delay < 0:
        engine_delay = 0

    message = (

        f"{tag} *(Score: {score}/{MAX_SCORE})*\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"🪙 *Pair:* `{SYMBOL}`\n"

        f"⏱ *Interval:* `{INTERVAL}`\n"

        f"🕐 *Candle Open:* `{candle_time}`\n"

        f"🔒 *Candle Close:* "
        f"`{candle_close_time}`\n"

        f"⚡ *Signal Generated:* "
        f"`{generated_time}`\n"

        f"⏱ *Engine Delay:* "
        f"`{engine_delay:.1f} sec`\n"

        f"💵 *Price:* "
        f"`${signal['close']:,.2f}`\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"📈 *EMA 20:* "
        f"`${signal['ema_20']:,.2f}`\n"

        f"📈 *EMA 50:* "
        f"`${signal['ema_50']:,.2f}`\n"

        f"📉 *EMA 200:* "
        f"`${signal['ema_200']:,.2f}`\n"

        f"📊 *Volume:* "
        f"`{signal['volume_ratio']:.2f}x`\n"

        f"📊 *Delta:* "
        f"`{signal['delta']:+,.1f} ETH`\n"

        f"📊 *Delta %:* "
        f"`{signal['delta_pct']:+.2f}%`\n"

        f"📊 *CVD Change:* "
        f"`{signal['cvd_change']:+,.1f}`\n"

        f"⚡ *OI State:* "
        f"`{signal['oi_status']}`\n"

        f"⚡ *OI Change:* "
        f"`{oi_change}`\n"

        f"📈 *MACD:* "
        f"`{signal['macd']:.4f}`\n"

        f"📉 *MACD Signal:* "
        f"`{signal['macd_signal']:.4f}`\n"

        f"📊 *MACD Histogram:* "
        f"`{signal['macd_hist']:.4f}`\n"

        f"💧 *Liquidity:* "
        f"`{signal['liquidity_status']}`\n"

        f"🧲 *Absorption:* "
        f"`{signal['absorption_status']}`\n"
        f"🚧 *Breakout / Hold:* "
        f"`{signal['breakout_status']}`\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"📋 *Phase 2 Confluences:*\n"

        +
        "\n".join(reasons)

        +
        "\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"⚠️ *WATCH ONLY*\n"

        f"Human chart verification required.\n"

        f"Level / breakout / retest confirmation "
        f"is still required before any trade."
    )

    return message


# ============================================================
# MAIN ENGINE
# ============================================================

def run_engine():

    print(
        f"🚀 PHASE 2 ETH Signal Engine started "
        f"for {SYMBOL} {INTERVAL}"
    )

    print(
        "📊 8-Factor Engine:"
        " EMA + Volume + OI + Delta/CVD +"
        " Liquidity + Absorption + MACD + Price Action"
    )

    last_processed_time = None

    last_alert_direction = None
    last_alert_score = None

    while True:

        try:

            # ------------------------------------------------
            # FETCH CANDLES
            # ------------------------------------------------

            df = fetch_klines()

            latest_closed_time = (
                df.iloc[-2]["open_time"]
            )

            # ------------------------------------------------
            # PROCESS ONLY ONCE PER CLOSED CANDLE
            # ------------------------------------------------

            if (
                latest_closed_time
                !=
                last_processed_time
            ):

                detection_at = utc_now()

                last_processed_time = (
                    latest_closed_time
                )

                # ------------------------------------------------
                # INDICATORS
                # ------------------------------------------------

                df = calculate_indicators(df)

                # ------------------------------------------------
                # OI
                # ------------------------------------------------

                df_oi = fetch_open_interest()

                # ------------------------------------------------
                # PHASE 2 SCORING
                # ------------------------------------------------

                signal = evaluate_scoring(
                    df,
                    df_oi
                )

                signal_generated_at = utc_now()

                long_score = signal["long_score"]
                short_score = signal["short_score"]

                # ------------------------------------------------
                # TIMING
                # ------------------------------------------------

                detection_delay = (
                    detection_at
                    -
                    signal["candle_close_time"]
                ).total_seconds()

                processing_delay = (
                    signal_generated_at
                    -
                    detection_at
                ).total_seconds()

                total_signal_delay = (
                    signal_generated_at
                    -
                    signal["candle_close_time"]
                ).total_seconds()

                if total_signal_delay < 0:
                    total_signal_delay = 0

                print(
                    "\n"
                    "========== PHASE 2 TIMING ==========\n"

                    f"Candle Open: "
                    f"{signal['candle_time']}\n"

                    f"Candle Close: "
                    f"{signal['candle_close_time']}\n"

                    f"Detected: "
                    f"{detection_at}\n"

                    f"Signal Generated: "
                    f"{signal_generated_at}\n"

                    f"Detection Delay: "
                    f"{detection_delay:.2f} sec\n"

                    f"Processing Delay: "
                    f"{processing_delay:.2f} sec\n"

                    f"TOTAL Signal Delay: "
                    f"{total_signal_delay:.2f} sec\n"

                    "===================================="
                )

                # ------------------------------------------------
                # OI LOG
                # ------------------------------------------------

                if np.isnan(
                    signal["oi_change_pct"]
                ):

                    oi_log = "N/A"

                else:

                    oi_log = (
                        f"{signal['oi_change_pct']:+.3f}%"
                    )

                # ------------------------------------------------
                # NORMAL LOG
                # ------------------------------------------------

                print(
                    "\n"
                    f"[PHASE 2 | {SYMBOL} {INTERVAL}]\n"

                    f"Close: "
                    f"${signal['close']:,.2f}\n"

                    f"EMA20: "
                    f"${signal['ema_20']:,.2f}\n"

                    f"EMA50: "
                    f"${signal['ema_50']:,.2f}\n"

                    f"EMA200: "
                    f"${signal['ema_200']:,.2f}\n"

                    f"Volume Ratio: "
                    f"{signal['volume_ratio']:.2f}x\n"

                    f"Delta: "
                    f"{signal['delta']:+,.1f} ETH "
                    f"({signal['delta_pct']:+.2f}%)\n"

                    f"CVD Change: "
                    f"{signal['cvd_change']:+,.1f}\n"

                    f"OI: "
                    f"{signal['oi_status']} "
                    f"({oi_log})\n"

                    f"MACD: "
                    f"{signal['macd']:.4f}\n"

                    f"MACD Signal: "
                    f"{signal['macd_signal']:.4f}\n"

                    f"Liquidity: "
                    f"{signal['liquidity_status']}\n"

                    f"Absorption: "
                    f"{signal['absorption_status']}\n"

                    f"Breakout / Hold: "
                    f"{signal['breakout_status']}\n"

                    f"Long Blocked: "
                    f"{signal['long_blocked']}\n"

                    f"Long Score: "
                    f"{long_score}/{MAX_SCORE}\n"

                    f"Short Score: "
                    f"{short_score}/{MAX_SCORE}"
                )

                # ------------------------------------------------
                # LONG
                # ------------------------------------------------

                if (
                    long_score >= MIN_SCORE
                    and
                    long_score > short_score
                    and
                    not signal["long_blocked"]
                ):

                    new_signal = (

                        last_alert_direction
                        !=
                        "LONG"

                        or
                        last_alert_score is None

                        or
                        long_score
                        >
                        last_alert_score
                    )

                    if new_signal:

                        message = build_message(
                            signal,
                            "LONG",
                            signal_generated_at
                        )

                        if send_telegram_alert(message):

                            telegram_sent_at = utc_now()

                            telegram_delay = (
                                telegram_sent_at
                                -
                                signal_generated_at
                            ).total_seconds()

                            last_alert_direction = "LONG"
                            last_alert_score = long_score

                            print(
                                "🟢 PHASE 2 LONG alert sent."
                            )

                            print(
                                "Telegram send delay: "
                                f"{telegram_delay:.2f} sec"
                            )

                # ------------------------------------------------
                # SHORT
                # ------------------------------------------------

                elif (
                    short_score >= MIN_SCORE
                    and
                    short_score > long_score
                ):

                    new_signal = (

                        last_alert_direction
                        !=
                        "SHORT"

                        or
                        last_alert_score is None

                        or
                        short_score
                        >
                        last_alert_score
                    )

                    if new_signal:

                        message = build_message(
                            signal,
                            "SHORT",
                            signal_generated_at
                        )

                        if send_telegram_alert(message):

                            telegram_sent_at = utc_now()

                            telegram_delay = (
                                telegram_sent_at
                                -
                                signal_generated_at
                            ).total_seconds()

                            last_alert_direction = "SHORT"
                            last_alert_score = short_score

                            print(
                                "🔴 PHASE 2 SHORT alert sent."
                            )

                            print(
                                "Telegram send delay: "
                                f"{telegram_delay:.2f} sec"
                            )

                # ------------------------------------------------
                # NO HIGH CONFLUENCE
                # ------------------------------------------------

                else:

                    if signal["long_blocked"]:
                        print(
                            "🚫 LONG BLOCKED: breakout resistance was not held. "
                            f"Reason: {signal['breakout_status']}"
                        )
                    else:
                        print(
                            "⚪ No Phase 2 high-confluence setup."
                        )

                    last_alert_direction = None
                    last_alert_score = None

            # ------------------------------------------------
            # POLL
            # ------------------------------------------------

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
