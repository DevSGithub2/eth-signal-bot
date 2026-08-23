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

# Existing Phase 1 threshold
MIN_SCORE = 5

# Phase 2 settings
LEVEL_LOOKBACK = 50
SWING_WINDOW = 3

# Distance allowed between price and an important level.
# 0.25% means price can be within 0.25% of the level.
LEVEL_TOLERANCE_PCT = 0.25

# Minimum wick/body relationship for rejection.
# 1.0 means wick >= body.
MIN_WICK_BODY_RATIO = 1.0

# Phase 2 rejection threshold
MIN_REJECTION_SCORE = 3


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

    # Volume average
    df["vol_sma_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    # Volume ratio
    df["volume_ratio"] = (
        df["volume"]
        /
        df["vol_sma_20"]
    )

    # Taker sell
    df["taker_sell_base"] = (
        df["volume"]
        -
        df["taker_buy_base"]
    )

    # Taker delta
    df["delta"] = (
        df["taker_buy_base"]
        -
        df["taker_sell_base"]
    )

    # Delta %
    df["delta_pct"] = (
        df["delta"]
        /
        df["volume"].replace(
            0,
            np.nan
        )
    ) * 100

    # CVD
    df["cvd"] = df["delta"].cumsum()

    # Candle structure
    df["body"] = (
        df["close"] - df["open"]
    ).abs()

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
        (
            current_oi
            -
            previous_oi
        )
        /
        previous_oi
    ) * 100

    price_change_pct = (
        (
            close
            -
            previous_close
        )
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
# PHASE 2 — LEVEL DETECTION
# ============================================================

def find_phase2_levels(df):

    latest = df.iloc[-2]

    levels = []

    # --------------------------------------------------------
    # EMA LEVELS
    # --------------------------------------------------------

    levels.append({
        "name": "EMA 20",
        "price": float(latest["ema_20"]),
        "type": "EMA"
    })

    levels.append({
        "name": "EMA 200",
        "price": float(latest["ema_200"]),
        "type": "EMA"
    })

    # --------------------------------------------------------
    # RECENT SWING HIGH / RESISTANCE
    # --------------------------------------------------------

    start = max(
        0,
        len(df) - LEVEL_LOOKBACK - 2
    )

    end = len(df) - 2

    recent = df.iloc[start:end]

    swing_highs = []

    for i in range(
        SWING_WINDOW,
        len(recent) - SWING_WINDOW
    ):

        high = float(
            recent.iloc[i]["high"]
        )

        left = recent.iloc[
            i - SWING_WINDOW:i
        ]["high"].max()

        right = recent.iloc[
            i + 1:i + 1 + SWING_WINDOW
        ]["high"].max()

        if (
            high >= left
            and
            high >= right
        ):
            swing_highs.append(high)

    if swing_highs:

        resistance = max(
            swing_highs
        )

        levels.append({
            "name": "Recent Resistance",
            "price": resistance,
            "type": "RESISTANCE"
        })

    # --------------------------------------------------------
    # RECENT SWING LOW / SUPPORT
    # --------------------------------------------------------

    swing_lows = []

    for i in range(
        SWING_WINDOW,
        len(recent) - SWING_WINDOW
    ):

        low = float(
            recent.iloc[i]["low"]
        )

        left = recent.iloc[
            i - SWING_WINDOW:i
        ]["low"].min()

        right = recent.iloc[
            i + 1:i + 1 + SWING_WINDOW
        ]["low"].min()

        if (
            low <= left
            and
            low <= right
        ):
            swing_lows.append(low)

    if swing_lows:

        support = min(
            swing_lows
        )

        levels.append({
            "name": "Recent Support",
            "price": support,
            "type": "SUPPORT"
        })

    return levels


# ============================================================
# PHASE 2 — REJECTION DETECTION
# ============================================================

def detect_rejection(
    df,
    levels
):

    latest = df.iloc[-2]
    previous = df.iloc[-3]
    previous2 = df.iloc[-4]

    close = float(latest["close"])
    open_price = float(latest["open"])
    high = float(latest["high"])
    low = float(latest["low"])

    body = max(
        float(latest["body"]),
        1e-9
    )

    upper_wick = float(
        latest["upper_wick"]
    )

    lower_wick = float(
        latest["lower_wick"]
    )

    volume_ratio = float(
        latest["volume_ratio"]
    )

    delta = float(
        latest["delta"]
    )

    candidates = []

    for level in levels:

        level_price = level["price"]

        distance_pct = (
            abs(close - level_price)
            /
            level_price
        ) * 100

        # ----------------------------------------------------
        # Price must be reasonably close to level
        # ----------------------------------------------------

        if distance_pct > LEVEL_TOLERANCE_PCT:
            continue

        # ----------------------------------------------------
        # BEARISH REJECTION
        #
        # Price trades at/above level
        # but closes below it.
        # ----------------------------------------------------

        bearish_touch = (
            high >= level_price
            * (1 - LEVEL_TOLERANCE_PCT / 100)
        )

        bearish_close = (
            close < level_price
        )

        bearish_candle = (
            close < open_price
        )

        strong_upper_wick = (
            upper_wick
            >=
            body * MIN_WICK_BODY_RATIO
        )

        if (
            bearish_touch
            and
            bearish_close
            and
            strong_upper_wick
        ):

            rejection_score = 0
            reasons = []

            rejection_score += 1
            reasons.append(
                "Price tested important level"
            )

            rejection_score += 1
            reasons.append(
                "Upper-wick rejection"
            )

            rejection_score += 1
            reasons.append(
                "Candle closed below level"
            )

            if bearish_candle:
                rejection_score += 1
                reasons.append(
                    "Bearish rejection candle"
                )

            if (
                volume_ratio >= 1.2
            ):
                rejection_score += 1
                reasons.append(
                    f"Volume {volume_ratio:.2f}x average"
                )

            if delta < 0:
                rejection_score += 1
                reasons.append(
                    "Negative taker delta"
                )

            # Check consecutive rejection/failure
            previous_close = float(
                previous["close"]
            )

            previous2_close = float(
                previous2["close"]
            )

            consecutive_failure = (
                previous_close < level_price
                and
                previous2_close < level_price
            )

            if consecutive_failure:
                rejection_score += 1
                reasons.append(
                    "Multi-candle level failure"
                )

            candidates.append({
                "direction": "SHORT",
                "level_name": level["name"],
                "level_type": level["type"],
                "level_price": level_price,
                "distance_pct": distance_pct,
                "rejection_score": rejection_score,
                "wick_ratio": (
                    upper_wick / body
                ),
                "reasons": reasons
            })

        # ----------------------------------------------------
        # BULLISH REJECTION
        #
        # Price trades at/below level
        # but closes above it.
        # ----------------------------------------------------

        bullish_touch = (
            low <= level_price
            * (1 + LEVEL_TOLERANCE_PCT / 100)
        )

        bullish_close = (
            close > level_price
        )

        bullish_candle = (
            close > open_price
        )

        strong_lower_wick = (
            lower_wick
            >=
            body * MIN_WICK_BODY_RATIO
        )

        if (
            bullish_touch
            and
            bullish_close
            and
            strong_lower_wick
        ):

            rejection_score = 0
            reasons = []

            rejection_score += 1
            reasons.append(
                "Price tested important level"
            )

            rejection_score += 1
            reasons.append(
                "Lower-wick rejection"
            )

            rejection_score += 1
            reasons.append(
                "Candle closed above level"
            )

            if bullish_candle:
                rejection_score += 1
                reasons.append(
                    "Bullish rejection candle"
                )

            if (
                volume_ratio >= 1.2
            ):
                rejection_score += 1
                reasons.append(
                    f"Volume {volume_ratio:.2f}x average"
                )

            if delta > 0:
                rejection_score += 1
                reasons.append(
                    "Positive taker delta"
                )

            previous_close = float(
                previous["close"]
            )

            previous2_close = float(
                previous2["close"]
            )

            consecutive_failure = (
                previous_close > level_price
                and
                previous2_close > level_price
            )

            if consecutive_failure:
                rejection_score += 1
                reasons.append(
                    "Multi-candle level failure"
                )

            candidates.append({
                "direction": "LONG",
                "level_name": level["name"],
                "level_type": level["type"],
                "level_price": level_price,
                "distance_pct": distance_pct,
                "rejection_score": rejection_score,
                "wick_ratio": (
                    lower_wick / body
                ),
                "reasons": reasons
            })

    if not candidates:
        return None

    # Strongest rejection only
    candidates.sort(
        key=lambda x: x["rejection_score"],
        reverse=True
    )

    strongest = candidates[0]

    if (
        strongest["rejection_score"]
        <
        MIN_REJECTION_SCORE
    ):
        return None

    return strongest


# ============================================================
# PHASE 1 SCORING
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

    # --------------------------------------------------------
    # FACTOR 1 — PRICE VS EMA 200
    # --------------------------------------------------------

    if close > ema_200:

        long_score += 1

        long_reasons.append(
            "✅ Price > 200 EMA"
        )

    elif close < ema_200:

        short_score += 1

        short_reasons.append(
            "✅ Price < 200 EMA"
        )

    # --------------------------------------------------------
    # FACTOR 2 — EMA 20 VS EMA 200
    # --------------------------------------------------------

    if ema_20 > ema_200:

        long_score += 1

        long_reasons.append(
            "✅ 20 EMA > 200 EMA"
        )

    elif ema_20 < ema_200:

        short_score += 1

        short_reasons.append(
            "✅ 20 EMA < 200 EMA"
        )

    # --------------------------------------------------------
    # FACTOR 3 — CLOSE VS EMA 20
    # --------------------------------------------------------

    if close > ema_20:

        long_score += 1

        long_reasons.append(
            "✅ Close > 20 EMA"
        )

    elif close < ema_20:

        short_score += 1

        short_reasons.append(
            "✅ Close < 20 EMA"
        )

    # --------------------------------------------------------
    # FACTOR 4 — VOLUME
    # --------------------------------------------------------

    volume_confirmed = (
        not np.isnan(volume_ratio)
        and
        volume_ratio > 1.0
    )

    volume_text = (
        f"📊 Volume {volume_ratio:.2f}x average"
    )

    if volume_confirmed:

        long_reasons.append(
            volume_text
        )

        short_reasons.append(
            volume_text
        )

    # --------------------------------------------------------
    # FACTOR 5 — OI
    # --------------------------------------------------------

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
            f"⚠️ OI Short Covering "
            f"({oi_change_pct:+.3f}%)"
        )

    elif oi_status == "Long Unwinding":

        short_reasons.append(
            f"⚠️ OI Long Unwinding "
            f"({oi_change_pct:+.3f}%)"
        )

    else:

        long_reasons.append(
            "⚪ OI Neutral / Unavailable"
        )

        short_reasons.append(
            "⚪ OI Neutral / Unavailable"
        )

    # --------------------------------------------------------
    # FACTOR 6 — TAKER DELTA
    # --------------------------------------------------------

    if delta > 0:

        long_score += 1

        long_reasons.append(
            f"✅ Taker Delta "
            f"{delta:+,.1f} ETH "
            f"({delta_pct:+.2f}%)"
        )

    elif delta < 0:

        short_score += 1

        short_reasons.append(
            f"✅ Taker Delta "
            f"{delta:+,.1f} ETH "
            f"({delta_pct:+.2f}%)"
        )

    else:

        long_reasons.append(
            "⚪ Taker Delta Neutral"
        )

        short_reasons.append(
            "⚪ Taker Delta Neutral"
        )

    # --------------------------------------------------------
    # VOLUME CONFIRMATION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PHASE 2 LEVEL DETECTION
    # --------------------------------------------------------

    levels = find_phase2_levels(df)

    rejection = detect_rejection(
        df,
        levels
    )

    # --------------------------------------------------------
    # PHASE 2 SCORE
    # --------------------------------------------------------

    phase2_score = 0

    if rejection:

        phase2_score = rejection[
            "rejection_score"
        ]

        if rejection["direction"] == "SHORT":

            short_reasons.append(
                "🔴 Phase 2 Resistance Rejection"
            )

        else:

            long_reasons.append(
                "🟢 Phase 2 Support Rejection"
            )

    return {

        "candle_time":
            candle_time,

        "candle_close_time":
            candle_close_time,

        "close":
            close,

        "ema_20":
            ema_20,

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

        "oi_status":
            oi_status,

        "oi_change_pct":
            oi_change_pct,

        "long_score":
            long_score,

        "short_score":
            short_score,

        "phase2_score":
            phase2_score,

        "rejection":
            rejection,

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

        tag = "🟢 LONG WATCH"

    else:

        score = signal["short_score"]

        reasons = signal["short_reasons"]

        tag = "🔴 SHORT WATCH"

    rejection = signal["rejection"]

    if np.isnan(
        signal["oi_change_pct"]
    ):
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

        f"{tag} *(Phase 1 Score: "
        f"{score}/6)*\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"🪙 *Pair:* `{SYMBOL}`\n"

        f"⏱ *Interval:* `{INTERVAL}`\n"

        f"🕐 *Candle Open:* "
        f"`{candle_time}`\n"

        f"🔒 *Candle Close:* "
        f"`{candle_close_time}`\n"

        f"⚡ *Signal Generated:* "
        f"`{generated_time}`\n"

        f"⏱ *Engine Delay:* "
        f"`{engine_delay:.1f} sec`\n"

        f"💵 *Price:* "
        f"`${signal['close']:,.2f}`\n"

        f"📈 *20 EMA:* "
        f"`${signal['ema_20']:,.2f}`\n"

        f"📉 *200 EMA:* "
        f"`${signal['ema_200']:,.2f}`\n"

        f"📊 *Volume:* "
        f"`{signal['volume_ratio']:.2f}x`\n"

        f"📊 *Taker Delta:* "
        f"`{signal['delta']:+,.1f} ETH`\n"

        f"📊 *Delta %:* "
        f"`{signal['delta_pct']:+.2f}%`\n"

        f"⚡ *OI:* "
        f"`{signal['oi_status']}`\n"

        f"⚡ *OI Change:* "
        f"`{oi_change}`\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"
    )

    # --------------------------------------------------------
    # PHASE 2 MESSAGE
    # --------------------------------------------------------

    if rejection:

        direction_text = (
            "🔴 RESISTANCE REJECTION"
            if rejection["direction"] == "SHORT"
            else
            "🟢 SUPPORT REJECTION"
        )

        message += (

            f"🎯 *PHASE 2: "
            f"{direction_text}*\n"

            f"📍 *Level:* "
            f"`{rejection['level_name']}`\n"

            f"💵 *Level Price:* "
            f"`${rejection['level_price']:,.2f}`\n"

            f"📏 *Distance:* "
            f"`{rejection['distance_pct']:.3f}%`\n"

            f"🕯️ *Wick/Body:* "
            f"`{rejection['wick_ratio']:.2f}x`\n"

            f"🎯 *Rejection Score:* "
            f"`{rejection['rejection_score']}`\n"

            f"📋 *Rejection Evidence:*\n"

            +
            "\n".join(
                f"• {r}"
                for r in rejection["reasons"]
            )

            +
            "\n\n"
        )

    else:

        message += (
            "⚪ *PHASE 2:* "
            "No measurable rejection detected\n\n"
        )

    message += (

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"📋 *Phase 1 Confluences:*\n"

        +
        "\n".join(
            reasons
        )

        +
        "\n\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"⚠️ *WATCH ONLY — Human chart "
        f"verification required.*\n"

        f"⚠️ *Phase 2 does NOT execute trades.*"
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

    print(
        "🧠 Phase 2 Support/Resistance "
        "Rejection Detection: ACTIVE"
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
            # PROCESS ONCE PER CLOSED CANDLE
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

                df_oi = (
                    fetch_open_interest()
                )

                # ------------------------------------------------
                # SCORING
                # ------------------------------------------------

                signal = evaluate_scoring(
                    df,
                    df_oi
                )

                signal_generated_at = (
                    utc_now()
                )

                long_score = signal[
                    "long_score"
                ]

                short_score = signal[
                    "short_score"
                ]

                phase2_score = signal[
                    "phase2_score"
                ]

                rejection = signal[
                    "rejection"
                ]

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
                    "========== TIMING ==========\n"

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

                    "============================"
                )

                # ------------------------------------------------
                # NORMAL LOG
                # ------------------------------------------------

                if np.isnan(
                    signal["oi_change_pct"]
                ):

                    oi_log = "N/A"

                else:

                    oi_log = (
                        f"{signal['oi_change_pct']:+.3f}%"
                    )

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
                    f"({oi_log})\n"

                    f"Long Score: "
                    f"{long_score}/6\n"

                    f"Short Score: "
                    f"{short_score}/6\n"

                    f"Phase 2 Rejection Score: "
                    f"{phase2_score}"
                )

                # ------------------------------------------------
                # PHASE 2 LOG
                # ------------------------------------------------

                if rejection:

                    print(
                        "\n"
                        "========== PHASE 2 ==========\n"

                        f"Direction: "
                        f"{rejection['direction']}\n"

                        f"Level: "
                        f"{rejection['level_name']}\n"

                        f"Level Price: "
                        f"${rejection['level_price']:,.2f}\n"

                        f"Distance: "
                        f"{rejection['distance_pct']:.3f}%\n"

                        f"Wick/Body: "
                        f"{rejection['wick_ratio']:.2f}x\n"

                        f"Rejection Score: "
                        f"{rejection['rejection_score']}\n"

                        "=============================="
                    )

                else:

                    print(
                        "⚪ Phase 2: "
                        "No measurable rejection."
                    )

                # =================================================
                # SHORT SIGNAL
                # =================================================

                short_phase2_confirmed = (
                    rejection is not None
                    and
                    rejection["direction"]
                    == "SHORT"
                    and
                    phase2_score
                    >=
                    MIN_REJECTION_SCORE
                )

                if (
                    short_score >= MIN_SCORE
                    and
                    short_score > long_score
                    and
                    short_phase2_confirmed
                ):

                    new_signal = (
                        last_alert_direction
                        !=
                        "SHORT"
                        or
                        last_alert_score
                        is None
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

                        if send_telegram_alert(
                            message
                        ):

                            telegram_sent_at = (
                                utc_now()
                            )

                            telegram_delay = (
                                telegram_sent_at
                                -
                                signal_generated_at
                            ).total_seconds()

                            last_alert_direction = (
                                "SHORT"
                            )

                            last_alert_score = (
                                short_score
                            )

                            print(
                                "🔴 SHORT Phase 2 "
                                "alert sent."
                            )

                            print(
                                "Telegram send delay: "
                                f"{telegram_delay:.2f} sec"
                            )

                # =================================================
                # LONG SIGNAL
                # =================================================

                elif (
                    long_score >= MIN_SCORE
                    and
                    long_score > short_score
                    and
                    rejection is not None
                    and
                    rejection["direction"]
                    == "LONG"
                    and
                    phase2_score
                    >=
                    MIN_REJECTION_SCORE
                ):

                    new_signal = (
                        last_alert_direction
                        !=
                        "LONG"
                        or
                        last_alert_score
                        is None
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

                        if send_telegram_alert(
                            message
                        ):

                            telegram_sent_at = (
                                utc_now()
                            )

                            telegram_delay = (
                                telegram_sent_at
                                -
                                signal_generated_at
                            ).total_seconds()

                            last_alert_direction = (
                                "LONG"
                            )

                            last_alert_score = (
                                long_score
                            )

                            print(
                                "🟢 LONG Phase 2 "
                                "alert sent."
                            )

                            print(
                                "Telegram send delay: "
                                f"{telegram_delay:.2f} sec"
                            )

                # =================================================
                # NO PHASE 2 SETUP
                # =================================================

                else:

                    print(
                        "⚪ No Phase 2 "
                        "high-confluence setup."
                    )

                    last_alert_direction = None
                    last_alert_score = None

            # ------------------------------------------------
            # POLL
            # ------------------------------------------------

            time.sleep(
                POLL_SECONDS
            )

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
