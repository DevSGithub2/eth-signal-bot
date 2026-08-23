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

# Check Binance every 3 seconds
POLL_SECONDS = 3

# Minimum score required for alert
MIN_SCORE = 5


# ============================================================
# PHASE 2 — LEVEL / REJECTION CONFIGURATION
# ============================================================

# Price can come this close to a level and still be considered
# a test of that level.
LEVEL_TOLERANCE_PCT = 0.0025

# Minimum wick/body relationship required for rejection.
# Example: 1.2 means wick must be at least 1.2x the candle body.
REJECTION_WICK_RATIO = 1.2

# Recent candles used for dynamic support/resistance.
LEVEL_LOOKBACK = 30

# Number of candles used to identify a local swing.
SWING_WINDOW = 2


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

            print(
                f"Telegram API error: {result}"
            )

            return False

        return True

    except requests.RequestException as e:

        print(
            f"Telegram request error: {e}"
        )

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

            print(
                "Insufficient OI data."
            )

            return pd.DataFrame()

        df_oi = pd.DataFrame(data)

        required_columns = [
            "timestamp",
            "sumOpenInterest",
            "sumOpenInterestValue"
        ]

        for col in required_columns:

            if col not in df_oi.columns:

                print(
                    f"Missing OI column: {col}"
                )

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
            .sort_values(
                "timestamp"
            )
            .reset_index(
                drop=True
            )
        )

        return df_oi

    except Exception as e:

        print(
            f"OI fetch error: {e}"
        )

        return pd.DataFrame()


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # ========================================================
    # EMA 20
    # ========================================================

    df["ema_20"] = (
        df["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    # ========================================================
    # EMA 50
    # ========================================================

    df["ema_50"] = (
        df["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    # ========================================================
    # EMA 200
    # ========================================================

    df["ema_200"] = (
        df["close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    # ========================================================
    # 20-period volume average
    # ========================================================

    df["vol_sma_20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    # ========================================================
    # Volume ratio
    # ========================================================

    df["volume_ratio"] = (
        df["volume"]
        /
        df["vol_sma_20"]
    )

    # ========================================================
    # Taker sell volume
    # ========================================================

    df["taker_sell_base"] = (
        df["volume"]
        -
        df["taker_buy_base"]
    )

    # ========================================================
    # Taker Delta
    # ========================================================

    df["delta"] = (
        df["taker_buy_base"]
        -
        df["taker_sell_base"]
    )

    # ========================================================
    # Delta percentage
    # ========================================================

    df["delta_pct"] = (
        df["delta"]
        /
        df["volume"].replace(
            0,
            np.nan
        )
    ) * 100

    # ========================================================
    # CVD
    # ========================================================

    df["cvd"] = (
        df["delta"]
        .cumsum()
    )

    return df


# ============================================================
# PHASE 2 — LEVEL DETECTION
# ============================================================

def detect_levels(df, latest_index):

    """
    Detect dynamic levels using only candles BEFORE
    the latest closed candle.

    Levels:
        EMA 20
        EMA 50
        EMA 200
        Recent Resistance
        Recent Support
    """

    levels = []

    if latest_index < LEVEL_LOOKBACK + 5:

        return levels

    latest = df.iloc[latest_index]

    # ========================================================
    # EMA LEVELS
    # ========================================================

    ema_levels = [
        (
            "EMA 20",
            float(latest["ema_20"])
        ),
        (
            "EMA 50",
            float(latest["ema_50"])
        ),
        (
            "EMA 200",
            float(latest["ema_200"])
        )
    ]

    for name, price in ema_levels:

        if np.isfinite(price):

            levels.append({
                "name": name,
                "price": price,
                "type": "EMA"
            })

    # ========================================================
    # HISTORICAL WINDOW
    # ========================================================

    start_index = max(
        0,
        latest_index - LEVEL_LOOKBACK
    )

    historical = df.iloc[
        start_index:latest_index
    ].copy()

    if historical.empty:

        return levels

    # ========================================================
    # RECENT RESISTANCE
    # ========================================================

    resistance_price = float(
        historical["high"].max()
    )

    resistance_time = historical.loc[
        historical["high"].idxmax(),
        "open_time"
    ]

    levels.append({
        "name": "Recent Resistance",
        "price": resistance_price,
        "type": "RESISTANCE",
        "time": resistance_time
    })

    # ========================================================
    # RECENT SUPPORT
    # ========================================================

    support_price = float(
        historical["low"].min()
    )

    support_time = historical.loc[
        historical["low"].idxmin(),
        "open_time"
    ]

    levels.append({
        "name": "Recent Support",
        "price": support_price,
        "type": "SUPPORT",
        "time": support_time
    })

    # ========================================================
    # LOCAL SWING HIGH
    # ========================================================

    swing_highs = []

    for i in range(
        max(start_index + SWING_WINDOW, 0),
        latest_index - SWING_WINDOW
    ):

        current_high = float(
            df.iloc[i]["high"]
        )

        left_highs = df.iloc[
            i - SWING_WINDOW:i
        ]["high"]

        right_highs = df.iloc[
            i + 1:i + 1 + SWING_WINDOW
        ]["high"]

        if (
            current_high >= left_highs.max()
            and
            current_high >= right_highs.max()
        ):

            swing_highs.append(
                (
                    current_high,
                    df.iloc[i]["open_time"]
                )
            )

    if swing_highs:

        swing_high_price, swing_high_time = (
            swing_highs[-1]
        )

        levels.append({
            "name": "Recent Swing High",
            "price": float(swing_high_price),
            "type": "RESISTANCE",
            "time": swing_high_time
        })

    # ========================================================
    # LOCAL SWING LOW
    # ========================================================

    swing_lows = []

    for i in range(
        max(start_index + SWING_WINDOW, 0),
        latest_index - SWING_WINDOW
    ):

        current_low = float(
            df.iloc[i]["low"]
        )

        left_lows = df.iloc[
            i - SWING_WINDOW:i
        ]["low"]

        right_lows = df.iloc[
            i + 1:i + 1 + SWING_WINDOW
        ]["low"]

        if (
            current_low <= left_lows.min()
            and
            current_low <= right_lows.min()
        ):

            swing_lows.append(
                (
                    current_low,
                    df.iloc[i]["open_time"]
                )
            )

    if swing_lows:

        swing_low_price, swing_low_time = (
            swing_lows[-1]
        )

        levels.append({
            "name": "Recent Swing Low",
            "price": float(swing_low_price),
            "type": "SUPPORT",
            "time": swing_low_time
        })

    return levels


# ============================================================
# PHASE 2 — REJECTION DETECTION
# ============================================================

def detect_rejection(
    candle,
    previous_candle,
    levels
):

    """
    Detect rejection of dynamic levels.

    Resistance rejection:

        Price reaches/crosses level
        +
        Upper wick is meaningful
        +
        Candle closes back below level

    Support rejection:

        Price reaches/crosses level
        +
        Lower wick is meaningful
        +
        Candle closes back above level

    This function does NOT affect the 6-factor score yet.
    """

    open_price = float(candle["open"])
    high = float(candle["high"])
    low = float(candle["low"])
    close = float(candle["close"])

    previous_close = float(
        previous_candle["close"]
    )

    body = abs(
        close - open_price
    )

    # Avoid zero-body candles causing problems.
    minimum_body = max(
        close * 0.00005,
        0.01
    )

    body_for_ratio = max(
        body,
        minimum_body
    )

    upper_wick = max(
        0.0,
        high - max(
            open_price,
            close
        )
    )

    lower_wick = max(
        0.0,
        min(
            open_price,
            close
        ) - low
    )

    upper_wick_ratio = (
        upper_wick
        /
        body_for_ratio
    )

    lower_wick_ratio = (
        lower_wick
        /
        body_for_ratio
    )

    best_resistance = None
    best_support = None

    # ========================================================
    # SEARCH LEVELS
    # ========================================================

    for level in levels:

        level_price = float(
            level["price"]
        )

        if not np.isfinite(level_price):

            continue

        distance_pct = (
            abs(
                close - level_price
            )
            /
            level_price
        )

        # ----------------------------------------------------
        # RESISTANCE
        # ----------------------------------------------------

        if level["type"] in (
            "EMA",
            "RESISTANCE"
        ):

            touched_resistance = (
                high
                >=
                level_price
                *
                (
                    1
                    -
                    LEVEL_TOLERANCE_PCT
                )
            )

            closed_below = (
                close
                <
                level_price
            )

            crossed_above = (
                high
                >
                level_price
            )

            meaningful_upper_wick = (
                upper_wick_ratio
                >=
                REJECTION_WICK_RATIO
            )

            if (
                touched_resistance
                and
                closed_below
                and
                meaningful_upper_wick
            ):

                rejection_strength = (
                    upper_wick_ratio
                )

                candidate = {
                    "detected": True,
                    "type": "RESISTANCE REJECTION",
                    "level_name": level["name"],
                    "level_type": level["type"],
                    "level_price": level_price,
                    "distance_pct": distance_pct,
                    "wick": upper_wick,
                    "wick_ratio": upper_wick_ratio,
                    "crossed_level": crossed_above,
                    "candle_bearish": (
                        close < open_price
                    ),
                    "previous_close": previous_close,
                    "follow_through": False,
                    "confirmed": False
                }

                if (
                    best_resistance is None
                    or
                    candidate["wick_ratio"]
                    >
                    best_resistance[
                        "wick_ratio"
                    ]
                ):

                    best_resistance = candidate

        # ----------------------------------------------------
        # SUPPORT
        # ----------------------------------------------------

        if level["type"] in (
            "EMA",
            "SUPPORT"
        ):

            touched_support = (
                low
                <=
                level_price
                *
                (
                    1
                    +
                    LEVEL_TOLERANCE_PCT
                )
            )

            closed_above = (
                close
                >
                level_price
            )

            crossed_below = (
                low
                <
                level_price
            )

            meaningful_lower_wick = (
                lower_wick_ratio
                >=
                REJECTION_WICK_RATIO
            )

            if (
                touched_support
                and
                closed_above
                and
                meaningful_lower_wick
            ):

                rejection_strength = (
                    lower_wick_ratio
                )

                candidate = {
                    "detected": True,
                    "type": "SUPPORT REJECTION",
                    "level_name": level["name"],
                    "level_type": level["type"],
                    "level_price": level_price,
                    "distance_pct": distance_pct,
                    "wick": lower_wick,
                    "wick_ratio": lower_wick_ratio,
                    "crossed_level": crossed_below,
                    "candle_bullish": (
                        close > open_price
                    ),
                    "previous_close": previous_close,
                    "follow_through": False,
                    "confirmed": False
                }

                if (
                    best_support is None
                    or
                    candidate["wick_ratio"]
                    >
                    best_support[
                        "wick_ratio"
                    ]
                ):

                    best_support = candidate

    # ========================================================
    # CURRENT CANDLE REJECTION
    # ========================================================

    if (
        best_resistance is not None
        and
        best_support is not None
    ):

        # If both are detected, choose the stronger one.
        if (
            best_resistance["wick_ratio"]
            >=
            best_support["wick_ratio"]
        ):

            return best_resistance

        return best_support

    if best_resistance is not None:

        return best_resistance

    if best_support is not None:

        return best_support

    return {
        "detected": False,
        "type": "NONE",
        "level_name": None,
        "level_type": None,
        "level_price": np.nan,
        "distance_pct": np.nan,
        "wick": np.nan,
        "wick_ratio": np.nan,
        "crossed_level": False,
        "confirmed": False,
        "follow_through": False
    }


# ============================================================
# PHASE 2 — CONFIRM PREVIOUS REJECTION
# ============================================================

def detect_follow_through(
    current_candle,
    previous_candle,
    levels
):

    """
    Checks whether the PREVIOUS candle rejected a level
    and the CURRENT candle followed through.

    This creates a one-candle-later confirmation.

    Bearish example:

        Previous candle rejects resistance
        Current candle closes lower

    Bullish example:

        Previous candle rejects support
        Current candle closes higher
    """

    previous_rejection = detect_rejection(
        previous_candle,
        None
        if previous_candle is None
        else previous_candle,
        levels
    )

    if not previous_rejection["detected"]:

        return {
            "confirmed": False,
            "type": "NONE",
            "level_name": None,
            "level_price": np.nan
        }

    current_open = float(
        current_candle["open"]
    )

    current_close = float(
        current_candle["close"]
    )

    previous_close = float(
        previous_candle["close"]
    )

    # ========================================================
    # RESISTANCE REJECTION FOLLOW-THROUGH
    # ========================================================

    if (
        previous_rejection["type"]
        ==
        "RESISTANCE REJECTION"
    ):

        bearish_follow = (
            current_close
            <
            previous_close
            and
            current_close
            <=
            current_open
        )

        if bearish_follow:

            previous_rejection[
                "confirmed"
            ] = True

            previous_rejection[
                "follow_through"
            ] = True

            return previous_rejection

    # ========================================================
    # SUPPORT REJECTION FOLLOW-THROUGH
    # ========================================================

    if (
        previous_rejection["type"]
        ==
        "SUPPORT REJECTION"
    ):

        bullish_follow = (
            current_close
            >
            previous_close
            and
            current_close
            >=
            current_open
        )

        if bullish_follow:

            previous_rejection[
                "confirmed"
            ] = True

            previous_rejection[
                "follow_through"
            ] = True

            return previous_rejection

    return {
        "confirmed": False,
        "type": "NONE",
        "level_name": None,
        "level_price": np.nan
    }


# ============================================================
# PHASE 2 — COMPLETE REJECTION ANALYSIS
# ============================================================

def analyze_phase2(
    df
):

    """
    Analyze the latest CLOSED candle.

    Returns:
        Current rejection
        Previous rejection confirmation
        Detected level
    """

    latest_index = len(df) - 2

    latest = df.iloc[
        latest_index
    ]

    previous = df.iloc[
        latest_index - 1
    ]

    levels = detect_levels(
        df,
        latest_index
    )

    current_rejection = detect_rejection(
        latest,
        previous,
        levels
    )

    previous_confirmation = (
        detect_follow_through(
            latest,
            previous,
            levels
        )
    )

    # ========================================================
    # PRIORITIZE CONFIRMED REJECTION
    # ========================================================

    if previous_confirmation.get(
        "confirmed",
        False
    ):

        final_rejection = (
            previous_confirmation
        )

        final_rejection[
            "signal_source"
        ] = "PREVIOUS CANDLE + FOLLOW-THROUGH"

    elif current_rejection.get(
        "detected",
        False
    ):

        final_rejection = (
            current_rejection
        )

        final_rejection[
            "signal_source"
        ] = "CURRENT CANDLE"

    else:

        final_rejection = {
            "detected": False,
            "confirmed": False,
            "follow_through": False,
            "type": "NONE",
            "level_name": None,
            "level_type": None,
            "level_price": np.nan,
            "distance_pct": np.nan,
            "wick": np.nan,
            "wick_ratio": np.nan,
            "signal_source": "NONE"
        }

    return {
        "rejection": final_rejection,
        "current_rejection": current_rejection,
        "confirmed_rejection": previous_confirmation,
        "levels": levels
    }


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
        .sort_values(
            "timestamp"
        )
        .copy()
    )

    target = pd.DataFrame({
        "candle_time": [
            candle_time
        ]
    })

    aligned = pd.merge_asof(
        target.sort_values(
            "candle_time"
        ),
        oi,
        left_on="candle_time",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(
            minutes=2
        )
    )

    if aligned.empty:

        return None

    row = aligned.iloc[0]

    if pd.isna(
        row["timestamp"]
    ):

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

    current_time = (
        current["timestamp"]
    )

    current_oi = float(
        current[
            "sumOpenInterest"
        ]
    )

    previous_rows = df_oi[
        df_oi["timestamp"]
        <
        current_time
    ]

    if previous_rows.empty:

        return result

    previous_oi = float(
        previous_rows.iloc[-1][
            "sumOpenInterest"
        ]
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

    result["change_pct"] = (
        oi_change_pct
    )

    result["oi_time"] = (
        current_time
    )

    # Price + OI matrix

    if (
        price_change_pct > 0
        and
        oi_change_pct > 0
    ):

        result["status"] = (
            "Long Buildup"
        )

    elif (
        price_change_pct < 0
        and
        oi_change_pct > 0
    ):

        result["status"] = (
            "Short Buildup"
        )

    elif (
        price_change_pct > 0
        and
        oi_change_pct < 0
    ):

        result["status"] = (
            "Short Covering"
        )

    elif (
        price_change_pct < 0
        and
        oi_change_pct < 0
    ):

        result["status"] = (
            "Long Unwinding"
        )

    else:

        result["status"] = (
            "Neutral"
        )

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

    candle_time = (
        latest["open_time"]
    )

    candle_close_time = (
        latest["close_time"]
    )

    close = float(
        latest["close"]
    )

    previous_close = float(
        previous["close"]
    )

    ema_20 = float(
        latest["ema_20"]
    )

    ema_50 = float(
        latest["ema_50"]
    )

    ema_200 = float(
        latest["ema_200"]
    )

    volume = float(
        latest["volume"]
    )

    volume_avg = float(
        latest["vol_sma_20"]
    )

    volume_ratio = float(
        latest["volume_ratio"]
    )

    delta = float(
        latest["delta"]
    )

    delta_pct = float(
        latest["delta_pct"]
    )

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
            "✅ 20 EMA > 200 EMA "
            "(Bullish Alignment)"
        )

    elif ema_20 < ema_200:

        short_score += 1

        short_reasons.append(
            "✅ 20 EMA < 200 EMA "
            "(Bearish Alignment)"
        )

    # ========================================================
    # FACTOR 3 — CLOSE VS EMA 20
    # ========================================================

    if close > ema_20:

        long_score += 1

        long_reasons.append(
            "✅ Closed Candle > 20 EMA "
            "(Momentum)"
        )

    elif close < ema_20:

        short_score += 1

        short_reasons.append(
            "✅ Closed Candle < 20 EMA "
            "(Momentum)"
        )

    # ========================================================
    # FACTOR 4 — VOLUME
    # ========================================================

    volume_confirmed = (
        not np.isnan(
            volume_ratio
        )
        and
        volume_ratio > 1.0
    )

    if volume_confirmed:

        volume_text = (
            f"📊 Volume "
            f"{volume_ratio:.2f}x "
            f"20-period average"
        )

        long_reasons.append(
            volume_text
        )

        short_reasons.append(
            volume_text
        )

    # ========================================================
    # FACTOR 5 — OI
    # ========================================================

    oi = calculate_oi_state(
        close=close,
        previous_close=previous_close,
        candle_time=candle_time,
        df_oi=df_oi
    )

    oi_status = oi[
        "status"
    ]

    oi_change_pct = oi[
        "change_pct"
    ]

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

    # ========================================================
    # PHASE 2 — REJECTION ANALYSIS
    # ========================================================

    phase2 = analyze_phase2(
        df
    )

    rejection = phase2[
        "rejection"
    ]

    # IMPORTANT:
    # Phase 2 does NOT modify long_score/short_score yet.
    # We first collect live data and test its accuracy.

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

        "oi_status":
            oi_status,

        "oi_change_pct":
            oi_change_pct,

        "long_score":
            long_score,

        "short_score":
            short_score,

        "long_reasons":
            long_reasons,

        "short_reasons":
            short_reasons,

        # ====================================================
        # PHASE 2 DATA
        # ====================================================

        "phase2_rejection":
            rejection,

        "phase2_current_rejection":
            phase2[
                "current_rejection"
            ],

        "phase2_confirmed_rejection":
            phase2[
                "confirmed_rejection"
            ],

        "phase2_levels":
            phase2[
                "levels"
            ]
    }


# ============================================================
# PHASE 2 — TELEGRAM TEXT
# ============================================================

def build_phase2_text(signal):

    rejection = signal[
        "phase2_rejection"
    ]

    if not rejection.get(
        "detected",
        False
    ):

        return (
            "🧭 *Phase 2 Level Reaction:*\n"
            "⚪ No confirmed level rejection"
        )

    level_name = rejection.get(
        "level_name"
    )

    level_price = rejection.get(
        "level_price"
    )

    wick_ratio = rejection.get(
        "wick_ratio",
        np.nan
    )

    distance_pct = rejection.get(
        "distance_pct",
        np.nan
    )

    crossed_level = rejection.get(
        "crossed_level",
        False
    )

    confirmed = rejection.get(
        "confirmed",
        False
    )

    signal_source = rejection.get(
        "signal_source",
        "UNKNOWN"
    )

    if (
        level_price is None
        or
        not np.isfinite(level_price)
    ):

        level_text = "N/A"

    else:

        level_text = (
            f"${level_price:,.2f}"
        )

    if np.isfinite(
        distance_pct
    ):

        distance_text = (
            f"{distance_pct * 100:.2f}%"
        )

    else:

        distance_text = "N/A"

    if np.isfinite(
        wick_ratio
    ):

        wick_text = (
            f"{wick_ratio:.2f}x body"
        )

    else:

        wick_text = "N/A"

    if confirmed:

        confirmation_text = (
            "✅ Confirmed + Follow-through"
        )

    else:

        confirmation_text = (
            "⚠️ Rejection detected — "
            "follow-through not confirmed"
        )

    crossed_text = (
        "YES"
        if crossed_level
        else
        "NO"
    )

    return (

        f"🧭 *Phase 2 Level Reaction:*\n"

        f"{'🔴' if 'RESISTANCE' in rejection['type'] else '🟢'} "
        f"*{rejection['type']}*\n"

        f"📍 Level: "
        f"`{level_name}`\n"

        f"💵 Level Price: "
        f"`{level_text}`\n"

        f"📏 Distance: "
        f"`{distance_text}`\n"

        f"🕯 Wick: "
        f"`{wick_text}`\n"

        f"↕️ Level Crossed: "
        f"`{crossed_text}`\n"

        f"🔎 Source: "
        f"`{signal_source}`\n"

        f"{confirmation_text}"
    )


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def build_message(
    signal,
    direction,
    generated_at
):

    if direction == "LONG":

        score = signal[
            "long_score"
        ]

        reasons = signal[
            "long_reasons"
        ]

        if score == 6:

            tag = (
                "🟢 STRONG LONG WATCH"
            )

        else:

            tag = (
                "🟡 LONG WATCH"
            )

    else:

        score = signal[
            "short_score"
        ]

        reasons = signal[
            "short_reasons"
        ]

        if score == 6:

            tag = (
                "🔴 STRONG SHORT WATCH"
            )

        else:

            tag = (
                "🟠 SHORT WATCH"
            )

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

    # Actual engine delay

    engine_delay = (
        generated_at
        -
        signal["candle_close_time"]
    ).total_seconds()

    if engine_delay < 0:

        engine_delay = 0

    # ========================================================
    # PHASE 2 TEXT
    # ========================================================

    phase2_text = build_phase2_text(
        signal
    )

    message = (

        f"{tag} *(Score: {score}/6)*\n"

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

        f"📊 *50 EMA:* "
        f"`${signal['ema_50']:,.2f}`\n"

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

        + "\n".join(
            reasons
        )

        + "\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"{phase2_text}\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"⚠️ *WATCH ONLY — Human chart "
        f"verification & level/retest "
        f"confirmation required.*"
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
        "🧭 Phase 2 Level/Rejection Detection: ENABLED"
    )

    print(
        f"📏 Level tolerance: "
        f"{LEVEL_TOLERANCE_PCT * 100:.2f}%"
    )

    print(
        f"🕯 Rejection wick ratio: "
        f"{REJECTION_WICK_RATIO:.2f}x"
    )

    last_processed_time = None

    last_alert_direction = None

    last_alert_score = None

    while True:

        try:

            # =================================================
            # FETCH CANDLES
            # =================================================

            df = fetch_klines()

            # Latest CLOSED candle

            latest_closed_time = (
                df.iloc[-2]["open_time"]
            )

            # =================================================
            # PROCESS ONLY ONCE PER CANDLE
            # =================================================

            if (
                latest_closed_time
                !=
                last_processed_time
            ):

                # Time bot detected
                # the new closed candle

                detection_at = utc_now()

                last_processed_time = (
                    latest_closed_time
                )

                # =================================================
                # CALCULATE INDICATORS
                # =================================================

                df = calculate_indicators(
                    df
                )

                # =================================================
                # FETCH OI
                # =================================================

                df_oi = (
                    fetch_open_interest()
                )

                # =================================================
                # EVALUATE SIGNAL
                # =================================================

                signal = evaluate_scoring(
                    df,
                    df_oi
                )

                # Signal generation time

                signal_generated_at = (
                    utc_now()
                )

                long_score = (
                    signal["long_score"]
                )

                short_score = (
                    signal["short_score"]
                )

                # =================================================
                # TIMING DIAGNOSTICS
                # =================================================

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

                # =================================================
                # NORMAL LOG
                # =================================================

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

                    f"50 EMA: "
                    f"${signal['ema_50']:,.2f}\n"

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
                    f"{short_score}/6"
                )

                # =================================================
                # PHASE 2 LOG
                # =================================================

                rejection = signal[
                    "phase2_rejection"
                ]

                print(
                    "\n"
                    "========== PHASE 2 ==========\n"
                )

                if rejection.get(
                    "detected",
                    False
                ):

                    print(
                        f"Reaction: "
                        f"{rejection['type']}"
                    )

                    print(
                        f"Level: "
                        f"{rejection['level_name']}"
                    )

                    if np.isfinite(
                        rejection[
                            "level_price"
                        ]
                    ):

                        print(
                            f"Level Price: "
                            f"${rejection['level_price']:,.2f}"
                        )

                    if np.isfinite(
                        rejection[
                            "wick_ratio"
                        ]
                    ):

                        print(
                            f"Wick Ratio: "
                            f"{rejection['wick_ratio']:.2f}x"
                        )

                    print(
                        f"Confirmed: "
                        f"{rejection.get('confirmed', False)}"
                    )

                    print(
                        f"Follow-through: "
                        f"{rejection.get('follow_through', False)}"
                    )

                else:

                    print(
                        "No level rejection detected."
                    )

                print(
                    "=============================="
                )

                # =================================================
                # LONG
                # =================================================

                if (
                    long_score >= MIN_SCORE
                    and
                    long_score > short_score
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

                        message = (
                            build_message(
                                signal,
                                "LONG",
                                signal_generated_at
                            )
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
                                "🟢 LONG alert sent."
                            )

                            print(
                                "Telegram send delay: "
                                f"{telegram_delay:.2f} sec"
                            )

                # =================================================
                # SHORT
                # =================================================

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
                        last_alert_score
                        is None

                        or
                        short_score
                        >
                        last_alert_score
                    )

                    if new_signal:

                        message = (
                            build_message(
                                signal,
                                "SHORT",
                                signal_generated_at
                            )
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
                                "🔴 SHORT alert sent."
                            )

                            print(
                                "Telegram send delay: "
                                f"{telegram_delay:.2f} sec"
                            )

                # =================================================
                # NO HIGH-CONFLUENCE SETUP
                # =================================================

                else:

                    print(
                        "⚪ No high-confluence setup."
                    )

                    last_alert_direction = None

                    last_alert_score = None

            # =================================================
            # POLL AGAIN
            # =================================================

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
