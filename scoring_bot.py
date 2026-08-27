import os
import time
from dataclasses import dataclass
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


# ============================================================
# BREAKOUT / REJECTION
# ============================================================

BREAKOUT_HOLD_TOLERANCE_PCT = 0.15


# ============================================================
# PRE-BREAKOUT WATCH
# ============================================================

PRE_BREAKOUT_DISTANCE_PCT = 0.20
PRE_BREAKOUT_MIN_CONFLUENCE = 4
PRE_BREAKOUT_MIN_VOLUME_RATIO = 0.80
PRE_BREAKOUT_ALERT_COOLDOWN_SECONDS = 60


# ============================================================
# PHASE 2 SCORING
# ============================================================

MIN_SCORE = 6
MAX_SCORE = 8


# ============================================================
# RSI SETTINGS
# ============================================================

RSI_PERIOD_FAST = 14
RSI_PERIOD_SLOW = 24

# RSI is used as momentum confirmation.
# It is NOT treated as a standalone trade trigger.
RSI_BEARISH_LEVEL = 45.0
RSI_STRONG_BEARISH_LEVEL = 40.0

RSI_BULLISH_LEVEL = 55.0


# ============================================================
# BULLISH ZONE REJECTION DETECTOR
# ============================================================

BULLISH_REJECTION_MIN_SCORE = 4
BULLISH_REJECTION_VOLUME_MULTIPLIER = 1.30
BULLISH_REJECTION_WICK_RATIO = 0.40


@dataclass
class RejectionResult:
    rejected: bool
    score: int
    reasons: list


def detect_bullish_zone_rejection(
    candle,
    ema20,
    ema50,
    ema200,
    resistance,
    volume,
    avg_volume,
    close_below_level=True,
):
    """
    Detect rejection of a bullish/resistance zone.

    IMPORTANT:
    This function only invalidates LONG.
    It does NOT create a SHORT signal by itself.
    """

    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    c = float(candle["close"])

    candle_range = max(h - l, 1e-9)

    upper_wick = h - max(o, c)

    upper_wick_ratio = (
        upper_wick / candle_range
    )

    score = 0
    reasons = []

    touched_zone = (
        h >= float(resistance)
    )

    if touched_zone:
        score += 1
        reasons.append(
            "Bullish zone/resistance tested"
        )

    if (
        touched_zone
        and
        upper_wick_ratio >= BULLISH_REJECTION_WICK_RATIO
    ):
        score += 1
        reasons.append(
            "Large upper wick rejection"
        )

    if (
        touched_zone
        and
        close_below_level
        and
        c < float(resistance)
    ):
        score += 2
        reasons.append(
            "Close back below bullish zone"
        )

    if (
        touched_zone
        and
        c < o
    ):
        score += 1
        reasons.append(
            "Bearish rejection candle"
        )

    if (
        avg_volume > 0
        and
        volume >= avg_volume * BULLISH_REJECTION_VOLUME_MULTIPLIER
    ):
        score += 1
        reasons.append(
            "High rejection volume"
        )

    if ema20 > ema50 > ema200:
        score += 1
        reasons.append(
            "Bullish EMA structure"
        )

    return RejectionResult(
        rejected=(
            score >= BULLISH_REJECTION_MIN_SCORE
        ),
        score=score,
        reasons=reasons,
    )


# ============================================================
# BINANCE FUTURES MARKET SCANNER
# ============================================================

MARKET_SCANNER_ENABLED = (
    os.getenv(
        "MARKET_SCANNER_ENABLED",
        "true"
    ).lower() == "true"
)

MARKET_SCANNER_MIN_VOLUME_USDT = float(
    os.getenv(
        "MARKET_SCANNER_MIN_VOLUME_USDT",
        "20000000"
    )
)

MARKET_SCANNER_TOP_N = int(
    os.getenv(
        "MARKET_SCANNER_TOP_N",
        "10"
    )
)

MARKET_SCANNER_INTERVAL_SECONDS = int(
    os.getenv(
        "MARKET_SCANNER_INTERVAL_SECONDS",
        "60"
    )
)

BINANCE_EXCHANGE_INFO_URL = (
    "https://fapi.binance.com/fapi/v1/exchangeInfo"
)

BINANCE_24H_TICKER_URL = (
    "https://fapi.binance.com/fapi/v1/ticker/24hr"
)


def get_futures_symbols():

    response = requests.get(
        BINANCE_EXCHANGE_INFO_URL,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return {
        s["symbol"]
        for s in data.get("symbols", [])
        if (
            s.get("contractType") == "PERPETUAL"
            and
            s.get("quoteAsset") == "USDT"
            and
            s.get("status") == "TRADING"
        )
    }


def get_24h_tickers():

    response = requests.get(
        BINANCE_24H_TICKER_URL,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return (
        data
        if isinstance(data, list)
        else []
    )


def scan_market(
    min_volume_usdt=MARKET_SCANNER_MIN_VOLUME_USDT,
    top_n=MARKET_SCANNER_TOP_N
):

    symbols = get_futures_symbols()
    tickers = get_24h_tickers()

    candidates = []

    for t in tickers:

        symbol = t.get("symbol")

        if symbol not in symbols:
            continue

        try:

            price = float(
                t["lastPrice"]
            )

            change = float(
                t["priceChangePercent"]
            )

            volume = float(
                t["quoteVolume"]
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ):
            continue

        if volume < min_volume_usdt:
            continue

        candidates.append({
            "symbol": symbol,
            "price": price,
            "change_24h": change,
            "volume_24h": volume,
        })

    candidates.sort(
        key=lambda x: x["volume_24h"],
        reverse=True
    )

    return candidates[:top_n]


def format_scanner_log(candidates):

    if not candidates:
        return (
            "No qualifying Futures candidates."
        )

    lines = [
        "\n===== BINANCE FUTURES MARKET SCANNER ====="
    ]

    for i, coin in enumerate(
        candidates,
        1
    ):

        lines.append(
            f"{i:02d}. {coin['symbol']} | "
            f"24h {coin['change_24h']:+.2f}% | "
            f"Volume ${coin['volume_24h']/1_000_000:.2f}M | "
            f"Price ${coin['price']:,.8f}"
        )

    return "\n".join(lines)


# ============================================================
# UTC TIME
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_alert(message: str):

    if (
        not TELEGRAM_BOT_TOKEN
        or
        not TELEGRAM_CHANNEL_ID
    ):

        print(
            "Telegram credentials are missing."
        )

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

        if not result.get(
            "ok",
            False
        ):

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
        or
        len(data) < 220
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
            or
            len(data) < 3
        ):

            print(
                "Insufficient OI data."
            )

            return pd.DataFrame()

        df_oi = pd.DataFrame(
            data
        )

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
            .sort_values("timestamp")
            .reset_index(drop=True)
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
    # EMA 20 / 50 / 200
    # ========================================================

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

    # ========================================================
    # VOLUME
    # ========================================================

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

    # ========================================================
    # TAKER DELTA
    # ========================================================

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
        df["volume"].replace(
            0,
            np.nan
        )
    ) * 100

    # ========================================================
    # CVD
    # ========================================================

    df["cvd"] = (
        df["delta"].cumsum()
    )

    df["cvd_change"] = (
        df["cvd"]
        -
        df["cvd"].shift(1)
    )

    # ========================================================
    # MACD 12 / 26 / 9
    # ========================================================

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
        ema_12
        -
        ema_26
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

    # ========================================================
    # RSI 14
    # ========================================================

    delta_price = (
        df["close"].diff()
    )

    gain = (
        delta_price.clip(lower=0)
    )

    loss = (
        -delta_price.clip(upper=0)
    )

    avg_gain = (
        gain.ewm(
            alpha=1 / RSI_PERIOD_FAST,
            adjust=False
        )
        .mean()
    )

    avg_loss = (
        loss.ewm(
            alpha=1 / RSI_PERIOD_FAST,
            adjust=False
        )
        .mean()
    )

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi_14"] = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    # ========================================================
    # RSI 24
    # ========================================================

    avg_gain_24 = (
        gain.ewm(
            alpha=1 / RSI_PERIOD_SLOW,
            adjust=False
        )
        .mean()
    )

    avg_loss_24 = (
        loss.ewm(
            alpha=1 / RSI_PERIOD_SLOW,
            adjust=False
        )
        .mean()
    )

    rs_24 = (
        avg_gain_24
        /
        avg_loss_24.replace(
            0,
            np.nan
        )
    )

    df["rsi_24"] = (
        100
        -
        (
            100
            /
            (1 + rs_24)
        )
    )

    # ========================================================
    # CANDLE STRUCTURE
    # ========================================================

    df["body"] = (
        (
            df["close"]
            -
            df["open"]
        ).abs()
    )

    df["range"] = (
        df["high"]
        -
        df["low"]
    )

    df["upper_wick"] = (
        df["high"]
        -
        df[["open", "close"]]
        .max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]]
        .min(axis=1)
        -
        df["low"]
    )

    # ========================================================
    # RECENT HIGH / LOW
    # EXCLUDES CURRENT CANDLE
    # ========================================================

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
        current["sumOpenInterest"]
    )

    previous_rows = df_oi[
        df_oi["timestamp"]
        <
        current_time
    ]

    if previous_rows.empty:
        return result

    previous_oi = float(
        previous_rows
        .iloc[-1]["sumOpenInterest"]
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
# LIQUIDITY / SWEEP DETECTION
# ============================================================

def detect_liquidity(df):

    latest = df.iloc[-2]

    recent_high = (
        latest["recent_high"]
    )

    recent_low = (
        latest["recent_low"]
    )

    high = float(
        latest["high"]
    )

    low = float(
        latest["low"]
    )

    close = float(
        latest["close"]
    )

    result = {
        "status": "Neutral",
        "long": False,
        "short": False,
        "level": np.nan
    }

    if (
        pd.isna(recent_high)
        or
        pd.isna(recent_low)
    ):
        return result

    # ========================================================
    # BEARISH LIQUIDITY SWEEP
    # ========================================================

    if (
        high > recent_high
        and
        close < recent_high
    ):

        result["status"] = (
            "Bearish Liquidity Sweep"
        )

        result["short"] = True
        result["level"] = recent_high

        return result

    # ========================================================
    # BULLISH LIQUIDITY SWEEP
    # ========================================================

    if (
        low < recent_low
        and
        close > recent_low
    ):

        result["status"] = (
            "Bullish Liquidity Sweep"
        )

        result["long"] = True
        result["level"] = recent_low

        return result

    # ========================================================
    # BREAKOUT CONTEXT
    # ========================================================

    if close > recent_high:

        result["status"] = (
            "High Liquidity Broken"
        )

        result["long"] = True
        result["level"] = recent_high

    elif close < recent_low:

        result["status"] = (
            "Low Liquidity Broken"
        )

        result["short"] = True
        result["level"] = recent_low

    return result


# ============================================================
# BREAKOUT HOLD / REJECTION
# ============================================================

def detect_breakout_rejection(df):

    latest = df.iloc[-2]
    previous = df.iloc[-3]

    result = {
        "status": (
            "No clear breakout / hold event"
        ),
        "short": False,
        "long_blocked": False,
        "level": np.nan,
        "failed_hold": False
    }

    latest_level = (
        latest["recent_high"]
    )

    previous_level = (
        previous["recent_high"]
    )

    if pd.isna(latest_level):
        return result

    level = float(
        latest_level
    )

    high = float(
        latest["high"]
    )

    low = float(
        latest["low"]
    )

    close = float(
        latest["close"]
    )

    body = float(
        latest["body"]
    )

    candle_range = float(
        latest["range"]
    )

    upper_wick = float(
        latest["upper_wick"]
    )

    if candle_range <= 0:
        return result

    # ========================================================
    # BULLISH BREAKOUT CONFIRMED
    # ========================================================

    if close > level:

        result["status"] = (
            "Bullish Breakout Confirmed"
        )

        result["level"] = level

        return result

    # ========================================================
    # HOLD / RETEST AFTER PREVIOUS BREAKOUT
    # ========================================================

    if not pd.isna(
        previous_level
    ):

        previous_level = float(
            previous_level
        )

        previous_close = float(
            previous["close"]
        )

        previous_broke_out = (
            previous_close
            >
            previous_level
        )

        if previous_broke_out:

            if close < previous_level:

                result["status"] = (
                    "Bearish Breakout Failure / Hold Lost"
                )

                result["short"] = True
                result["long_blocked"] = True
                result["level"] = (
                    previous_level
                )

                result["failed_hold"] = True

                return result

            if low <= previous_level:

                result["status"] = (
                    "Bullish Breakout Retest / HOLD Confirmed"
                )

            else:

                result["status"] = (
                    "Bullish Breakout HOLD Confirmed"
                )

            result["level"] = (
                previous_level
            )

            return result

    # ========================================================
    # REJECTION WITHOUT CONFIRMED PRIOR BREAKOUT
    # ========================================================

    rejection_close = (
        close < level
    )

    took_level = (
        high > level
    )

    wick_rejection = (
        upper_wick
        >=
        max(
            body * 1.2,
            candle_range * 0.20
        )
    )

    if (
        took_level
        and
        rejection_close
        and
        wick_rejection
    ):

        result["status"] = (
            "Bearish Resistance Rejection / Hold Failed"
        )

        result["short"] = True
        result["level"] = level

        if (
            close
            <
            level
            *
            (
                1
                -
                BREAKOUT_HOLD_TOLERANCE_PCT
                /
                100
            )
        ):

            result["long_blocked"] = True

    return result


# ============================================================
# PRE-BREAKOUT WATCH
# ============================================================

def detect_pre_breakout_watch(df):

    if len(df) < 25:

        return {
            "active": False,
            "status": "Insufficient data",
            "level": np.nan,
            "distance_pct": np.nan,
            "confluence": 0,
            "reasons": []
        }

    closed = df.iloc[-2]
    live = df.iloc[-1]

    live_price = float(
        live["close"]
    )

    local_resistance = float(
        df["high"].iloc[-9:-1].max()
    )

    broad_resistance = (
        closed["recent_high"]
    )

    resistance_candidates = [
        local_resistance
    ]

    if not pd.isna(
        broad_resistance
    ):

        resistance_candidates.append(
            float(broad_resistance)
        )

    above_price = [
        level
        for level
        in resistance_candidates
        if level > live_price
    ]

    if not above_price:

        return {
            "active": False,
            "status": (
                "No nearby resistance detected"
            ),
            "level": np.nan,
            "distance_pct": np.nan,
            "confluence": 0,
            "reasons": []
        }

    resistance = min(
        above_price
    )

    distance_pct = (
        (
            resistance
            -
            live_price
        )
        /
        resistance
    ) * 100

    near_resistance = (
        distance_pct >= 0
        and
        distance_pct
        <= PRE_BREAKOUT_DISTANCE_PCT
    )

    if (
        float(live["high"])
        >
        resistance
    ):

        near_resistance = False

    reasons = []
    confluence = 0

    ema_bullish = (
        float(closed["ema_20"])
        >
        float(closed["ema_50"])
        and
        float(closed["ema_50"])
        >
        float(closed["ema_200"])
    )

    if ema_bullish:

        confluence += 1

        reasons.append(
            "EMA 20 > 50 > 200"
        )

    if (
        float(closed["close"])
        >
        float(closed["ema_20"])
    ):

        confluence += 1

        reasons.append(
            "Last close above EMA20"
        )

    volume_ratio = float(
        closed["volume_ratio"]
    )

    if (
        not np.isnan(volume_ratio)
        and
        volume_ratio
        >= PRE_BREAKOUT_MIN_VOLUME_RATIO
    ):

        confluence += 1

        reasons.append(
            f"Volume {volume_ratio:.2f}x average"
        )

    if (
        float(closed["macd"])
        >
        float(closed["macd_signal"])
        and
        float(closed["macd_hist"])
        > 0
    ):

        confluence += 1

        reasons.append(
            "MACD bullish"
        )

    if (
        float(closed["delta"])
        > 0
        and
        float(closed["cvd_change"])
        > 0
    ):

        confluence += 1

        reasons.append(
            "Positive Delta + CVD"
        )

    if (
        float(closed["close"])
        >
        float(closed["open"])
    ):

        confluence += 1

        reasons.append(
            "Last candle bullish"
        )

    active = (
        near_resistance
        and
        confluence
        >= PRE_BREAKOUT_MIN_CONFLUENCE
    )

    if active:

        status = (
            "PRE-BREAKOUT WATCH — resistance approaching"
        )

    elif near_resistance:

        status = (
            "Near resistance — weak confluence"
        )

    else:

        status = (
            "No pre-breakout setup"
        )

    return {
        "active": active,
        "status": status,
        "level": resistance,
        "distance_pct": distance_pct,
        "confluence": confluence,
        "reasons": reasons
    }


def build_pre_breakout_message(
    watch,
    generated_at
):

    level = watch["level"]
    distance = watch["distance_pct"]
    reasons = watch["reasons"]

    return (
        f"🟡 *PRE-BREAKOUT WATCH*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 *Pair:* `{SYMBOL}`\n"
        f"⏱ *Interval:* `{INTERVAL}`\n"
        f"⚡ *Detected:* "
        f"`{generated_at.strftime('%H:%M:%S UTC')}`\n"
        f"🎯 *Resistance:* "
        f"`${level:,.2f}`\n"
        f"📏 *Distance:* "
        f"`{distance:.2f}%`\n"
        f"🔥 *Bullish Confluence:* "
        f"`{watch['confluence']}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 *Why it is being watched:*\n"
        +
        "\n".join(
            f"✅ {r}"
            for r in reasons
        )
        +
        "\n━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ *WATCH ONLY — NOT A LONG SIGNAL*\n"
        f"Wait for breakout close above "
        f"`${level:,.2f}` and hold/retest confirmation."
    )


# ============================================================
# ABSORPTION / TRAP
# ============================================================

def detect_absorption(df):

    latest = df.iloc[-2]

    body = float(
        latest["body"]
    )

    candle_range = float(
        latest["range"]
    )

    volume_ratio = float(
        latest["volume_ratio"]
    )

    delta_pct = float(
        latest["delta_pct"]
    )

    upper_wick = float(
        latest["upper_wick"]
    )

    lower_wick = float(
        latest["lower_wick"]
    )

    result = {
        "status": "Neutral",
        "long": False,
        "short": False
    }

    if candle_range <= 0:
        return result

    body_ratio = (
        body / candle_range
    )

    # ========================================================
    # BULLISH ABSORPTION
    # ========================================================

    if (
        delta_pct < -5
        and
        volume_ratio >= 1.3
        and
        lower_wick >= body * 1.2
        and
        body_ratio < 0.60
    ):

        result["status"] = (
            "Bullish Absorption"
        )

        result["long"] = True

    # ========================================================
    # BEARISH ABSORPTION
    # ========================================================

    elif (
        delta_pct > 5
        and
        volume_ratio >= 1.3
        and
        upper_wick >= body * 1.2
        and
        body_ratio < 0.60
    ):

        result["status"] = (
            "Bearish Absorption"
        )

        result["short"] = True

    return result


# ============================================================
# SCORING
# ============================================================

def evaluate_scoring(
    df,
    df_oi
):

    latest = df.iloc[-2]
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

    cvd_change = (
        float(latest["cvd_change"])
        if not pd.isna(
            latest["cvd_change"]
        )
        else 0
    )

    macd = float(
        latest["macd"]
    )

    macd_signal = float(
        latest["macd_signal"]
    )

    macd_hist = float(
        latest["macd_hist"]
    )

    rsi_14 = float(
        latest["rsi_14"]
    )

    rsi_24 = float(
        latest["rsi_24"]
    )

    long_score = 0
    short_score = 0

    long_reasons = []
    short_reasons = []

    # ========================================================
    # FACTOR 1 — EMA STRUCTURE / PRESSURE
    # ========================================================

    # --------------------------------------------------------
    # Bullish remains strict.
    # --------------------------------------------------------

    bullish_ema = (
        ema_20 > ema_50
        and
        ema_50 > ema_200
    )

    # --------------------------------------------------------
    # NEW:
    # Bearish pressure is recognized earlier.
    #
    # We do NOT require EMA50 < EMA200.
    #
    # Price below EMA20 + EMA20 below EMA50
    # is enough to establish active bearish EMA pressure.
    # --------------------------------------------------------

    bearish_ema_pressure = (
        close < ema_20
        and
        ema_20 < ema_50
    )

    bearish_ema_full = (
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

    elif bearish_ema_full:

        short_score += 1

        short_reasons.append(
            "✅ EMA structure bearish "
            "(20 < 50 < 200)"
        )

    elif bearish_ema_pressure:

        short_score += 1

        short_reasons.append(
            "✅ Bearish EMA pressure "
            "(Price < EMA20 < EMA50)"
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

        # ====================================================
        # IMPORTANT FIX
        #
        # Price falling + OI falling means long positions
        # are being closed/liquidated.
        #
        # This is NOT fresh short buildup, but it is genuine
        # bearish pressure and should contribute to a SHORT
        # setup when other bearish evidence agrees.
        # ====================================================

        short_score += 1

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

    liquidity = detect_liquidity(
        df
    )

    liquidity_status = (
        liquidity["status"]
    )

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

    absorption = detect_absorption(
        df
    )

    absorption_status = (
        absorption["status"]
    )

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
    # FACTOR 7 — MACD
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
    # FACTOR 8 — PRICE ACTION + RSI
    # ========================================================

    candle_open = float(
        latest["open"]
    )

    breakout = detect_breakout_rejection(
        df
    )

    breakout_status = (
        breakout["status"]
    )

    breakout_level = (
        breakout["level"]
    )

    long_blocked = (
        breakout["long_blocked"]
    )

    # --------------------------------------------------------
    # Bullish-zone rejection
    # --------------------------------------------------------

    rejection_level = (
        breakout_level
    )

    if pd.isna(
        rejection_level
    ):

        rejection_level = (
            latest["recent_high"]
        )

    rejection = RejectionResult(
        False,
        0,
        []
    )

    if not pd.isna(
        rejection_level
    ):

        rejection = (
            detect_bullish_zone_rejection(
                candle={
                    "open": latest["open"],
                    "high": latest["high"],
                    "low": latest["low"],
                    "close": latest["close"],
                },
                ema20=ema_20,
                ema50=ema_50,
                ema200=ema_200,
                resistance=float(
                    rejection_level
                ),
                volume=volume,
                avg_volume=volume_avg,
                close_below_level=True,
            )
        )

        if rejection.rejected:

            long_blocked = True

    # ========================================================
    # RSI STATES
    # ========================================================

    rsi_bearish = (
        rsi_14 < RSI_BEARISH_LEVEL
        and
        rsi_24 < 50
    )

    rsi_strong_bearish = (
        rsi_14 < RSI_STRONG_BEARISH_LEVEL
        and
        rsi_24 < RSI_BEARISH_LEVEL
    )

    rsi_bullish = (
        rsi_14 > RSI_BULLISH_LEVEL
        and
        rsi_24 > 50
    )

    # --------------------------------------------------------
    # Bullish price confirmation
    # --------------------------------------------------------

    price_bullish = (
        close > candle_open
        and
        close > ema_20
        and
        rsi_bullish
        and
        breakout_status in (
            "Bullish Breakout Confirmed",
            "Bullish Breakout HOLD Confirmed",
            "Bullish Breakout Retest / HOLD Confirmed"
        )
    )

    # --------------------------------------------------------
    # Bearish price confirmation
    #
    # IMPORTANT FIX:
    #
    # Price below EMA20 + bearish RSI now participates in the
    # price-action factor.
    # --------------------------------------------------------

    price_bearish = (
        close < candle_open
        and
        close < ema_20
        and
        rsi_bearish
    )

    # --------------------------------------------------------
    # Breakout failure / resistance rejection
    # --------------------------------------------------------

    if breakout["short"]:

        short_score += 1

        if pd.isna(
            breakout_level
        ):

            level_text = (
                "dynamic resistance"
            )

        else:

            level_text = (
                f"${float(breakout_level):,.2f}"
            )

        short_reasons.append(
            f"⚠️ {breakout_status} "
            f"at {level_text}"
        )

        long_reasons.append(
            f"🚫 Long blocked: "
            f"resistance not held "
            f"({level_text})"
        )

    elif price_bearish:

        short_score += 1

        if rsi_strong_bearish:

            short_reasons.append(
                f"✅ Bearish price + RSI "
                f"(Price < EMA20, "
                f"RSI14 {rsi_14:.1f}, "
                f"RSI24 {rsi_24:.1f})"
            )

        else:

            short_reasons.append(
                f"✅ Bearish price + RSI "
                f"(Price < EMA20, "
                f"RSI14 {rsi_14:.1f}, "
                f"RSI24 {rsi_24:.1f})"
            )

    elif price_bullish:

        long_score += 1

        long_reasons.append(
            "✅ Bullish price + RSI "
            "(close > EMA20)"
        )

    else:

        long_reasons.append(
            "⚪ Price/RSI confirmation mixed"
        )

        short_reasons.append(
            "⚪ Price/RSI confirmation mixed"
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

        "rsi_14":
            rsi_14,

        "rsi_24":
            rsi_24,

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

        "bullish_rejection":
            rejection.rejected,

        "bullish_rejection_score":
            rejection.score,

        "bullish_rejection_reasons":
            rejection.reasons,

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

            tag = (
                "🟢 STRONG LONG WATCH"
            )

        else:

            tag = (
                "🟡 LONG WATCH"
            )

    else:

        score = signal["short_score"]
        reasons = signal["short_reasons"]

        if score >= 7:

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

    engine_delay = (
        generated_at
        -
        signal["candle_close_time"]
    ).total_seconds()

    if engine_delay < 0:
        engine_delay = 0

    message = (

        f"{tag} "
        f"*(Score: {score}/{MAX_SCORE})*\n"

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

        f"📊 *RSI 14:* "
        f"`{signal['rsi_14']:.2f}`\n"

        f"📊 *RSI 24:* "
        f"`{signal['rsi_24']:.2f}`\n"

        f"💧 *Liquidity:* "
        f"`{signal['liquidity_status']}`\n"

        f"🧲 *Absorption:* "
        f"`{signal['absorption_status']}`\n"

        f"🚧 *Breakout / Hold:* "
        f"`{signal['breakout_status']}`\n"

        f"🔴 *Bullish Rejection:* "
        f"`{signal['bullish_rejection']} "
        f"(score "
        f"{signal['bullish_rejection_score']})`\n"

        f"━━━━━━━━━━━━━━━━━━━━\n"

        f"📋 *Phase 2 Confluences:*\n"

        +
        "\n".join(
            reasons
        )

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
        " Liquidity + Absorption + MACD +"
        " Price Action/RSI"
    )

    print(
        "🟡 Pre-breakout live watch enabled: "
        f"within {PRE_BREAKOUT_DISTANCE_PCT:.2f}% "
        f"of resistance"
    )

    print(
        "🔴 Bearish SHORT detection enhanced:"
        " EMA pressure + OI unwinding + RSI"
    )

    last_processed_time = None

    last_alert_direction = None
    last_alert_score = None

    last_pre_breakout_key = None
    last_pre_breakout_sent_at = None

    last_market_scan_at = None

    while True:

        try:

            # =================================================
            # FETCH CANDLES
            # =================================================

            df = fetch_klines()

            # =================================================
            # MARKET SCANNER
            # =================================================

            if MARKET_SCANNER_ENABLED:

                now = utc_now()

                scanner_due = (
                    last_market_scan_at is None
                    or
                    (
                        now
                        -
                        last_market_scan_at
                    ).total_seconds()
                    >=
                    MARKET_SCANNER_INTERVAL_SECONDS
                )

                if scanner_due:

                    try:

                        candidates = (
                            scan_market()
                        )

                        print(
                            format_scanner_log(
                                candidates
                            )
                        )

                        last_market_scan_at = (
                            now
                        )

                    except Exception as scanner_error:

                        print(
                            "⚠️ Futures scanner error: "
                            f"{scanner_error}"
                        )

                        last_market_scan_at = (
                            now
                        )

            # =================================================
            # LIVE PRE-BREAKOUT WATCH
            # =================================================

            try:

                live_df = (
                    calculate_indicators(
                        df
                    )
                )

                pre_breakout = (
                    detect_pre_breakout_watch(
                        live_df
                    )
                )

                if pre_breakout["active"]:

                    current_live_candle = (
                        live_df
                        .iloc[-1]["open_time"]
                    )

                    watch_key = (
                        current_live_candle,
                        round(
                            float(
                                pre_breakout["level"]
                            ),
                            4
                        )
                    )

                    now = utc_now()

                    cooldown_ok = (
                        last_pre_breakout_sent_at
                        is None
                        or
                        (
                            now
                            -
                            last_pre_breakout_sent_at
                        ).total_seconds()
                        >=
                        PRE_BREAKOUT_ALERT_COOLDOWN_SECONDS
                    )

                    if (
                        watch_key
                        !=
                        last_pre_breakout_key
                        and
                        cooldown_ok
                    ):

                        watch_message = (
                            build_pre_breakout_message(
                                pre_breakout,
                                now
                            )
                        )

                        if send_telegram_alert(
                            watch_message
                        ):

                            last_pre_breakout_key = (
                                watch_key
                            )

                            last_pre_breakout_sent_at = (
                                now
                            )

                            print(
                                "🟡 PRE-BREAKOUT WATCH "
                                "alert sent. "
                                f"Resistance: "
                                f"${pre_breakout['level']:,.2f}"
                            )

            except Exception as pre_error:

                print(
                    "⚠️ Pre-breakout watch error: "
                    f"{pre_error}"
                )

            # =================================================
            # CLOSED CANDLE
            # =================================================

            latest_closed_time = (
                df.iloc[-2]["open_time"]
            )

            # =================================================
            # PROCESS ONLY ONCE PER CLOSED CANDLE
            # =================================================

            if (
                latest_closed_time
                !=
                last_processed_time
            ):

                detection_at = (
                    utc_now()
                )

                last_processed_time = (
                    latest_closed_time
                )

                # =================================================
                # INDICATORS
                # =================================================

                df = calculate_indicators(
                    df
                )

                # =================================================
                # OI
                # =================================================

                df_oi = (
                    fetch_open_interest()
                )

                # =================================================
                # SCORING
                # =================================================

                signal = (
                    evaluate_scoring(
                        df,
                        df_oi
                    )
                )

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
                # TIMING
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

                # =================================================
                # OI LOG
                # =================================================

                if np.isnan(
                    signal["oi_change_pct"]
                ):

                    oi_log = "N/A"

                else:

                    oi_log = (
                        f"{signal['oi_change_pct']:+.3f}%"
                    )

                # =================================================
                # NORMAL LOG
                # =================================================

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

                    f"RSI14: "
                    f"{signal['rsi_14']:.2f}\n"

                    f"RSI24: "
                    f"{signal['rsi_24']:.2f}\n"

                    f"Liquidity: "
                    f"{signal['liquidity_status']}\n"

                    f"Absorption: "
                    f"{signal['absorption_status']}\n"

                    f"Breakout / Hold: "
                    f"{signal['breakout_status']}\n"

                    f"Long Blocked: "
                    f"{signal['long_blocked']}\n"

                    f"Bullish Rejection: "
                    f"{signal['bullish_rejection']} "
                    f"(score "
                    f"{signal['bullish_rejection_score']})\n"

                    f"Long Score: "
                    f"{long_score}/{MAX_SCORE}\n"

                    f"Short Score: "
                    f"{short_score}/{MAX_SCORE}"
                )

                # =================================================
                # LONG
                # =================================================

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
                                "🟢 PHASE 2 LONG alert sent."
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
                                "🔴 PHASE 2 SHORT alert sent."
                            )

                            print(
                                "Telegram send delay: "
                                f"{telegram_delay:.2f} sec"
                            )

                # =================================================
                # NO HIGH CONFLUENCE
                # =================================================

                else:

                    if signal[
                        "bullish_rejection"
                    ]:

                        print(
                            "🔴 BULLISH ZONE REJECTION: "
                            "LONG INVALIDATED. "
                            f"Score: "
                            f"{signal['bullish_rejection_score']}/7 | "
                            f"Reasons: "
                            f"{', '.join(signal['bullish_rejection_reasons'])}"
                        )

                    elif signal[
                        "long_blocked"
                    ]:

                        print(
                            "🚫 LONG BLOCKED: "
                            "breakout resistance was not held. "
                            f"Reason: "
                            f"{signal['breakout_status']}"
                        )

                    else:

                        print(
                            "⚪ No Phase 2 "
                            "high-confluence setup."
                        )

                    last_alert_direction = None
                    last_alert_score = None

            # =================================================
            # POLL
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
