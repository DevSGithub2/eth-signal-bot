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

# Breakout / rejection detection
# A breakout must hold above the detected resistance.
# If price breaks above resistance but the CLOSED candle returns
# below that level, the bot treats it as a failed hold / rejection.
BREAKOUT_HOLD_TOLERANCE_PCT = 0.15

# ------------------------------------------------------------
# PRE-BREAKOUT WATCH
# ------------------------------------------------------------
# The normal engine waits for a CLOSED candle for confirmation.
# This live watch warns when the CURRENT candle is approaching
# resistance with bullish confluence, so a fast breakout is not
# first reported only after the move has already happened.
PRE_BREAKOUT_DISTANCE_PCT = 0.20
PRE_BREAKOUT_MIN_CONFLUENCE = 4
PRE_BREAKOUT_MIN_VOLUME_RATIO = 0.80
PRE_BREAKOUT_ALERT_COOLDOWN_SECONDS = 60

# Phase 2 = 8 factor system
MIN_SCORE = 6
MAX_SCORE = 8

# ============================================================
# BULLISH ZONE REJECTION DETECTOR
# ============================================================
# This is a LONG invalidation detector. It does NOT create a
# SHORT signal by itself. A separate bearish setup must qualify
# before the normal SHORT engine can alert.
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
    """Detect rejection of a bullish/resistance zone.

    IMPORTANT: this function only invalidates LONG. It does not
    create a SHORT signal.
    """
    o = float(candle["open"])
    h = float(candle["high"])
    l = float(candle["low"])
    c = float(candle["close"])

    candle_range = max(h - l, 1e-9)
    upper_wick = h - max(o, c)
    upper_wick_ratio = upper_wick / candle_range

    score = 0
    reasons = []

    touched_zone = h >= float(resistance)
    if touched_zone:
        score += 1
        reasons.append("Bullish zone/resistance tested")

    if touched_zone and upper_wick_ratio >= BULLISH_REJECTION_WICK_RATIO:
        score += 1
        reasons.append("Large upper wick rejection")

    if touched_zone and close_below_level and c < float(resistance):
        score += 2
        reasons.append("Close back below bullish zone")

    if touched_zone and c < o:
        score += 1
        reasons.append("Bearish rejection candle")

    if avg_volume > 0 and volume >= avg_volume * BULLISH_REJECTION_VOLUME_MULTIPLIER:
        score += 1
        reasons.append("High rejection volume")

    if ema20 > ema50 > ema200:
        score += 1
        reasons.append("Bullish EMA structure")

    return RejectionResult(
        rejected=score >= BULLISH_REJECTION_MIN_SCORE,
        score=score,
        reasons=reasons,
    )


# ============================================================
# BINANCE FUTURES MARKET SCANNER
# ============================================================
MARKET_SCANNER_ENABLED = os.getenv("MARKET_SCANNER_ENABLED", "true").lower() == "true"
MARKET_SCANNER_MIN_VOLUME_USDT = float(os.getenv("MARKET_SCANNER_MIN_VOLUME_USDT", "20000000"))
MARKET_SCANNER_TOP_N = int(os.getenv("MARKET_SCANNER_TOP_N", "10"))
MARKET_SCANNER_INTERVAL_SECONDS = int(os.getenv("MARKET_SCANNER_INTERVAL_SECONDS", "60"))
BINANCE_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_24H_TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"

def get_futures_symbols():
    response = requests.get(BINANCE_EXCHANGE_INFO_URL, timeout=10)
    response.raise_for_status()
    data = response.json()
    return {
        s["symbol"] for s in data.get("symbols", [])
        if s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("status") == "TRADING"
    }

def get_24h_tickers():
    response = requests.get(BINANCE_24H_TICKER_URL, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []

def scan_market(min_volume_usdt=MARKET_SCANNER_MIN_VOLUME_USDT, top_n=MARKET_SCANNER_TOP_N):
    symbols = get_futures_symbols()
    tickers = get_24h_tickers()
    candidates = []
    for t in tickers:
        symbol = t.get("symbol")
        if symbol not in symbols:
            continue
        try:
            price = float(t["lastPrice"])
            change = float(t["priceChangePercent"])
            volume = float(t["quoteVolume"])
        except (KeyError, TypeError, ValueError):
            continue
        if volume < min_volume_usdt:
            continue
        candidates.append({
            "symbol": symbol,
            "price": price,
            "change_24h": change,
            "volume_24h": volume,
        })
    candidates.sort(key=lambda x: x["volume_24h"], reverse=True)
    return candidates[:top_n]

def format_scanner_log(candidates):
    if not candidates:
        return "No qualifying Futures candidates."
    lines = ["\n===== BINANCE FUTURES MARKET SCANNER ====="]
    for i, coin in enumerate(candidates, 1):
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
# PRE-BREAKOUT WATCH — LIVE FORMING CANDLE
# ============================================================

def detect_pre_breakout_watch(df):
    """
    Early warning only — NOT a confirmed LONG signal.

    The normal engine evaluates the last CLOSED candle. This
    function additionally looks at the CURRENT forming candle and
    warns when price is close to the resistance defined by the last
    closed candle's recent_high.

    Confirmation still requires a closed candle above resistance
    and/or a valid hold/retest. The watch never authorizes a trade.
    """
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

    live_price = float(live["close"])

    # Use BOTH a short local resistance and the broader 20-candle
    # resistance. For fast moves, the immediate local level is often
    # the one that matters first (for example 2431/2432), while the
    # 20-candle high can be much farther away.
    local_resistance = float(df["high"].iloc[-9:-1].max())
    broad_resistance = closed["recent_high"]

    resistance_candidates = [local_resistance]
    if not pd.isna(broad_resistance):
        resistance_candidates.append(float(broad_resistance))

    # Pick the nearest resistance ABOVE the current live price.
    above_price = [
        level for level in resistance_candidates
        if level > live_price
    ]

    if not above_price:
        return {
            "active": False,
            "status": "No nearby resistance detected",
            "level": np.nan,
            "distance_pct": np.nan,
            "confluence": 0,
            "reasons": []
        }

    resistance = min(above_price)
    distance_pct = ((resistance - live_price) / resistance) * 100

    near_resistance = (
        distance_pct >= 0
        and
        distance_pct <= PRE_BREAKOUT_DISTANCE_PCT
    )

    # If the CURRENT candle has already traded through this level,
    # it is no longer a pre-breakout state.
    if float(live["high"]) > resistance:
        near_resistance = False

    reasons = []
    confluence = 0

    ema_bullish = (
        float(closed["ema_20"]) > float(closed["ema_50"])
        and
        float(closed["ema_50"]) > float(closed["ema_200"])
    )

    if ema_bullish:
        confluence += 1
        reasons.append("EMA 20 > 50 > 200")

    if float(closed["close"]) > float(closed["ema_20"]):
        confluence += 1
        reasons.append("Last close above EMA20")

    volume_ratio = float(closed["volume_ratio"])
    if not np.isnan(volume_ratio) and volume_ratio >= PRE_BREAKOUT_MIN_VOLUME_RATIO:
        confluence += 1
        reasons.append(f"Volume {volume_ratio:.2f}x average")

    if (
        float(closed["macd"]) > float(closed["macd_signal"])
        and
        float(closed["macd_hist"]) > 0
    ):
        confluence += 1
        reasons.append("MACD bullish")

    if (
        float(closed["delta"]) > 0
        and
        float(closed["cvd_change"]) > 0
    ):
        confluence += 1
        reasons.append("Positive Delta + CVD")

    if float(closed["close"]) > float(closed["open"]):
        confluence += 1
        reasons.append("Last candle bullish")

    active = (
        near_resistance
        and
        confluence >= PRE_BREAKOUT_MIN_CONFLUENCE
    )

    if active:
        status = "PRE-BREAKOUT WATCH — resistance approaching"
    elif near_resistance:
        status = "Near resistance — weak confluence"
    else:
        status = "No pre-breakout setup"

    return {
        "active": active,
        "status": status,
        "level": resistance,
        "distance_pct": distance_pct,
        "confluence": confluence,
        "reasons": reasons
    }


def build_pre_breakout_message(watch, generated_at):
    level = watch["level"]
    distance = watch["distance_pct"]
    reasons = watch["reasons"]

    return (
        f"🟡 *PRE-BREAKOUT WATCH*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 *Pair:* `{SYMBOL}`\n"
        f"⏱ *Interval:* `{INTERVAL}`\n"
        f"⚡ *Detected:* `{generated_at.strftime('%H:%M:%S UTC')}`\n"
        f"🎯 *Resistance:* `${level:,.2f}`\n"
        f"📏 *Distance:* `{distance:.2f}%`\n"
        f"🔥 *Bullish Confluence:* `{watch['confluence']}`\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📋 *Why it is being watched:*\n"
        + "\n".join(f"✅ {r}" for r in reasons)
        + "\n━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ *WATCH ONLY — NOT A LONG SIGNAL*\n"
        f"Wait for breakout close above `${level:,.2f}` and hold/retest confirmation."
    )


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

    # New bullish-zone rejection detector. Use the latest closed
    # candle and the same dynamic resistance used by the price-action
    # engine. This ONLY blocks LONG; it does not add a SHORT point.
    rejection_level = breakout_level
    if pd.isna(rejection_level):
        rejection_level = latest["recent_high"]

    rejection = RejectionResult(False, 0, [])
    if not pd.isna(rejection_level):
        rejection = detect_bullish_zone_rejection(
            candle={
                "open": latest["open"],
                "high": latest["high"],
                "low": latest["low"],
                "close": latest["close"],
            },
            ema20=ema_20,
            ema50=ema_50,
            ema200=ema_200,
            resistance=float(rejection_level),
            volume=volume,
            avg_volume=volume_avg,
            close_below_level=True,
        )
        if rejection.rejected:
            long_blocked = True

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
        f"🔴 *Bullish Rejection:* "
        f"`{signal['bullish_rejection']} "
        f"(score {signal['bullish_rejection_score']})`\n"

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
    print(
        "🟡 Pre-breakout live watch enabled: "
        f"within {PRE_BREAKOUT_DISTANCE_PCT:.2f}% of resistance"
    )

    last_processed_time = None

    last_alert_direction = None
    last_alert_score = None

    # Pre-breakout watch state. Separate from confirmed signals.
    last_pre_breakout_key = None
    last_pre_breakout_sent_at = None

    # Market scanner state. Scanner is discovery-only for now; it does
    # not replace the ETH scoring engine or auto-trade selected coins.
    last_market_scan_at = None

    while True:

        try:

            # ------------------------------------------------
            # FETCH CANDLES
            # ------------------------------------------------

            df = fetch_klines()

            # ------------------------------------------------
            # BINANCE FUTURES MARKET SCANNER
            # ------------------------------------------------
            if MARKET_SCANNER_ENABLED:
                now = utc_now()
                scanner_due = (
                    last_market_scan_at is None
                    or (now - last_market_scan_at).total_seconds() >= MARKET_SCANNER_INTERVAL_SECONDS
                )
                if scanner_due:
                    try:
                        candidates = scan_market()
                        print(format_scanner_log(candidates))
                        last_market_scan_at = now
                    except Exception as scanner_error:
                        print(f"⚠️ Futures scanner error: {scanner_error}")
                        last_market_scan_at = now

            # ------------------------------------------------
            # LIVE PRE-BREAKOUT WATCH
            # ------------------------------------------------
            # Runs on the CURRENT forming candle. It does not replace
            # the confirmed closed-candle engine below.
            try:
                live_df = calculate_indicators(df)
                pre_breakout = detect_pre_breakout_watch(live_df)

                if pre_breakout["active"]:
                    current_live_candle = live_df.iloc[-1]["open_time"]
                    watch_key = (
                        current_live_candle,
                        round(float(pre_breakout["level"]), 4)
                    )
                    now = utc_now()

                    cooldown_ok = (
                        last_pre_breakout_sent_at is None
                        or
                        (now - last_pre_breakout_sent_at).total_seconds()
                        >= PRE_BREAKOUT_ALERT_COOLDOWN_SECONDS
                    )

                    if (
                        watch_key != last_pre_breakout_key
                        and
                        cooldown_ok
                    ):
                        watch_message = build_pre_breakout_message(
                            pre_breakout,
                            now
                        )

                        if send_telegram_alert(watch_message):
                            last_pre_breakout_key = watch_key
                            last_pre_breakout_sent_at = now
                            print(
                                "🟡 PRE-BREAKOUT WATCH alert sent. "
                                f"Resistance: ${pre_breakout['level']:,.2f}"
                            )

            except Exception as pre_error:
                print(
                    f"⚠️ Pre-breakout watch error: {pre_error}"
                )

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

                    f"Bullish Rejection: "
                    f"{signal['bullish_rejection']} "
                    f"(score {signal['bullish_rejection_score']})\n"

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

                    if signal["bullish_rejection"]:
                        print(
                            "🔴 BULLISH ZONE REJECTION: LONG INVALIDATED. "
                            f"Score: {signal['bullish_rejection_score']}/7 | "
                            f"Reasons: {', '.join(signal['bullish_rejection_reasons'])}"
                        )
                    elif signal["long_blocked"]:
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
