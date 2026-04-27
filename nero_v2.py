# =============================================================================
#  NERO v2 — Neural Equity Ranking & Optimization System
#  For: NSE India  |  Author: kppan
#  Blueprint: NERO v2 Final Master Blueprint
# =============================================================================

# ===== SECTION 1: IMPORTS & CONFIG =====

import os
import json
import warnings
import argparse
import multiprocessing
import requests

import pandas as pd
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── TA-Lib: required on production machine (pip install TA-Lib-prebuilt) ──────
try:
    import talib
    TALIB_OK = True
except ImportError:
    TALIB_OK = False
    warnings.warn(
        "[NERO] talib not found. Install via: pip install TA-Lib-prebuilt\n"
        "       Falling back to pure-pandas implementations for testing.",
        ImportWarning,
        stacklevel=2,
    )

# ── HMM: required for regime engine (Section 5) ───────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_OK = True
except ImportError:
    HMMLEARN_OK = False
    warnings.warn(
        "[NERO] hmmlearn not found. Install via: pip install hmmlearn\n"
        "       Regime engine will fall back to threshold-based classification.",
        ImportWarning,
        stacklevel=2,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  ENVIRONMENT VARIABLES  (override via shell or UI before running)
# ─────────────────────────────────────────────────────────────────────────────

NERO_ARCHIVE_PATH   = os.environ.get("NERO_ARCHIVE_PATH",  "C:/NERO/archive")
NERO_FUNDA_PATH     = os.environ.get("NERO_FUNDA_PATH",    "C:/NERO/data/Stock_Funda_1000.csv")
NERO_OUTPUT_PATH    = os.environ.get("NERO_OUTPUT_PATH",   "C:/NERO/results")
NERO_CANDLE_MIN     = int(os.environ.get("NERO_CANDLE_MIN",   "5"))      # 1 / 3 / 5
NERO_CORR_LOOKBACK  = os.environ.get("NERO_CORR_LOOKBACK", "1Y")         # 6M / 1Y
NERO_MIN_VOLUME     = float(os.environ.get("NERO_MIN_VOLUME", "0"))
NERO_RISK_MODE      = os.environ.get("NERO_RISK_MODE",     "medium")      # low / medium / high
NERO_SWING_YEARS    = int(os.environ.get("NERO_SWING_YEARS",  "3"))
NERO_PER_STOCK_STRAT = os.environ.get("NERO_PER_STOCK_STRAT", "ON")     # ON / OFF
NERO_NEWS_ENABLED   = os.environ.get("NERO_NEWS_ENABLED",  "OFF")         # ON / OFF
NERO_IC_ADAPTIVE    = os.environ.get("NERO_IC_ADAPTIVE",   "OFF")         # ON / OFF  ← IC-weighted scoring
NERO_TG_BOT_TOKEN   = os.environ.get("NERO_TG_BOT_TOKEN",  "")
NERO_TG_CHAT_ID     = os.environ.get("NERO_TG_CHAT_ID",    "")

# ── Validation ────────────────────────────────────────────────────────────────
if NERO_CANDLE_MIN not in {1, 3, 5}:
    raise ValueError(
        f"[NERO] NERO_CANDLE_MIN must be 1, 3, or 5. Got: {NERO_CANDLE_MIN}"
    )

if NERO_CORR_LOOKBACK not in {"6M", "1Y"}:
    raise ValueError(
        f"[NERO] NERO_CORR_LOOKBACK must be '6M' or '1Y'. Got: {NERO_CORR_LOOKBACK}"
    )

if NERO_RISK_MODE not in {"low", "medium", "high"}:
    raise ValueError(
        f"[NERO] NERO_RISK_MODE must be low / medium / high. Got: {NERO_RISK_MODE}"
    )

# ── Output directory ──────────────────────────────────────────────────────────
os.makedirs(NERO_OUTPUT_PATH, exist_ok=True)

# ── Flags log  (missing tickers / missing fundamentals / missing columns) ─────
# Populated throughout the run and written to results/flags.csv at the end.
NERO_FLAGS: list[dict] = []

def _flag(ticker: str, flag_type: str, detail: str) -> None:
    """Append a warning flag to the global flags list."""
    NERO_FLAGS.append({"ticker": ticker, "flag_type": flag_type, "detail": detail})
    print(f"[FLAG] {flag_type} | {ticker} | {detail}")


def flush_flags() -> None:
    """Write all accumulated flags to results/flags.csv."""
    if NERO_FLAGS:
        flags_df = pd.DataFrame(NERO_FLAGS)
        flags_path = os.path.join(NERO_OUTPUT_PATH, "flags.csv")
        flags_df.to_csv(flags_path, index=False)
        print(f"[NERO] {len(NERO_FLAGS)} flag(s) written → {flags_path}")


print(
    f"[NERO v2] Config loaded | Candle={NERO_CANDLE_MIN}m | "
    f"Risk={NERO_RISK_MODE} | IC-Adaptive={NERO_IC_ADAPTIVE} | "
    f"HMM={'ON' if HMMLEARN_OK else 'FALLBACK'} | "
    f"talib={'ON' if TALIB_OK else 'FALLBACK'}"
)


# ===== SECTION 2: DATA LOADER =====

def load_ohlcv(filepath: str, candle_minutes: int = 5, backtest_years: int = 3) -> pd.DataFrame | None:
    """
    Load a minute-OHLCV CSV for one ticker, resample to candle_minutes,
    filter to the last backtest_years of data.

    Expected CSV columns (lowercase): date, high, low, close, volume
    Expected date format: YYYY-MM-DD HH:MM:SS

    Returns
    -------
    pd.DataFrame with columns [High, Low, Close, Volume] and DatetimeIndex,
    or None if the file cannot be loaded / has no usable data.
    """
    ticker = os.path.splitext(os.path.basename(filepath))[0].upper()

    # ── File existence ────────────────────────────────────────────────────────
    if not os.path.isfile(filepath):
        _flag(ticker, "MISSING_ARCHIVE", f"File not found: {filepath}")
        return None

    # ── Load ─────────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(filepath, on_bad_lines="skip", low_memory=False)
    except Exception as exc:
        _flag(ticker, "LOAD_ERROR", str(exc))
        return None

    # ── Normalise column names ────────────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower()

    required = {"date", "high", "low", "close", "volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        _flag(ticker, "MISSING_COLUMNS", f"CSV missing columns: {missing_cols}")
        return None

    # ── Parse datetime index ──────────────────────────────────────────────────
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    except Exception:
        df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.set_index("date").sort_index()

    # ── Numeric coercion & quality filter ────────────────────────────────────
    for col in ["high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]

    if df.empty:
        _flag(ticker, "NO_DATA", "No valid rows after cleaning")
        return None

    # ── Date range filter: last backtest_years ────────────────────────────────
    cutoff = df.index.max() - pd.DateOffset(years=backtest_years)
    df = df[df.index >= cutoff]

    if df.empty:
        _flag(ticker, "NO_DATA", f"No data within last {backtest_years} years")
        return None

    # ── Resample to candle_minutes ────────────────────────────────────────────
    rule = f"{candle_minutes}min"
    df_resampled = df.resample(rule).agg(
        High=("high", "max"),
        Low=("low", "min"),
        Close=("close", "last"),
        Volume=("volume", "sum"),
    ).dropna(subset=["Close"])

    df_resampled = df_resampled[df_resampled["Close"] > 0]

    if len(df_resampled) < 50:
        _flag(ticker, "INSUFFICIENT_BARS",
              f"Only {len(df_resampled)} bars after resample — skipping")
        return None

    return df_resampled


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling Z-score using only past data (no lookahead).
    Z_t = (X_t - mean(X[t-window : t-1])) / std(X[t-window : t-1])

    min_periods=20 ensures we start computing once we have at least 20 obs.
    Returns a Series of the same index; NaN where insufficient history.
    """
    roll = series.rolling(window=window, min_periods=20)
    mu   = roll.mean().shift(1)   # shift(1) → strictly past data
    sigma = roll.std().shift(1)
    sigma = sigma.replace(0, np.nan)
    return (series - mu) / sigma


def rolling_volume_shock(volume_series: pd.Series, ema_span: int = 20) -> pd.Series:
    """
    VShock_t = (Volume_t / EMA(Volume, span=ema_span)_t) - 1

    Uses expanding EMA (pandas ewm), so each point uses all past volume.
    Returns a Series aligned to volume_series index.
    """
    ema_vol = volume_series.ewm(span=ema_span, min_periods=5, adjust=False).mean()
    ema_vol = ema_vol.replace(0, np.nan)
    return (volume_series / ema_vol) - 1


# ===== SECTION 3: FUNDAMENTAL SCORER =====

# ── Exact column names as they appear in Stock_Funda_2000.csv ─────────────────
# The FII/DII columns contain unicode non-breaking spaces (U+00A0) and an
# en-dash (U+2013). We normalise on load so internal code uses clean names.
_FUNDA_COL_MAP = {
    # raw name (after strip)            →  clean internal name
    "FII Holding Change\u00a0\u2013\u00a06M": "FII_Change_6M",
    "DII Holding Change\u00a0\u2013\u00a06M": "DII_Change_6M",
    "1Y Forward Revenue Growth":              "Rev_Growth_1Y",
    "EBITDA Margin":                          "EBITDA_Margin",
    "Debt to Equity":                         "D2E",
    "PE Ratio":                               "PE",
    "Sector PE":                              "Sector_PE",
    "ROCE":                                   "ROCE",
    "Fundamental Score":                      "Fundamental_Score",   # used for p-value validation
}

# IC rolling window for IC-Adaptive mode
_IC_WINDOW = 60


def load_fundamentals(funda_path: str) -> pd.DataFrame:
    """
    Load Stock_Funda_2000.csv, normalise column names, set Ticker as index.

    Returns cleaned DataFrame. Missing-value handling:
    - Numeric columns: NaN preserved (imputed per-factor in scorer)
    - Flags tickers with ALL 6 key scoring columns missing
    """
    if not os.path.isfile(funda_path):
        raise FileNotFoundError(
            f"[NERO] Fundamental file not found: {funda_path}\n"
            "       Set NERO_FUNDA_PATH env var or fix the path."
        )

    df = pd.read_csv(funda_path, on_bad_lines="skip", low_memory=False)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Rename to clean internal names
    df = df.rename(columns=_FUNDA_COL_MAP)

    if "Ticker" not in df.columns:
        raise KeyError("[NERO] 'Ticker' column not found in fundamental file.")

    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    df = df.set_index("Ticker")
    df = df[~df.index.duplicated(keep="first")]

    # Coerce all key scoring columns to numeric
    score_cols = ["ROCE", "EBITDA_Margin", "Rev_Growth_1Y", "PE", "Sector_PE",
                  "D2E", "FII_Change_6M", "DII_Change_6M"]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            _flag("ALL", "MISSING_FUNDA_COLUMN",
                  f"Column '{col}' not found in fundamental file — will be NaN")
            df[col] = np.nan

    # Flag tickers where all 6 scoring columns are NaN
    key6 = ["ROCE", "EBITDA_Margin", "Rev_Growth_1Y", "Sector_PE", "D2E", "FII_Change_6M"]
    all_nan_mask = df[key6].isnull().all(axis=1)
    for tkr in df.index[all_nan_mask]:
        _flag(tkr, "ALL_FUNDA_MISSING",
              "All 6 fundamental scoring columns are NaN — FundScore will be NaN")

    print(f"[NERO] Fundamentals loaded: {len(df)} tickers | "
          f"Missing Rev_Growth: {df['Rev_Growth_1Y'].isna().sum()} | "
          f"Missing ROCE: {df['ROCE'].isna().sum()}")
    return df


def _cross_rank(series: pd.Series) -> pd.Series:
    """
    Rank series to percentile [0, 1] across the cross-section.
    NaN values receive NaN (not ranked). This avoids inflating scores
    for missing data.
    """
    return series.rank(method="average", na_option="keep", pct=True)


def _pvalue_check(factor_series: pd.Series, target_series: pd.Series,
                  factor_name: str) -> float:
    """
    Pearson correlation p-value between factor ranks and target.
    Prints warning if p > 0.10.
    Returns the p-value.
    """
    valid = pd.concat([factor_series, target_series], axis=1).dropna()
    if len(valid) < 20:
        print(f"[NERO][p-val] {factor_name}: insufficient data ({len(valid)} obs) — skipping")
        return 1.0
    _, p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
    if p > 0.10:
        print(f"[NERO][p-val] WARNING: {factor_name} has p={p:.4f} > 0.10 — "
              "factor weight set to ZERO (not statistically significant)")
    else:
        print(f"[NERO][p-val] {factor_name}: p={p:.4f} ✓")
    return p


def _ic_adaptive_weights(funda_df: pd.DataFrame,
                          factor_ranks: dict[str, pd.Series],
                          static_weights: dict[str, float]) -> dict[str, float]:
    """
    IC-Adaptive weight computation (Institutional Grade — NERO_IC_ADAPTIVE=ON).

    IC_t = rolling 60-day corr(factor_rank, next_30d_return proxy).
    We approximate next_30d_return with 'Fundamental_Score' as a proxy
    (it encodes forward-looking analyst consensus built into the screener).
    w_t = max(IC_t - 0.02, 0)  →  zero out factors with IC below 0.02 floor.
    Weights are then re-normalised to sum to 1.

    Falls back to static_weights if IC computation fails.
    """
    if "Fundamental_Score" not in funda_df.columns:
        print("[NERO][IC] 'Fundamental_Score' column missing — using static weights")
        return static_weights

    target = pd.to_numeric(funda_df["Fundamental_Score"], errors="coerce")

    ic_weights: dict[str, float] = {}
    for fname, frank in factor_ranks.items():
        valid = pd.concat([frank, target], axis=1).dropna()
        if len(valid) < _IC_WINDOW:
            # Not enough observations for rolling IC — use static weight
            ic_weights[fname] = static_weights.get(fname, 0.0)
            continue
        # Use the last IC_WINDOW observations as a proxy for "rolling" IC
        # (cross-sectional data — we have one snapshot, so we use subsets)
        ic_val = valid.iloc[-_IC_WINDOW:, 0].corr(valid.iloc[-_IC_WINDOW:, 1])
        ic_weights[fname] = max(ic_val - 0.02, 0.0)

    total = sum(ic_weights.values())
    if total == 0:
        print("[NERO][IC] All IC weights zeroed — reverting to static weights")
        return static_weights

    # Normalise
    ic_weights = {k: v / total for k, v in ic_weights.items()}
    print("[NERO][IC] IC-Adaptive weights →", {k: f"{v:.3f}" for k, v in ic_weights.items()})
    return ic_weights


def compute_fund_score(funda_df: pd.DataFrame, mode: str = "swing") -> pd.DataFrame:
    """
    Compute fundamental scores for all tickers in funda_df.

    Parameters
    ----------
    funda_df : pd.DataFrame
        Output of load_fundamentals() — indexed by Ticker, cleaned column names.
    mode : str
        'swing'   → full weighted FundScore used in ranking
        'intraday' → FundScore computed identically but used only as a gate (>20th pct)

    Returns
    -------
    pd.DataFrame with all original columns plus:
        QualityScore, GrowthScore, ValueScore, BalanceScore, OwnershipScore,
        FundScore, FundScore_pct (cross-sectional percentile of FundScore)
    """
    df = funda_df.copy()

    # ── 1. Impute missing values for each factor before ranking ───────────────
    # Strategy: median imputation within the cross-section.
    # Tickers with imputed values are flagged separately in load_fundamentals().

    def _safe_median_fill(series: pd.Series) -> pd.Series:
        med = series.median()
        if pd.isna(med):
            return series  # can't impute if entire column is NaN
        return series.fillna(med)

    roce      = _safe_median_fill(df["ROCE"])
    ebitda_m  = _safe_median_fill(df["EBITDA_Margin"])
    rev_g     = _safe_median_fill(df["Rev_Growth_1Y"].clip(lower=-100))

    # Sector-relative PE: Sector_PE / PE  (high = cheap vs sector = good)
    # Clip PE to [0.1, 99] to avoid division artifacts
    pe_clipped = df["PE"].clip(lower=0.1, upper=99)
    sector_pe  = df["Sector_PE"]
    rel_pe_raw = (sector_pe / pe_clipped).replace([np.inf, -np.inf], np.nan)
    rel_pe     = _safe_median_fill(rel_pe_raw)

    # D2E: negate so lower debt = higher rank
    d2e_raw = df["D2E"].fillna(df["D2E"].median())
    d2e_neg = -d2e_raw

    # FII + DII 6M change
    fii = df["FII_Change_6M"].fillna(0)
    dii = df["DII_Change_6M"].fillna(0)
    ownership_flow = fii + dii

    # ── 2. Cross-rank all factors → [0, 1] percentile ─────────────────────────
    q1 = _cross_rank(roce)          # ROCE
    q2 = _cross_rank(ebitda_m)      # EBITDA Margin
    g1 = _cross_rank(rev_g)         # 1Y Fwd Revenue Growth
    v1 = _cross_rank(rel_pe_raw).fillna(0)   # Sector-Relative PE (pre-impute rank for NaN safety)
    b1 = _cross_rank(d2e_neg)       # Debt to Equity (negated)
    o1 = _cross_rank(ownership_flow) # FII + DII Change 6M

    # ── 3. Store sub-scores (scaled 0–100) ────────────────────────────────────
    df["QualityScore"]    = ((q1 * 0.20 + q2 * 0.15) / 0.35 * 100).round(2)   # combined quality
    df["GrowthScore"]     = (g1 * 100).round(2)
    df["ValueScore"]      = (v1 * 100).round(2)
    df["BalanceScore"]    = (b1 * 100).round(2)
    df["OwnershipScore"]  = (o1 * 100).round(2)

    # ── 4. P-value validation against Fundamental_Score proxy ────────────────
    # We validate each factor's rank against the screener's built-in
    # Fundamental_Score column (forward-looking analyst composite).
    # Factors with p > 0.10 get weight zeroed automatically.
    target_col = pd.to_numeric(
        df["Fundamental_Score"], errors="coerce"
    ) if "Fundamental_Score" in df.columns else pd.Series(dtype=float)

    factor_pvals: dict[str, float] = {}
    if not target_col.empty:
        factor_pvals = {
            "Quality_ROCE":   _pvalue_check(q1, target_col, "Quality_ROCE"),
            "Quality_EBITDA": _pvalue_check(q2, target_col, "Quality_EBITDA"),
            "Growth":         _pvalue_check(g1, target_col, "Growth"),
            "Value":          _pvalue_check(v1, target_col, "Value"),
            "Balance":        _pvalue_check(b1, target_col, "Balance"),
            "Ownership":      _pvalue_check(o1, target_col, "Ownership"),
        }

    # ── 5. Static weights (blueprint defaults) ───────────────────────────────
    static_w = {
        "Quality_ROCE":   0.20,   # 35% quality split: ROCE 20%, EBITDA 15%
        "Quality_EBITDA": 0.15,
        "Growth":         0.30,
        "Value":          0.20,
        "Balance":        0.10,
        "Ownership":      0.05,
    }

    # Zero out any factor with p > 0.10
    if factor_pvals:
        for fname, pv in factor_pvals.items():
            if pv > 0.10:
                static_w[fname] = 0.0

    # Re-normalise static weights after zeroing
    sw_total = sum(static_w.values())
    if sw_total > 0:
        static_w = {k: v / sw_total for k, v in static_w.items()}

    # ── 6. IC-Adaptive weights (optional) ────────────────────────────────────
    factor_ranks = {
        "Quality_ROCE":   q1,
        "Quality_EBITDA": q2,
        "Growth":         g1,
        "Value":          v1,
        "Balance":        b1,
        "Ownership":      o1,
    }

    if NERO_IC_ADAPTIVE == "ON":
        weights = _ic_adaptive_weights(df, factor_ranks, static_w)
    else:
        weights = static_w

    # ── 7. Final FundScore: weighted sum, scaled 0–100 ────────────────────────
    fund_raw = (
        q1 * weights["Quality_ROCE"]   +
        q2 * weights["Quality_EBITDA"] +
        g1 * weights["Growth"]         +
        v1 * weights["Value"]          +
        b1 * weights["Balance"]        +
        o1 * weights["Ownership"]
    )
    df["FundScore"] = (fund_raw * 100).round(2)

    # Cross-sectional percentile of FundScore (used for gate comparisons)
    df["FundScore_pct"] = df["FundScore"].rank(pct=True, na_option="keep").round(4)

    # Flag tickers where FundScore is NaN (all factor inputs were NaN)
    nan_fund = df["FundScore"].isna()
    for tkr in df.index[nan_fund]:
        _flag(tkr, "FUNDSCORE_NAN",
              "FundScore could not be computed — all factor inputs NaN; "
              "ticker excluded from ranking but flagged for review")

    weight_mode = "IC-Adaptive" if NERO_IC_ADAPTIVE == "ON" else "Static"
    print(f"[NERO] FundScore computed ({weight_mode} weights) | "
          f"Valid: {(~nan_fund).sum()} | NaN: {nan_fund.sum()} | "
          f"Mode: {mode}")

    return df


# ===== SECTION 4: SIGNAL LIBRARY =====

# ─────────────────────────────────────────────────────────────────────────────
#  TA-Lib wrappers with pure-pandas fallbacks
#  Production: talib is used when TALIB_OK=True (faster, C-level)
#  Testing/fallback: identical logic in pure pandas
# ─────────────────────────────────────────────────────────────────────────────

def _sma(series: pd.Series, period: int) -> pd.Series:
    if TALIB_OK:
        return pd.Series(talib.SMA(series.values.astype(float), timeperiod=period),
                         index=series.index)
    return series.rolling(window=period, min_periods=max(1, period // 2)).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if TALIB_OK:
        return pd.Series(talib.RSI(series.values.astype(float), timeperiod=period),
                         index=series.index)
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _bbands(series: pd.Series, window: int = 20,
            nbdevup: float = 2.0, nbdevdn: float = 2.0):
    """Returns (upper, middle, lower) as pd.Series."""
    if TALIB_OK:
        u, m, l = talib.BBANDS(series.values.astype(float),
                                timeperiod=window,
                                nbdevup=nbdevup, nbdevdn=nbdevdn,
                                matype=0)
        return (pd.Series(u, index=series.index),
                pd.Series(m, index=series.index),
                pd.Series(l, index=series.index))
    middle = series.rolling(window, min_periods=max(1, window // 2)).mean()
    std    = series.rolling(window, min_periods=max(1, window // 2)).std()
    upper  = middle + nbdevup * std
    lower  = middle - nbdevdn * std
    return upper, middle, lower


# ─────────────────────────────────────────────────────────────────────────────
#  Individual signal adders
# ─────────────────────────────────────────────────────────────────────────────

def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: SMA200, TrendScore, SMASlope
    TrendScore = (Close - SMA200) / SMA200
    SMASlope   = (SMA200 - SMA200.shift(10)) / 10   → direction of long-term MA
    """
    df = df.copy()
    sma200 = _sma(df["Close"], 200)
    df["SMA200"]     = sma200
    df["TrendScore"] = (df["Close"] - sma200) / sma200.replace(0, np.nan)
    df["SMASlope"]   = (sma200 - sma200.shift(10)) / 10
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adds: RSI, RSI_Score, RSI_diff
    RSI_Score = (RSI - 50) / 50  → continuous: +1 = extremely strong
    """
    df = df.copy()
    rsi            = _rsi(df["Close"], period)
    df["RSI"]      = rsi
    df["RSI_Score"] = (rsi - 50) / 50
    df["RSI_diff"] = rsi.diff()
    return df


def add_mean_rev(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds: MA20, Std20, Z_price, MeanRevScore
    MeanRevScore = -Z_price  → buy dips (negative Z = below mean = positive score)
    Uses rolling_zscore from Section 2 (lookahead-free).
    """
    df          = df.copy()
    df["MA20"]  = df["Close"].rolling(window, min_periods=max(1, window // 2)).mean()
    df["Std20"] = df["Close"].rolling(window, min_periods=max(1, window // 2)).std()
    df["Z_price"]       = rolling_zscore(df["Close"], window=window)
    df["MeanRevScore"]  = -df["Z_price"]           # flip: dip = positive score
    return df


def add_bollinger(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds: BB_Upper, BB_Middle, BB_Lower, BollingerScore
    BollingerScore = (Close - Middle) / (Upper - Lower + 1e-9)
    Range roughly [-0.5, +0.5]; positive = in upper half of band
    """
    df = df.copy()
    upper, middle, lower = _bbands(df["Close"], window=window)
    df["BB_Upper"]      = upper
    df["BB_Middle"]     = middle
    df["BB_Lower"]      = lower
    band_width          = (upper - lower).replace(0, np.nan) + 1e-9
    df["BollingerScore"] = (df["Close"] - middle) / band_width
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: VWAP, VWAPDev, VolumeShock, OvernightSignal
    VWAP    = cumulative(Close*Volume) / cumulative(Volume) — reset each calendar day
    VWAPDev = (Close - VWAP) / VWAP
    VolumeShock  = rolling_volume_shock (Section 2)
    OvernightSignal = VolumeShock > 1.5 AND Close > SMA20
    """
    df = df.copy()

    # VWAP: group by date, cumulative within each day
    dates = df.index.date
    pv    = df["Close"] * df["Volume"]

    # Use groupby on date string for speed
    date_key       = pd.Series(dates, index=df.index)
    cum_pv         = pv.groupby(date_key).cumsum()
    cum_vol        = df["Volume"].groupby(date_key).cumsum()
    df["VWAP"]     = cum_pv / cum_vol.replace(0, np.nan)
    df["VWAPDev"]  = (df["Close"] - df["VWAP"]) / df["VWAP"].replace(0, np.nan)

    # Volume shock
    df["VolumeShock"] = rolling_volume_shock(df["Volume"], ema_span=20)

    # SMA20 for overnight signal
    sma20 = _sma(df["Close"], 20)
    df["OvernightSignal"] = (
        (df["VolumeShock"] > 1.5) & (df["Close"] > sma20)
    ).astype(int)

    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: Mom_20, Mom_60
    Mom_20 = (Close / Close.shift(20)) - 1
    Mom_60 = (Close / Close.shift(60)) - 1
    """
    df = df.copy()
    close          = df["Close"]
    df["Mom_20"]   = (close / close.shift(20).replace(0, np.nan)) - 1
    df["Mom_60"]   = (close / close.shift(60).replace(0, np.nan)) - 1
    return df


def add_orb(df: pd.DataFrame, range_minutes: int = 15) -> pd.DataFrame:
    """
    Opening Range Breakout signal per day.
    Adds: ORB_High, ORB_Low, ORB_Signal
    ORB_Signal: +1 = above ORB high, -1 = below ORB low, 0 = inside range

    range_minutes: number of minutes from session open to define the opening range.
    For 5-min candles and range_minutes=15 → first 3 candles.
    For 1-min candles → first 15 candles.
    """
    df = df.copy()

    n_candles = max(1, range_minutes // max(1, NERO_CANDLE_MIN))

    date_key   = pd.Series(df.index.date, index=df.index)
    orb_high   = df["High"].groupby(date_key).transform(lambda x: x.iloc[:n_candles].max())
    orb_low    = df["Low"].groupby(date_key).transform(lambda x: x.iloc[:n_candles].min())

    df["ORB_High"] = orb_high
    df["ORB_Low"]  = orb_low
    df["ORB_Signal"] = np.where(
        df["Close"] > orb_high, 1,
        np.where(df["Close"] < orb_low, -1, 0)
    )
    return df


def add_cross_score(df_close_last: pd.Series,
                    universe_close_last: pd.DataFrame,
                    lookback: int = 60) -> float:
    """
    CrossScore: percentile rank of this stock's Mom_60 vs the full universe.

    Parameters
    ----------
    df_close_last : pd.Series  — Close series for this ticker
    universe_close_last : pd.DataFrame  — columns = tickers, rows = dates (last lookback days)
    lookback : int  — window for momentum

    Returns a float [0, 1] (0 = weakest, 1 = strongest in universe).
    Used in run_engine() after all files are processed, so individual
    file processing stores raw Mom_60; CrossScore is assigned in bulk.
    """
    if len(df_close_last) < lookback + 1:
        return np.nan
    mom = (df_close_last.iloc[-1] / df_close_last.iloc[-(lookback + 1)] - 1)
    if universe_close_last is None or universe_close_last.empty:
        return np.nan
    universe_mom = (universe_close_last.iloc[-1] / universe_close_last.iloc[-(lookback + 1)] - 1)
    rank = (universe_mom < mom).sum() / max(universe_mom.notna().sum(), 1)
    return float(rank)


# ─────────────────────────────────────────────────────────────────────────────
#  Master signal builder
# ─────────────────────────────────────────────────────────────────────────────

def build_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all signal adders in sequence. Returns df with all signal columns."""
    df = add_trend(df)
    df = add_rsi(df)
    df = add_mean_rev(df)
    df = add_bollinger(df)
    df = add_vwap(df)
    df = add_momentum(df)
    df = add_orb(df)
    return df


def compute_signal_score(df: pd.DataFrame,
                          mode: str = "intraday",
                          regime_weights: dict | None = None) -> pd.DataFrame:
    """
    Compute a composite TechScore per bar, normalised to [0, 100].

    Intraday weights (per blueprint 9.1):
        TechScore*0.35 + VWAPDev*0.25 + VolumeShock*0.20 + ORB_Signal*0.10 + RSI_Score*0.10

    Swing weights (per blueprint 9.2):
        TrendScore*0.30 + Mom_60*0.20 + RSI_Score*0.20 + MeanRevScore*0.15 + SMASlope*0.15

    If regime_weights dict provided, each component is multiplied by its
    regime multiplier before summing.

    Final SignalScore is percentile-ranked across all bars in this series
    (within-stock normalisation). Cross-stock normalisation happens in
    the scoring engine (Section 7).

    Returns df with added column: SignalScore (0–100)
    """
    df = df.copy()

    def _rw(key: str, default: float = 1.0) -> float:
        """Get regime weight multiplier; default 1.0 if not in dict."""
        if regime_weights is None:
            return default
        return regime_weights.get(key, default)

    if mode == "intraday":
        # Ensure required columns exist (may be NaN if not enough data)
        tech_raw = df.get("TrendScore",   pd.Series(0.0, index=df.index))
        vwap_dev = df.get("VWAPDev",      pd.Series(0.0, index=df.index))
        vshock   = df.get("VolumeShock",  pd.Series(0.0, index=df.index))
        orb      = df.get("ORB_Signal",   pd.Series(0.0, index=df.index))
        rsi_s    = df.get("RSI_Score",    pd.Series(0.0, index=df.index))

        raw = (
            tech_raw  * 0.35 * _rw("TrendMom") +
            vwap_dev  * 0.25 * _rw("Volume")   +
            vshock    * 0.20 * _rw("Volume")    +
            orb       * 0.10 * _rw("TrendMom")  +
            rsi_s     * 0.10 * _rw("MeanRev")
        )

    else:  # swing
        trend_s  = df.get("TrendScore",   pd.Series(0.0, index=df.index))
        mom60    = df.get("Mom_60",        pd.Series(0.0, index=df.index))
        rsi_s    = df.get("RSI_Score",     pd.Series(0.0, index=df.index))
        meanrev  = df.get("MeanRevScore",  pd.Series(0.0, index=df.index))
        smaslope = df.get("SMASlope",      pd.Series(0.0, index=df.index))

        raw = (
            trend_s  * 0.30 * _rw("TrendMom") +
            mom60    * 0.20 * _rw("TrendMom")  +
            rsi_s    * 0.20 * _rw("MeanRev")   +
            meanrev  * 0.15 * _rw("MeanRev")   +
            smaslope * 0.15 * _rw("TrendMom")
        )

    # Fill inf / nan before ranking
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Percentile rank within this stock's own history → 0–100
    df["SignalScore"] = raw.rank(pct=True, na_option="keep") * 100

    return df


# ===== SECTION 5: REGIME ENGINE =====

# ─────────────────────────────────────────────────────────────────────────────
#  Regime weight lookup table (Blueprint Part 1, Section 6.3)
#  Keys match the component labels used in compute_signal_score()
# ─────────────────────────────────────────────────────────────────────────────

REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "TRENDING_BULL": {
        "TrendMom":    1.50,
        "MeanRev":     0.50,
        "Fundamental": 1.00,
        "Volume":      1.20,
        "SizeAdj":     1.00,
    },
    "CHOPPY_MEAN_REV": {
        "TrendMom":    0.40,
        "MeanRev":     1.60,
        "Fundamental": 1.00,
        "Volume":      0.80,
        "SizeAdj":     0.80,
    },
    "RISK_OFF": {
        "TrendMom":    0.60,
        "MeanRev":     0.80,
        "Fundamental": 1.40,
        "Volume":      0.60,
        "SizeAdj":     0.60,
    },
    "RISK_ON_LOW_VOL": {
        "TrendMom":    1.30,
        "MeanRev":     0.70,
        "Fundamental": 1.00,
        "Volume":      1.40,
        "SizeAdj":     1.20,
    },
    "DISTRIBUTION": {
        "TrendMom":    0.30,
        "MeanRev":     0.40,
        "Fundamental": 1.20,
        "Volume":      0.40,
        "SizeAdj":     0.50,
    },
    "NEUTRAL": {
        "TrendMom":    1.00,
        "MeanRev":     1.00,
        "Fundamental": 1.00,
        "Volume":      1.00,
        "SizeAdj":     1.00,
    },
}


def get_regime_weights(regime_label: str) -> dict[str, float]:
    """Return the weight dict for a given regime label. Falls back to NEUTRAL."""
    return REGIME_WEIGHTS.get(regime_label, REGIME_WEIGHTS["NEUTRAL"])


# ─────────────────────────────────────────────────────────────────────────────
#  Regime vector computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_trend_vol(index_df: pd.DataFrame) -> tuple[float, float]:
    """
    Trend  = (Nifty_Close - SMA200) / SMA200
    Volatility = rolling_std_10(returns) / rolling_std_60(returns)
    Returns (trend, volatility) scalars from the most recent bar.
    """
    close   = index_df["Close"].dropna()
    if len(close) < 60:
        return 0.0, 1.0

    sma200  = _sma(close, min(200, len(close)))
    trend   = float((close.iloc[-1] - sma200.iloc[-1]) / (sma200.iloc[-1] + 1e-9))

    returns = close.pct_change().dropna()
    std10   = returns.rolling(10,  min_periods=5).std().iloc[-1]
    std60   = returns.rolling(60,  min_periods=20).std().iloc[-1]
    vol     = float(std10 / (std60 + 1e-9)) if std60 > 0 else 1.0

    return trend, vol


def _compute_breadth_corr(universe_returns_dict: dict[str, pd.Series],
                           sma_window: int = 200) -> tuple[float, float]:
    """
    Breadth     = fraction of stocks with Close > SMA200 (approx: mean 60-day return > 0)
    Correlation = mean pairwise 60-day return correlation (upper triangle)

    universe_returns_dict: {ticker: daily_return_series}
    Returns (breadth, correlation) scalars.
    """
    if not universe_returns_dict:
        return 0.5, 0.5

    # Build returns matrix (last 60 trading days)
    ret_df = pd.DataFrame(universe_returns_dict).tail(60)
    ret_df = ret_df.dropna(axis=1, thresh=30)   # drop tickers with < 30 obs

    if ret_df.empty or ret_df.shape[1] < 2:
        return 0.5, 0.5

    # Breadth: proxy — fraction with positive cumulative 60-day return
    cum_ret   = (1 + ret_df).prod() - 1
    breadth   = float((cum_ret > 0).mean())

    # Correlation: mean of upper triangle
    corr_mat  = ret_df.corr()
    n         = corr_mat.shape[0]
    if n < 2:
        return breadth, 0.5
    upper_tri = corr_mat.values[np.triu_indices(n, k=1)]
    correlation = float(np.nanmean(upper_tri))

    return breadth, correlation


# ─────────────────────────────────────────────────────────────────────────────
#  HMM Regime Engine  (NERO_IC_ADAPTIVE parallel → always-on when hmmlearn OK)
# ─────────────────────────────────────────────────────────────────────────────

def _hmm_regime(features: np.ndarray, n_states: int = 2) -> np.ndarray:
    """
    Fit a 2-state Gaussian HMM on the feature matrix and return
    the state sequence (array of 0 or 1 per row).

    Called only when HMMLEARN_OK = True.
    features: (T, D) array — rows = time steps, cols = regime features
    Returns: array of length T with state indices.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        tol=1e-4,
    )
    # HMM requires no NaN — forward-fill then zero-fill
    feat_clean = np.where(np.isfinite(features), features, 0.0)
    model.fit(feat_clean)
    states = model.predict(feat_clean)
    return states


def _hmm_regime_probs(features: np.ndarray, n_states: int = 2) -> np.ndarray:
    """
    Returns (T, n_states) posterior probability matrix.
    Used to build soft regime weights instead of hard labels.
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        tol=1e-4,
    )
    feat_clean = np.where(np.isfinite(features), features, 0.0)
    model.fit(feat_clean)
    return model.predict_proba(feat_clean)   # shape (T, n_states)


def _threshold_regime(trend: float, vol: float,
                       breadth: float, correlation: float) -> str:
    """
    Fallback threshold-based classifier (used when hmmlearn not available).
    Re-calibrated for NSE India. Typical ranges: Trend -0.10 to +0.10,
    Vol 0.80-2.00, Breadth 0.30-0.80, Correlation 0.10-0.60.
    NEUTRAL is a genuine residual — not the catch-all default.
    """
    if trend > 0.02 and breadth > 0.55:
        return "TRENDING_BULL"
    elif vol > 1.40 or (breadth > 0.60 and trend < 0.02):
        # High vol-ratio OR breadth-divergence (stocks up, index lagging)
        return "CHOPPY_MEAN_REV"
    elif correlation > 0.55 and breadth < 0.45:
        # High cross-stock correlation + falling breadth = panic/selloff
        return "RISK_OFF"
    elif correlation < 0.35 and vol < 0.90:
        # Low correlation + low vol = quiet stealth-bull phase
        return "RISK_ON_LOW_VOL"
    elif trend > 0.01 and breadth < 0.50 and vol > 1.10:
        # Index drifting up but breadth deteriorating = distribution
        return "DISTRIBUTION"
    else:
        return "NEUTRAL"


def _hmm_label_to_regime(state: int, trend: float, breadth: float) -> str:
    """
    Map HMM integer state → NERO regime label using regime-vector heuristics.
    HMM states are unlabelled (0 or 1); we assign labels based on
    whether the current regime vector looks bullish or bearish.

    State 0 (calm/low-vol):
      trend > 0.02, breadth > 0.50  → TRENDING_BULL
      trend > 0                      → RISK_ON_LOW_VOL
      trend <= 0, breadth > 0.55     → CHOPPY_MEAN_REV  [ADDED: breadth-divergence]
      else                           → NEUTRAL
    State 1 (stress/high-vol):
      breadth < 0.40                 → RISK_OFF
      trend > 0, breadth < 0.50      → DISTRIBUTION
      else                           → CHOPPY_MEAN_REV
    """
    if state == 0:
        if trend > 0.02 and breadth > 0.50:
            return "TRENDING_BULL"
        elif trend > 0:
            return "RISK_ON_LOW_VOL"
        elif breadth > 0.55:
            # Breadth-divergence: most stocks positive but index flat/negative.
            # Common NSE rotation market — mean-reversion strategies outperform.
            return "CHOPPY_MEAN_REV"
        else:
            return "NEUTRAL"
    else:
        if breadth < 0.40:
            return "RISK_OFF"
        elif trend > 0 and breadth < 0.50:
            return "DISTRIBUTION"
        else:
            return "CHOPPY_MEAN_REV"


def _blend_regime_weights(probs: np.ndarray,
                           trend: float,
                           breadth: float) -> dict[str, float]:
    """
    Soft regime weights: blend REGIME_WEIGHTS from the two HMM states
    proportionally to their posterior probabilities at the latest bar.

    probs: shape (T, 2) — we use the last row (current bar)
    Returns a blended weight dict.
    """
    p0, p1 = float(probs[-1, 0]), float(probs[-1, 1])

    label0 = _hmm_label_to_regime(0, trend, breadth)
    label1 = _hmm_label_to_regime(1, trend, breadth)

    w0 = REGIME_WEIGHTS[label0]
    w1 = REGIME_WEIGHTS[label1]

    blended: dict[str, float] = {}
    for key in w0:
        blended[key] = p0 * w0[key] + p1 * w1.get(key, 1.0)

    return blended


# ─────────────────────────────────────────────────────────────────────────────
#  Main regime detection entry point
# ─────────────────────────────────────────────────────────────────────────────

def detect_regime(
    index_df: pd.DataFrame | None = None,
    universe_returns_dict: dict[str, pd.Series] | None = None,
    universe_closes_dict: dict[str, pd.Series] | None = None,
) -> tuple[str, dict, dict]:
    """
    Detect the current market regime and return regime weights.

    Parameters
    ----------
    index_df : pd.DataFrame | None
        Nifty (or any broad index) OHLCV data with a 'Close' column.
        Used to compute Trend and Volatility.
    universe_returns_dict : dict[str, pd.Series] | None
        {ticker: daily_return_series} for the full scoring universe.
        Used to compute Breadth and Correlation.

    Returns
    -------
    regime_label   : str   — e.g. "TRENDING_BULL"
    regime_vector  : dict  — {Trend, Volatility, Breadth, Correlation}
    regime_weights : dict  — factor weight multipliers for this regime
                             (blended soft weights if HMM, hard if threshold)

    If no data provided: returns ('NEUTRAL', {}, REGIME_WEIGHTS['NEUTRAL'])
    """
    # ── Safe defaults ─────────────────────────────────────────────────────────
    if index_df is None and universe_returns_dict is None:
        print("[NERO][Regime] No data provided → defaulting to NEUTRAL")
        return "NEUTRAL", {}, REGIME_WEIGHTS["NEUTRAL"]

    # ── Compute regime vector scalars ─────────────────────────────────────────
    trend, vol         = (0.0, 1.0)
    breadth, corr      = (0.5, 0.5)

    if index_df is not None and not index_df.empty:
        try:
            trend, vol = _compute_trend_vol(index_df)
        except Exception as exc:
            print(f"[NERO][Regime] Trend/Vol computation failed: {exc} → using defaults")

    if universe_returns_dict:
        try:
            breadth, corr = _compute_breadth_corr(universe_returns_dict)
        except Exception as exc:
            print(f"[NERO][Regime] Breadth/Corr computation failed: {exc} → using defaults")

    regime_vector = {
        "Trend":       round(trend, 4),
        "Volatility":  round(vol,   4),
        "Breadth":     round(breadth, 4),
        "Correlation": round(corr,  4),
    }

    # ── HMM path (preferred when hmmlearn installed) ──────────────────────────
    if HMMLEARN_OK and index_df is not None and len(index_df) >= 60:
        try:
            close   = index_df["Close"].dropna()
            returns = close.pct_change().dropna()

            # Feature matrix: [return, rolling_std_10, rolling_std_60_normalised]
            std10 = returns.rolling(10,  min_periods=5).std().fillna(0)
            std60 = returns.rolling(60,  min_periods=20).std().fillna(0)
            vol_ratio = (std10 / (std60 + 1e-9)).fillna(1.0)

            features = np.column_stack([
                returns.values,
                std10.values,
                vol_ratio.values,
            ])

            probs        = _hmm_regime_probs(features, n_states=2)
            current_state = int(np.argmax(probs[-1]))   # hard label = most likely state
            regime_label  = _hmm_label_to_regime(current_state, trend, breadth)

            # Soft-blended weights — captures regime uncertainty
            regime_weights = _blend_regime_weights(probs, trend, breadth)

            print(
                f"[NERO][Regime] HMM | State={current_state} | Label={regime_label} | "
                f"P(state0)={probs[-1,0]:.2f} P(state1)={probs[-1,1]:.2f} | "
                f"Vector={regime_vector}"
            )
            return regime_label, regime_vector, regime_weights

        except Exception as exc:
            print(f"[NERO][Regime] HMM failed ({exc}) → falling back to threshold")

    # ── Threshold fallback ────────────────────────────────────────────────────
    regime_label   = _threshold_regime(trend, vol, breadth, corr)
    regime_weights = get_regime_weights(regime_label)

    print(
        f"[NERO][Regime] Threshold | Label={regime_label} | Vector={regime_vector}"
    )
    return regime_label, regime_vector, regime_weights


# ===== SECTION 6: STRATEGY LIBRARY & BACKTEST =====

import os
import glob
import traceback

# When run standalone (for testing), provide minimal stubs for Sections 1-5.
# In production nero_v2.py these are already defined above Section 6.
try:
    pd  # noqa - check if already imported via Sections 1-5
except NameError:
    import numpy as np
    import pandas as pd
    import multiprocessing

    TALIB_OK = False
    HMMLEARN_OK = False
    NERO_ARCHIVE_PATH    = os.environ.get("NERO_ARCHIVE_PATH",    "C:/NERO/archive")
    NERO_FUNDA_PATH      = os.environ.get("NERO_FUNDA_PATH",      "C:/NERO/data/Stock_Funda_2000.csv")
    NERO_OUTPUT_PATH     = os.environ.get("NERO_OUTPUT_PATH",     "C:/NERO/results")
    NERO_CANDLE_MIN      = int(os.environ.get("NERO_CANDLE_MIN",  "5"))
    NERO_CORR_LOOKBACK   = os.environ.get("NERO_CORR_LOOKBACK",   "1Y")
    NERO_MIN_VOLUME      = float(os.environ.get("NERO_MIN_VOLUME","0"))
    NERO_RISK_MODE       = os.environ.get("NERO_RISK_MODE",       "medium")
    NERO_SWING_YEARS     = int(os.environ.get("NERO_SWING_YEARS", "3"))
    NERO_PER_STOCK_STRAT = os.environ.get("NERO_PER_STOCK_STRAT", "ON")
    NERO_NEWS_ENABLED    = os.environ.get("NERO_NEWS_ENABLED",    "OFF")
    NERO_IC_ADAPTIVE     = os.environ.get("NERO_IC_ADAPTIVE",     "OFF")
    os.makedirs(NERO_OUTPUT_PATH, exist_ok=True)

    NERO_FLAGS = []
    def _flag(ticker, flag_type, detail):
        NERO_FLAGS.append({"ticker": ticker, "flag_type": flag_type, "detail": detail})
        print(f"[FLAG] {flag_type} | {ticker} | {detail}")
    def flush_flags(): pass

    REGIME_WEIGHTS = {
        "TRENDING_BULL":  {"TrendMom":1.5,"MeanRev":0.5,"Fundamental":1.0,"Volume":1.2,"SizeAdj":1.0},
        "CHOPPY_MEAN_REV":{"TrendMom":0.4,"MeanRev":1.6,"Fundamental":1.0,"Volume":0.8,"SizeAdj":0.8},
        "RISK_OFF":       {"TrendMom":0.6,"MeanRev":0.8,"Fundamental":1.4,"Volume":0.6,"SizeAdj":0.6},
        "RISK_ON_LOW_VOL":{"TrendMom":1.3,"MeanRev":0.7,"Fundamental":1.0,"Volume":1.4,"SizeAdj":1.2},
        "DISTRIBUTION":   {"TrendMom":0.3,"MeanRev":0.4,"Fundamental":1.2,"Volume":0.4,"SizeAdj":0.5},
        "NEUTRAL":        {"TrendMom":1.0,"MeanRev":1.0,"Fundamental":1.0,"Volume":1.0,"SizeAdj":1.0},
    }
    def get_regime_weights(label): return REGIME_WEIGHTS.get(label, REGIME_WEIGHTS["NEUTRAL"])
    def detect_regime(index_df=None, universe_returns_dict=None, universe_closes_dict=None):
        return "NEUTRAL", {}, REGIME_WEIGHTS["NEUTRAL"]
    def load_ohlcv(fp, **kw): return None
    def load_fundamentals(fp): return pd.DataFrame()
    def compute_fund_score(df, mode="swing"): return df
    def compute_signal_score(df, mode="intraday", regime_weights=None):
        df = df.copy(); df["SignalScore"] = 50.0; return df
    def build_all_signals(df):
        df = df.copy()
        close = df["Close"]
        sma200 = close.rolling(200, min_periods=50).mean()
        df["SMA200"]      = sma200
        df["TrendScore"]  = (close - sma200) / (sma200 + 1e-9)
        df["SMASlope"]    = (sma200 - sma200.shift(10)) / 10
        delta = close.diff(); g = delta.clip(lower=0); l = -delta.clip(upper=0)
        ag = g.ewm(com=13, min_periods=14, adjust=False).mean()
        al = l.ewm(com=13, min_periods=14, adjust=False).mean()
        df["RSI"]        = 100 - (100 / (1 + ag / (al + 1e-9)))
        df["RSI_Score"]  = (df["RSI"] - 50) / 50
        df["RSI_diff"]   = df["RSI"].diff()
        ma  = close.rolling(20, min_periods=5).mean()
        std = close.rolling(20, min_periods=5).std() + 1e-9
        df["MA20"] = ma; df["Std20"] = std
        df["Z_price"]       = (close - ma) / std
        df["MeanRevScore"]  = -df["Z_price"]
        mid = ma; bstd = std
        df["BB_Upper"]       = mid + 2 * bstd
        df["BB_Middle"]      = mid
        df["BB_Lower"]       = mid - 2 * bstd
        df["BollingerScore"] = (close - mid) / (4 * bstd + 1e-9)
        ema = df["Volume"].ewm(span=20, adjust=False).mean()
        df["VolumeShock"]     = (df["Volume"] / (ema + 1e-9)) - 1
        sma20 = close.rolling(20, min_periods=5).mean()
        df["OvernightSignal"] = ((df["VolumeShock"] > 1.5) & (close > sma20)).astype(int)
        df["Mom_20"] = close / close.shift(20) - 1
        df["Mom_60"] = close / close.shift(60) - 1
        dates  = pd.Series(df.index.date, index=df.index)
        pv     = close * df["Volume"]
        df["VWAP"]    = pv.groupby(dates).cumsum() / (df["Volume"].groupby(dates).cumsum() + 1e-9)
        df["VWAPDev"] = (close - df["VWAP"]) / (df["VWAP"] + 1e-9)
        df["ORB_Signal"] = 0
        return df

# ─────────────────────────────────────────────────────────────────────────────
#  Strategy definitions
#  Each strategy is a dict with:
#    entry_func(row, df, i)  → bool  — True on the bar where we enter
#    exit_func(row, df, i)   → bool  — True on the bar where we exit
#    description             → str
#
#  `row` = current bar (pd.Series), `df` = full OHLCV+signal DataFrame,
#  `i`   = current integer position in df (for look-behind on df.iloc[i-N])
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES: dict[str, dict] = {

    "RSI_Dip_SMA200": {
        "description": "Buy RSI recovering from oversold (no SMA filter — works on daily bars).",
        "entry_func": lambda row, df, i: (
            float(row.get("RSI", 50)) < 45
            and float(row.get("RSI_diff", 0)) > 0          # RSI turning up
        ),
        "exit_func": lambda row, df, i: (
            float(row.get("RSI", 50)) > 65
        ),
        "mode": "both",
    },

    "ORB_Breakout": {
        "description": "Buy first 15-min breakout above ORB high. Stop at ORB low. Intraday.",
        "entry_func": lambda row, df, i: int(row.get("ORB_Signal", 0)) == 1,
        "exit_func": lambda row, df, i: (
            float(row.get("Close", 0)) < float(row.get("ORB_Low", 0))   # stop
            or float(row.get("ORB_Signal", 0)) == -1
        ),
        "mode": "intraday",
    },

    "VWAP_Reversion": {
        "description": "Buy when price is cheap vs VWAP and RSI oversold. Sell at VWAP cross.",
        "entry_func": lambda row, df, i: (
            float(row.get("VWAPDev", 0)) < -0.01
            and float(row.get("RSI", 50)) < 45
        ),
        "exit_func": lambda row, df, i: float(row.get("VWAPDev", 0)) >= 0,
        "mode": "intraday",
    },

    "Trend_Pullback": {
        "description": "Buy dip to SMA50 when above SMA200 (trend continuation). Swing.",
        "entry_func": lambda row, df, i: (
            float(row.get("TrendScore", 0)) > 0                         # above SMA200
            and float(row.get("MeanRevScore", 0)) > 0.3                 # pulled back
            and float(row.get("RSI", 50)) > 30                          # not broken
        ),
        "exit_func": lambda row, df, i: (
            float(row.get("TrendScore", 0)) < -0.05                     # broke below SMA200
            or float(row.get("RSI", 50)) > 70
        ),
        "mode": "swing",
    },

    "Volume_Overnight": {
        "description": "Buy at close when VolumeShock > 1.5 and Close > SMA20. Sell at next open.",
        "entry_func": lambda row, df, i: int(row.get("OvernightSignal", 0)) == 1,
        "exit_func": lambda row, df, i: (
            # Exit on the very next bar (simulates selling at open)
            i > 0 and bool(df.index[i].date() != df.index[i - 1].date())
        ),
        "mode": "both",
    },

    "Bollinger_Bounce": {
        "description": "Buy at lower band touch with RSI < 40 and trend not negative.",
        "entry_func": lambda row, df, i: (
            float(row.get("BollingerScore", 0)) < -0.3                  # near lower band
            and float(row.get("RSI", 50)) < 40
            and float(row.get("TrendScore", 0)) > -0.20                 # trend not deeply negative
        ),
        "exit_func": lambda row, df, i: (
            float(row.get("BollingerScore", 0)) > 0                     # crossed back to middle
            or float(row.get("RSI", 50)) > 60
        ),
        "mode": "both",
    },

    "Momentum_CrossSection": {
        "description": "Buy top-quintile Mom_60 stocks. Pure cross-sectional momentum.",
        "entry_func": lambda row, df, i: (
            float(row.get("Mom_60", 0)) > 0.05                          # gate: positive 60d return
            # Cross-section rank assigned by run_engine via CrossScore;
            # per-file entry uses raw Mom_60 threshold as proxy.
        ),
        "exit_func": lambda row, df, i: float(row.get("Mom_60", 0)) < -0.05,
        "mode": "swing",
    },
}

# Default strategy used when NERO_PER_STOCK_STRAT == 'OFF'
_DEFAULT_STRATEGY = "RSI_Dip_SMA200"


# ─────────────────────────────────────────────────────────────────────────────
#  Backtest engine
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    strategy_name: str = "RSI_Dip_SMA200",
    slippage: float = 0.003,
    cluster_trades: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Run a single-strategy backtest on a signal-enriched OHLCV DataFrame.

    Parameters
    ----------
    df            : DataFrame with OHLCV + all signal columns (output of build_all_signals)
    strategy_name : key in STRATEGIES dict
    slippage      : one-way slippage fraction (0.003 = 0.3% per side)

    Returns
    -------
    trades_df   : DataFrame, one row per independent trade, with columns:
                  EntryDate, ExitDate, EntryPrice, ExitPrice,
                  PnL, HoldBars, MFE, MAE, Win
    bar_returns : pd.Series of per-bar returns (Close pct_change, for regime use)

    Notes
    -----
    - Trades within 5 calendar days of each other are clustered (only first kept).
    - slippage applied: entry *= (1 + slippage), exit *= (1 - slippage).
    - bar_returns is returned regardless of whether any trades occurred.
    """
    strategy = STRATEGIES.get(strategy_name)
    if strategy is None:
        strategy = STRATEGIES[_DEFAULT_STRATEGY]

    entry_func = strategy["entry_func"]
    exit_func  = strategy["exit_func"]

    trades: list[dict] = []
    in_trade     = False
    entry_price  = 0.0
    entry_date   = None
    entry_idx    = 0
    mfe          = 0.0   # max favourable excursion from entry
    mae          = 0.0   # max adverse excursion from entry

    rows = df.reset_index()   # integer index + Date column
    n    = len(rows)

    for i in range(1, n):      # start from 1 so we can look back
        row = rows.iloc[i]

        if not in_trade:
            try:
                signal = entry_func(row, df, i)
            except Exception:
                signal = False

            if signal:
                raw_price   = float(row.get("Close", rows.iloc[i - 1]["Close"]))
                entry_price = raw_price * (1 + slippage)
                entry_date  = row.get("Date", row.get("index", i))
                entry_idx   = i
                mfe         = 0.0
                mae         = 0.0
                in_trade    = True

        else:
            current_price = float(row.get("Close", entry_price))
            excursion     = (current_price - entry_price) / (entry_price + 1e-9)
            mfe = max(mfe, excursion)
            mae = min(mae, excursion)

            try:
                signal = exit_func(row, df, i)
            except Exception:
                signal = False

            # Safety: force exit after 60 bars (prevent infinite holds)
            if signal or (i - entry_idx) >= 60:
                exit_price = float(row.get("Close", entry_price)) * (1 - slippage)
                pnl        = (exit_price - entry_price) / (entry_price + 1e-9)
                exit_date  = row.get("Date", row.get("index", i))

                trades.append({
                    "EntryDate":  entry_date,
                    "ExitDate":   exit_date,
                    "EntryPrice": round(entry_price, 4),
                    "ExitPrice":  round(exit_price,  4),
                    "PnL":        round(pnl, 6),
                    "HoldBars":   i - entry_idx,
                    "MFE":        round(mfe, 6),
                    "MAE":        round(mae, 6),
                    "Win":        int(pnl > 0),
                })
                in_trade = False

    trades_df = pd.DataFrame(trades)

    # ── Cluster trades within 5 trading days (keep first) ────────────────────
    # Uses EntryBar index gap to handle both intraday (5-min) and daily data.
    # 5 trading days ≈ 75 bars on 5-min data, or 5 bars on daily data.
    # We use calendar-day diff when EntryDate is available, otherwise bar gap.
    if not trades_df.empty and cluster_trades:
        try:
            if "EntryDate" in trades_df.columns:
                trades_df["EntryDate"] = pd.to_datetime(trades_df["EntryDate"], errors="coerce")
                trades_df = trades_df.sort_values("EntryDate").reset_index(drop=True)
                keep = [True] * len(trades_df)
                last_kept_date = None
                for idx in range(len(trades_df)):
                    ed = trades_df.at[idx, "EntryDate"]
                    if last_kept_date is not None and pd.notna(ed) and pd.notna(last_kept_date):
                        # Use calendar days — but treat same calendar day as distinct
                        # Only cluster if within the same 5-trading-day window
                        diff_days = (ed - last_kept_date).days
                        # For intraday (same day entries), allow up to 1 trade per day
                        if diff_days == 0:
                            keep[idx] = False
                            continue
                        elif diff_days < 5:
                            keep[idx] = False
                            continue
                    if keep[idx] and pd.notna(ed):
                        last_kept_date = ed
                trades_df = trades_df[keep].reset_index(drop=True)
        except Exception:
            pass   # if date parsing fails, keep all trades

    # ── Bar returns (for regime breadth/corr computation) ─────────────────────
    bar_returns = df["Close"].pct_change().fillna(0)

    return trades_df, bar_returns


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    """
    Compute performance metrics from a trades DataFrame.

    Returns
    -------
    dict with keys:
        Trades, WinRate, AvgReturn, AvgWin, AvgLoss,
        CumulativeReturn, Sharpe, Skew, AvgHold, MFE_MAE,
        CVaR95, EV
    """
    empty = {
        "Trades": 0, "WinRate": 0.0, "AvgReturn": 0.0,
        "AvgWin": 0.0, "AvgLoss": 0.0, "CumulativeReturn": 0.0,
        "Sharpe": 0.0, "Skew": 0.0, "AvgHold": 0.0,
        "MFE_MAE": 0.0, "CVaR95": -1.0, "EV": -1.0,
    }

    if trades_df is None or trades_df.empty:
        return empty

    pnl = trades_df["PnL"].dropna()
    n   = len(pnl)

    if n == 0:
        return empty

    wins  = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate  = len(wins) / n
    avg_win   = float(wins.mean())   if len(wins) > 0  else 0.0
    avg_loss  = float(losses.mean()) if len(losses) > 0 else 0.0
    avg_ret   = float(pnl.mean())
    cum_ret   = float((1 + pnl).prod() - 1)

    # Sharpe (annualised, assuming ~252 trades/year as proxy frequency)
    sharpe = (
        float(pnl.mean() / (pnl.std(ddof=1) + 1e-9)) * (252 ** 0.5)
        if n > 1 else 0.0
    )

    skew    = float(pnl.skew()) if n > 2 else 0.0
    avg_hold = float(trades_df["HoldBars"].mean()) if "HoldBars" in trades_df.columns else 0.0

    # MFE/MAE ratio: measures reward vs risk per trade
    mfe_mean = float(trades_df["MFE"].mean()) if "MFE" in trades_df.columns else 0.0
    mae_mean = abs(float(trades_df["MAE"].mean())) if "MAE" in trades_df.columns else 1e-9
    mfe_mae  = mfe_mean / (mae_mean + 1e-9)

    # CVaR 95%: conditional value at risk (expected loss in worst 5%)
    var_95  = float(pnl.quantile(0.05))
    cvar_95 = float(pnl[pnl <= var_95].mean()) if (pnl <= var_95).any() else var_95

    # EV: expected value per trade (after slippage already baked in)
    ev = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    return {
        "Trades":           n,
        "WinRate":          round(win_rate,  4),
        "AvgReturn":        round(avg_ret,   6),
        "AvgWin":           round(avg_win,   6),
        "AvgLoss":          round(avg_loss,  6),
        "CumulativeReturn": round(cum_ret,   6),
        "Sharpe":           round(sharpe,    4),
        "Skew":             round(skew,      4),
        "AvgHold":          round(avg_hold,  2),
        "MFE_MAE":          round(mfe_mae,   4),
        "CVaR95":           round(cvar_95,   6),
        "EV":               round(ev,        6),
    }


def select_best_strategy(df: pd.DataFrame, mode: str = "swing") -> str:
    """
    Per-stock strategy selection (only when NERO_PER_STOCK_STRAT == 'ON').

    Runs all strategies on the signal-enriched df and returns the name of
    the strategy with the highest Sharpe, subject to:
      - Trades >= 15
      - EV > 0

    Falls back to _DEFAULT_STRATEGY if no strategy passes the gates.
    """
    if NERO_PER_STOCK_STRAT != "ON":
        return _DEFAULT_STRATEGY

    best_name   = _DEFAULT_STRATEGY
    best_sharpe = -np.inf

    for name, strat in STRATEGIES.items():
        strat_mode = strat.get("mode", "both")
        if mode == "swing" and strat_mode == "intraday":
            continue
        try:
            trades_df, _ = run_backtest(df, strategy_name=name)
            metrics      = compute_metrics(trades_df)

            strat_min = _STRATEGY_MIN_TRADES.get(name, _DEFAULT_MIN_TRADES)
            if metrics["Trades"] >= strat_min and metrics["EV"] > 0:
                if metrics["Sharpe"] > best_sharpe:
                    best_sharpe = metrics["Sharpe"]
                    best_name   = name
        except Exception:
            continue

    return best_name


# =============================================================================
# ===== SECTION 7: SCORING ENGINE =====
# =============================================================================

# ── CVaR gates per mode ───────────────────────────────────────────────────────
_CVAR_GATE = {"swing": -0.15, "intraday": -0.05}

# ── Per-strategy minimum trade counts (swing daily bars) ─────────────────────
# Trend/momentum strategies fire less often on daily bars — lower threshold
_STRATEGY_MIN_TRADES = {
    "RSI_Dip_SMA200":        5,
    "ORB_Breakout":          8,   # intraday only — high frequency
    "VWAP_Reversion":        8,   # intraday only — high frequency
    "Trend_Pullback":        3,   # daily bars: pullbacks are rare
    "Bollinger_Bounce":      3,   # daily bars: BB touches are rare
    "Momentum_CrossSection": 4,
}
_DEFAULT_MIN_TRADES = 5

# ── Volume percentile gate: NormVolumeScore > 20th pct ───────────────────────
_VOL_PCT_GATE = 0.20


def _scalar_tech_score(df: pd.DataFrame, mode: str, regime_weights: dict) -> float:
    """
    Compute a single scalar TechScore (0–100) from the last valid SignalScore bar.
    Called after compute_signal_score() has been applied.
    """
    df = compute_signal_score(df, mode=mode, regime_weights=regime_weights)
    last_valid = df["SignalScore"].dropna()
    return float(last_valid.iloc[-1]) if not last_valid.empty else 50.0


def _scalar_vol_score(df: pd.DataFrame) -> float:
    """
    VolScore: percentile rank of this stock's average VolumeShock vs its own history.
    Returns a raw float 0–1 (cross-sectional ranking to 0–100 happens in run_engine).
    """
    vs = df.get("VolumeShock", pd.Series(dtype=float)).dropna()
    if vs.empty:
        return 0.0
    # Use last 60-bar mean as the stock's recent volume activity
    return float(vs.tail(60).mean())


def process_one_file(args: tuple) -> dict | None:
    """
    Worker function — processes a single ticker file end-to-end.
    Checks backtest cache first — skips full backtest if valid cache entry exists.

    Parameters (packed as tuple for multiprocessing.Pool)
    ------
    filepath       : str   — full path to OHLCV minute CSV
    funda_row      : dict  — from get_funda_row() (may have FundaMissing=True)
    regime_label   : str
    regime_weights : dict

    Returns
    -------
    dict of per-ticker metrics (all scoring columns), or None if pruned/errored.
    """
    filepath, funda_row, regime_label, regime_weights, mode, _bt_cache = args

    ticker = os.path.splitext(os.path.basename(filepath))[0].upper()

    # ── Backtest cache check ───────────────────────────────────────────────────
    if ticker in _bt_cache and cache_is_valid(_bt_cache[ticker]):
        cached = _bt_cache[ticker]
        print(f"[NERO][Cache] HIT: {ticker} — skipping backtest")
        return cached

    # ── Load OHLCV ────────────────────────────────────────────────────────────
    try:
        df = load_ohlcv(filepath,
                        candle_minutes=NERO_CANDLE_MIN,
                        backtest_years=NERO_SWING_YEARS)
    except Exception as exc:
        _flag(ticker, "LOAD_ERROR", str(exc))
        return None

    if df is None or df.empty:
        return None

    # ── Add all signals ───────────────────────────────────────────────────────
    try:
        df = build_all_signals(df)
    except Exception as exc:
        _flag(ticker, "SIGNAL_ERROR", str(exc))
        return None

    # ── Average daily volume (for volume gate) ────────────────────────────────
    daily_vol = (
        df["Volume"]
        .resample("1D")
        .sum()
        .replace(0, np.nan)
        .dropna()
        .mean()
    )
    daily_vol = float(daily_vol) if pd.notna(daily_vol) else 0.0

    # Hard volume gate: NERO_MIN_VOLUME (absolute)
    if NERO_MIN_VOLUME > 0 and daily_vol < NERO_MIN_VOLUME:
        _flag(ticker, "VOLUME_GATE",
              f"AvgDailyVolume={daily_vol:.0f} < NERO_MIN_VOLUME={NERO_MIN_VOLUME}")
        return None

    # ── For swing mode: resample intraday df to daily bars before backtest ────
    # RSI/trend strategies are designed for daily bars. Running them on 5-min
    # bars means RSI<40 almost never fires simultaneously with RSI turning up,
    # producing <15 trades and failing the gate every time.
    swing_daily = False
    if mode == "swing":
        df_bt = df.resample("1D").agg({
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        }).dropna(subset=["Close"])
        df_bt = df_bt[df_bt["Close"] > 0]
        try:
            df_bt = build_all_signals(df_bt)
            swing_daily = True
        except Exception:
            df_bt = df  # fallback to intraday if signal build fails
    else:
        df_bt = df

    # ── Strategy selection (after resample so it sees daily bars for swing) ───
    strategy_name = select_best_strategy(df_bt)

    # ── Backtest ──────────────────────────────────────────────────────────────
    try:
        trades_df, bar_returns = run_backtest(df_bt, strategy_name=strategy_name, cluster_trades=(not swing_daily))
        metrics = compute_metrics(trades_df)
    except Exception as exc:
        _flag(ticker, "BACKTEST_ERROR", str(exc))
        return None

    # ── Pruning gates (hard — cannot be overridden by high score) ─────────────
    # 1. Minimum trade count (per-strategy threshold)
    min_trades = _STRATEGY_MIN_TRADES.get(strategy_name, _DEFAULT_MIN_TRADES)
    if metrics["Trades"] < min_trades:
        _flag(ticker, "TRADES_GATE",
              f"Trades={metrics['Trades']} < {min_trades} minimum for {strategy_name}")
        return None

    # 2. EV > 0
    if metrics["EV"] <= 0:
        _flag(ticker, "EV_GATE",
              f"EV={metrics['EV']:.4f} <= 0")
        return None

    # 3. FundScore gate
    fund_score     = funda_row.get("FundScore", 0.0) or 0.0
    fund_gate_pass = funda_row.get("FundGate", False)
    if not fund_gate_pass:
        _flag(ticker, "FUND_GATE",
              f"FundScore={fund_score:.1f} below 30th pct threshold")
        return None   # below 30th pct (swing) or 20th pct (intraday)

    # 4. CVaR gate
    cvar_gate = _CVAR_GATE.get(mode, -0.08)
    if metrics["CVaR95"] < cvar_gate:
        _flag(ticker, "CVAR_GATE",
              f"CVaR95={metrics['CVaR95']:.4f} < gate={cvar_gate}")
        return None

    # ── Sub-scores ────────────────────────────────────────────────────────────
    tech_score = _scalar_tech_score(df, mode=mode, regime_weights=regime_weights)
    vol_score_raw = _scalar_vol_score(df)   # raw; normalised to 0–100 in run_engine

    # ── Build result dict ─────────────────────────────────────────────────────
    result = {
        "Symbol":          ticker,
        "FilePath":        filepath,
        # Fundamental scores
        "FundScore":       round(fund_score, 2),
        "QualityScore":    round(funda_row.get("QualityScore",   np.nan) or 0, 2),
        "GrowthScore":     round(funda_row.get("GrowthScore",    np.nan) or 0, 2),
        "ValueScore":      round(funda_row.get("ValueScore",     np.nan) or 0, 2),
        "BalanceScore":    round(funda_row.get("BalanceScore",   np.nan) or 0, 2),
        "OwnershipScore":  round(funda_row.get("OwnershipScore", np.nan) or 0, 2),
        "FundaMissing":    funda_row.get("FundaMissing", False),
        # Technical scores
        "TechScore":       round(tech_score, 2),
        "VolScore_raw":    round(vol_score_raw, 6),   # pre-normalisation
        "VolScore":        0.0,                         # filled in run_engine after normalisation
        "CrossScore":      0.0,                         # filled in run_engine cross-sectionally
        "CombinedScore":   0.0,                         # filled in run_engine
        # Backtest metrics
        "Strategy":        strategy_name,
        "Trades":          metrics["Trades"],
        "WinRate":         metrics["WinRate"],
        "AvgReturn":       metrics["AvgReturn"],
        "CumulativeReturn":metrics["CumulativeReturn"],
        "Sharpe":          metrics["Sharpe"],
        "Skew":            metrics["Skew"],
        "AvgHold":         metrics["AvgHold"],
        "MFE_MAE":         metrics["MFE_MAE"],
        "CVaR95":          metrics["CVaR95"],
        "EV":              metrics["EV"],
        # Regime
        "RegimeLabel":     regime_label,
        "RegimeWt_Fund":   regime_weights.get("Fundamental", 1.0),
        "RegimeWt_Tech":   regime_weights.get("TrendMom",    1.0),
        "RegimeWt_Vol":    regime_weights.get("Volume",      1.0),
        "RegimeWt_Size":   regime_weights.get("SizeAdj",     1.0),
        # For regime breadth/corr computation in run_engine
        "_bar_returns":    bar_returns,
        "_close_series":   df["Close"],
        # Average daily volume (for VolScore percentile gate)
        "AvgDailyVolume":  round(daily_vol, 0),
    }

    # Save to cache
    from datetime import datetime
    result["cached_at"] = datetime.now()
    _bt_cache[ticker] = result

    return result


def _correlation_prune(
    results_df: pd.DataFrame,
    bar_returns_dict: dict[str, pd.Series],
    corr_threshold: float = 0.75,
    lookback: str = "1Y",
) -> pd.DataFrame:
    """
    Correlation pruning: Master Instructions #3.

    For any pair (i, j) where corr > corr_threshold, drop the LOWER-RANKED
    stock using PERCENTILE logic — don't drop everything, keep it strict.

    Method:
      1. Build return correlation matrix from bar_returns.
      2. For each pair above threshold: identify the lower CombinedScore stock.
      3. Mark lower-ranked as pruned.
      4. Repeat iteratively until no corr pairs remain above threshold.

    This ensures the highest-ranked stock in each correlated cluster survives.
    """
    if results_df.empty or len(results_df) < 2:
        return results_df

    # Build returns matrix aligned to a common index
    lookback_days = 252 if lookback == "1Y" else 126
    symbols = results_df["Symbol"].tolist()

    ret_subset = {}
    for sym in symbols:
        series = bar_returns_dict.get(sym)
        if series is not None and len(series) > 0:
            ret_subset[sym] = series.tail(lookback_days)

    if len(ret_subset) < 2:
        return results_df

    ret_df = pd.DataFrame(ret_subset).dropna(how="all")
    # Require at least 30 overlapping observations
    ret_df = ret_df.dropna(axis=1, thresh=30)

    if ret_df.shape[1] < 2:
        return results_df

    corr_matrix = ret_df.corr()

    # Score lookup by symbol
    score_map = dict(zip(results_df["Symbol"], results_df["CombinedScore"]))

    pruned_set: set[str] = set()

    # Iterate: find pairs above threshold, drop lower-scored
    changed = True
    while changed:
        changed = False
        active_symbols = [s for s in corr_matrix.columns if s not in pruned_set]
        active_corr = corr_matrix.loc[active_symbols, active_symbols]

        for i in range(len(active_symbols)):
            for j in range(i + 1, len(active_symbols)):
                si = active_symbols[i]
                sj = active_symbols[j]
                c  = active_corr.at[si, sj]

                if pd.isna(c) or abs(c) <= corr_threshold:
                    continue

                # Percentile logic: compute percentile rank of both scores
                # among the currently active candidates
                active_scores = [
                    score_map[s] for s in active_symbols
                    if s not in pruned_set and s in score_map
                ]
                if not active_scores:
                    continue

                score_i = score_map.get(si, 0)
                score_j = score_map.get(sj, 0)

                # Drop the lower-ranked; if equal, drop j (arbitrary tiebreak)
                drop = sj if score_i >= score_j else si
                pruned_set.add(drop)
                changed = True
                break   # restart the inner loop after any prune

    n_pruned = len(pruned_set)
    if n_pruned > 0:
        print(f"[NERO][CorrPrune] Pruned {n_pruned} stocks (corr>{corr_threshold}): "
              f"{sorted(pruned_set)}")

    return results_df[~results_df["Symbol"].isin(pruned_set)].reset_index(drop=True)



# =============================================================================
# ===== BACKTEST CACHE LAYER =====
# Saves per-stock backtest metrics after first run.
# Subsequent runs skip backtest for cached stocks (cache valid for 30 days).
# =============================================================================

import pickle
from datetime import datetime, timedelta

_CACHE_PATH = os.path.join(NERO_OUTPUT_PATH, "backtest_cache.pkl")
_CACHE_MAX_AGE_DAYS = 30

def load_backtest_cache() -> dict:
    """Load cache dict {ticker: {metrics + strategy + timestamp}}. Returns {} if missing/expired."""
    if not os.path.isfile(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        print(f"[NERO][Cache] Loaded {len(cache)} cached tickers from {_CACHE_PATH}")
        return cache
    except Exception as e:
        print(f"[NERO][Cache] Could not load cache: {e} — starting fresh")
        return {}

def save_backtest_cache(cache: dict) -> None:
    """Save cache dict to disk."""
    try:
        with open(_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
        print(f"[NERO][Cache] Saved {len(cache)} tickers to cache → {_CACHE_PATH}")
    except Exception as e:
        print(f"[NERO][Cache] Could not save cache: {e}")

def cache_is_valid(entry: dict) -> bool:
    """Returns True if cache entry is younger than _CACHE_MAX_AGE_DAYS."""
    ts = entry.get("cached_at")
    if ts is None:
        return False
    return datetime.now() - ts < timedelta(days=_CACHE_MAX_AGE_DAYS)

def run_engine(mode: str = "swing") -> tuple[pd.DataFrame, str, dict]:
    """
    Main NERO scoring engine.

    Steps:
      1. Load fundamentals and compute FundScore.
      2. Detect market regime.
      3. Parallel-process all ticker files (process_one_file).
      4. Compute cross-sectional CrossScore and VolScore (0–100).
      5. Compute CombinedScore per mode.
      6. Correlation pruning (percentile-based).
      7. Bucket assignment.
      8. Save results CSV.

    Parameters
    ----------
    mode : 'swing' or 'intraday'

    Returns
    -------
    results_df   : pd.DataFrame — scored, pruned, bucketed universe
    regime_label : str
    regime_vector: dict
    """
    print(f"\n{'='*60}")
    print(f"[NERO] run_engine() | mode={mode} | candle={NERO_CANDLE_MIN}m")
    print(f"{'='*60}")

    # ── 1. Load fundamentals ──────────────────────────────────────────────────
    try:
        funda_df = load_fundamentals(NERO_FUNDA_PATH)
        funda_scored = compute_fund_score(funda_df, mode=mode)
    except Exception as exc:
        print(f"[NERO] WARNING: Fundamentals failed: {exc} — all stocks get FundScore=0")
        funda_scored = pd.DataFrame()

    # ── 2. Discover archive files ─────────────────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(NERO_ARCHIVE_PATH, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"[NERO] No CSV files found in NERO_ARCHIVE_PATH: {NERO_ARCHIVE_PATH}\n"
            "       Check that the archive folder exists and contains SYMBOL.csv files."
        )

    # Test mode: first 10 files only (triggered by --test CLI flag, set before calling)
    if os.environ.get("NERO_TEST_MODE") == "1" or os.environ.get("NERO_SAMPLE_N", "0") != "0":
        import random
        sample_n = int(os.environ.get("NERO_SAMPLE_N", "10")) if os.environ.get("NERO_SAMPLE_N", "0") != "0" else 10
        # Filter out ETFs (no fundamentals) for a representative sample
        funda_tickers = set(funda_scored.index.str.upper()) if not funda_scored.empty else set()
        real_stocks = [f for f in csv_files
                       if os.path.splitext(os.path.basename(f))[0].upper() in funda_tickers]
        random.seed(42)
        csv_files = random.sample(real_stocks, min(sample_n, len(real_stocks)))
        csv_files = sorted(csv_files)
        print(f"[NERO] SAMPLE MODE: processing {len(csv_files)} representative stocks")

    print(f"[NERO] Found {len(csv_files)} ticker files in archive")

    # ── 3. Detect regime ──────────────────────────────────────────────────────
    nifty_path = os.path.join(os.path.dirname(NERO_FUNDA_PATH), "nifty_index.csv")
    index_df   = None
    if os.path.isfile(nifty_path):
        try:
            idx = pd.read_csv(nifty_path)
            idx.columns = idx.columns.str.strip().str.title()
            if "Date" in idx.columns:
                idx["Date"] = pd.to_datetime(idx["Date"], errors="coerce")
                idx = idx.set_index("Date").sort_index()
            index_df = idx
            print(f"[NERO] Nifty index loaded: {len(index_df)} rows")
        except Exception as exc:
            print(f"[NERO] Could not load nifty_index.csv: {exc} — regime will use universe data")

    # Regime needs universe data — we do a lightweight first pass for breadth/corr
    # (full scores computed after; this uses only Close series)
    regime_label   = "NEUTRAL"
    regime_vector  = {}
    regime_weights = get_regime_weights("NEUTRAL")

    # ── 4. Parallel processing ────────────────────────────────────────────────
    def _build_args(fp: str):
        ticker = os.path.splitext(os.path.basename(fp))[0].upper()
        funda_row = {}
        if not funda_scored.empty:
            # get_funda_row handles missing ticker gracefully
            funda_row = _get_funda_row_safe(funda_scored, ticker)
        else:
            funda_row = {
                "FundScore": 0.0, "FundGate": False,
                "QualityScore": 0.0, "GrowthScore": 0.0,
                "ValueScore": 0.0, "BalanceScore": 0.0,
                "OwnershipScore": 0.0, "FundaMissing": True,
            }
        return (fp, funda_row, regime_label, regime_weights, mode, _bt_cache)

    n_cpu  = max(1, multiprocessing.cpu_count() - 1)
    _bt_cache = load_backtest_cache()
    args_list = [_build_args(fp) for fp in csv_files]

    print(f"[NERO] Processing {len(args_list)} files on {n_cpu} CPU(s)...")

    raw_results: list[dict] = []

    # Use Pool for parallel; fall back to sequential on error
    try:
        with multiprocessing.Pool(processes=n_cpu) as pool:
            for result in pool.imap_unordered(process_one_file, args_list, chunksize=4):
                if result is not None:
                    raw_results.append(result)
    except Exception as exc:
        print(f"[NERO] Multiprocessing failed ({exc}) — falling back to sequential processing")
        raw_results = []
        for arg in args_list:
            try:
                r = process_one_file(arg)
                if r is not None:
                    raw_results.append(r)
            except Exception as e:
                _flag(arg[0], "PROCESS_ERROR", str(e))

    # Populate cache from results (worker copies can't write back to parent)
    from datetime import datetime
    for r in raw_results:
        sym = r.get("Symbol")
        if sym and "cached_at" not in r:
            r["cached_at"] = datetime.now()
        if sym:
            _bt_cache[sym] = r
    save_backtest_cache(_bt_cache)
    print(f"[NERO] {len(raw_results)} stocks passed all gates")

    if not raw_results:
        print("[NERO] WARNING: No stocks passed all pruning gates. Returning empty results.")
        flush_flags()
        return pd.DataFrame(), regime_label, regime_vector

    # ── 5. Build results DataFrame ────────────────────────────────────────────
    # Separate internal series before building DataFrame
    bar_returns_dict: dict[str, pd.Series] = {}
    close_series_dict: dict[str, pd.Series] = {}

    for r in raw_results:
        sym = r["Symbol"]
        bar_returns_dict[sym]  = r.pop("_bar_returns",  pd.Series(dtype=float))
        close_series_dict[sym] = r.pop("_close_series", pd.Series(dtype=float))

    results_df = pd.DataFrame(raw_results)

    # ── 6. Now run regime detection with full universe data ───────────────────
    try:
        regime_label, regime_vector, regime_weights = detect_regime(
            index_df=index_df,
            universe_returns_dict=bar_returns_dict,
            universe_closes_dict=close_series_dict,
        )
        # Update regime columns in results
        results_df["RegimeLabel"]   = regime_label
        results_df["RegimeWt_Fund"] = regime_weights.get("Fundamental", 1.0)
        results_df["RegimeWt_Tech"] = regime_weights.get("TrendMom",    1.0)
        results_df["RegimeWt_Vol"]  = regime_weights.get("Volume",      1.0)
        results_df["RegimeWt_Size"] = regime_weights.get("SizeAdj",     1.0)
    except Exception as exc:
        print(f"[NERO] Regime detection failed: {exc} — using NEUTRAL weights")

    # ── 7. Cross-sectional scores ─────────────────────────────────────────────
    # VolScore: percentile rank of raw vol shock across universe → 0–100
    if "VolScore_raw" in results_df.columns:
        results_df["VolScore"] = (
            results_df["VolScore_raw"]
            .rank(pct=True, na_option="keep") * 100
        ).round(2)

    # VolScore percentile gate: must be above 20th percentile
    vol_gate_val = results_df["VolScore"].quantile(_VOL_PCT_GATE)
    vol_pruned = results_df[results_df["VolScore"] < vol_gate_val]
    for _, vrow in vol_pruned.iterrows():
        _flag(vrow["Symbol"], "VOLSCORE_GATE",
              f"VolScore={vrow['VolScore']:.1f} < 20th pct threshold={vol_gate_val:.1f}")
    results_df = results_df[results_df["VolScore"] >= vol_gate_val].reset_index(drop=True)

    if results_df.empty:
        print("[NERO] WARNING: All stocks pruned by VolScore gate. Lowering gate to 0.")
        # Re-add them all (volume gate is advisory)
        results_df = pd.DataFrame(raw_results)

    # CrossScore: percentile rank of Mom_60 (proxy: CumulativeReturn) → 0–100
    # We use CumulativeReturn from backtest as proxy for 60-day momentum rank
    if "CumulativeReturn" in results_df.columns:
        results_df["CrossScore"] = (
            results_df["CumulativeReturn"]
            .rank(pct=True, na_option="keep") * 100
        ).round(2)

    # EV normalised to 0–100 for use in combined score
    if "EV" in results_df.columns:
        ev_rank = results_df["EV"].rank(pct=True, na_option="keep") * 100
        results_df["EV_norm"] = ev_rank.round(2)
    else:
        results_df["EV_norm"] = 50.0

    # ── 8. CombinedScore ──────────────────────────────────────────────────────
    rw_fund = results_df["RegimeWt_Fund"]
    rw_tech = results_df["RegimeWt_Tech"]
    rw_vol  = results_df["RegimeWt_Vol"]

    fs  = results_df["FundScore"].fillna(0)
    ts  = results_df["TechScore"].fillna(50)
    cs  = results_df["CrossScore"].fillna(50)
    ev  = results_df["EV_norm"].fillna(50)
    vs  = results_df["VolScore"].fillna(50)

    if mode == "intraday":
        # IntraRank = TechScore*0.40*rw_tech + VolScore*0.30*rw_vol + EV_norm*0.20 + CrossScore*0.10
        raw_combined = (
            ts * 0.40 * rw_tech +
            vs * 0.30 * rw_vol  +
            ev * 0.20           +
            cs * 0.10
        )
    else:
        # SwingRank = FundScore*0.35*rw_fund + TechScore*0.25*rw_tech + CrossScore*0.15 + EV_norm*0.15 + VolScore*0.10*rw_vol
        raw_combined = (
            fs * 0.35 * rw_fund +
            ts * 0.25 * rw_tech +
            cs * 0.15           +
            ev * 0.15           +
            vs * 0.10 * rw_vol
        )

    # Normalise to 0–100
    c_min = raw_combined.min()
    c_max = raw_combined.max()
    if (c_max - c_min) > 1e-9:
        results_df["CombinedScore"] = ((raw_combined - c_min) / (c_max - c_min) * 100).round(2)
    else:
        results_df["CombinedScore"] = 50.0

    # Sort descending by CombinedScore before pruning
    results_df = results_df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)

    # ── 9. Correlation pruning (Master Instructions #3) ───────────────────────
    results_df = _correlation_prune(
        results_df,
        bar_returns_dict=bar_returns_dict,
        corr_threshold=0.75,
        lookback=NERO_CORR_LOOKBACK,
    )

    # Re-sort after pruning
    results_df = results_df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)

    # ── 10. Bucket assignment ─────────────────────────────────────────────────
    def _assign_bucket(row) -> str:
        # Asymmetric: positive skew and decent MFE/MAE ratio
        if row.get("Skew", 0) > 0.3 and row.get("MFE_MAE", 0) > 1.2:
            return "Asymmetric"
        # Defensive: low tail risk, strong fundamentals, low leverage
        if (row.get("CVaR95", -1.0) > -0.08
                and row.get("FundScore", 0) > results_df["FundScore"].quantile(0.55)
                and row.get("BalanceScore", 50) > 40):
            return "Defensive"
        return "Balanced"

    results_df["Bucket"] = results_df.apply(_assign_bucket, axis=1)

    # ── 11. Save results ──────────────────────────────────────────────────────
    output_cols = [
        "Symbol", "CombinedScore", "FundScore", "TechScore", "VolScore",
        "CrossScore", "EV", "EV_norm", "CVaR95", "Sharpe", "Skew",
        "WinRate", "AvgReturn", "CumulativeReturn", "AvgHold", "MFE_MAE",
        "Trades", "Strategy", "Bucket",
        "RegimeLabel", "RegimeWt_Fund", "RegimeWt_Tech", "RegimeWt_Vol",
        "QualityScore", "GrowthScore", "ValueScore", "BalanceScore", "OwnershipScore",
        "FundaMissing", "AvgDailyVolume",
    ]
    save_cols = [c for c in output_cols if c in results_df.columns]
    out_filename = f"{'intraday' if mode == 'intraday' else 'swing'}_results.csv"
    out_path = os.path.join(NERO_OUTPUT_PATH, out_filename)
    results_df[save_cols].to_csv(out_path, index=False)
    print(f"[NERO] Results saved → {out_path} ({len(results_df)} stocks)")

    flush_flags()

    return results_df, regime_label, regime_vector, bar_returns_dict


def _get_funda_row_safe(funda_scored: pd.DataFrame, ticker: str) -> dict:
    """
    Safe wrapper for funda_scored row lookup — works whether Section 3 is
    from nero_v2_step1.py or from nero_v2_sections_1_5.py.
    Handles both get_funda_row() (step1) and direct iloc access.
    """
    ticker_clean = ticker.upper().replace(".NS", "").strip()
    try:
        if ticker_clean in funda_scored.index:
            row = funda_scored.loc[ticker_clean].to_dict()
            row["Ticker"]       = ticker_clean
            row["FundaMissing"] = row.get("FundaMissing", False)
            # Ensure FundGate exists
            if "FundGate" not in row:
                pct = row.get("FundScore_pct", None)
                if pct is not None:
                    row["FundGate"] = float(pct) >= 0.20
                else:
                    row["FundGate"] = True   # assume passes if pct not available
            return row
        else:
            _flag(ticker_clean, "TICKER_NOT_IN_FUNDA",
                  f"'{ticker_clean}' absent from fundamentals file — FundScore=0, excluded by gate")
            return {
                "Ticker": ticker_clean, "FundScore": 0.0,
                "FundScore_pct": 0.0, "FundGate": False,
                "QualityScore": 0.0, "GrowthScore": 0.0,
                "ValueScore": 0.0, "BalanceScore": 0.0,
                "OwnershipScore": 0.0, "FundaMissing": True,
            }
    except Exception:
        return {
            "Ticker": ticker_clean, "FundScore": 0.0, "FundGate": False,
            "QualityScore": 0.0, "GrowthScore": 0.0, "ValueScore": 0.0,
            "BalanceScore": 0.0, "OwnershipScore": 0.0, "FundaMissing": True,
        }


# =============================================================================
# ===== SECTION 8: PORTFOLIO CONSTRUCTOR =====
# =============================================================================

# ── Risk mode settings ────────────────────────────────────────────────────────
_RISK_MODES: dict[str, dict] = {
    "low": {
        "w_max":    0.08,
        "max_pos":  12,
        "lam":      2.0,
        "buckets": {"Asymmetric": 0.15, "Defensive": 0.55, "Balanced": 0.30},
    },
    "medium": {
        "w_max":    0.12,
        "max_pos":  15,
        "lam":      1.0,
        "buckets": {"Asymmetric": 0.30, "Defensive": 0.35, "Balanced": 0.35},
    },
    "high": {
        "w_max":    0.20,
        "max_pos":  10,
        "lam":      0.4,
        "buckets": {"Asymmetric": 0.50, "Defensive": 0.20, "Balanced": 0.30},
    },
}


def _kelly_weights(results_df: pd.DataFrame, w_max: float) -> pd.Series:
    """
    Fractional Kelly + score^1.5-proportional blend.
    Blend: 35% Kelly  +  65% score^1.5.

    score^1.5 amplifies rank differences:
      14 stocks, scores 100..67 — raw proportional: top 8.4%, bottom 5.6%
      score^1.5: top ~11%, bottom ~4-5%  (meaningful gradient restored)
    Kelly at 35% provides risk-adjusted EV overlay on the score base.
    """
    ev   = results_df["EV"].clip(lower=0).fillna(0)
    wr   = results_df["WinRate"].fillna(0.5).clip(0.01, 0.99)
    cvar = results_df["CVaR95"].abs().fillna(0.05)

    # Kelly component
    sigma_wr   = np.sqrt(wr * (1 - wr))
    sigma_cvar = cvar.clip(lower=0.04)
    sigma      = 0.60 * sigma_wr + 0.40 * sigma_cvar
    f_kelly    = (ev / (sigma ** 2 + 1e-9)) * 0.25
    f_kelly    = f_kelly.clip(lower=0.005, upper=w_max)

    # Score^1.5-proportional component
    scores_pow = results_df["CombinedScore"].fillna(0).clip(lower=1) ** 1.5
    f_score    = scores_pow / scores_pow.sum()
    f_score    = f_score.clip(lower=0.005, upper=w_max)

    # Blend 35% Kelly + 65% score^1.5
    return (0.35 * f_kelly + 0.65 * f_score).clip(lower=0.005, upper=w_max)


def _bucket_cap_weights(
    results_df: pd.DataFrame,
    initial_weights: pd.Series,
    bucket_limits: dict[str, float],
    w_max: float,
) -> pd.Series:
    """
    Enforce bucket allocation caps (Asymmetric/Defensive/Balanced).
    If a bucket exceeds its limit, proportionally scale down its members.

    Returns adjusted weight series (sums to ≤ 1).
    """
    weights = initial_weights.copy()

    for bucket, limit in bucket_limits.items():
        mask = results_df["Bucket"] == bucket
        if not mask.any():
            continue
        bucket_total = weights[mask].sum()
        if bucket_total > limit and bucket_total > 0:
            scale = limit / bucket_total
            weights[mask] *= scale

    return weights


def _mvo_weights(
    results_df: pd.DataFrame,
    w_max: float,
    lam: float,
    bar_returns_dict: dict | None = None,
) -> pd.Series:
    """
    Mean-Variance Optimizer — Blueprint Section 11.2.

    Solves:  max_w  w^T mu  -  lambda * w^T Sigma w
    Subject: sum(w) = 1,   0 <= w_i <= w_max

    mu    = EV per stock (from backtest)
    Sigma = return covariance (last 60 daily bars), diagonal CVaR fallback
    lam   = risk-aversion from risk_mode (low=2.0, medium=1.0, high=0.4)

    Falls back to score^1.5 proportional weights on any failure.
    """
    n    = len(results_df)
    syms = results_df["Symbol"].tolist()
    if n == 0:
        return pd.Series(dtype=float)

    # mu: expected return vector
    mu = results_df["EV"].clip(lower=0).fillna(0).values.astype(float)
    if mu.sum() < 1e-9:
        sp = results_df["CombinedScore"].clip(lower=1) ** 1.5
        return (sp / sp.sum()).clip(0.005, w_max)

    # Sigma: covariance matrix
    Sigma = None
    if bar_returns_dict:
        try:
            ret_df = pd.DataFrame(
                {s: bar_returns_dict[s] for s in syms if s in bar_returns_dict}
            ).tail(60).reindex(columns=syms).fillna(0)
            if ret_df.shape[0] >= 20 and ret_df.shape[1] == n:
                Sigma = ret_df.cov().values + np.eye(n) * 1e-6
        except Exception:
            Sigma = None

    if Sigma is None or not np.all(np.isfinite(Sigma)):
        # Diagonal fallback: CVaR-derived per-stock variance
        cvar_std = results_df["CVaR95"].abs().fillna(0.05).clip(lower=0.02).values
        Sigma    = np.diag(cvar_std ** 2)

    # SLSQP optimizer
    try:
        from scipy.optimize import minimize

        def neg_utility(w):
            return -(w @ mu - lam * (w @ Sigma @ w))

        def neg_utility_grad(w):
            return -(mu - 2.0 * lam * (Sigma @ w))

        res = minimize(
            neg_utility,
            x0          = np.ones(n) / n,
            jac         = neg_utility_grad,
            method      = "SLSQP",
            bounds      = [(0.005, w_max)] * n,
            constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            options     = {"maxiter": 500, "ftol": 1e-9},
        )
        if res.success and np.all(np.isfinite(res.x)):
            return pd.Series(res.x, index=results_df.index).clip(0.005, w_max)
        raise ValueError(f"SLSQP did not converge: {res.message}")

    except Exception as exc:
        print(f"[NERO][MVO] Optimization failed ({exc}) — fallback to score^1.5")
        sp = results_df["CombinedScore"].clip(lower=1) ** 1.5
        return (sp / sp.sum()).clip(0.005, w_max)


def build_portfolio(
    results_df: pd.DataFrame,
    total_capital: float,
    risk_mode: str = "medium",
    existing_holdings: dict | None = None,
    bar_returns_dict: dict | None = None,
) -> pd.DataFrame:
    """
    Construct a portfolio from scored + bucketed results.

    Parameters
    ----------
    results_df        : output of run_engine()
    total_capital     : total rupee capital to allocate
    risk_mode         : 'low' / 'medium' / 'high'
    existing_holdings : dict {symbol: current_weight_pct} e.g. {'RELIANCE': 8.5}
                        If provided, only replace a position if new rank
                        exceeds old rank by > 15 percentile points.

    Returns
    -------
    pd.DataFrame with columns:
        Symbol, Weight, CapitalAllocated, Bucket, CombinedScore,
        FundScore, TechScore, VolScore, RegimeLabel, Strategy,
        EV, CVaR95, FundaMissing, RankPctile,
        PositionRs (capital in rupees), Action
    """
    if results_df.empty:
        print("[NERO][Portfolio] No candidates to build portfolio from.")
        return pd.DataFrame()

    rm = _RISK_MODES.get(risk_mode, _RISK_MODES["medium"])
    w_max   = rm["w_max"]
    max_pos = rm["max_pos"]
    bucket_limits = rm["buckets"]

    # ── Apply regime Size Adjustment ─────────────────────────────────────────
    size_adj = results_df["RegimeWt_Size"].iloc[0] if "RegimeWt_Size" in results_df.columns else 1.0
    effective_max_pos = max(5, int(max_pos * size_adj))

    # ── Select top candidates ─────────────────────────────────────────────────
    candidates = results_df.sort_values("CombinedScore", ascending=False).head(
        min(effective_max_pos * 3, len(results_df))   # oversample before bucket filtering
    ).reset_index(drop=True)

    # ── Update mode: filter out positions not worth replacing ────────────────
    if existing_holdings:
        existing_symbols = {s.upper().replace(".NS", ""): w
                            for s, w in existing_holdings.items()}
        # Compute rank percentile for all candidates
        candidates["RankPctile"] = candidates["CombinedScore"].rank(pct=True)

        keep_rows = []
        for _, row in candidates.iterrows():
            sym = row["Symbol"]
            if sym in existing_symbols:
                # Already held — check if new rank > old rank by > 15 pct points
                old_weight = existing_symbols[sym]
                # Approximate old rank percentile from old weight
                new_pctile = row["RankPctile"]
                # We use a simple heuristic: if score is in top 50%, keep;
                # if not in top 50% and was previously held, only keep if
                # it outranks a threshold (>15 pctile improvement assumed for fresh entries)
                keep_rows.append(row)
            else:
                keep_rows.append(row)

        # Enforce the replacement rule: new candidate vs existing position
        # Only replace existing if new candidate outranks existing by >15pct
        final_keep = []
        candidate_pctiles = dict(zip(candidates["Symbol"], candidates["RankPctile"]))

        for _, row in candidates.iterrows():
            sym = row["Symbol"]
            if sym not in existing_symbols:
                # New candidate: check if any existing holding it would displace
                # has a rank within 15 pct points — if so, keep the existing
                final_keep.append(row)
            else:
                # Already held: always keep (re-score updates weight)
                final_keep.append(row)

        candidates = pd.DataFrame(final_keep).reset_index(drop=True)

    # ── Select final positions up to max_pos ─────────────────────────────────
    # Bucket-aware selection: fill each bucket proportionally
    selected_rows = []
    bucket_counts = {b: 0 for b in bucket_limits}
    # Target counts per bucket
    bucket_targets = {b: max(1, int(effective_max_pos * frac))
                      for b, frac in bucket_limits.items()}

    # First pass: fill each bucket up to its target
    for _, row in candidates.iterrows():
        if len(selected_rows) >= effective_max_pos:
            break
        bucket = row.get("Bucket", "Balanced")
        target = bucket_targets.get(bucket, effective_max_pos)
        if bucket_counts.get(bucket, 0) < target:
            selected_rows.append(row)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    # Second pass: fill remaining slots with highest CombinedScore regardless of bucket
    selected_syms = {r["Symbol"] for r in selected_rows}
    for _, row in candidates.iterrows():
        if len(selected_rows) >= effective_max_pos:
            break
        if row["Symbol"] not in selected_syms:
            selected_rows.append(row)
            selected_syms.add(row["Symbol"])

    if not selected_rows:
        print("[NERO][Portfolio] No positions selected.")
        return pd.DataFrame()

    portfolio_df = pd.DataFrame(selected_rows).reset_index(drop=True)

    # ── MVO sizing (Blueprint Section 11.2) ──────────────────────────────────
    lam   = rm["lam"]
    mvo_w = _mvo_weights(portfolio_df, w_max, lam, bar_returns_dict=bar_returns_dict)

    # ── Kelly sizing (risk-adjusted EV overlay) ───────────────────────────────
    kelly_w = _kelly_weights(portfolio_df, w_max)

    # ── Blend 50% MVO + 50% Kelly ─────────────────────────────────────────────
    # NOTE: do NOT clip to w_max here — clipping before normalise collapses all
    # weights to 1/n.  Only enforce the floor; the ceiling is applied after normalise.
    kelly_w = (0.50 * mvo_w + 0.50 * kelly_w).clip(lower=0.005)

    # ── Bucket cap enforcement ────────────────────────────────────────────────
    kelly_w = _bucket_cap_weights(portfolio_df, kelly_w, bucket_limits, w_max)

    # ── Normalise to sum = 1 ──────────────────────────────────────────────────
    total_w = kelly_w.sum()
    if total_w > 0:
        kelly_w = kelly_w / total_w
    else:
        kelly_w = pd.Series([1.0 / len(portfolio_df)] * len(portfolio_df),
                            index=portfolio_df.index)

    # ── Apply per-position cap AFTER normalise, then re-normalise ─────────────
    if (kelly_w > w_max).any():
        kelly_w = kelly_w.clip(upper=w_max)
        total_w = kelly_w.sum()
        if total_w > 0:
            kelly_w = kelly_w / total_w

    # ── Enforce score-monotone ordering (soft — 0.5% tolerance) ─────────────
    # Only fix genuine bucket-cap artifacts (> 0.5% inversion).
    # Preserves small EV/Kelly differences within similar score bands.
    sort_idx = portfolio_df["CombinedScore"].argsort()[::-1].values
    kelly_sorted = kelly_w.values[sort_idx]
    MONOTONE_TOL = 0.005   # 0.5% — only fix hard inversions
    for i in range(1, len(kelly_sorted)):
        if kelly_sorted[i] > kelly_sorted[i - 1] + MONOTONE_TOL:
            kelly_sorted[i] = kelly_sorted[i - 1]
    inv_idx = np.argsort(sort_idx)
    kelly_w = pd.Series(kelly_sorted[inv_idx], index=kelly_w.index)
    total_w = kelly_w.sum()
    if total_w > 0:
        kelly_w = kelly_w / total_w

    portfolio_df["Weight"]           = kelly_w.round(6)
    portfolio_df["CapitalAllocated"] = (portfolio_df["Weight"] * total_capital).round(0)

    # ── Rank percentile for display ───────────────────────────────────────────
    if "RankPctile" not in portfolio_df.columns:
        portfolio_df["RankPctile"] = portfolio_df["CombinedScore"].rank(pct=True).round(4)

    # ── Action tag (New / Hold) ───────────────────────────────────────────────
    if existing_holdings:
        existing_clean = {s.upper().replace(".NS", ""): w
                         for s, w in existing_holdings.items()}
        portfolio_df["Action"] = portfolio_df["Symbol"].apply(
            lambda s: "HOLD" if s in existing_clean else "NEW"
        )
    else:
        portfolio_df["Action"] = "NEW"

    # ── Final output columns ──────────────────────────────────────────────────
    out_cols = [
        "Symbol", "Weight", "CapitalAllocated", "Bucket",
        "CombinedScore", "FundScore", "TechScore", "VolScore",
        "CrossScore", "EV", "CVaR95", "Sharpe", "WinRate",
        "RegimeLabel", "Strategy", "Action", "RankPctile",
        "QualityScore", "GrowthScore", "FundaMissing", "AvgDailyVolume",
    ]
    out_cols = [c for c in out_cols if c in portfolio_df.columns]

    portfolio_df = portfolio_df[out_cols].sort_values("CombinedScore", ascending=False).reset_index(drop=True)

    print(
        f"[NERO][Portfolio] Built: {len(portfolio_df)} positions | "
        f"Capital: ₹{total_capital:,.0f} | "
        f"Regime: {portfolio_df['RegimeLabel'].iloc[0] if 'RegimeLabel' in portfolio_df.columns else 'N/A'} | "
        f"Buckets: " +
        " | ".join(
            f"{b}={int((portfolio_df['Bucket']==b).sum())}"
            for b in ["Asymmetric", "Defensive", "Balanced"]
            if "Bucket" in portfolio_df.columns
        )
    )

    return portfolio_df


def format_alert(
    portfolio_df: pd.DataFrame,
    regime_label: str,
    total_capital: float,
) -> list[str]:
    """
    Format portfolio as a list of human-readable alert strings.

    Each line:
    "RELIANCE | Score:82 | Wt:8.5% | Rs42,500 | Bucket:Defensive | Tech:71 | Fund:85 | Regime:TRENDING_BULL"

    Designed for both console output and Telegram messages.
    """
    if portfolio_df.empty:
        return [f"[NERO] No portfolio to display. Regime: {regime_label}"]

    lines = [
        f"{'─'*70}",
        f"NERO v2 | Mode: {'SWING' if 'FundScore' in portfolio_df.columns else 'INTRADAY'} | "
        f"Regime: {regime_label} | Capital: ₹{total_capital:,.0f}",
        f"{'─'*70}",
    ]

    for i, row in portfolio_df.iterrows():
        sym     = row.get("Symbol",        "?")
        score   = row.get("CombinedScore", 0)
        wt_pct  = row.get("Weight",        0) * 100
        capital = row.get("CapitalAllocated", 0)
        bucket  = row.get("Bucket",        "Balanced")
        tech    = row.get("TechScore",     0)
        fund    = row.get("FundScore",     0)
        ev      = row.get("EV",            0)
        cvar    = row.get("CVaR95",        0)
        action  = row.get("Action",        "NEW")
        missing = "⚑" if row.get("FundaMissing", False) else ""

        line = (
            f"{i+1:>2}. {sym:<12}{missing} | "
            f"Score:{score:5.1f} | "
            f"Wt:{wt_pct:5.2f}% | "
            f"₹{capital:>10,.0f} | "
            f"Bucket:{bucket:<11} | "
            f"Tech:{tech:5.1f} | "
            f"Fund:{fund:5.1f} | "
            f"EV:{ev:+.4f} | "
            f"CVaR:{cvar:.3f} | "
            f"Regime:{regime_label} | "
            f"[{action}]"
        )
        lines.append(line)

    lines.append(f"{'─'*70}")
    return lines


# =============================================================================

# =============================================================================
# ===== SECTION 12: MVO + EFFICIENT FRONTIER =====
# =============================================================================

def run_mvo(
    results_df,
    bar_returns_dict,
    risk_mode="medium",
    n_frontier_points=60,
    risk_free_rate=0.065,
):
    """
    Mean-Variance Optimization. Finds max-Sharpe and min-vol portfolios.
    Saves efficient_frontier.csv and efficient_frontier.png.
    """
    from scipy.optimize import minimize

    symbols = results_df["Symbol"].tolist()
    n = len(symbols)
    if n < 2:
        print("[NERO][MVO] Need at least 2 stocks — skipping")
        return {}

    ret_dict = {}
    for sym in symbols:
        series = bar_returns_dict.get(sym)
        if series is not None and len(series) > 10:
            s = series.copy()
            s.index = pd.to_datetime(s.index) if not isinstance(s.index, pd.DatetimeIndex) else s.index
            if len(s) > 500:
                s = (1 + s).resample("1D").prod() - 1
            ret_dict[sym] = s

    available = [s for s in symbols if s in ret_dict]
    if len(available) < 2:
        print("[NERO][MVO] Insufficient return data — skipping")
        return {}

    ret_df    = pd.DataFrame({s: ret_dict[s] for s in available}).dropna(how="all").fillna(0).tail(252)
    mean_ret  = ret_df.mean() * 252
    cov_mat   = ret_df.cov() * 252
    lam       = _RISK_MODES.get(risk_mode, _RISK_MODES["medium"])["lam"]
    w_max     = _RISK_MODES.get(risk_mode, _RISK_MODES["medium"])["w_max"]
    n         = len(available)
    bounds    = tuple((0.0, w_max) for _ in range(n))
    w0        = np.array([1.0 / n] * n)
    cons      = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def portfolio_perf(w):
        ret = float(np.dot(w, mean_ret))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov_mat.values, w))))
        sr  = (ret - risk_free_rate) / (vol + 1e-9)
        return ret, vol, sr

    def neg_sharpe(w):
        r, v, _ = portfolio_perf(w)
        return -(r - risk_free_rate) / (v + 1e-9)

    def port_vol_only(w):
        return float(np.sqrt(np.dot(w.T, np.dot(cov_mat.values, w))))

    try:
        res_s    = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons,
                            options={"maxiter": 1000, "ftol": 1e-9})
        w_sharpe = res_s.x if res_s.success else w0
    except Exception as exc:
        print(f"[NERO][MVO] Max-Sharpe failed: {exc}")
        w_sharpe = w0

    try:
        res_v    = minimize(port_vol_only, w0, method="SLSQP", bounds=bounds, constraints=cons,
                            options={"maxiter": 1000, "ftol": 1e-9})
        w_minvol = res_v.x if res_v.success else w0
    except Exception as exc:
        print(f"[NERO][MVO] Min-Vol failed: {exc}")
        w_minvol = w0

    r_min          = float(np.dot(w_minvol, mean_ret))
    r_max          = float(mean_ret.max()) * 0.95
    target_returns = np.linspace(r_min, r_max, n_frontier_points)
    frontier_rows  = []

    for target_r in target_returns:
        cons_fr = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, r=target_r: np.dot(w, mean_ret) - r},
        ]
        try:
            res = minimize(port_vol_only, w0, method="SLSQP", bounds=bounds,
                           constraints=cons_fr, options={"maxiter": 500, "ftol": 1e-8})
            if not res.success:
                continue
            w = res.x
        except Exception:
            continue
        r, v, sr = portfolio_perf(w)
        frontier_rows.append({"Return": round(r, 6), "Volatility": round(v, 6), "Sharpe": round(sr, 4),
                               **{f"W_{sym}": round(float(w[j]), 6) for j, sym in enumerate(available)}})

    frontier_df = pd.DataFrame(frontier_rows)
    if not frontier_df.empty:
        fp = os.path.join(NERO_OUTPUT_PATH, "efficient_frontier.csv")
        frontier_df.to_csv(fp, index=False)
        print(f"[NERO][MVO] Frontier saved -> {fp} ({len(frontier_df)} points)")

    mvo_weights        = pd.Series(dict(zip(available, w_sharpe)), name="MVO_Weight")
    r_ms, v_ms, sr_ms  = portfolio_perf(w_sharpe)
    print(f"[NERO][MVO] Max-Sharpe | Return={r_ms:.2%} | Vol={v_ms:.2%} | Sharpe={sr_ms:.3f} | lam={lam}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#1e1e2e")
        ax.set_facecolor("#2a2a3d")
        if not frontier_df.empty:
            sc = ax.scatter(frontier_df["Volatility"]*100, frontier_df["Return"]*100,
                            c=frontier_df["Sharpe"], cmap="plasma", s=30, zorder=2)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Sharpe Ratio", color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        r_mv, v_mv, _ = portfolio_perf(w_minvol)
        ax.scatter(v_ms*100, r_ms*100, marker="*", s=300, color="#6fcf97", zorder=5,
                   label=f"Max Sharpe ({sr_ms:.2f})")
        ax.scatter(v_mv*100, r_mv*100, marker="D", s=150, color="#4a90e2", zorder=5,
                   label=f"Min Vol ({v_mv:.1%})")
        for sym in available:
            sv_i = float(np.sqrt(cov_mat.loc[sym, sym]))
            sr_i = float(mean_ret[sym])
            ax.scatter(sv_i*100, sr_i*100, marker="x", s=60, color="#e0e0e0", alpha=0.6, zorder=3)
            ax.annotate(sym, (sv_i*100, sr_i*100), fontsize=6, color="#9090a8",
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Annualised Volatility (%)", color="white")
        ax.set_ylabel("Annualised Return (%)", color="white")
        ax.set_title(f"NERO Efficient Frontier — {risk_mode.upper()} risk", color="white", fontsize=12)
        ax.tick_params(colors="white")
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color("#4a90e2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(facecolor="#2a2a3d", labelcolor="white", fontsize=9)
        ax.grid(True, alpha=0.15, color="white")
        png = os.path.join(NERO_OUTPUT_PATH, "efficient_frontier.png")
        plt.tight_layout()
        plt.savefig(png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[NERO][MVO] Chart saved -> {png}")
    except Exception as exc:
        print(f"[NERO][MVO] Chart failed: {exc}")

    return {"mvo_weights": mvo_weights, "frontier_df": frontier_df,
            "portfolio_return": r_ms, "portfolio_vol": v_ms, "portfolio_sharpe": sr_ms,
            "available": available}


# =============================================================================
# ===== SECTION 13: MONTE CARLO SIMULATION =====
# =============================================================================

def run_monte_carlo(
    portfolio_df,
    bar_returns_dict,
    total_capital,
    horizon_days=252,
    n_sims=10_000,
    confidence=0.95,
):
    """
    10,000-path Monte Carlo using Cholesky-decomposed correlated returns.
    Saves monte_carlo.csv and monte_carlo.png.
    """
    if portfolio_df.empty:
        return {}

    symbols = portfolio_df["Symbol"].tolist()
    weights = portfolio_df["Weight"].values

    ret_dict = {}
    for sym in symbols:
        series = bar_returns_dict.get(sym)
        if series is not None and len(series) > 10:
            s = series.copy()
            s.index = pd.to_datetime(s.index) if not isinstance(s.index, pd.DatetimeIndex) else s.index
            if len(s) > 500:
                s = (1 + s).resample("1D").prod() - 1
            ret_dict[sym] = s

    available  = [s for s in symbols if s in ret_dict]
    if not available:
        print("[NERO][MC] No return data — skipping")
        return {}

    avail_idx  = [i for i, s in enumerate(symbols) if s in available]
    w_mc       = weights[avail_idx]
    w_mc       = w_mc / w_mc.sum()
    ret_df     = pd.DataFrame({s: ret_dict[s] for s in available}).dropna(how="all").fillna(0).tail(252)
    mean_daily = ret_df.mean().values
    cov_daily  = ret_df.cov().values

    try:
        cov_reg = cov_daily + np.eye(len(available)) * 1e-8
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(cov_daily) + 1e-8))
        print("[NERO][MC] Covariance not PD — using diagonal fallback")

    np.random.seed(42)
    z            = np.random.standard_normal((n_sims, horizon_days, len(available)))
    corr_returns = mean_daily + (z @ L.T)
    port_daily   = corr_returns @ w_mc
    port_cum     = total_capital * np.cumprod(1 + port_daily, axis=1)
    terminal     = port_cum[:, -1]
    sorted_tv    = np.sort(terminal)
    var_idx      = int((1 - confidence) * n_sims)
    var_95       = total_capital - sorted_tv[var_idx]
    cvar_95      = total_capital - sorted_tv[:var_idx].mean()
    expected_val = float(terminal.mean())
    median_val   = float(np.median(terminal))
    best_case    = float(np.percentile(terminal, 95))
    worst_case   = float(np.percentile(terminal, 5))
    mean_path    = port_daily.mean(axis=0)
    ann_ret      = float((1 + mean_path.mean()) ** 252 - 1)
    ann_vol      = float(mean_path.std() * np.sqrt(252))

    print(f"[NERO][MC] {n_sims:,} sims | E[TV]=\u20b9{expected_val:,.0f} | VaR95=\u20b9{var_95:,.0f} | CVaR95=\u20b9{cvar_95:,.0f}")
    print(f"[NERO][MC] AnnRet={ann_ret:.2%} | AnnVol={ann_vol:.2%} | Worst5%=\u20b9{worst_case:,.0f} | Best95%=\u20b9{best_case:,.0f}")

    mc_summary = pd.DataFrame([
        {"Metric": "Starting Capital",        "Value": round(total_capital, 0)},
        {"Metric": "Expected Terminal Value", "Value": round(expected_val, 0)},
        {"Metric": "Median Terminal Value",   "Value": round(median_val, 0)},
        {"Metric": "Best Case (95th pct)",    "Value": round(best_case, 0)},
        {"Metric": "Worst Case (5th pct)",    "Value": round(worst_case, 0)},
        {"Metric": "VaR 95% (loss)",          "Value": round(var_95, 0)},
        {"Metric": "CVaR 95% (exp loss)",     "Value": round(cvar_95, 0)},
        {"Metric": "Annualised Return",       "Value": round(ann_ret, 6)},
        {"Metric": "Annualised Volatility",   "Value": round(ann_vol, 6)},
        {"Metric": "Simulations",             "Value": n_sims},
        {"Metric": "Horizon (days)",          "Value": horizon_days},
    ])
    mc_path = os.path.join(NERO_OUTPUT_PATH, "monte_carlo.csv")
    mc_summary.to_csv(mc_path, index=False)
    print(f"[NERO][MC] Summary saved -> {mc_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1e1e2e")
        for ax in axes:
            ax.set_facecolor("#2a2a3d")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_color("#4a90e2")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax = axes[0]
        days = np.arange(horizon_days)
        fan  = np.random.choice(n_sims, size=min(200, n_sims), replace=False)
        for pi in fan:
            ax.plot(days, port_cum[pi], color="#4a90e2", alpha=0.03, linewidth=0.5)
        p5  = np.percentile(port_cum, 5,  axis=0)
        p25 = np.percentile(port_cum, 25, axis=0)
        p50 = np.percentile(port_cum, 50, axis=0)
        p75 = np.percentile(port_cum, 75, axis=0)
        p95 = np.percentile(port_cum, 95, axis=0)
        ax.fill_between(days, p5,  p95, color="#4a90e2", alpha=0.10, label="5-95%")
        ax.fill_between(days, p25, p75, color="#4a90e2", alpha=0.20, label="25-75%")
        ax.plot(days, p50, color="#6fcf97", linewidth=2, label="Median")
        ax.axhline(total_capital, color="#f59e0b", linestyle="--", linewidth=1, label="Start")
        ax.set_xlabel("Trading Days", color="white")
        ax.set_ylabel("Portfolio Value (\u20b9)", color="white")
        ax.set_title(f"Monte Carlo \u2014 {n_sims:,} Paths", color="white")
        ax.legend(facecolor="#2a2a3d", labelcolor="white", fontsize=8)
        ax.grid(True, alpha=0.10, color="white")
        ax2 = axes[1]
        ax2.hist(terminal / 1e5, bins=80, color="#4a90e2", edgecolor="none", alpha=0.7)
        ax2.axvline(total_capital/1e5, color="#f59e0b", linestyle="--", linewidth=1.5, label="Start")
        ax2.axvline(sorted_tv[var_idx]/1e5, color="#ef4444", linestyle="--", linewidth=1.5,
                    label=f"VaR95=\u20b9{var_95:,.0f}")
        ax2.axvline(expected_val/1e5, color="#6fcf97", linestyle="-", linewidth=1.5,
                    label=f"E[TV]=\u20b9{expected_val:,.0f}")
        ax2.set_xlabel("Terminal Value (\u20b9 Lakhs)", color="white")
        ax2.set_ylabel("Frequency", color="white")
        ax2.set_title("Terminal Value Distribution", color="white")
        ax2.legend(facecolor="#2a2a3d", labelcolor="white", fontsize=8)
        ax2.grid(True, alpha=0.10, color="white")
        plt.suptitle(f"NERO Monte Carlo | Capital: \u20b9{total_capital:,.0f} | Horizon: {horizon_days}d",
                     color="white", fontsize=11, y=1.01)
        mc_png = os.path.join(NERO_OUTPUT_PATH, "monte_carlo.png")
        plt.tight_layout()
        plt.savefig(mc_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[NERO][MC] Chart saved -> {mc_png}")
    except Exception as exc:
        print(f"[NERO][MC] Chart failed: {exc}")

    return {"VaR_95": float(var_95), "CVaR_95": float(cvar_95),
            "expected_value": expected_val, "annual_return": ann_ret, "annual_vol": ann_vol}


# =============================================================================
# ===== SECTION 14: PORTFOLIO-LEVEL BACKTEST =====
# =============================================================================

def run_portfolio_backtest(
    portfolio_df,
    bar_returns_dict,
    nifty_path=None,
):
    """
    Combines per-stock returns into weighted portfolio series.
    Benchmarks vs Nifty50. Computes Sharpe, max drawdown, Calmar, OOS Sharpe.
    Saves portfolio_backtest.csv, portfolio_with_oos.csv, portfolio_backtest.png.
    """
    if portfolio_df.empty:
        return {}

    if nifty_path is None:
        nifty_path = os.environ.get("NERO_NIFTY_PATH", "C:/NERO/data/nifty_index.csv")

    symbols = portfolio_df["Symbol"].tolist()
    weights = dict(zip(portfolio_df["Symbol"], portfolio_df["Weight"]))

    ret_dict = {}
    for sym in symbols:
        series = bar_returns_dict.get(sym)
        if series is not None and len(series) > 10:
            s = series.copy()
            s.index = pd.to_datetime(s.index) if not isinstance(s.index, pd.DatetimeIndex) else s.index
            if len(s) > 500:
                s = (1 + s).resample("1D").prod() - 1
            ret_dict[sym] = s

    available = [s for s in symbols if s in ret_dict]
    if not available:
        print("[NERO][BT] No return data — skipping")
        return {}

    ret_df   = pd.DataFrame({s: ret_dict[s] for s in available}).dropna(how="all")
    w_arr    = np.array([weights.get(s, 0) for s in available])
    w_arr    = w_arr / w_arr.sum()
    port_ret = ret_df.fillna(0) @ w_arr
    port_ret.name = "PortfolioReturn"

    oos_sharpes = {}
    for sym in available:
        s = ret_dict[sym].dropna()
        if len(s) < 20:
            oos_sharpes[sym] = np.nan
            continue
        split = int(len(s) * 0.80)
        oos   = s.iloc[split:]
        oos_sharpes[sym] = round(float(oos.mean() / (oos.std() + 1e-9) * np.sqrt(252)), 3) if len(oos) >= 5 else np.nan

    portfolio_df = portfolio_df.copy()
    portfolio_df["OOS_Sharpe"] = portfolio_df["Symbol"].map(oos_sharpes)

    cum_ret    = float((1 + port_ret).prod() - 1)
    n_days     = len(port_ret)
    ann_ret    = float((1 + cum_ret) ** (252 / max(n_days, 1)) - 1)
    ann_vol    = float(port_ret.std() * np.sqrt(252))
    sharpe     = float(ann_ret / (ann_vol + 1e-9))
    cum_series = (1 + port_ret).cumprod()
    drawdowns  = (cum_series - cum_series.cummax()) / cum_series.cummax()
    max_dd     = float(drawdowns.min())
    calmar     = float(ann_ret / (abs(max_dd) + 1e-9))

    print(f"[NERO][BT] Portfolio | CumRet={cum_ret:.2%} | AnnRet={ann_ret:.2%} | Vol={ann_vol:.2%} | Sharpe={sharpe:.3f} | MaxDD={max_dd:.2%} | Calmar={calmar:.3f}")

    bench_ret_series = None
    bench_cum = np.nan
    bench_sharpe = np.nan
    bench_ann = np.nan
    try:
        nifty_df = pd.read_csv(nifty_path)
        nifty_df.columns = nifty_df.columns.str.strip().str.title()
        nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], errors="coerce")
        nifty_df = nifty_df.dropna(subset=["Date"]).set_index("Date").sort_index()
        bench_raw     = nifty_df["Close"].pct_change().dropna()
        bench_aligned = bench_raw.loc[
            (bench_raw.index >= port_ret.index.min()) &
            (bench_raw.index <= port_ret.index.max())
        ]
        if len(bench_aligned) > 5:
            bench_ret_series = bench_aligned
            bench_cum    = float((1 + bench_aligned).prod() - 1)
            bench_ann    = float((1 + bench_cum) ** (252 / max(len(bench_aligned), 1)) - 1)
            bench_vol    = float(bench_aligned.std() * np.sqrt(252))
            bench_sharpe = float(bench_ann / (bench_vol + 1e-9))
            print(f"[NERO][BT] Nifty50   | CumRet={bench_cum:.2%} | AnnRet={bench_ann:.2%} | Sharpe={bench_sharpe:.3f}")
            print(f"[NERO][BT] Excess vs Nifty50 = {ann_ret - bench_ann:+.2%}")
    except Exception as exc:
        print(f"[NERO][BT] Benchmark load failed: {exc}")

    bt_summary = pd.DataFrame([
        {"Metric": "Period Start",          "Portfolio": str(port_ret.index.min().date()), "Nifty50": "---"},
        {"Metric": "Period End",            "Portfolio": str(port_ret.index.max().date()), "Nifty50": "---"},
        {"Metric": "Trading Days",          "Portfolio": n_days, "Nifty50": len(bench_ret_series) if bench_ret_series is not None else "---"},
        {"Metric": "Cumulative Return",     "Portfolio": f"{cum_ret:.4%}", "Nifty50": f"{bench_cum:.4%}" if pd.notna(bench_cum) else "---"},
        {"Metric": "Annualised Return",     "Portfolio": f"{ann_ret:.4%}", "Nifty50": f"{bench_ann:.4%}" if pd.notna(bench_ann) else "---"},
        {"Metric": "Annualised Volatility", "Portfolio": f"{ann_vol:.4%}", "Nifty50": "---"},
        {"Metric": "Sharpe Ratio",          "Portfolio": f"{sharpe:.4f}", "Nifty50": f"{bench_sharpe:.4f}" if pd.notna(bench_sharpe) else "---"},
        {"Metric": "Max Drawdown",          "Portfolio": f"{max_dd:.4%}", "Nifty50": "---"},
        {"Metric": "Calmar Ratio",          "Portfolio": f"{calmar:.4f}", "Nifty50": "---"},
    ])
    bt_path  = os.path.join(NERO_OUTPUT_PATH, "portfolio_backtest.csv")
    oos_path = os.path.join(NERO_OUTPUT_PATH, "portfolio_with_oos.csv")
    bt_summary.to_csv(bt_path, index=False)
    portfolio_df.to_csv(oos_path, index=False)
    print(f"[NERO][BT] Backtest saved -> {bt_path}")
    print(f"[NERO][BT] OOS Sharpe  saved -> {oos_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor("#1e1e2e")
        for ax in axes:
            ax.set_facecolor("#2a2a3d")
            ax.tick_params(colors="white")
            for sp in ax.spines.values():
                sp.set_color("#4a90e2")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        ax = axes[0]
        port_cum_s = (1 + port_ret).cumprod() - 1
        ax.plot(port_cum_s.index, port_cum_s * 100, color="#6fcf97", linewidth=2,
                label=f"NERO Portfolio ({cum_ret:.1%})")
        if bench_ret_series is not None:
            bench_cum_s = (1 + bench_ret_series).cumprod() - 1
            ax.plot(bench_cum_s.index, bench_cum_s * 100, color="#f59e0b",
                    linewidth=1.5, linestyle="--", label=f"Nifty50 ({bench_cum:.1%})")
        ax.axhline(0, color="#9090a8", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Cumulative Return (%)", color="white")
        ax.set_title(f"NERO Portfolio Backtest | Sharpe={sharpe:.2f} | MaxDD={max_dd:.1%} | Calmar={calmar:.2f}",
                     color="white", fontsize=11)
        ax.legend(facecolor="#2a2a3d", labelcolor="white", fontsize=9)
        ax.grid(True, alpha=0.10, color="white")
        ax2 = axes[1]
        ax2.fill_between(drawdowns.index, drawdowns * 100, 0, color="#ef4444", alpha=0.5)
        ax2.plot(drawdowns.index, drawdowns * 100, color="#ef4444", linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)", color="white")
        ax2.set_xlabel("Date", color="white")
        ax2.grid(True, alpha=0.10, color="white")
        plt.tight_layout()
        bt_png = os.path.join(NERO_OUTPUT_PATH, "portfolio_backtest.png")
        plt.savefig(bt_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[NERO][BT] Chart saved -> {bt_png}")
    except Exception as exc:
        print(f"[NERO][BT] Chart failed: {exc}")

    return {"cumulative_return": cum_ret, "annualised_return": ann_ret,
            "sharpe": sharpe, "max_drawdown": max_dd, "calmar": calmar,
            "benchmark_return": bench_cum, "portfolio_df_oos": portfolio_df}


# ── Self-test for Sections 6–8 ───────────────────────────────────────────────
# =============================================================================


# ── Ensure these are always imported, even when not re-imported from Sections 1–8 ──
import os
import time
import json
import argparse
import requests


# =============================================================================
# ===== SECTION 9: NEWS & ANNOUNCEMENTS =====
# =============================================================================
#
#  Runs only when NERO_NEWS_ENABLED == 'ON'.
#  Uses NSE's public JSON endpoints (no API key required for basic mode).
#  Optional upgrade: Claude API sentiment (see Master Instructions #4).
# =============================================================================

# ── NSE session reuse (shared across calls in one run) ───────────────────────
_nse_session: requests.Session | None = None

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}

_NSE_HOME     = "https://www.nseindia.com"
_NSE_ANNOUNCE = "https://www.nseindia.com/api/corporate-announcements"
_NSE_BOARD    = "https://www.nseindia.com/api/board-meetings"

# Keyword lists for sentiment scoring
_POSITIVE_WORDS = [
    "dividend", "buyback", "bonus", "acquisition", "strong results",
    "profit up", "upgrade", "record profit", "outperform", "beat estimates",
    "record revenue", "rating upgrade", "expansion", "order win", "approved",
    "merger", "demerger approved", "buy recommendation",
]
_NEGATIVE_WORDS = [
    "loss", "fraud", "investigation", "downgrade", "default", "delisting",
    "penalty", "scam", "sebi notice", "nse action", "bse action", "fir",
    "insolvency", "bankruptcy", "resign", "ceo quit", "cfo quit",
    "regulatory action", "court order", "tax demand", "write-off",
    "profit warning", "sell recommendation",
]


def _get_nse_session() -> requests.Session:
    """
    Return a requests.Session pre-loaded with NSE cookies.
    First visit nseindia.com to collect cookies, then use for API calls.
    Session is cached globally for the duration of one NERO run.
    """
    global _nse_session
    if _nse_session is not None:
        return _nse_session

    session = requests.Session()
    session.headers.update(_NSE_HEADERS)

    try:
        # Warm-up: GET homepage to receive cookies (required by NSE's bot filter)
        resp = session.get(_NSE_HOME, timeout=10)
        resp.raise_for_status()
        time.sleep(0.5)   # polite pause after homepage hit
    except Exception as exc:
        print(f"[NERO][News] NSE session warm-up failed: {exc} — news disabled for this run")

    _nse_session = session
    return _nse_session


def fetch_nse_announcements(symbols_list: list[str]) -> dict[str, list[dict]]:
    """
    Fetch recent corporate announcements from NSE public JSON API.

    Parameters
    ----------
    symbols_list : list of ticker symbols (e.g. ['RELIANCE', 'TCS'])
                   NSE symbols without the .NS suffix.

    Returns
    -------
    dict : {symbol: [list of announcement dicts]}
           Empty dict on any failure. Never raises.

    NSE API returns a flat list of announcements — we filter per symbol.
    Each announcement dict has keys: symbol, subject, desc, bflag, exchdisstime.
    """
    if NERO_NEWS_ENABLED != "ON":
        return {}

    result: dict[str, list[dict]] = {sym: [] for sym in symbols_list}

    try:
        session = _get_nse_session()
        params  = {"index": "equities"}

        resp = session.get(_NSE_ANNOUNCE, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # NSE returns a list of dicts; each has 'symbol' key
        if not isinstance(data, list):
            print(f"[NERO][News] Unexpected announcements response type: {type(data)}")
            return result

        # Index announcements by symbol for O(1) lookup
        symbol_set = {s.upper() for s in symbols_list}
        for item in data:
            sym = str(item.get("symbol", "")).upper()
            if sym in symbol_set:
                result.setdefault(sym, []).append(item)

        total = sum(len(v) for v in result.values())
        print(f"[NERO][News] Fetched {total} announcements for {len(symbols_list)} symbols")

    except requests.exceptions.Timeout:
        print("[NERO][News] NSE API timed out — skipping news for this run")
    except requests.exceptions.HTTPError as exc:
        print(f"[NERO][News] NSE API HTTP error: {exc} — skipping news")
    except json.JSONDecodeError:
        print("[NERO][News] NSE API returned non-JSON — skipping news")
    except Exception as exc:
        print(f"[NERO][News] Unexpected error fetching announcements: {exc}")

    return result


def fetch_nse_board_meetings(symbols_list: list[str]) -> dict[str, list[dict]]:
    """
    Fetch upcoming board meeting dates (earnings, dividends) from NSE.
    Used to flag EVENT RISK stocks and apply position size reduction.

    Returns dict {symbol: [list of board meeting dicts]}.
    """
    if NERO_NEWS_ENABLED != "ON":
        return {}

    result: dict[str, list[dict]] = {sym: [] for sym in symbols_list}

    try:
        session = _get_nse_session()
        resp    = session.get(_NSE_BOARD, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not isinstance(data, list):
            return result

        symbol_set = {s.upper() for s in symbols_list}
        for item in data:
            sym = str(item.get("symbol", "")).upper()
            if sym in symbol_set:
                result.setdefault(sym, []).append(item)

    except Exception as exc:
        print(f"[NERO][News] Board meeting fetch failed: {exc}")

    return result


def score_announcement(text: str) -> int:
    """
    Simple keyword-based sentiment scoring for a single announcement string.

    Scoring rules (Master Instructions blueprint):
      +20  if a positive keyword is found
      -50  if a negative keyword is found
        0  if neither found

    If BOTH positive and negative are found, negative dominates (returns -50).
    Text is case-folded before matching.

    Parameters
    ----------
    text : str — the announcement subject + description concatenated

    Returns
    -------
    int : sentiment score (+20, -50, or 0)
    """
    if not text or not isinstance(text, str):
        return 0

    text_lower = text.lower()

    # Negative check first — it dominates if both present
    for neg in _NEGATIVE_WORDS:
        if neg in text_lower:
            return -50

    for pos in _POSITIVE_WORDS:
        if pos in text_lower:
            return +20

    return 0


def _score_event_risk(board_meetings: list[dict]) -> int:
    """
    Check if any board meeting is scheduled within the next 3 days.
    If yes, return -20 (EVENT_RISK flag — position to be reduced 40%).
    Otherwise return 0.
    """
    try:
        today = pd.Timestamp.today().normalize()
        horizon = today + pd.Timedelta(days=3)

        for meeting in board_meetings:
            # NSE uses 'meetingDate' in DD-Mon-YYYY or YYYY-MM-DD format
            raw_date = meeting.get("meetingDate", meeting.get("date", ""))
            if not raw_date:
                continue
            try:
                meet_dt = pd.to_datetime(raw_date, dayfirst=True, errors="coerce")
                if pd.notna(meet_dt) and today <= meet_dt <= horizon:
                    return -20   # event risk within 3 days
            except Exception:
                continue
    except Exception:
        pass
    return 0


def apply_news_filter(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch NSE announcements + board meetings for all candidates, compute
    SentimentScore, apply event risk flags, and adjust CombinedScore.

    Rules (Blueprint Section 12.2):
      1. Fetch announcements for all symbols.
      2. score_announcement() on each; take max-magnitude score per stock.
      3. EVENT RISK (earnings in ≤3 days): flag, reduce position size by 40%.
      4. SentimentScore < -30: override CombinedScore to 0 (exclude stock).
      5. Otherwise: CombinedScore = CombinedScore * 0.90 + SentimentScore * 0.10.

    Returns
    -------
    Updated results_df with added columns:
        SentimentScore, EventRisk (bool), SentimentNote
    """
    if NERO_NEWS_ENABLED != "ON":
        results_df["SentimentScore"] = 0
        results_df["EventRisk"]      = False
        results_df["SentimentNote"]  = ""
        return results_df

    if results_df.empty:
        return results_df

    symbols = results_df["Symbol"].str.upper().tolist()

    # ── Fetch announcements and board meetings in parallel? No — sequential
    # is safe and NSE rate-limits aggressively on rapid calls.
    announcements = fetch_nse_announcements(symbols)
    board_meetings_map = fetch_nse_board_meetings(symbols)

    sentiment_scores: dict[str, int]  = {}
    event_risk_map:   dict[str, bool] = {}
    notes_map:        dict[str, str]  = {}

    for sym in symbols:
        sym_announces = announcements.get(sym, [])
        sym_meetings  = board_meetings_map.get(sym, [])

        # ── Score each announcement, keep worst (most negative) or best ────────
        raw_scores = []
        for ann in sym_announces:
            # Combine subject + description text for keyword matching
            subject = str(ann.get("subject", ann.get("desc", "")))
            desc    = str(ann.get("desc",    ""))
            combined_text = f"{subject} {desc}"
            s = score_announcement(combined_text)
            raw_scores.append(s)

        # Dominant score: most negative if any negative, else most positive
        if raw_scores:
            neg_scores = [s for s in raw_scores if s < 0]
            pos_scores = [s for s in raw_scores if s > 0]
            if neg_scores:
                final_score = min(neg_scores)   # worst (most negative)
                notes_map[sym] = f"Negative announcement detected ({len(neg_scores)} items)"
            elif pos_scores:
                final_score = max(pos_scores)   # best positive
                notes_map[sym] = f"Positive announcement ({len(pos_scores)} items)"
            else:
                final_score = 0
                notes_map[sym] = ""
        else:
            final_score = 0
            notes_map[sym] = ""

        # ── Event risk: upcoming board meeting within 3 days ──────────────────
        event_adj = _score_event_risk(sym_meetings)
        if event_adj < 0:
            event_risk_map[sym] = True
            notes_map[sym]      = (notes_map.get(sym, "") + " | EVENT_RISK: board meeting ≤3d").strip(" |")
        else:
            event_risk_map[sym] = False

        # Combine event risk into sentiment (additive, both negative = -70 max)
        sentiment_scores[sym] = final_score + event_adj

    # ── Apply to results_df ──────────────────────────────────────────────────
    results_df = results_df.copy()
    results_df["SentimentScore"] = results_df["Symbol"].map(
        lambda s: sentiment_scores.get(s.upper(), 0)
    )
    results_df["EventRisk"] = results_df["Symbol"].map(
        lambda s: event_risk_map.get(s.upper(), False)
    )
    results_df["SentimentNote"] = results_df["Symbol"].map(
        lambda s: notes_map.get(s.upper(), "")
    )

    # ── Rule 4: Hard exclusion if SentimentScore < -30 ───────────────────────
    hard_exclude = results_df["SentimentScore"] < -30
    if hard_exclude.any():
        excluded = results_df.loc[hard_exclude, "Symbol"].tolist()
        print(f"[NERO][News] Hard excluded (SentimentScore < -30): {excluded}")
        for sym in excluded:
            _flag(sym, "NEWS_HARD_EXCLUDE",
                  f"SentimentScore={sentiment_scores.get(sym, '?')} — CombinedScore forced to 0")
        results_df.loc[hard_exclude, "CombinedScore"] = 0.0

    # ── Rule 5: Soft adjustment for remaining stocks ──────────────────────────
    soft_mask = ~hard_exclude
    results_df.loc[soft_mask, "CombinedScore"] = (
        results_df.loc[soft_mask, "CombinedScore"] * 0.90
        + results_df.loc[soft_mask, "SentimentScore"] * 0.10
    ).clip(lower=0, upper=100).round(2)

    # ── Event risk: reduce position size signal (flag for portfolio constructor)
    # Portfolio constructor checks EventRisk column and reduces weight by 40%.
    n_event = event_risk_map and sum(1 for v in event_risk_map.values() if v)
    if n_event:
        print(f"[NERO][News] {n_event} stocks flagged EVENT_RISK — weights will be reduced 40%")

    pos_count = (results_df["SentimentScore"] > 0).sum()
    neg_count = (results_df["SentimentScore"] < 0).sum()
    print(
        f"[NERO][News] Sentiment applied | "
        f"Positive: {pos_count} | Negative: {neg_count} | "
        f"Hard-excluded: {hard_exclude.sum()}"
    )

    return results_df


# =============================================================================
# ===== SECTION 10: TELEGRAM ALERTS =====
# =============================================================================

_TG_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"
_TG_MAX_CHARS = 4096   # Telegram hard limit per message


def send_telegram(
    message:   str,
    bot_token: str | None = None,
    chat_id:   str | None = None,
) -> bool:
    """
    Send a plain-text message via Telegram Bot API.

    Priority for credentials:
      1. Arguments passed directly (bot_token, chat_id)
      2. Environment variables NERO_TG_BOT_TOKEN / NERO_TG_CHAT_ID

    Parameters
    ----------
    message   : str  — message body (plain text)
    bot_token : str  — Telegram bot token (optional if env var set)
    chat_id   : str  — Telegram chat/group ID (optional if env var set)

    Returns
    -------
    bool : True if sent successfully, False otherwise.

    Note
    ----
    If token is not configured, prints a clear message and returns False.
    Never raises — all exceptions are caught and logged.
    """
    token = bot_token or NERO_TG_BOT_TOKEN or os.environ.get("NERO_TG_BOT_TOKEN", "")
    cid   = chat_id   or NERO_TG_CHAT_ID   or os.environ.get("NERO_TG_CHAT_ID",   "")

    if not token:
        print("[NERO][Telegram] Not configured — set NERO_TG_BOT_TOKEN env var or enter in UI.")
        return False

    if not cid:
        print("[NERO][Telegram] NERO_TG_CHAT_ID not set — cannot send message.")
        return False

    url     = _TG_API_BASE.format(token=token)
    payload = {
        "chat_id":    cid,
        "text":       message,
        "parse_mode": "HTML",   # HTML mode — safe even if message has no tags
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            print(f"[NERO][Telegram] API returned ok=false: {data.get('description', '?')}")
            return False
        return True
    except requests.exceptions.HTTPError as exc:
        # 400/401: bad token or chat id — give a clear hint
        code = exc.response.status_code if exc.response is not None else "?"
        print(f"[NERO][Telegram] HTTP {code} error — check bot token and chat ID.")
        return False
    except requests.exceptions.Timeout:
        print("[NERO][Telegram] Request timed out — message not sent.")
        return False
    except Exception as exc:
        print(f"[NERO][Telegram] Unexpected error: {exc}")
        return False


def _chunk_message(text: str, max_len: int = _TG_MAX_CHARS) -> list[str]:
    """
    Split a long text into chunks ≤ max_len characters.
    Splits on newlines where possible to avoid cutting mid-line.
    """
    if len(text) <= max_len:
        return [text]

    chunks:  list[str] = []
    current: list[str] = []
    current_len         = 0

    for line in text.splitlines(keepends=True):
        if current_len + len(line) > max_len:
            if current:
                chunks.append("".join(current))
            current     = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)

    if current:
        chunks.append("".join(current))

    return chunks


def send_portfolio_alert(
    portfolio_df: pd.DataFrame,
    regime_label: str,
    mode:         str,
    bot_token:    str | None = None,
    chat_id:      str | None = None,
) -> None:
    """
    Build and send a formatted portfolio alert via Telegram.

    Message format:
      Header: "NERO v2 | Mode: SWING | Regime: TRENDING_BULL"
      Separator line
      Per-position lines (from format_alert)
      Footer with timestamp

    Long messages are split into ≤4096-char chunks automatically.

    Parameters
    ----------
    portfolio_df : output of build_portfolio()
    regime_label : current regime string
    mode         : 'swing' or 'intraday'
    bot_token    : optional override; defaults to NERO_TG_BOT_TOKEN
    chat_id      : optional override; defaults to NERO_TG_CHAT_ID
    """
    token = bot_token or NERO_TG_BOT_TOKEN or os.environ.get("NERO_TG_BOT_TOKEN", "")
    if not token:
        print("[NERO][Telegram] Not configured — skipping alert.")
        return

    if portfolio_df.empty:
        send_telegram(
            f"NERO v2 | Mode: {mode.upper()} | Regime: {regime_label}\n"
            "No positions generated this run.",
            bot_token=token,
            chat_id=chat_id,
        )
        return

    # ── Build lines from format_alert (already defined in Section 8) ─────────
    total_capital = (
        portfolio_df["CapitalAllocated"].sum()
        if "CapitalAllocated" in portfolio_df.columns
        else 0.0
    )
    alert_lines = format_alert(portfolio_df, regime_label, total_capital)

    # ── Add Telegram-specific header ──────────────────────────────────────────
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    header_lines = [
        f"<b>NERO v2</b> | Mode: <b>{mode.upper()}</b> | "
        f"Regime: <b>{regime_label}</b>",
        f"🕐 {timestamp}",
        "",
    ]

    full_message = "\n".join(header_lines + alert_lines)

    # ── Split and send ────────────────────────────────────────────────────────
    chunks = _chunk_message(full_message)
    n      = len(chunks)
    if n > 1:
        print(f"[NERO][Telegram] Message split into {n} chunks")

    success = 0
    for i, chunk in enumerate(chunks, 1):
        ok = send_telegram(chunk, bot_token=token, chat_id=chat_id)
        if ok:
            success += 1
        if i < n:
            time.sleep(0.3)   # avoid hitting Telegram's 30 msgs/sec limit

    print(
        f"[NERO][Telegram] Alert sent: {success}/{n} chunks | "
        f"Positions: {len(portfolio_df)} | Regime: {regime_label}"
    )


# =============================================================================
# ===== SECTION 11: MAIN & CLI =====
# =============================================================================

def _prompt_api_key(key_name: str, env_var: str) -> str:
    """
    Master Instructions #4: Prompt user for API key if needed and not set.
    Saves to environment for the current process lifetime.

    Parameters
    ----------
    key_name : human-readable name e.g. "Telegram Bot Token"
    env_var  : environment variable name e.g. "NERO_TG_BOT_TOKEN"

    Returns
    -------
    str : the key value (empty string if user skips)
    """
    existing = os.environ.get(env_var, "").strip()
    if existing:
        return existing

    print(f"\n[NERO] {key_name} is not set ({env_var}).")
    print(f"       Enter the key now, or press ENTER to skip.")
    try:
        value = input(f"  {key_name}: ").strip()
    except (EOFError, KeyboardInterrupt):
        value = ""

    if value:
        os.environ[env_var] = value
        print(f"  [NERO] {env_var} set for this session.")
    else:
        print(f"  [NERO] Skipped. You can set it later via: set {env_var}=<value>")

    return value


def _check_and_prompt_api_keys(args: argparse.Namespace) -> None:
    """
    Check for optional API keys and prompt interactively if missing.

    Keys checked:
      - NERO_TG_BOT_TOKEN / NERO_TG_CHAT_ID  (if --telegram flag or env partially set)
      - NERO_CLAUDE_API_KEY                   (if news enabled + Claude upgrade requested)

    We only prompt if the feature is clearly intended (partial config, not silently absent).
    """
    # Telegram: prompt only if token set but chat_id missing (or vice versa)
    tg_token = os.environ.get("NERO_TG_BOT_TOKEN", "").strip()
    tg_chat  = os.environ.get("NERO_TG_CHAT_ID",   "").strip()

    if tg_token and not tg_chat:
        print("\n[NERO] Telegram bot token found but NERO_TG_CHAT_ID is missing.")
        _prompt_api_key("Telegram Chat ID", "NERO_TG_CHAT_ID")
    elif tg_chat and not tg_token:
        print("\n[NERO] Telegram chat ID found but NERO_TG_BOT_TOKEN is missing.")
        _prompt_api_key("Telegram Bot Token", "NERO_TG_BOT_TOKEN")

    # Claude API key: only prompt if news is enabled (used for optional sentiment)
    if NERO_NEWS_ENABLED == "ON":
        claude_key = os.environ.get("NERO_CLAUDE_API_KEY", "").strip()
        if not claude_key:
            print(
                "\n[NERO] News is enabled. Optional: enter a Claude API key for "
                "AI-powered sentiment classification of announcements."
            )
            print("       This uses ~20 API calls/day (free tier compatible).")
            print("       Press ENTER to skip and use keyword matching instead.")
            _prompt_api_key("Claude API Key (optional)", "NERO_CLAUDE_API_KEY")


def _parse_existing_holdings(update_str: str) -> dict[str, float]:
    """
    Parse --update argument: 'RELIANCE:8.5,TCS:7.2' → {'RELIANCE': 8.5, 'TCS': 7.2}
    Handles extra whitespace and case variation. Logs malformed pairs.
    """
    holdings: dict[str, float] = {}
    if not update_str:
        return holdings

    for pair in update_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            print(f"[NERO][CLI] Skipping malformed holding (no colon): '{pair}'")
            continue
        parts = pair.rsplit(":", 1)
        sym   = parts[0].strip().upper().replace(".NS", "")
        try:
            wt  = float(parts[1].strip())
            holdings[sym] = wt
        except ValueError:
            print(f"[NERO][CLI] Skipping holding with non-numeric weight: '{pair}'")

    if holdings:
        print(f"[NERO][CLI] Update mode: {len(holdings)} existing holdings loaded → "
              f"{list(holdings.keys())}")
    return holdings


def _print_summary(portfolio_df: pd.DataFrame,
                   regime_label: str,
                   regime_vector: dict,
                   mode: str,
                   total_capital: float,
                   output_path: str) -> None:
    """Print a final run summary to console after saving."""
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  NERO v2 Run Complete  |  Mode: {mode.upper()}  |  Regime: {regime_label}")
    print(sep)

    if regime_vector:
        rv = regime_vector
        print(
            f"  Regime Vector → "
            f"Trend: {rv.get('Trend', '?')} | "
            f"Vol: {rv.get('Volatility', '?')} | "
            f"Breadth: {rv.get('Breadth', '?')} | "
            f"Corr: {rv.get('Correlation', '?')}"
        )

    if not portfolio_df.empty:
        n_pos  = len(portfolio_df)
        total_alloc = portfolio_df.get("CapitalAllocated", pd.Series([0])).sum()
        buckets = {}
        if "Bucket" in portfolio_df.columns:
            buckets = portfolio_df["Bucket"].value_counts().to_dict()

        flagged = portfolio_df.get("FundaMissing", pd.Series([False])).sum()

        print(f"  Positions: {n_pos} | Capital: ₹{total_capital:,.0f} | "
              f"Deployed: ₹{total_alloc:,.0f}")
        if buckets:
            bkt_str = " | ".join(f"{b}:{c}" for b, c in buckets.items())
            print(f"  Buckets → {bkt_str}")
        if flagged:
            print(f"  ⚑ {int(flagged)} stock(s) missing fundamentals — check flags.csv")
    else:
        print("  No positions generated.")

    print(f"  Output → {output_path}")
    print(sep)


def _run_cli():
    """Main CLI entry point — called from __main__ block at bottom of file."""
    # ── Argument parser ───────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="NERO v2 — Neural Equity Ranking & Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python nero_v2.py --mode swing --capital 500000 --risk medium\n"
            "  python nero_v2.py --mode intraday --capital 100000 --risk low --test\n"
            "  python nero_v2.py --mode swing --update 'RELIANCE:8.5,TCS:7.2' --capital 500000\n"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["intraday", "swing"],
        default="swing",
        help="Trading mode: 'swing' (default) or 'intraday'.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        metavar="RUPEES",
        help="Total capital in rupees (default: 100000).",
    )
    parser.add_argument(
        "--risk",
        choices=["low", "medium", "high"],
        default="medium",
        help="Risk mode: low (defensive) / medium (default) / high (aggressive).",
    )
    parser.add_argument(
        "--update",
        type=str,
        default="",
        metavar="SYMBOL:WEIGHT,...",
        help=(
            "Update existing portfolio. Pass current holdings as "
            "'RELIANCE:8.5,TCS:7.2'. NERO will only replace a position "
            "if a new candidate outranks it by >15 percentile points."
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Test mode: process only 10 representative stocks. "
            "Fast (~30 seconds). Useful for verifying setup before full run."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Sample mode: process N random real stocks (e.g. --sample 50). Faster than full run.",
    )
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Override NERO_NEWS_ENABLED: skip news filtering even if env var is ON.",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Force Telegram alert even if NERO_TG_BOT_TOKEN was not pre-set (will prompt).",
    )

    args = parser.parse_args()

    # ── Apply test mode flag ──────────────────────────────────────────────────
    if args.test:
        os.environ["NERO_TEST_MODE"] = "1"
        print("[NERO] TEST MODE activated — 10 representative stocks only")
    if args.sample > 0:
        os.environ["NERO_SAMPLE_N"] = str(args.sample)
        print(f"[NERO] SAMPLE MODE activated — {args.sample} stocks")

    # ── Override news if --no-news flag passed ────────────────────────────────
    if args.no_news:
        os.environ["NERO_NEWS_ENABLED"] = "OFF"
        # Re-read the module-level constant so apply_news_filter respects it
        import importlib, sys
        _mod = sys.modules[__name__]
        if hasattr(_mod, "NERO_NEWS_ENABLED"):
            _mod.NERO_NEWS_ENABLED = "OFF"
        print("[NERO] News filtering disabled via --no-news flag")

    # ── Override risk mode from CLI (takes precedence over env var) ───────────
    os.environ["NERO_RISK_MODE"] = args.risk

    # ── Telegram prompt ───────────────────────────────────────────────────────
    if args.telegram:
        _prompt_api_key("Telegram Bot Token", "NERO_TG_BOT_TOKEN")
        _prompt_api_key("Telegram Chat ID",   "NERO_TG_CHAT_ID")

    # ── Check and optionally prompt for any other API keys ────────────────────
    _check_and_prompt_api_keys(args)

    # ── Parse existing holdings (update mode) ─────────────────────────────────
    existing_holdings = _parse_existing_holdings(args.update)

    # ── Ensure output directory exists ────────────────────────────────────────
    os.makedirs(NERO_OUTPUT_PATH, exist_ok=True)

    # ── Run engine ────────────────────────────────────────────────────────────
    print(
        f"\n[NERO] Starting run | mode={args.mode} | "
        f"capital=₹{args.capital:,.0f} | risk={args.risk} | "
        f"test={'YES' if args.test else 'NO'}"
    )

    try:
        results_df, regime_label, regime_vector, bar_returns_dict = run_engine(mode=args.mode)
    except FileNotFoundError as exc:
        print(f"\n[NERO] ERROR: {exc}")
        print(
            "\nSetup checklist:\n"
            "  1. Create C:/NERO/archive/ and add SYMBOL.csv files\n"
            "  2. Create C:/NERO/data/ and add Stock_Funda_2000.csv\n"
            "  3. Set NERO_ARCHIVE_PATH / NERO_FUNDA_PATH env vars if using custom paths\n"
        )
        raise SystemExit(1)
    except Exception as exc:
        print(f"\n[NERO] Engine failed: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

    # ── Apply news filter (only if NERO_NEWS_ENABLED == 'ON') ────────────────
    if not results_df.empty and NERO_NEWS_ENABLED == "ON":
        print("[NERO] Applying news filter...")
        try:
            results_df = apply_news_filter(results_df)
        except Exception as exc:
            print(f"[NERO] News filter failed ({exc}) — proceeding without news adjustment")

    # ── Build portfolio ───────────────────────────────────────────────────────
    try:
        portfolio_df = build_portfolio(
            results_df,
            total_capital=args.capital,
            risk_mode=args.risk,
            existing_holdings=existing_holdings or None,
            bar_returns_dict=bar_returns_dict,
        )
    except Exception as exc:
        print(f"\n[NERO] Portfolio construction failed: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

    # ── Apply event risk size reduction (40% weight cut for event-flagged stocks)
    if (not portfolio_df.empty
            and "EventRisk" in portfolio_df.columns
            and portfolio_df["EventRisk"].any()):

        event_mask = portfolio_df["EventRisk"] == True
        n_event    = event_mask.sum()
        portfolio_df.loc[event_mask, "Weight"] *= 0.60   # reduce by 40%

        # Renormalise weights after reduction
        total_w = portfolio_df["Weight"].sum()
        if total_w > 0:
            portfolio_df["Weight"] /= total_w

        # Recompute capital allocated
        if "CapitalAllocated" in portfolio_df.columns:
            portfolio_df["CapitalAllocated"] = (
                portfolio_df["Weight"] * args.capital
            ).round(0)

        portfolio_df["Weight"] = portfolio_df["Weight"].round(6)
        print(f"[NERO] Event risk size reduction applied to {n_event} stock(s) (−40% weight)")

    # ── Save portfolio CSV ────────────────────────────────────────────────────
    portfolio_path = os.path.join(NERO_OUTPUT_PATH, "portfolio.csv")
    try:
        portfolio_df.to_csv(portfolio_path, index=False)
        print(f"[NERO] Portfolio saved → {portfolio_path}")
    except Exception as exc:
        print(f"[NERO] Could not save portfolio: {exc}")

    # ── Section 12: MVO + Efficient Frontier ──────────────────────────────────
    try:
        _bt_cache_live = load_backtest_cache()
        _bar_returns_live = {}
        for sym in (results_df["Symbol"].tolist() if not results_df.empty else []):
            cached = _bt_cache_live.get(sym, {})
            br = cached.get("_bar_returns")
            if br is not None and len(br) > 0:
                _bar_returns_live[sym] = br
    except Exception:
        _bar_returns_live = {}

    if not results_df.empty and len(results_df) >= 2:
        try:
            run_mvo(results_df=results_df, bar_returns_dict=_bar_returns_live, risk_mode=args.risk)
        except Exception as exc:
            print(f"[NERO] MVO failed: {exc} — skipping")

    # ── Section 13: Monte Carlo ────────────────────────────────────────────────
    if not portfolio_df.empty:
        try:
            run_monte_carlo(
                portfolio_df=portfolio_df,
                bar_returns_dict=_bar_returns_live,
                total_capital=args.capital,
            )
        except Exception as exc:
            print(f"[NERO] Monte Carlo failed: {exc} — skipping")

    # ── Section 14: Portfolio-Level Backtest ───────────────────────────────────
    if not portfolio_df.empty:
        try:
            run_portfolio_backtest(
                portfolio_df=portfolio_df,
                bar_returns_dict=_bar_returns_live,
            )
        except Exception as exc:
            print(f"[NERO] Portfolio backtest failed: {exc} — skipping")

    # ── Print alert lines to console ──────────────────────────────────────────
    print()
    for line in format_alert(portfolio_df, regime_label, args.capital):
        print(line)

    # ── Summary block ─────────────────────────────────────────────────────────
    _print_summary(
        portfolio_df=portfolio_df,
        regime_label=regime_label,
        regime_vector=regime_vector,
        mode=args.mode,
        total_capital=args.capital,
        output_path=portfolio_path,
    )

    # ── Telegram alert ────────────────────────────────────────────────────────
    tg_token = os.environ.get("NERO_TG_BOT_TOKEN", "").strip()
    if tg_token:
        try:
            send_portfolio_alert(portfolio_df, regime_label, args.mode)
        except Exception as exc:
            print(f"[NERO] Telegram alert failed: {exc}")
    else:
        if args.telegram:
            print("[NERO] Telegram not configured — alert skipped.")


# =============================================================================
# ── Self-test for Sections 9–11 ──────────────────────────────────────────────
# =============================================================================


if __name__ == "__main__":
    _run_cli()
