"""
Build MA & 60min features aligned to P_60min datetime anchors.

Columns produced
----------------
── MA (马来棕榈油) ──────────────────────────────────────────────
  ma_open / ma_high / ma_low / ma_close / ma_close_ffill / ma_tick_count
  x_ma_bar_ret          : MA return within the *same* P bar window
  x_ma_ret_{h}          : MA close-to-close lookback (h ∈ HORIZONS)
  x_p_ret_{h}           : P  close-to-close lookback
  x_spread_{h}          : x_ma_ret_{h} − x_p_ret_{h}
  x_ma_midday_ret       : MA 11:30-13:30, broadcast to P 13:25-15:00
  x_ma_afternoon_ret    : MA 15:00-18:00, broadcast to P 21:00 + morning

No-future-data guarantee
------------------------
All shift() / ffill() operations are backward-looking.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = Path(r"G:\data_prepare\TQSDK")
P_PATH       = BASE_DIR / "P_60min.parquet"
MA_1MIN_PATH = BASE_DIR / "马棕油主连.csv"

OUT_MA_60MIN_PATH = BASE_DIR / "ma_60min_from_p_time.csv"
OUT_MERGED_PATH   = BASE_DIR / "P_60min_with_ma_features_from_scratch.csv"

# ---------------------------------------------------------------------------
# Lookback horizons shared by MA and P  (unit: 60-min bars)
# ---------------------------------------------------------------------------
HORIZONS: dict[str, int] = {
    "1h":  1,
    "2h":  2,
    "4h":  4,
    "6h":  6,
    "8h":  8,
    "12h": 12,
    "20h": 20,
}


# ===========================================================================
# Data preparation helpers
# ===========================================================================

def prepare_p_dataframe(df_p: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort P 60min dataframe."""
    required_cols = {"datetime"}
    missing = required_cols - set(df_p.columns)
    if missing:
        raise ValueError(f"Missing required columns in P_60min: {sorted(missing)}")

    df = df_p.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    if df["datetime"].isna().any():
        raise ValueError(f"Found {df['datetime'].isna().sum()} rows with invalid P datetime")
    if df["datetime"].duplicated().any():
        raise ValueError(f"Found {df['datetime'].duplicated().sum()} duplicated datetimes in P_60min")

    if not df["datetime"].is_monotonic_increasing:
        df = df.sort_values("datetime", kind="stable").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def prepare_ohlc_1min_dataframe(df_raw: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate and sort any 1min OHLC dataframe."""
    required_cols = {"datetime", "open", "high", "low", "close"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"[{label}] Missing columns: {sorted(missing)}")

    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].isna().any():
        raise ValueError(f"[{label}] Found {df['datetime'].isna().sum()} invalid datetimes")

    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)
    return df


# ===========================================================================
# Core: aggregate 1min bars to P-aligned 60min bars
# ===========================================================================

def build_60min_from_p_datetime(
    df_p: pd.DataFrame,
    df_1min: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Aggregate `df_1min` OHLC ticks into bars whose *end* times match
    the P 60min datetime anchors.

    For each P bar i  →  interval = (p[i-1], p[i]]  (i > 0)
                                   = single tick before p[0]  (i == 0)

    Returns a DataFrame indexed on the same datetimes as df_p, with columns:
        {prefix}_open, {prefix}_high, {prefix}_low,
        {prefix}_close, {prefix}_close_ffill, {prefix}_tick_count

    `{prefix}_close_ffill` is a true row-by-row forward-fill of
    {prefix}_close, so that every P bar carries the last known price even
    during sessions when the instrument is not trading.
    This is strictly backward-looking — no future data is introduced.
    """
    p_time = df_p["datetime"].to_numpy(dtype="datetime64[ns]")
    t1     = df_1min["datetime"].to_numpy(dtype="datetime64[ns]")
    o1     = df_1min["open"].to_numpy(dtype=np.float64)
    h1     = df_1min["high"].to_numpy(dtype=np.float64)
    l1     = df_1min["low"].to_numpy(dtype=np.float64)
    c1     = df_1min["close"].to_numpy(dtype=np.float64)

    n = len(p_time)
    out_o = np.full(n, np.nan)
    out_h = np.full(n, np.nan)
    out_l = np.full(n, np.nan)
    out_c = np.full(n, np.nan)
    out_k = np.zeros(n, dtype=np.int64)

    right_idx = np.searchsorted(t1, p_time, side="right")
    left_idx  = np.zeros(n, dtype=np.int64)
    if n > 1:
        left_idx[1:] = np.searchsorted(t1, p_time[:-1], side="right")

    for i in range(n):
        r = int(right_idx[i])
        if i == 0:
            if r == 0:
                continue
            l = r - 1
        else:
            l = int(left_idx[i])

        if l >= r:
            continue

        out_o[i] = o1[l]
        out_h[i] = np.nanmax(h1[l:r])
        out_l[i] = np.nanmin(l1[l:r])
        out_c[i] = c1[r - 1]
        out_k[i] = r - l

    # True row-by-row ffill (strictly backward-looking)
    out_c_ffill = pd.Series(out_c).ffill().to_numpy()

    return pd.DataFrame({
        "datetime":              df_p["datetime"].values,
        f"{prefix}_open":        out_o,
        f"{prefix}_high":        out_h,
        f"{prefix}_low":         out_l,
        f"{prefix}_close":       out_c,
        f"{prefix}_close_ffill": out_c_ffill,
        f"{prefix}_tick_count":  out_k,
    })


# ===========================================================================
# Feature engineering
# ===========================================================================

def add_return_columns(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MA / P lookback returns and cross-asset spreads.

    All operations are backward-looking (shift forward in time index).

    x_ma_bar_ret
        MA return within the same P bar window = (ma_close − ma_open) / ma_open.
        NaN for P bars where MA was not trading — expected.

    x_ma_ret_{h} / x_p_ret_{h} / x_spread_{h}
        Close-to-close returns and MA-vs-P spread over h hours.
        Use *_close_ffill to avoid NaN cascades across closed sessions.
    """
    # ── MA intra-bar return ──────────────────────────────────────────────
    df_merged["x_ma_bar_ret"] = (
        (df_merged["ma_close"] - df_merged["ma_open"]) / df_merged["ma_open"]
    )

    ma_ffill = df_merged["ma_close_ffill"]
    p_close  = df_merged["y_close"]

    for label, n in HORIZONS.items():
        ma_prev = ma_ffill.shift(n)
        p_prev  = p_close.shift(n)

        df_merged[f"x_ma_ret_{label}"] = (ma_ffill - ma_prev) / ma_prev
        df_merged[f"x_p_ret_{label}"]  = (p_close  - p_prev)  / p_prev
        df_merged[f"x_spread_{label}"] = (
            df_merged[f"x_ma_ret_{label}"] - df_merged[f"x_p_ret_{label}"]
        )

    return df_merged


def insert_session_anchor_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Insert 13:25 and 21:00 anchor bars, values forward-filled from prior row."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    T_1335 = pd.Timestamp("13:35").time()
    T_2110 = pd.Timestamp("21:10").time()

    existing_dts = set(df["datetime"])
    insert_spec = []
    for i, row in df.iterrows():
        t = row["datetime"].time()
        d = row["datetime"].date()
        if t == T_1335:
            new_dt = pd.Timestamp(f"{d} 13:25:00")
            if new_dt not in existing_dts:
                insert_spec.append((i, new_dt))
        if t == T_2110:
            new_dt = pd.Timestamp(f"{d} 21:00:00")
            if new_dt not in existing_dts:
                insert_spec.append((i, new_dt))

    if not insert_spec:
        return df

    new_rows = []
    for target_idx, new_dt in insert_spec:
        if target_idx == 0:
            continue
        prev_row = df.loc[target_idx - 1].copy()
        prev_row["datetime"] = new_dt
        if "bar_period" in prev_row.index:
            prev_row["bar_period"] = "interpolated"
        new_rows.append(prev_row)

    if not new_rows:
        return df

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)
    print(f"  inserted {len(new_rows)} anchor bars (13:25 / 21:00)")
    return df


def add_ma_session_returns(
    df_merged: pd.DataFrame, df_ma_1min: pd.DataFrame
) -> pd.DataFrame:
    """
    Broadcast MA intra-session returns to relevant P bar windows:
      x_ma_midday_ret    : MA 11:30-13:30  →  P 13:25-15:00
      x_ma_afternoon_ret : MA 15:00-18:00  →  P 21:00 + next morning ≤ 11:35
    """
    df = df_merged.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    df_ma = df_ma_1min.copy()
    df_ma["datetime"] = pd.to_datetime(df_ma["datetime"])
    df_ma["date"] = df_ma["datetime"].dt.date
    df_ma["time"] = df_ma["datetime"].dt.time

    daily_returns: dict = {}
    for date, day_data in df_ma.groupby("date"):
        day_data = day_data.sort_values("datetime")

        mid = day_data[
            (day_data["time"] >= pd.Timestamp("11:30").time())
            & (day_data["time"] <= pd.Timestamp("13:30").time())
        ]
        midday_ret = (
            (mid.iloc[-1]["close"] / mid.iloc[0]["close"] - 1)
            if len(mid) >= 2 else 0.0
        )

        aft = day_data[
            (day_data["time"] >= pd.Timestamp("15:00").time())
            & (day_data["time"] < pd.Timestamp("18:00").time())
        ]
        afternoon_ret = (
            (aft.iloc[-1]["close"] / aft.iloc[0]["close"] - 1)
            if len(aft) >= 2 else 0.0
        )

        daily_returns[date] = {
            "midday_ret": midday_ret,
            "afternoon_ret": afternoon_ret,
        }

    dt       = df["datetime"]
    t        = dt.dt.time
    cal_date = dt.dt.date

    T_1325 = pd.Timestamp("13:25").time()
    T_1500 = pd.Timestamp("15:00").time()
    T_2100 = pd.Timestamp("21:00").time()
    T_1135 = pd.Timestamp("11:35").time()

    midday_mask  = (t >= T_1325) & (t <= T_1500)
    night_mask   = t >= T_2100
    morning_mask = t <= T_1135

    df["x_ma_midday_ret"] = 0.0
    df.loc[midday_mask, "x_ma_midday_ret"] = cal_date[midday_mask].map(
        lambda d: daily_returns.get(d, {}).get("midday_ret", 0.0)
    )

    df["x_ma_afternoon_ret"] = 0.0
    df.loc[night_mask, "x_ma_afternoon_ret"] = cal_date[night_mask].map(
        lambda d: daily_returns.get(d, {}).get("afternoon_ret", 0.0)
    )
    df.loc[morning_mask, "x_ma_afternoon_ret"] = cal_date[morning_mask].map(
        lambda d: daily_returns.get(
            (pd.Timestamp(d) - pd.Timedelta(days=1)).date(), {}
        ).get("afternoon_ret", 0.0)
    )
    return df


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # -----------------------------------------------------------------------
    # Step 1: load
    # -----------------------------------------------------------------------
    print("Step 1/5: load source data...")
    df_p_raw = pd.read_parquet(P_PATH).reset_index()
    df_ma_1min_raw = pd.read_csv(
        MA_1MIN_PATH,
        usecols=["datetime", "open", "high", "low", "close"],
        parse_dates=["datetime"],
    )

    df_p       = prepare_p_dataframe(df_p_raw)
    df_ma_1min = prepare_ohlc_1min_dataframe(df_ma_1min_raw, "MA")

    print(f"  P rows       : {len(df_p):,}")
    print(f"  MA 1min rows : {len(df_ma_1min):,}")
    print(f"  P  range     : {df_p['datetime'].min()} -> {df_p['datetime'].max()}")
    print(f"  MA range     : {df_ma_1min['datetime'].min()} -> {df_ma_1min['datetime'].max()}")

    # -----------------------------------------------------------------------
    # Step 2: build aligned 60min bars for MA
    # -----------------------------------------------------------------------
    print("\nStep 2/5: build aligned 60min bars...")

    df_ma_60min = build_60min_from_p_datetime(df_p, df_ma_1min, prefix="ma")

    df_ma_60min.to_csv(OUT_MA_60MIN_PATH, index=False)
    print(f"  saved MA -> {OUT_MA_60MIN_PATH}")

    # -----------------------------------------------------------------------
    # Step 3: merge P + MA
    # -----------------------------------------------------------------------
    print("\nStep 3/5: merge P + MA by datetime...")
    df_merged = df_p.merge(df_ma_60min, on="datetime", how="left", validate="one_to_one")
    print(f"  merged rows             : {len(df_merged):,}")
    print(f"  ma_close_ffill non-null : {int(df_merged['ma_close_ffill'].notna().sum()):,}")

    # -----------------------------------------------------------------------
    # Step 4: return / spread features
    # -----------------------------------------------------------------------
    print("\nStep 4/5: generate return/spread columns (no future leak)...")
    df_merged = add_return_columns(df_merged)

    nn = int(df_merged["x_ma_bar_ret"].notna().sum())
    print(f"  x_ma_bar_ret : non-null={nn:,}  "
          f"mean={df_merged['x_ma_bar_ret'].mean():.6f}  "
          f"std={df_merged['x_ma_bar_ret'].std():.6f}")
    print(f"  (NaN for P bars with no MA ticks — expected)")

    for label in HORIZONS:
        for col in [f"x_ma_ret_{label}", f"x_p_ret_{label}", f"x_spread_{label}"]:
            nn = int(df_merged[col].notna().sum())
            print(f"  {col}: non-null={nn:,}")

    # -----------------------------------------------------------------------
    # Step 5: MA session returns + anchor bars
    # -----------------------------------------------------------------------
    print("\nStep 5/5: add MA session return factors...")
    df_merged = add_ma_session_returns(df_merged, df_ma_1min)
    for col in ["x_ma_midday_ret", "x_ma_afternoon_ret"]:
        nz = int((df_merged[col] != 0).sum())
        print(f"  {col}: non-zero={nz:,}  "
              f"mean={df_merged[col].mean():.6f}  std={df_merged[col].std():.6f}")

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    cols = ["datetime"] + [c for c in df_merged.columns if c != "datetime"]
    df_merged = df_merged[cols]

    # 只保留有 MA 数据的行（MA 数据起始晚于 P）
    # df_merged = df_merged[df_merged["ma_close_ffill"].notna()].reset_index(drop=True)

    df_merged.to_csv(OUT_MERGED_PATH, index=False)
    print(f"\nDone. saved -> {OUT_MERGED_PATH}  (rows: {len(df_merged):,})")
    print(f"  total feature columns: {len(df_merged.columns):,}")


if __name__ == "__main__":
    main()