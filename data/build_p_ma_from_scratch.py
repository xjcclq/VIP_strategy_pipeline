"""
Build MA & P-aligned features.

Columns produced
----------------
── MA (马来棕榈油) ──────────────────────────────────────────────
  ma_open / ma_high / ma_low / ma_close / ma_close_ffill / ma_tick_count

  x_ma_bar_ret          : MA return within the *same* P bar window
                          = (ma_close − ma_open) / ma_open
                          NaN when MA has no ticks in that P bar window.

  x_ma_ret_{h}          : MA close-to-close return over past N hours,
                          anchored to the P bar_end datetime.
                          Looks up: ma_close_ffill at bar_end  vs
                                    ma_close_ffill at (bar_end − N hours),
                          using merge_asof(direction="backward") on the
                          per-minute MA series — no future data.

  x_p_ret_{h}           : P close-to-close return over past N hours,
                          using the same time-based anchor.
                          P close at bar_end vs P close at (bar_end − N hours),
                          found via merge_asof on the P bar series.

  x_spread_{h}          : x_ma_ret_{h} − x_p_ret_{h}

  x_ma_midday_ret       : MA 11:30→13:30 return, attached to P bars
                          whose bar_end is in [13:25, 15:00].
  x_ma_afternoon_ret    : MA 15:00→close return, attached to P bars
                          whose bar_end is in night session (≥21:00)
                          or next-morning session (≤11:35).

No-future-data guarantee
------------------------
* build_60min_from_p_datetime  : interval = (p[i-1], p[i]], strictly left-open.
* ma_close_ffill               : row-by-row pandas ffill, backward only.
* All shift / merge_asof       : direction="backward", no look-ahead.
* HORIZONS lookup              : bar_end − N hours, then merge_asof backward.
* Session returns              : computed from MA 1min data whose timestamps
                                 are strictly before the P bar_end they attach to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_path = Path(__file__).resolve()
BASE_DIR = script_path.parents[0]
# BASE_DIR     = Path(r"F:\VIP\data")
P_PATH       = BASE_DIR / "P.parquet"
MA_1MIN_PATH = BASE_DIR / "马棕油主连.csv"

OUT_MA_60MIN_PATH = BASE_DIR / "ma_aligned_from_p_time.csv"
OUT_MERGED_PATH   = BASE_DIR / "P_with_ma_features.csv"

# ---------------------------------------------------------------------------
# Lookback horizons: label → actual hours
# The P bars are NOT equal-length (slots: 60min, 75min, 180min, 40min, ...),
# so shift(n) is WRONG. We compute returns by looking back N *real* hours
# from each bar_end timestamp, using merge_asof on the MA 1min series.
# ---------------------------------------------------------------------------
HORIZONS: dict[str, float] = {
    "1h":  1.0,
    "2h":  2.0,
    "4h":  4.0,
    "6h":  6.0,
    "8h":  8.0,
    "12h": 12.0,
    "20h": 20.0,
}


# ===========================================================================
# Data preparation helpers
# ===========================================================================

def prepare_p_dataframe(df_p: pd.DataFrame) -> pd.DataFrame:
    """Validate and sort P bar dataframe (index = datetime after reset_index)."""
    df = df_p.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    if df["datetime"].isna().any():
        raise ValueError(f"Found {df['datetime'].isna().sum()} rows with invalid P datetime")
    if df["datetime"].duplicated().any():
        raise ValueError(f"Found {df['datetime'].duplicated().sum()} duplicated datetimes in P")

    df = df.sort_values("datetime", kind="stable").reset_index(drop=True)
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
# Core: aggregate 1min bars to P-aligned bars
# ===========================================================================

def build_aligned_from_p_datetime(
    df_p: pd.DataFrame,
    df_1min: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Aggregate df_1min OHLC into bars whose *end* times match P bar_end datetimes.

    Interval for bar i:  (p_time[i-1], p_time[i]]   (left-open, right-closed)
    For i==0:            single tick at or before p_time[0]

    Returns DataFrame with same length as df_p, columns:
        {prefix}_open, {prefix}_high, {prefix}_low,
        {prefix}_close, {prefix}_close_ffill, {prefix}_tick_count

    {prefix}_close_ffill: row-by-row forward-fill of {prefix}_close.
        Strictly backward-looking. Fills NaN (e.g. MA during P night session)
        with the last known MA price.
    """
    p_time = df_p["datetime"].to_numpy(dtype="datetime64[ns]")
    t1     = df_1min["datetime"].to_numpy(dtype="datetime64[ns]")
    o1     = df_1min["open"].to_numpy(dtype=np.float64)
    h1     = df_1min["high"].to_numpy(dtype=np.float64)
    l1     = df_1min["low"].to_numpy(dtype=np.float64)
    c1     = df_1min["close"].to_numpy(dtype=np.float64)

    n      = len(p_time)
    out_o  = np.full(n, np.nan)
    out_h  = np.full(n, np.nan)
    out_l  = np.full(n, np.nan)
    out_c  = np.full(n, np.nan)
    out_k  = np.zeros(n, dtype=np.int64)

    # right_idx[i] = first index in t1 strictly after p_time[i]
    right_idx        = np.searchsorted(t1, p_time, side="right")
    left_idx         = np.zeros(n, dtype=np.int64)
    if n > 1:
        # left_idx[i] = first index in t1 strictly after p_time[i-1]
        left_idx[1:] = np.searchsorted(t1, p_time[:-1], side="right")

    for i in range(n):
        r = int(right_idx[i])
        l = (r - 1) if i == 0 else int(left_idx[i])
        # 必须同时满足: l >= 0 且 l < r，才有有效区间
        if l < 0 or l >= r:
            continue
        out_o[i] = o1[l]
        out_h[i] = np.nanmax(h1[l:r])
        out_l[i] = np.nanmin(l1[l:r])
        out_c[i] = c1[r - 1]
        out_k[i] = r - l

    out_c_ffill = pd.Series(out_c).ffill().to_numpy()

    return pd.DataFrame({
        "datetime":               df_p["datetime"].values,
        f"{prefix}_open":         out_o,
        f"{prefix}_high":         out_h,
        f"{prefix}_low":          out_l,
        f"{prefix}_close":        out_c,
        f"{prefix}_close_ffill":  out_c_ffill,
        f"{prefix}_tick_count":   out_k,
    })


# ===========================================================================
# Build a minute-level MA close series (for horizon lookback)
# ===========================================================================

def build_ma_minute_ffill(df_ma_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Build a complete per-minute MA close series with forward-fill.

    This is used as the "price tape" for looking up MA price at any
    arbitrary past timestamp (bar_end − N hours) via merge_asof.

    Returns DataFrame with columns: [datetime, ma_close_ffill_1min]
    Sorted by datetime ascending.
    """
    df = df_ma_1min[["datetime", "close"]].copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    df["ma_close_ffill_1min"] = df["close"].ffill()
    return df[["datetime", "ma_close_ffill_1min"]]


# ===========================================================================
# Feature engineering — time-based horizon returns (NO shift(n))
# ===========================================================================

def add_return_columns(
    df_merged: pd.DataFrame,
    df_ma_minute: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute MA / P lookback returns and cross-asset spreads.

    Method
    ------
    For each P bar with bar_end = T and horizon = H hours:

      MA return:
        current_price = ma_close_ffill at T  (already in df_merged)
        past_price    = ma_close_ffill_1min at the largest 1min timestamp
                        that is <= T − H hours
                        → merge_asof(direction="backward") on df_ma_minute
        x_ma_ret = (current_price − past_price) / past_price

      P return:
        current_price = y_close at T  (already in df_merged)
        past_price    = y_close at the largest P bar_end
                        that is <= T − H hours
                        → merge_asof(direction="backward") on df_merged itself
        x_p_ret = (current_price − past_price) / past_price

      spread = x_ma_ret − x_p_ret

    All operations are strictly backward-looking.
    No future data is introduced.

    x_ma_bar_ret
        Intra-bar MA return = (ma_close − ma_open) / ma_open.
        NaN for bars where MA has no ticks (e.g. P night session bars).
    """
    df = df_merged.copy()

    # ── intra-bar MA return ──────────────────────────────────────────────
    df["x_ma_bar_ret"] = (
        (df["ma_close"] - df["ma_open"]) / df["ma_open"]
    )

    # ── prepare lookup tables ────────────────────────────────────────────
    # MA minute tape: sorted, used for merge_asof
    ma_tape = df_ma_minute.sort_values("datetime").reset_index(drop=True)

    # P bar tape: (datetime, y_close) for P-side lookback
    p_tape = df[["datetime", "y_close"]].copy().sort_values("datetime").reset_index(drop=True)

    bar_times    = df["datetime"].values           # bar_end timestamps
    ma_cur       = df["ma_close_ffill"].values     # MA price at each bar_end
    p_cur        = df["y_close"].values            # P  price at each bar_end

    # ── 诊断：打印关键时间范围，帮助定位空列原因 ────────────────────────
    bar_times_pd = pd.to_datetime(bar_times)
    print(f"\n[DIAG] bar_times range  : {bar_times_pd.min()} → {bar_times_pd.max()}")
    print(f"[DIAG] ma_tape datetime : {ma_tape['datetime'].min()} → {ma_tape['datetime'].max()}")
    print(f"[DIAG] p_tape  datetime : {p_tape['datetime'].min()} → {p_tape['datetime'].max()}")
    print(f"[DIAG] ma_cur  non-null : {np.sum(~np.isnan(ma_cur.astype(float)))}")
    print(f"[DIAG] p_cur   non-null : {np.sum(~np.isnan(p_cur.astype(float)))}")

    # 检查 bar_times 和 ma_tape 的 dtype 是否一致
    print(f"[DIAG] bar_times dtype  : {bar_times.dtype}")
    print(f"[DIAG] ma_tape dt dtype : {ma_tape['datetime'].dtype}")
    print(f"[DIAG] p_tape  dt dtype : {p_tape['datetime'].dtype}")

    # ── per-horizon returns ──────────────────────────────────────────────
    for label, hours in HORIZONS.items():
        # 用 pd.Timedelta 避免 int64 溢出和浮点精度问题
        delta = pd.Timedelta(hours=hours)
        past_times = bar_times_pd - delta

        # 诊断：只对第一个 horizon 详细打印
        if label == "1h":
            print(f"\n[DIAG] horizon={label}  delta={delta}")
            print(f"[DIAG] past_times range : {past_times.min()} → {past_times.max()}")
            # 检查有多少 past_time 落在 ma_tape 范围内
            ma_min = ma_tape["datetime"].min()
            ma_max = ma_tape["datetime"].max()
            in_range = ((past_times >= ma_min) & (past_times <= ma_max)).sum()
            print(f"[DIAG] past_times in MA tape range [{ma_min}, {ma_max}]: {in_range}/{len(past_times)}")
            p_min = p_tape["datetime"].min()
            p_max = p_tape["datetime"].max()
            in_range_p = ((past_times >= p_min) & (past_times <= p_max)).sum()
            print(f"[DIAG] past_times in P  tape range [{p_min}, {p_max}]: {in_range_p}/{len(past_times)}")

        # ── MA past price: merge_asof on 1min tape ───────────────────────
        lookup_ma = pd.DataFrame({
            "datetime": past_times,
            "bar_end":  bar_times_pd,
        }).sort_values("datetime").reset_index(drop=True)

        # 确保两侧 datetime 都是 datetime64[ns]
        lookup_ma["datetime"] = pd.to_datetime(lookup_ma["datetime"])

        merged_ma = pd.merge_asof(
            lookup_ma,
            ma_tape,
            on="datetime",
            direction="backward",
        )

        if label == "1h":
            hit = merged_ma["ma_close_ffill_1min"].notna().sum()
            print(f"[DIAG] merge_asof MA  hits (non-null): {hit}/{len(merged_ma)}")
            print(f"[DIAG] merged_ma sample:\n{merged_ma.head(5).to_string()}")

        # 还原原始顺序
        merged_ma = merged_ma.sort_values("bar_end").reset_index(drop=True)
        ma_past = merged_ma["ma_close_ffill_1min"].values

        # ── P past price: merge_asof on P bar tape ───────────────────────
        lookup_p = pd.DataFrame({
            "datetime": past_times,
            "bar_end":  bar_times_pd,
        }).sort_values("datetime").reset_index(drop=True)

        lookup_p["datetime"] = pd.to_datetime(lookup_p["datetime"])

        merged_p = pd.merge_asof(
            lookup_p,
            p_tape,
            on="datetime",
            direction="backward",
        )

        if label == "1h":
            hit_p = merged_p["y_close"].notna().sum()
            print(f"[DIAG] merge_asof P   hits (non-null): {hit_p}/{len(merged_p)}")
            print(f"[DIAG] merged_p sample:\n{merged_p.head(5).to_string()}")

        merged_p = merged_p.sort_values("bar_end").reset_index(drop=True)
        p_past = merged_p["y_close"].values

        ma_cur_f = ma_cur.astype(float)
        p_cur_f  = p_cur.astype(float)
        ma_past_f = ma_past.astype(float)
        p_past_f  = p_past.astype(float)

        ma_ret = np.where(
            (ma_past_f > 0) & np.isfinite(ma_past_f) & np.isfinite(ma_cur_f),
            (ma_cur_f - ma_past_f) / ma_past_f,
            np.nan,
        )
        p_ret = np.where(
            (p_past_f > 0) & np.isfinite(p_past_f) & np.isfinite(p_cur_f),
            (p_cur_f - p_past_f) / p_past_f,
            np.nan,
        )

        df[f"x_ma_ret_{label}"] = ma_ret
        df[f"x_p_ret_{label}"]  = p_ret
        df[f"x_spread_{label}"] = np.where(
            np.isfinite(ma_ret) & np.isfinite(p_ret),
            ma_ret - p_ret,
            np.nan,
        )

        if label == "1h":
            print(f"[DIAG] x_ma_ret_1h non-null: {np.sum(np.isfinite(ma_ret))}/{len(ma_ret)}")
            print(f"[DIAG] x_p_ret_1h  non-null: {np.sum(np.isfinite(p_ret))}/{len(p_ret)}")

    return df


# ===========================================================================
# MA intra-session returns broadcast to P bars
# ===========================================================================

def add_ma_session_returns(
    df_merged: pd.DataFrame,
    df_ma_1min: pd.DataFrame,
) -> pd.DataFrame:
    """
    Broadcast MA intra-session returns to relevant P bar windows.

    x_ma_midday_ret
        MA return from 11:30 to 13:30 on the *same calendar date* as
        the P bar_end. Attached to P bars whose bar_end time is in
        [13:30, 15:00]. (MA 13:30 close is known before P 14:15 bar_end.)

    x_ma_afternoon_ret
        MA return from 15:00 to the last tick before 18:00 on date D.
        Attached to:
          • P night bars: bar_end time >= 21:00  (same calendar date D,
            since night bar_end = prev_day 22:00/22:55 per BAR_SLOTS,
            and MA 18:00 < 22:00 — no future leak)
          • P morning bars: bar_end time in [09:00, 11:35] on date D+1
            (prev trading day's afternoon; use last *actual* trading day
            before D+1 to handle weekends correctly)

    No-future-data guarantee:
        midday_ret  uses MA data up to 13:30; earliest P bar_end is 14:15. ✓
        afternoon_ret uses MA data up to <18:00; P night bar_end >= 21:00. ✓
        Morning attachment uses *previous* day's afternoon — strictly past. ✓
        Weekend gap is handled by building a sorted list of MA trading dates
        and looking up the most recent date before the target date.
    """
    df = df_merged.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    df_ma = df_ma_1min.copy()
    df_ma["datetime"] = pd.to_datetime(df_ma["datetime"])
    df_ma["date"]     = df_ma["datetime"].dt.date
    df_ma["time"]     = df_ma["datetime"].dt.time

    T_1130 = pd.Timestamp("11:30").time()
    T_1330 = pd.Timestamp("13:30").time()
    T_1500 = pd.Timestamp("15:00").time()
    T_1800 = pd.Timestamp("18:00").time()

    # ── build daily session return table ────────────────────────────────
    # Keys: calendar date (datetime.date)
    # Values: {"midday_ret": float, "afternoon_ret": float}
    daily_returns: dict = {}
    ma_trading_dates_sorted: list = []   # sorted list of dates MA actually traded

    for date, day_data in df_ma.groupby("date"):
        day_data = day_data.sort_values("datetime")

        mid = day_data[
            (day_data["time"] >= T_1130) & (day_data["time"] <= T_1330)
        ]
        midday_ret = (
            (mid.iloc[-1]["close"] / mid.iloc[0]["close"] - 1)
            if len(mid) >= 2 else np.nan
        )

        aft = day_data[
            (day_data["time"] >= T_1500) & (day_data["time"] < T_1800)
        ]
        afternoon_ret = (
            (aft.iloc[-1]["close"] / aft.iloc[0]["close"] - 1)
            if len(aft) >= 2 else np.nan
        )

        daily_returns[date] = {
            "midday_ret":   midday_ret,
            "afternoon_ret": afternoon_ret,
        }
        ma_trading_dates_sorted.append(date)

    ma_trading_dates_sorted.sort()
    ma_dates_arr = np.array(ma_trading_dates_sorted, dtype="object")

    def _prev_ma_date(cal_date) -> object | None:
        """
        Return the most recent MA trading date strictly before cal_date.
        Handles weekends / holidays correctly — no hardcoded day-of-week logic.
        """
        import bisect
        idx = bisect.bisect_left(ma_dates_arr.tolist(), cal_date)
        return ma_dates_arr[idx - 1] if idx > 0 else None

    # ── attach to P bars ─────────────────────────────────────────────────
    dt       = df["datetime"]
    t_time   = dt.dt.time
    cal_date = dt.dt.date

    T_1330_t = pd.Timestamp("13:30").time()
    T_1455_t = pd.Timestamp("14:55").time()   # last P bar_end in day session
    T_2100_t = pd.Timestamp("21:00").time()
    T_1135_t = pd.Timestamp("11:35").time()

    # midday: P bar_end in [13:30, 14:55]  (day session, after MA 13:30 close)
    midday_mask = (t_time >= T_1330_t) & (t_time <= T_1455_t)

    # night: P bar_end >= 21:00 (bar_end = prev_day 22:00 / 22:55)
    night_mask = t_time >= T_2100_t

    # morning: P bar_end in [09:00, 11:35] (day session start)
    morning_mask = t_time <= T_1135_t

    df["x_ma_midday_ret"]   = np.nan
    df["x_ma_afternoon_ret"] = np.nan

    # midday: same cal_date as bar_end
    if midday_mask.any():
        df.loc[midday_mask, "x_ma_midday_ret"] = (
            cal_date[midday_mask]
            .map(lambda d: daily_returns.get(d, {}).get("midday_ret", np.nan))
            .values
        )

    # night: same cal_date (bar_end is prev_day 22:xx, MA afternoon is prev_day 15-18) ✓
    if night_mask.any():
        df.loc[night_mask, "x_ma_afternoon_ret"] = (
            cal_date[night_mask]
            .map(lambda d: daily_returns.get(d, {}).get("afternoon_ret", np.nan))
            .values
        )

    # morning: look up the *previous MA trading date* before cal_date
    # (handles weekends: Monday morning → Friday's afternoon_ret)
    if morning_mask.any():
        df.loc[morning_mask, "x_ma_afternoon_ret"] = (
            cal_date[morning_mask]
            .map(lambda d: daily_returns.get(
                _prev_ma_date(d), {}
            ).get("afternoon_ret", np.nan))
            .values
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

    print(f"  P  rows      : {len(df_p):,}")
    print(f"  MA 1min rows : {len(df_ma_1min):,}")
    print(f"  P  range     : {df_p['datetime'].min()} → {df_p['datetime'].max()}")
    print(f"  MA range     : {df_ma_1min['datetime'].min()} → {df_ma_1min['datetime'].max()}")

    # -----------------------------------------------------------------------
    # Step 2: build MA bars aligned to P bar_end times
    # -----------------------------------------------------------------------
    print("\nStep 2/5: aggregate MA 1min → P-aligned bars...")
    df_ma_aligned = build_aligned_from_p_datetime(df_p, df_ma_1min, prefix="ma")
    df_ma_aligned.to_csv(OUT_MA_60MIN_PATH, index=False)
    print(f"  saved → {OUT_MA_60MIN_PATH}")
    print(f"  ma_close non-null  : {int(df_ma_aligned['ma_close'].notna().sum()):,}")
    print(f"  ma_close_ffill non-null : {int(df_ma_aligned['ma_close_ffill'].notna().sum()):,}")

    # -----------------------------------------------------------------------
    # Step 3: merge P + MA aligned bars
    # -----------------------------------------------------------------------
    print("\nStep 3/5: merge P + MA aligned bars...")
    df_merged = df_p.merge(
        df_ma_aligned, on="datetime", how="left", validate="one_to_one"
    )
    print(f"  merged rows : {len(df_merged):,}")

    # -----------------------------------------------------------------------
    # Step 4: time-based horizon return features
    # -----------------------------------------------------------------------
    print("\nStep 4/5: compute time-based horizon returns (no shift, no future leak)...")

    # Build per-minute MA close tape for horizon lookback
    df_ma_minute = build_ma_minute_ffill(df_ma_1min)

    df_merged = add_return_columns(df_merged, df_ma_minute)

    print(f"  x_ma_bar_ret : non-null={int(df_merged['x_ma_bar_ret'].notna().sum()):,}"
          f"  mean={df_merged['x_ma_bar_ret'].mean():.6f}"
          f"  std={df_merged['x_ma_bar_ret'].std():.6f}")
    for label in HORIZONS:
        for col in [f"x_ma_ret_{label}", f"x_p_ret_{label}", f"x_spread_{label}"]:
            nn = int(df_merged[col].notna().sum())
            print(f"  {col}: non-null={nn:,}")

    # -----------------------------------------------------------------------
    # Step 5: MA intra-session returns
    # -----------------------------------------------------------------------
    print("\nStep 5/5: add MA session return factors...")
    df_merged = add_ma_session_returns(df_merged, df_ma_1min)
    for col in ["x_ma_midday_ret", "x_ma_afternoon_ret"]:
        nn = int(df_merged[col].notna().sum())
        nz = int(df_merged[col].ne(0).sum())
        print(f"  {col}: non-null={nn:,}  non-zero={nz:,}"
              f"  mean={df_merged[col].mean():.6f}"
              f"  std={df_merged[col].std():.6f}")

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    # Rename open → x_open to avoid collision with MA open
    if "open" in df_merged.columns:
        df_merged = df_merged.rename(columns={"open": "x_open"})

    # Drop VPIN columns that may not exist in new pipeline
    vpin_cols = [
        c for c in df_merged.columns
        if "vpin" in c.lower()
    ]
    if vpin_cols:
        df_merged.drop(columns=vpin_cols, inplace=True)
        print(f"\n  dropped {len(vpin_cols)} vpin columns: {vpin_cols}")

    # datetime first
    cols = ["datetime"] + [c for c in df_merged.columns if c != "datetime"]
    df_merged = df_merged[cols]

    df_merged.to_csv(OUT_MERGED_PATH, index=False)
    print(f"\nDone. saved → {OUT_MERGED_PATH}  (rows: {len(df_merged):,}  cols: {len(df_merged.columns):,})")


if __name__ == "__main__":
    main()