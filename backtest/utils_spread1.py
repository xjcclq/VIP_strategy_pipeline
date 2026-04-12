"""
utils_spread.py — 棕榈油回测工具函数（绝对价差版）
- 标签：price[t+fwd] - price[t]，不除以分母
- PnL ：position * (price[t+1] - price[t])，绝对点数
- 绘图：蓝色实线=资金曲线（元） / 橙色实线=原始价格（bar级完整）

【新增改动】（所有新增代码均以 "# ★新增" 注释标明）
  ① 标签反转 (_compute_labels_abs)：若 fwd 之后出现更强反向运动，用反转标签替换
     参数：check_days（向后额外检查bar数，0=不启用）、reversal_multiplier（反转强度倍数）
  ② 固定止损 (_backtest / apply_strength_filter)：浮亏超过 stop_loss_pts 点强制平仓
     参数：stop_loss_pts（止损点数，0=不启用）
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler


# ── 中文字体 ──────────────────────────────────────────────────────────────────

def _setup_chinese_font():
    import platform
    s = platform.system()
    plt.rcParams["font.sans-serif"] = (
        ["SimHei", "Microsoft YaHei"] if s == "Windows" else
        ["Arial Unicode MS", "PingFang SC"] if s == "Darwin" else
        ["DejaVu Sans", "WenQuanYi Micro Hei"]
    )
    plt.rcParams["axes.unicode_minus"] = False

_setup_chinese_font()


# ── 目录 / 时间戳 ─────────────────────────────────────────────────────────────

def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_output_directories(project_root, timestamp, tag="backtest") -> str:
    out = Path(project_root) / "output" / f"{tag}_{timestamp}"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_palm_oil_data(data_file) -> pd.DataFrame:
    path, suffix = Path(data_file), Path(data_file).suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(data_file)
    elif suffix == ".xlsx":
        df = pd.read_excel(data_file)
    elif suffix == ".csv":
        df = None
        for enc in ["utf-8", "gbk", "gb18030", "latin1"]:
            try:
                df = pd.read_csv(data_file, encoding=enc, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError(f"无法读取: {data_file}")
    else:
        raise ValueError(f"不支持格式: {suffix}")

    for col in ["date", "datetime", "trading_date", "Unnamed: 0"]:
        if col in df.columns:
            df.rename(columns={col: "date"}, inplace=True)
            df["date"] = df["date"].ffill()
            df.set_index("date", inplace=True)
            break
    df.index = pd.to_datetime(df.index)
    x_cols = [c for c in df.columns if c.startswith("x_")]
    for c in x_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── 因子数据准备 ──────────────────────────────────────────────────────────────

def prepare_factor_data(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    selected_factors: Optional[List[str]] = None,
    lag: int = 1,
    factor_lags: Optional[List[int]] = None,
    add_session_features: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    price_cols = [c for c in df.columns if c.lower().startswith("y_")]
    if not price_cols:
        raise ValueError("数据中无 y_ 价格列")

    avail = [c for c in df.columns if c.lower().startswith("x_")]
    if selected_factors:
        base_sel     = [f for f in selected_factors if f in avail]
        session_cols = []
        # session_cols = [c for c in ["x_is_night", "x_session_sin"] if c in avail]
        factor_cols  = list(dict.fromkeys(base_sel + session_cols))
        missing = [f for f in selected_factors if f not in avail]
        if missing:
            logging.warning(f"因子不存在: {missing}")
    else:
        factor_cols = avail

    close_col = next((c for c in price_cols if c.lower() == "y_close"), None)
    price_col = close_col or price_cols[0]
    pd_ = df[price_col].copy()
    fd  = df[factor_cols].copy()

    if factor_lags and len(factor_lags) == len(factor_cols):
        parts, names, max_lag = [], [], max(factor_lags)
        for col, cl in zip(factor_cols, factor_lags):
            parts.append(fd[col]); names.append(col)
            for i in range(1, cl):
                parts.append(fd[col].shift(i)); names.append(f"{col}_{i}")
        fd = pd.concat(parts, axis=1); fd.columns = names
        factor_cols = names
        fd, pd_ = fd.iloc[max_lag - 1:], pd_.iloc[max_lag - 1:]
    elif lag > 1:
        lagged = [fd]
        for i in range(1, lag):
            s = fd.shift(i); s.columns = [f"{c}_{i}" for c in factor_cols]; lagged.append(s)
        fd = pd.concat(lagged, axis=1); factor_cols = list(fd.columns)
        fd, pd_ = fd.iloc[lag - 1:], pd_.iloc[lag - 1:]

    if start_date:
        fd, pd_ = fd.loc[start_date:], pd_.loc[start_date:]

    fd = fd.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    all_nan_cols = fd.columns[fd.isna().all()].tolist()
    if all_nan_cols:
        logging.warning(f"丢弃全NaN因子列({len(all_nan_cols)}): {all_nan_cols}")
        fd = fd.drop(columns=all_nan_cols)
        factor_cols = [c for c in factor_cols if c not in set(all_nan_cols)]

    valid = ~(fd.isna().any(axis=1) | fd.isin([np.inf, -np.inf]).any(axis=1) | pd_.isna())
    return fd[valid], pd_[valid].copy(), factor_cols


# ── 标签：绝对价差（★新增：支持反转标签检测） ─────────────────────────────────

def _compute_labels_abs(price_data: pd.Series, fwd: int,
                        check_days: int = 0,
                        reversal_multiplier: float = 1.2) -> pd.Series:
    """
    绝对价差标签，支持反转检测。

    基础标签：label[t] = price[t+fwd] - price[t]

    ★新增：反转标签逻辑（check_days > 0 时启用）
      在 fwd 之后继续向前看 check_days 根 bar，
      如果出现方向相反且幅度超过 base * reversal_multiplier 的运动，
      则用该反转值替换原标签。
      这使模型学习到"不要在即将反转的位置开仓"。

    参数：
      price_data: 价格序列
      fwd: 前看窗口（bar数）
      check_days: ★新增 反转检查额外bar数（0=不启用，与原逻辑完全一致）
      reversal_multiplier: ★新增 反转强度阈值倍数
    """
    # 基础标签（与原逻辑完全一致）
    base = price_data.shift(-fwd) - price_data

    # ★新增：如果 check_days == 0，走原逻辑，直接返回
    if check_days <= 0:
        return base

    # ★新增：反转标签检测——逐步向后看，发现更强反转则替换
    labels = base.copy()
    replaced = pd.Series(False, index=price_data.index)

    for k in range(1, check_days + 1):
        # 从 t 到 t+fwd+k 的绝对价差
        rk = price_data.shift(-(fwd + k)) - price_data

        # 有效条件：base 和 rk 都非空、非 inf、base 非零
        valid = (base.notna() & rk.notna()
                 & ~np.isinf(base) & ~np.isinf(rk)
                 & (base != 0))

        # 替换条件：尚未被替换 & 方向相反 & 反转幅度超过阈值
        replace = (valid & ~replaced
                   & (base * rk < 0)
                   & (rk.abs() > reversal_multiplier * base.abs()))

        labels = labels.where(~replace, rk)
        replaced = replaced | replace

    return labels


# ── WLS 模型 ──────────────────────────────────────────────────────────────────

class _WLS:
    def __init__(self, alpha=0.0, feature_names=None,
                 weight_method="park", rolling_window=60, ewma_halflife=21):
        self.alpha = alpha; self.feature_names = feature_names
        self.weight_method = weight_method
        self.rolling_window = rolling_window; self.ewma_halflife = ewma_halflife
        self.coef_ = None; self.intercept_ = 0.0

    def _weights_park(self, X, y):
        res  = y - LinearRegression().fit(X, y).predict(X)
        logv = np.log(res ** 2 + 1e-6)
        Xc   = np.hstack([np.ones((len(X), 1)), X])
        beta = np.linalg.lstsq(Xc, logv, rcond=None)[0]
        w    = 1.0 / np.sqrt(np.exp(Xc @ beta) + 1e-6)
        return w / w.mean()

    def _weights_rolling(self, X, y):
        res   = y - LinearRegression().fit(X, y).predict(X)
        s     = pd.Series(res)
        sigma = (s.rolling(self.rolling_window,
                           min_periods=max(5, self.rolling_window // 4))
                  .std().fillna(s.expanding(min_periods=5).std()).fillna(s.std()).values)
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)
        return (1.0 / sigma) / (1.0 / sigma).mean()

    def _weights_ewma(self, X, y):
        n = len(y); resid = np.zeros(n); min_fit = max(X.shape[1] + 2, 30)
        for t in range(min_fit, n):
            m = LinearRegression().fit(X[:t], y[:t])
            resid[t] = y[t] - m.predict(X[t:t+1])[0]
        if min_fit > 0:
            m_full = LinearRegression().fit(X, y)
            resid[:min_fit] = y[:min_fit] - m_full.predict(X[:min_fit])
        resid_sq = pd.Series(resid ** 2)
        var_est  = (resid_sq.ewm(halflife=self.ewma_halflife, min_periods=5).mean()
                    .fillna(resid_sq.expanding(min_periods=1).mean()).values)
        var_est  = np.where(var_est < 1e-12, 1e-12, var_est)
        w = 1.0 / (np.sqrt(var_est) + 1e-8)
        w = np.clip(w, np.percentile(w, 1), np.percentile(w, 99))
        return w / w.mean()

    def _weights(self, X, y):
        if self.weight_method == "rolling": return self._weights_rolling(X, y)
        if self.weight_method == "ewma":    return self._weights_ewma(X, y)
        return self._weights_park(X, y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).flatten()
        W  = np.diag(self._weights(X, y))
        Xd = np.hstack([np.ones((len(X), 1)), X])
        A  = Xd.T @ W @ Xd + np.eye(Xd.shape[1]) * max(self.alpha, 1e-8)
        A[0, 0] -= max(self.alpha, 1e-8)
        try:    beta = np.linalg.solve(A, Xd.T @ W @ y)
        except: beta = np.linalg.pinv(A) @ (Xd.T @ W @ y)
        self.intercept_, self.coef_ = beta[0], beta[1:]
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def _train_wls(X_train, y_train, use_scaler=True,
               weight_method="park", rolling_window=60, ewma_halflife=21):
    scaler = None; X = X_train.values
    if use_scaler:
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        X = scaler.fit_transform(X)
    model = _WLS(feature_names=list(X_train.columns), weight_method=weight_method,
                 rolling_window=rolling_window, ewma_halflife=ewma_halflife)
    model.fit(X, y_train.values)
    return model, scaler


# ── 资金账户计算 ──────────────────────────────────────────────────────────────

def _calc_account_pnl(df: pd.DataFrame, price: np.ndarray,
                      multiplier: float, commission_rate: float,
                      init_capital: float, rebal_days: int) -> pd.DataFrame:
    n = len(df)
    pos = df["position"].values; pdiff = df["price_diff"].values
    prices = price; is_sw = df["is_switch"].values

    lot_arr = np.ones(n); trade_pnl_arr = np.zeros(n)
    comm_arr = np.zeros(n); net_pnl_arr = np.zeros(n); equity_arr = np.zeros(n)

    equity = init_capital; lot = 1.0
    dates       = pd.to_datetime(df.index).normalize()
    unique_days = sorted(set(dates))
    day_idx     = {d: i for i, d in enumerate(unique_days)}
    last_rebal  = 0

    for i in range(n):
        cur_day = day_idx[dates[i]]
        if cur_day - last_rebal >= rebal_days:
            lot = 1.0; last_rebal = cur_day
        lot_arr[i] = lot
        if is_sw[i]:
            equity_arr[i] = equity; continue
        gross = pos[i] * pdiff[i] * multiplier * lot
        trade_pnl_arr[i] = gross
        prev_pos = pos[i - 1] if i > 0 else 0.0
        fee = 0.0
        if (prev_pos == 0 and pos[i] != 0) or (prev_pos != 0 and pos[i] == 0):
            fee = 3500 * lot * commission_rate * multiplier
        comm_arr[i] = fee; net_pnl_arr[i] = gross - fee
        equity += net_pnl_arr[i]; equity_arr[i] = equity

    df = df.copy()
    df["lot_size"]       = lot_arr
    df["trade_pnl_cny"]  = trade_pnl_arr
    df["commission_cny"] = comm_arr
    df["net_pnl_cny"]    = net_pnl_arr
    df["equity"]         = equity_arr
    return df


# ── 回测核心（★新增：固定止损） ──────────────────────────────────────────────

def _backtest(trade_df: pd.DataFrame, args) -> pd.DataFrame:
    fwd           = max(1, int(getattr(args, "fwd", 1)))
    reg_threshold = float(getattr(args, "reg_threshold", 0.0))
    ct            = getattr(args, "close_threshold", [0, 0])
    close_mode    = str(getattr(args, "close_mode", "fixed") or "fixed").lower()
    if close_mode == "zscore": close_mode = "zscore_threshold"
    long_close  = ct[0] if isinstance(ct, (list, tuple)) else 0
    short_close = ct[1] if isinstance(ct, (list, tuple)) and len(ct) > 1 else 0

    # ★新增：读取固定止损参数（价格点数，0=不启用）
    stop_loss_pts = float(getattr(args, "stop_loss_pts", 0.0))

    n = len(trade_df); predictions = trade_df["prediction"].values
    price = trade_df["price"].values; open_mask = np.ones(n, dtype=bool)

    use_pred_close   = close_mode in {"threshold", "hybrid"}
    use_zscore_close = close_mode in {"zscore_threshold", "zscore_hybrid"}
    threshold_only   = close_mode in {"threshold", "zscore_threshold"}
    close_zscore = np.zeros(n); cz_thr = 0.0
    if use_zscore_close:
        cz_win = int(getattr(args, "close_zscore_window", 50)) or 50
        cz_thr = float(getattr(args, "close_zscore_threshold", 0.0))
        ps = trade_df["prediction"]
        zm = ps.rolling(cz_win, min_periods=2).mean()
        zs = ps.rolling(cz_win, min_periods=2).std().replace(0, np.nan)
        close_zscore = ((ps - zm) / zs).ffill().fillna(0.0).values

    positions = np.zeros(n); i = 0
    while i < n:
        if open_mask[i]:
            pred = predictions[i]
            signal = (1.0 if pred > reg_threshold else -1.0 if pred < -reg_threshold else 0.0) \
                     if reg_threshold > 0 else \
                     (1.0 if pred > 0 else -1.0 if pred < 0 else 0.0)
            if signal != 0:
                positions[i] = signal; j = i + 1
                max_hold = n if threshold_only else min(i + fwd, n)
                # ★新增：记录开仓价格，用于止损判断
                entry_price = price[i]
                while j < max_hold:
                    sc = False
                    # ★新增：固定止损——浮亏超过 stop_loss_pts 强制平仓
                    if stop_loss_pts > 0:
                        unrealized = signal * (price[j] - entry_price)
                        if unrealized <= -stop_loss_pts:
                            sc = True
                    if not sc and use_pred_close:
                        if signal > 0 and predictions[j] <= -long_close: sc = True
                        elif signal < 0 and predictions[j] >= short_close: sc = True
                    if not sc and use_zscore_close:
                        z = close_zscore[j]
                        if signal > 0 and z <= cz_thr: sc = True
                        elif signal < 0 and z >= -cz_thr: sc = True
                    if sc: i = j + 1; break
                    positions[j] = signal; j += 1
                else:
                    i = max_hold
                continue
        i += 1

    positions = np.roll(positions, 1); positions[0] = 0
    df = trade_df.copy(); df["position"] = positions
    price_diff = pd.Series(price, index=df.index).diff().fillna(0.0)
    sw = set(pd.to_datetime(getattr(args, "contract_switch_dates", [])))
    df["is_switch"]     = df.index.isin(sw)
    df["price_diff"]    = price_diff
    df["strategy_pnl"]  = df["position"] * price_diff
    df["benchmark_pnl"] = price_diff
    df.loc[df["is_switch"], "strategy_pnl"]  = 0.0
    df.loc[df["is_switch"], "benchmark_pnl"] = 0.0
    df["cum_strategy"]  = df["strategy_pnl"].cumsum()
    df["cum_benchmark"] = df["benchmark_pnl"].cumsum()
    prev_pos = df["position"].shift(1).fillna(0)
    df["daily_volume"] = ((prev_pos == 0) & (df["position"] != 0)).astype(int)

    df = _calc_account_pnl(df, price,
                           float(getattr(args, "multiplier",    10.0)),
                           float(getattr(args, "commission",    2e-4)),
                           float(getattr(args, "init_capital",  7000.0)),
                           int(getattr(args, "rebalance_days",  30)))
    return df[["prediction", "price", "price_diff",
               "strategy_pnl", "benchmark_pnl",
               "cum_strategy", "cum_benchmark",
               "daily_volume", "position", "is_switch",
               "lot_size", "trade_pnl_cny", "commission_cny", "net_pnl_cny", "equity"]]


# ── 指标计算 ──────────────────────────────────────────────────────────────────

def _performance(results_df: pd.DataFrame, args) -> dict:
    split_point  = getattr(args, "split_point", None)
    init_capital = float(getattr(args, "init_capital", 7000.0))

    df    = results_df.copy()
    _key  = pd.to_datetime(df.index).normalize()
    daily = df.groupby(_key).agg(
        strat_pnl=("strategy_pnl",  "sum"), bench_pnl=("benchmark_pnl","sum"),
        net_cny  =("net_pnl_cny",   "sum"), comm_cny =("commission_cny","sum"),
        is_switch=("is_switch","any"),       has_pos  =("position", lambda x: (x!=0).any()),
    )
    valid  = daily[~daily["is_switch"]]
    _r_raw     = valid["strat_pnl"].values.astype(np.float64)
    _r_cny_raw = valid["net_cny"].values.astype(np.float64)
    r      = np.where(np.isfinite(_r_raw),     _r_raw,     0.0)
    r_cny  = np.where(np.isfinite(_r_cny_raw), _r_cny_raw, 0.0)
    n_days = len(r)

    empty = dict(total_pnl=0, daily_avg_pnl=0, annual_pnl=0, sharpe_ratio=0,
                 sortino_ratio=0, calmar_ratio=0, max_drawdown_pts=0, volatility_daily=0,
                 win_rate=0, profit_loss_ratio=0, benchmark_total_pnl=0, trade_count=0,
                 init_capital=init_capital, total_net_cny=0, annual_net_cny=0,
                 total_commission=0, sharpe_cny=0, max_drawdown_cny=0,
                 annual_return_pct=0, split_point=split_point)
    if n_days == 0: return empty

    mu, sigma = r.mean(), r.std()
    ann_pnl = mu * 252; sr = mu / sigma * np.sqrt(252) if sigma > 0 else 0
    cum = np.cumsum(r); peak = np.maximum.accumulate(cum); mdd_pts = (cum - peak).min()
    neg = r[r < 0]; ds = neg.std() if len(neg) else 0
    sortino = mu / ds * np.sqrt(252) if ds > 0 else 0
    calmar  = ann_pnl / abs(mdd_pts) if (mdd_pts != 0 and np.isfinite(mdd_pts)) else 0
    _act_raw = valid[valid["has_pos"]]["strat_pnl"].values.astype(np.float64)
    active   = np.where(np.isfinite(_act_raw), _act_raw, 0.0)
    pos_r, neg_r = active[active > 0], active[active < 0]
    wr  = len(pos_r) / (len(pos_r) + len(neg_r)) if (len(pos_r) + len(neg_r)) > 0 else 0
    plr = pos_r.mean() / abs(neg_r.mean()) if (len(pos_r) and len(neg_r) and np.isfinite(neg_r.mean()) and neg_r.mean() != 0) else 0

    mu_c = r_cny.mean(); sig_c = r_cny.std()
    sr_c = mu_c / sig_c * np.sqrt(252) if sig_c > 0 else 0
    cum_c = np.cumsum(r_cny); peak_c = np.maximum.accumulate(cum_c); mdd_c = (cum_c - peak_c).min()
    ann_ret_pct = mu_c * 252 / init_capital * 100

    return dict(
        total_pnl=results_df["strategy_pnl"].sum(), daily_avg_pnl=mu,
        annual_pnl=ann_pnl, sharpe_ratio=sr, sortino_ratio=sortino,
        calmar_ratio=calmar, max_drawdown_pts=mdd_pts, volatility_daily=sigma,
        win_rate=wr, profit_loss_ratio=plr,
        benchmark_total_pnl=results_df["benchmark_pnl"].sum(),
        trade_count=int(results_df["daily_volume"].sum()),
        init_capital=init_capital,
        total_net_cny=results_df["net_pnl_cny"].sum(),
        annual_net_cny=mu_c * 252, total_commission=results_df["commission_cny"].sum(),
        sharpe_cny=sr_c, max_drawdown_cny=mdd_c, annual_return_pct=ann_ret_pct,
        split_point=split_point,
    )

def print_model_coefficients(model, scaler=None):
    """打印 WLS 线性回归的系数"""
    if model is None or model.coef_ is None:
        print("模型未训练，无系数可打印。")
        return

    names = model.feature_names or [f"x{i}" for i in range(len(model.coef_))]
    coefs = list(zip(names, model.coef_))
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'='*60}")
    print(f"WLS 回归系数  (截距 intercept = {model.intercept_:.6f})")
    print(f"{'='*60}")
    print(f"{'因子名':<30} {'系数':>15}")
    print(f"{'-'*45}")
    for name, c in coefs:
        print(f"{name:<30} {c:>15.6f}")
    print(f"{'='*60}\n")

def recalc_performance(results_df, args): return _performance(results_df, args)


# ── 信号强度过滤（★新增：固定止损） ──────────────────────────────────────────

def apply_strength_filter(results_df, args):
    entry_pct = float(getattr(args, "entry_strength_pct", 0.3))
    tw        = int(getattr(args, "threshold_window", 500))
    reg_thr   = float(getattr(args, "reg_threshold", 0.0))
    close_mode= str(getattr(args, "close_mode", "fixed") or "fixed").lower()
    fwd       = max(1, int(getattr(args, "fwd", 1)))
    ct        = getattr(args, "close_threshold", [0, 0])
    long_close  = ct[0] if isinstance(ct, (list, tuple)) else 0
    short_close = ct[1] if isinstance(ct, (list, tuple)) and len(ct) > 1 else 0

    # ★新增：读取固定止损参数
    stop_loss_pts = float(getattr(args, "stop_loss_pts", 0.0))

    pred = results_df["prediction"].copy(); n = len(pred)
    price = results_df["price"].values  # ★新增：需要价格序列做止损判断
    abs_pred = pred.abs(); q_level = 1.0 - entry_pct
    rolling_thr = (abs_pred.rolling(tw, min_periods=max(30, tw // 5)).quantile(q_level)
                   .fillna(abs_pred.expanding(min_periods=30).quantile(q_level))
                   .fillna(abs_pred.quantile(q_level)))

    use_pred_close   = close_mode in {"threshold", "hybrid"}
    use_zscore_close = close_mode in {"zscore_threshold", "zscore_hybrid"}
    threshold_only   = close_mode in {"threshold", "zscore_threshold"}
    close_zscore = pd.Series(0.0, index=pred.index); cz_thr = 0.0
    if use_zscore_close:
        cz_win = int(getattr(args, "close_zscore_window", tw))
        cz_thr = float(getattr(args, "close_zscore_threshold", 0.0))
        zm = pred.rolling(cz_win, min_periods=2).mean()
        zs = pred.rolling(cz_win, min_periods=2).std().replace(0, np.nan)
        close_zscore = ((pred - zm) / zs).ffill().fillna(0.0)

    signal = pd.Series(0.0, index=pred.index); i = 0
    while i < n:
        pv = pred.iloc[i]; thr = rolling_thr.iloc[i]
        if abs(pv) >= thr and abs(pv) > reg_thr:
            pos = float(np.sign(pv)); signal.iloc[i] = pos
            j = i + 1; max_hold = n if threshold_only else min(i + fwd, n)
            # ★新增：记录开仓价格
            entry_price = price[i]
            while j < max_hold:
                sc = False
                # ★新增：固定止损——浮亏超过 stop_loss_pts 强制平仓
                if stop_loss_pts > 0:
                    unrealized = pos * (price[j] - entry_price)
                    if unrealized <= -stop_loss_pts:
                        sc = True
                if not sc and use_pred_close:
                    if pos > 0 and pred.iloc[j] <= -long_close: sc = True
                    elif pos < 0 and pred.iloc[j] >= short_close: sc = True
                if not sc and use_zscore_close:
                    z = close_zscore.iloc[j]
                    if pos > 0 and z <= -abs(cz_thr): sc = True
                    elif pos < 0 and z >= abs(cz_thr): sc = True
                if sc: i = j + 1; break
                signal.iloc[j] = pos; j += 1
            else: i = max_hold
            continue
        i += 1

    res = results_df.copy()
    res["signal"] = signal; res["position"] = signal.shift(1).fillna(0)
    price_diff = res["price"].diff().fillna(0.0)
    res["price_diff"]    = price_diff
    res["strategy_pnl"]  = res["position"] * price_diff
    res["benchmark_pnl"] = price_diff
    if getattr(args, "contract_switch_dates", []):
        sw = set(pd.to_datetime(args.contract_switch_dates))
        res.loc[res.index.isin(sw), "strategy_pnl"]  = 0.0
        res.loc[res.index.isin(sw), "benchmark_pnl"] = 0.0
    res["cum_strategy"]  = res["strategy_pnl"].cumsum()
    res["cum_benchmark"] = res["benchmark_pnl"].cumsum()
    prev_pos = res["position"].shift(1).fillna(0)
    res["daily_volume"] = ((prev_pos == 0) & (res["position"] != 0)).astype(int)
    res = _calc_account_pnl(res, res["price"].values,
                            float(getattr(args, "multiplier",   10.0)),
                            float(getattr(args, "commission",   2e-4)),
                            float(getattr(args, "init_capital", 7000.0)),
                            int(getattr(args, "rebalance_days", 30)))
    return res


# ── 滑动窗口回测（★新增：传递反转标签参数） ──────────────────────────────────

def _run_sliding_window_backtest(factor_data, price_data, args):
    n, tw, fwd = len(factor_data), args.train_window, args.fwd
    freq, mode = args.retrain_freq, args.mode
    use_sc     = getattr(args, "use_scaler", True)

    predictions = np.full(n, np.nan)
    model = scaler = first_model = first_sc = None
    in_sample_done = False

    # ★新增：读取反转标签参数，传入 _compute_labels_abs
    check_days          = int(getattr(args, "check_days", 0))
    reversal_multiplier = float(getattr(args, "reversal_multiplier", 1.2))
    all_labels = _compute_labels_abs(price_data, fwd,
                                     check_days=check_days,
                                     reversal_multiplier=reversal_multiplier)

    factor_arr  = factor_data.values
    nan_mask    = np.isnan(factor_arr).any(axis=1) | np.isinf(factor_arr).any(axis=1)
    sw_set      = set(pd.to_datetime(getattr(args, "contract_switch_dates", [])))
    sw_mask     = factor_data.index.isin(sw_set)
    retrain_pts = [i for i in range(tw, n) if (i - tw) % freq == 0]

    for rp_idx, rp in enumerate(retrain_pts):
        train_end = rp - fwd - 5
        if train_end <= 0: continue
        train_start = max(0, train_end - tw) if mode == "rolling" else 0
        X_all = factor_data.iloc[train_start:train_end]
        y_all = all_labels.iloc[train_start:train_end]
        sw_slice = sw_mask[train_start:train_end]
        valid = ~(X_all.isna().any(axis=1) | np.isinf(X_all.values).any(axis=1)
                  | y_all.isna() | np.isinf(y_all) | sw_slice)
        Xv, yv = X_all[valid], y_all[valid]
        if valid.sum() < 50: continue

        wm = getattr(args, "weight_method", "park")
        rw = getattr(args, "rolling_window", 60)
        hl = getattr(args, "ewma_halflife",  21)
        model, scaler = _train_wls(Xv, yv, use_sc, weight_method=wm,
                                   rolling_window=rw, ewma_halflife=hl)
        if first_model is None: first_model, first_sc = model, scaler

        if not in_sample_done and first_model is not None:
            Xin = factor_data.iloc[:tw]; vin = ~nan_mask[:tw]
            if vin.any():
                try:
                    Xin_v = Xin[vin].values
                    Xs = first_sc.transform(Xin_v) if first_sc else Xin_v
                    predictions[:tw][vin] = first_model.predict(Xs)
                except Exception: pass
            in_sample_done = True

        pred_end = retrain_pts[rp_idx + 1] if rp_idx + 1 < len(retrain_pts) else n
        vp_mask  = ~nan_mask[rp:pred_end]
        if vp_mask.any() and model is not None:
            try:
                Xb = factor_data.iloc[rp:pred_end][vp_mask].values
                Xs = scaler.transform(Xb) if scaler else Xb
                predictions[rp:pred_end][vp_mask] = model.predict(Xs)
            except Exception: pass

    trade_df = pd.DataFrame({"prediction": predictions, "price": price_data.values},
                             index=price_data.index)
    results_df  = _backtest(trade_df, args)
    performance = _performance(results_df, args)

    if model is not None:
        print_model_coefficients(model, scaler)
    return results_df, performance


def run_backtest_reg(factor_data, price_data, args):
    args.train_window = min(args.train_window, len(factor_data) - 1)
    args.split_point  = factor_data.index[args.train_window - 1]
    return _run_sliding_window_backtest(factor_data, price_data, args)


# ── 打印性能表 ────────────────────────────────────────────────────────────────

def print_performance_table(results_df, args):
    split        = getattr(args, "split_point", None)
    fwd          = getattr(args, "fwd", 1)
    init_capital = float(getattr(args, "init_capital", 7000.0))
    multiplier   = float(getattr(args, "multiplier",   10.0))
    commission   = float(getattr(args, "commission",   2e-4))

    def _row(df):
        if len(df) == 0:
            return {k: np.nan for k in ["Sharpe(点)","总PnL(点)","年化PnL(点)",
                                         "最大回撤(点)","胜率","盈亏比",
                                         "净PnL(元)","年化收益%","Sharpe(元)",
                                         "最大回撤(元)","总手续费","交易次数"]}
        _key  = pd.to_datetime(df.index).normalize()
        daily = df.groupby(_key).agg(
            r=("strategy_pnl","sum"), r_cny=("net_pnl_cny","sum"),
            comm=("commission_cny","sum"), sw=("is_switch","any"),
            hp=("position", lambda x: (x!=0).any()))
        rv = daily[~daily["sw"]]
        _r_raw     = rv["r"].values.astype(np.float64)
        _r_cny_raw = rv["r_cny"].values.astype(np.float64)
        r     = np.where(np.isfinite(_r_raw),     _r_raw,     0.0)
        r_cny = np.where(np.isfinite(_r_cny_raw), _r_cny_raw, 0.0)
        mu = r.mean(); sigma = r.std()
        sr = mu / sigma * np.sqrt(252) if sigma > 0 else 0
        cum = np.cumsum(r); peak = np.maximum.accumulate(cum); mdd = (cum - peak).min()
        _act_raw = daily[~daily["sw"] & daily["hp"]]["r"].values.astype(np.float64)
        active   = np.where(np.isfinite(_act_raw), _act_raw, 0.0)
        pos_r, neg_r = active[active > 0], active[active < 0]
        wr  = len(pos_r) / (len(pos_r) + len(neg_r)) if (len(pos_r) + len(neg_r)) > 0 else 0
        plr = pos_r.mean() / abs(neg_r.mean()) if (len(pos_r) and len(neg_r) and np.isfinite(neg_r.mean()) and neg_r.mean() != 0) else 0
        mu_c = r_cny.mean(); sig_c = r_cny.std()
        sr_c = mu_c / sig_c * np.sqrt(252) if sig_c > 0 else 0
        cum_c = np.cumsum(r_cny); peak_c = np.maximum.accumulate(cum_c)
        mdd_c = (cum_c - peak_c).min()
        n_trades = int(((df["position"].shift(1).fillna(0) == 0) & (df["position"] != 0)).sum())
        return {
            "Sharpe(点)": sr, "总PnL(点)": r.sum(), "年化PnL(点)": mu * 252,
            "最大回撤(点)": mdd, "胜率": wr, "盈亏比": plr,
            "净PnL(元)": r_cny.sum(), "年化收益%": mu_c * 252 / init_capital * 100,
            "Sharpe(元)": sr_c, "最大回撤(元)": mdd_c,
            "总手续费": rv["comm"].sum(), "交易次数": n_trades,
        }

    in_df  = results_df[results_df.index < split] if split else results_df
    out_df = pd.DataFrame()
    if split and split in results_df.index:
        si = results_df.index.get_loc(split)
        if isinstance(si, slice): si = si.stop - 1 if si.stop is not None else len(results_df) - 1
        elif isinstance(si, np.ndarray): si = int(np.flatnonzero(si)[-1])
        out_df = results_df.iloc[si + fwd + 2:]

    rows = [("全样本", _row(results_df)), ("样本内", _row(in_df)),
            ("样本外", _row(out_df) if len(out_df) else {k: np.nan for k in _row(in_df)})]
    cols   = ["Sharpe(点)","总PnL(点)","年化PnL(点)","最大回撤(点)","胜率","盈亏比",
              "净PnL(元)","年化收益%","Sharpe(元)","最大回撤(元)","总手续费","交易次数"]
    widths = [12, 14, 14, 14, 10, 10, 14, 12, 12, 14, 12, 10]

    print(f"\n{'='*140}")
    print(f"回测性能指标  初始资金={init_capital:.0f}元  乘数={multiplier:.0f}  手续费={commission*1e4:.1f}万分之一")
    print(f"{'='*140}")
    print(f"{'样本':^10}" + "".join(f"{c:^{w}}" for c, w in zip(cols, widths)))
    print("-" * 140)
    for name, m in rows:
        line = f"{name:^10}"
        for c, w in zip(cols, widths):
            v = m[c]
            if pd.isna(v) or (isinstance(v, float) and np.isinf(v)): line += f"{'--':^{w}}"
            elif c == "胜率":       line += f"{v*100:^{w}.2f}%"
            elif c == "年化收益%":  line += f"{v:^{w}.2f}%"
            elif c == "交易次数":   line += f"{int(v):^{w}}"
            else:                   line += f"{v:^{w}.2f}"
        print(line)
    print("=" * 140 + "\n")


# ── 保存结果 ──────────────────────────────────────────────────────────────────

def save_results(args, factor_cols, output_dir, performance, results_df):
    os.makedirs(output_dir, exist_ok=True)
    df_save = results_df.copy()
    idx = pd.to_datetime(df_save.index)
    df_save.insert(0, "datetime", idx)
    df_save.insert(1, "date", idx.date)
    df_save.insert(2, "time", idx.strftime("%H:%M"))
    df_save.index.name = None
    df_save.to_csv(os.path.join(output_dir, "results_df.csv"), index=False)
    with open(os.path.join(output_dir, "performance_summary.txt"), "w", encoding="utf-8") as f:
        f.write("因子回测性能摘要（绝对价差版）\n" + "=" * 50 + "\n")
        for k, v in performance.items():
            if not isinstance(v, (list, dict)): f.write(f"{k}: {v}\n")
    params = {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
              for k, v in vars(args).items() if k not in ("contract_switch_dates",)}
    params["factor_cols"] = list(factor_cols)
    with open(os.path.join(output_dir, "backtest_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    _plot_results(results_df, performance, args, output_dir)


# ── 绘图 ─────────────────────────────────────────────────────────────────────

def _plot_single(results_df, args, output_dir, mode="all"):
    """
    左轴（蓝色实线）：资金曲线（元），日线
    右轴（橙色实线）：原始价格，bar级完整，不截断，不聚合
    """
    split = getattr(args, "split_point", None)
    fwd   = getattr(args, "fwd", 1)

    if mode == "in" and split:
        df, sfx, ttl = results_df[results_df.index < split], "_in_sample", "（样本内）"
    elif mode == "out" and split and split in results_df.index:
        si = results_df.index.get_loc(split)
        if isinstance(si, slice):
            si = si.stop - 1 if si.stop is not None else len(results_df) - 1
        elif isinstance(si, np.ndarray):
            si = int(np.flatnonzero(si)[-1])
        df, sfx, ttl = results_df.iloc[si + fwd + 2:], "_out_of_sample", "（样本外）"
    else:
        df, sfx, ttl = results_df, "", ""
    if len(df) == 0:
        return

    init_capital = float(getattr(args, "init_capital", 7000.0))

    _key  = pd.to_datetime(df.index).normalize()
    daily = df.groupby(_key).agg(
        r_cny=("net_pnl_cny",   "sum"),
        r_pts=("strategy_pnl",  "sum"),
        comm =("commission_cny","sum"),
        sw   =("is_switch",     "any"),
        hp   =("position",      lambda x: (x != 0).any()),
    )
    rv     = daily[~daily["sw"]]
    r_cny  = rv["r_cny"].values
    r_pts  = rv["r_pts"].values
    n_days = len(r_cny)
    if n_days == 0:
        return

    equity_daily = init_capital + np.cumsum(r_cny)
    raw_price = df["price"].values
    n_bars    = len(raw_price)
    bar_x = np.linspace(0, n_days - 1, n_bars)

    mu_c = r_cny.mean(); sig_c = r_cny.std()
    sr_c = mu_c / sig_c * np.sqrt(252) if sig_c > 0 else 0
    peak_e  = np.maximum.accumulate(equity_daily)
    mdd_cny = (equity_daily - peak_e).min()
    ann_ret_pct = mu_c * 252 / init_capital * 100

    mu_p = r_pts.mean(); sig_p = r_pts.std()
    sr_p = mu_p / sig_p * np.sqrt(252) if sig_p > 0 else 0
    cum_pts = np.cumsum(r_pts)
    peak_p  = np.maximum.accumulate(cum_pts)
    mdd_pts = (cum_pts - peak_p).min()

    _act_raw     = rv[rv["hp"]]["r_pts"].values.astype(np.float64)
    active       = np.where(np.isfinite(_act_raw), _act_raw, 0.0)
    pos_r, neg_r = active[active > 0], active[active < 0]
    wr  = len(pos_r) / len(active) if len(active) else 0
    plr = pos_r.mean() / abs(neg_r.mean()) if (len(pos_r) and len(neg_r) and np.isfinite(neg_r.mean()) and neg_r.mean() != 0) else 0

    fig = plt.figure(figsize=(16, 10))
    ax  = fig.add_axes([0.07, 0.32, 0.86, 0.60])
    ax2 = ax.twinx()
    fig.suptitle(f"回测结果{ttl}", fontsize=16, fontweight="bold", y=0.97)

    ax2.plot(bar_x, raw_price, color="#F28522", linewidth=0.9,
             alpha=0.50, label="原始价格", zorder=1)
    ax2.set_ylabel("价格", fontsize=11, color="#F28522")
    ax2.tick_params(axis="y", labelcolor="#F28522", labelsize=9)
    ax2.spines["right"].set_color("#F28522")
    p_min, p_max = raw_price.min(), raw_price.max()
    p_pad = (p_max - p_min) * 0.10
    ax2.set_ylim(p_min - p_pad, p_max + p_pad)

    ax.plot(np.arange(n_days), equity_daily, color="#2878B5", linewidth=2.2,
            label="资金曲线（元）", zorder=3)
    ax.axhline(init_capital, color="#AAAAAA", linestyle="--",
               linewidth=0.9, alpha=0.7, zorder=2)
    ax.set_ylabel("资金（元）", fontsize=11, color="#2878B5")
    ax.tick_params(axis="y", labelcolor="#2878B5", labelsize=9)
    ax.spines["left"].set_color("#2878B5")

    if mode == "all" and split is not None:
        try:
            split_day = pd.Timestamp(split).normalize()
            sp_pos    = int(pd.DatetimeIndex(rv.index)
                           .get_indexer([split_day], method="nearest")[0])
            if 0 <= sp_pos < n_days:
                ax.axvline(sp_pos, color="#E84F4F", linestyle="--",
                           linewidth=1.6, alpha=0.85, label="样本分割", zorder=4)
        except Exception:
            pass

    ticks = list(range(0, n_days, max(1, n_days // 10)))
    dates = list(rv.index)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [pd.Timestamp(dates[i]).strftime("%Y-%m-%d") for i in ticks],
        rotation=30, ha="right", fontsize=10,
    )
    ax.set_xlim(-n_days * 0.01, n_days * 1.01)
    ax2.set_xlim(-n_days * 0.01, n_days * 1.01)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=10, loc="upper left", framealpha=0.85)

    col_names = ["Sharpe(元)", "年化收益%", "最大回撤(元)",
                 "Sharpe(点)", "总PnL(点)", "最大回撤(点)",
                 "胜率", "盈亏比", "总手续费(元)", "交易次数"]
    vals = [
        f"{sr_c:.2f}", f"{ann_ret_pct:.1f}%", f"{mdd_cny:.0f}",
        f"{sr_p:.2f}", f"{cum_pts[-1]:.1f}",  f"{mdd_pts:.1f}",
        f"{wr*100:.1f}%", f"{plr:.2f}",
        f"{rv['comm'].sum():.0f}", f"{int(daily['hp'].sum())}",
    ]
    ncols  = len(col_names)
    ax_tbl = fig.add_axes([0.07, 0.03, 0.91, 0.21])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=[col_names, vals],
                       cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for j in range(ncols):
        tbl[(0, j)].set_facecolor("#DDEEFF")
        tbl[(0, j)].set_text_props(weight="bold")
        tbl[(0, j)].set_edgecolor("#AAAAAA")
        tbl[(1, j)].set_facecolor("#FFFFFF")
        tbl[(1, j)].set_edgecolor("#AAAAAA")

    plt.savefig(os.path.join(output_dir, f"backtest_summary{sfx}.png"),
                dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_results(results_df, performance, args, output_dir):
    _plot_single(results_df, args, output_dir, "all")
    if performance.get("split_point"):
        _plot_single(results_df, args, output_dir, "in")
        _plot_single(results_df, args, output_dir, "out")