"""
utils2.py — 棕榈油回测工具函数（WLS 专用）
支持 bar 级别数据，_performance 聚合到日线后计算指标
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

import numpy as np
import pandas as pd
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Fix-1 & Fix-2 & Fix-3
# ══════════════════════════════════════════════════════════════════════════════

def apply_strength_filter(results_df, args):
    """
    只交易预测强度前 entry_strength_pct 的信号。

    参数说明（相比旧版的语义修正）
    ─────────────────────────────────────────────────────────────────
    entry_strength_pct = 0.3  → 只交易最强的 30%（过滤掉弱的 70%）
                       = 0.7  → 只交易最强的 70%（过滤掉弱的 30%，力度弱）
    建议值：0.2 ~ 0.4

    Fix-1：滚动分位数阈值
      entry_threshold[t] = rolling(abs(pred), threshold_window).quantile(1 - entry_strength_pct)
      样本外阈值跟随近期预测分布自动调整，解决量纲漂移问题。

    Fix-2：补全 zscore 系列平仓逻辑
      zscore_threshold / zscore_hybrid 现在能正确在强度过滤里触发平仓。

    Fix-3：开仓同时尊重 reg_threshold
      只有 abs(pred) >= entry_threshold AND abs(pred) >= reg_threshold 才开仓，
      两个阈值不再互相覆盖。
    """
    entry_pct = float(getattr(args, "entry_strength_pct", 0.3))
    threshold_window = int(getattr(args, "threshold_window", 500))  # Fix-1 滚动窗口
    reg_threshold = float(getattr(args, "reg_threshold", 0.0))
    close_mode = str(getattr(args, "close_mode", "fixed") or "fixed").lower()
    fwd = max(1, int(getattr(args, "fwd", 1)))
    ct = getattr(args, "close_threshold", [0, 0])
    long_close = ct[0] if isinstance(ct, (list, tuple)) else 0
    short_close = ct[1] if isinstance(ct, (list, tuple)) and len(ct) > 1 else 0

    pred = results_df["prediction"].copy()
    n = len(pred)

    # ── Fix-1：滚动分位数阈值 ─────────────────────────────────────────────
    #   abs_pred.rolling(w).quantile(1 - pct)
    #   头部不足 w 时用扩展窗口兜底，保证从第一根 bar 就有阈值
    abs_pred = pred.abs()
    q_level = 1.0 - entry_pct  # pct=0.3 → 取 0.7 分位数

    rolling_thr = (
        abs_pred
        .rolling(threshold_window, min_periods=max(30, threshold_window // 5))
        .quantile(q_level)
        .fillna(abs_pred.expanding(min_periods=30).quantile(q_level))  # 头部扩展兜底
        .fillna(abs_pred.quantile(q_level))  # 极端兜底
    )
    #
    # print(f"\n[强度过滤 Fix] entry_pct={entry_pct}  q_level={q_level:.0%}  "
    #       f"threshold_window={threshold_window}  close_mode={close_mode}")
    # print(f"  阈值统计: min={rolling_thr.min():.6f}  "
    #       f"mean={rolling_thr.mean():.6f}  max={rolling_thr.max():.6f}")

    # ── Fix-2：平仓模式解析 ───────────────────────────────────────────────
    use_pred_close = close_mode in {"threshold", "hybrid"}
    use_zscore_close = close_mode in {"zscore_threshold", "zscore_hybrid"}
    threshold_only = close_mode in {"threshold", "zscore_threshold"}

    # 预计算 zscore 序列（Fix-2：zscore 模式现在真的用到了）
    close_zscore = pd.Series(0.0, index=pred.index)
    cz_thr = 0.0
    if use_zscore_close:
        cz_win = int(getattr(args, "close_zscore_window", threshold_window))
        cz_thr = float(getattr(args, "close_zscore_threshold", 0.0))
        zm = pred.rolling(cz_win, min_periods=2).mean()
        zs = pred.rolling(cz_win, min_periods=2).std().replace(0, np.nan)
        close_zscore = ((pred - zm) / zs).ffill().fillna(0.0)
        print(f"  zscore close: window={cz_win}  threshold={cz_thr}")

    # ── 主循环 ────────────────────────────────────────────────────────────
    signal = pd.Series(0.0, index=pred.index)
    i = 0
    while i < n:
        pv = pred.iloc[i]
        thr = rolling_thr.iloc[i]

        # Fix-3：同时满足 entry_threshold 和 reg_threshold 才开仓
        if abs(pv) >= thr and abs(pv) > reg_threshold:
            pos = float(np.sign(pv))
            signal.iloc[i] = pos
            j = i + 1
            max_hold = n if threshold_only else min(i + fwd, n)

            while j < max_hold:
                should_close = False

                # 预测值阈值平仓
                if use_pred_close:
                    if pos > 0 and pred.iloc[j] <= -long_close:
                        should_close = True
                    elif pos < 0 and pred.iloc[j] >= short_close:
                        should_close = True

                # Fix-2：zscore 平仓（原来缺失）
                if not should_close and use_zscore_close:
                    z = close_zscore.iloc[j]
                    if pos > 0 and z <= -abs(cz_thr):
                        should_close = True
                    elif pos < 0 and z >= abs(cz_thr):
                        should_close = True

                if should_close:
                    i = j + 1
                    break
                signal.iloc[j] = pos
                j += 1
            else:
                i = max_hold
            continue
        i += 1

    # ── 重算收益 ──────────────────────────────────────────────────────────
    res = results_df.copy()
    res["signal"] = signal
    res["position"] = signal.shift(1).fillna(0)
    trade_open = res["open"] if "open" in res.columns else res["price"]
    res["price_return"] = trade_open.pct_change(fill_method=None).shift(-1).fillna(0)
    res["strategy_return"] = res["position"] * res["price_return"]

    if getattr(args, "contract_switch_dates", []):
        sw = set(pd.to_datetime(args.contract_switch_dates))
        res.loc[res.index.isin(sw), "strategy_return"] = 0.0

    res["cumulative_value"] = 1 + res["strategy_return"].cumsum()

    # orig = int((pred.abs() > reg_threshold).sum())
    # kept = int((signal != 0).sum())
    # total = n
    # print(f"[强度过滤 Fix] 超 reg_thr 的信号={orig}  "
    #       f"过滤后有仓={kept}  过滤比={max(0, 1 - kept / orig) * 100:.1f}%  "
    #       f"占全部 bar={kept / total * 100:.1f}%")
    return res


# ══════════════════════════════════════════════════════════════════════════════
# 对应修改 _backtest()：开仓后尊重强度过滤，不重复叠加阈值
# 主要改动：把 reg_threshold 的"方向判断"和"强度判断"分离
# ══════════════════════════════════════════════════════════════════════════════

def _backtest_fixed(trade_df, args):
    """
    修正版 _backtest：
      · reg_threshold 仅用于决定方向（>0 开多，< 0 开空，±reg_threshold 之间不开）
      · 不再兼顾"强度"，强度过滤统一由 apply_strength_filter 负责
      · 其余逻辑与原版完全一致
    """
    from utils2 import _compute_reversal_labels  # 保持对原始 utils 的引用

    fwd = max(1, int(getattr(args, "fwd", 1)))
    reg_threshold = float(getattr(args, "reg_threshold", 0.0))
    close_threshold = getattr(args, "close_threshold", [0, 0])
    close_mode = str(getattr(args, "close_mode", "fixed") or "fixed").lower()

    long_close = close_threshold[0] if isinstance(close_threshold, (list, tuple)) else 0
    short_close = (close_threshold[1]
                   if isinstance(close_threshold, (list, tuple)) and len(close_threshold) > 1
                   else 0)

    n = len(trade_df)
    predictions = trade_df["prediction"].values
    open_mask = np.ones(n, dtype=bool)  # 强度过滤由外部统一处理，这里全开

    use_pred_close = close_mode in {"threshold", "hybrid"}
    use_zscore_close = close_mode in {"zscore_threshold", "zscore_hybrid"}
    threshold_only = close_mode in {"threshold", "zscore_threshold"}

    close_zscore = np.zeros(n, dtype=float)
    cz_thr = 0.0
    if use_zscore_close:
        cz_win = int(getattr(args, "close_zscore_window", 50))
        cz_thr = float(getattr(args, "close_zscore_threshold", 0.0))
        ps = trade_df["prediction"]
        zm = ps.rolling(cz_win, min_periods=2).mean()
        zs = ps.rolling(cz_win, min_periods=2).std().replace(0, np.nan)
        close_zscore = ((ps - zm) / zs).ffill().fillna(0.0).values

    positions = np.zeros(n)
    i = 0
    while i < n:
        if open_mask[i]:
            pred = predictions[i]
            # reg_threshold 仅决定方向，不叠加强度
            if reg_threshold > 0:
                signal = (1.0 if pred > reg_threshold else
                          -1.0 if pred < -reg_threshold else 0.0)
            else:
                signal = 1.0 if pred > 0 else -1.0 if pred < 0 else 0.0

            if signal != 0:
                positions[i] = signal
                j = i + 1
                max_hold = n if threshold_only else min(i + fwd, n)

                while j < max_hold:
                    should_close = False
                    if use_pred_close:
                        if signal > 0 and predictions[j] <= -long_close:
                            should_close = True
                        elif signal < 0 and predictions[j] >= short_close:
                            should_close = True
                    if not should_close and use_zscore_close:
                        z = close_zscore[j]
                        if signal > 0 and z <= -abs(cz_thr):
                            should_close = True
                        elif signal < 0 and z >= abs(cz_thr):
                            should_close = True
                    if should_close:
                        i = j + 1
                        break
                    positions[j] = signal
                    j += 1
                else:
                    i = max_hold
                continue
        i += 1

    positions = np.roll(positions, 1)
    positions[0] = 0

    df = trade_df.copy()
    df["position"] = positions
    df["price_return"] = df["price"].pct_change(fill_method=None).fillna(0)

    sw = set(pd.to_datetime(getattr(args, "contract_switch_dates", [])))
    df["is_switch"] = df.index.isin(sw)

    df["strategy_return"] = df["position"] * df["price_return"]
    df.loc[df["is_switch"], "strategy_return"] = 0.0
    df["actual_return"] = df["price_return"].where(~df["is_switch"], 0.0)
    df["cumulative_value"] = 1 + df["strategy_return"].cumsum()
    df["daily_volume"] = (df["position"].diff().fillna(0) != 0).astype(int)

    return df[["prediction", "actual_return", "strategy_return",
               "daily_volume", "position", "cumulative_value", "price", "is_switch"]]

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
                df = pd.read_csv(data_file, encoding=enc); break
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
    # print(f"数据加载: {df.shape}  {df.index[0]} ~ {df.index[-1]}")
    return df


# ── 因子数据准备 ──────────────────────────────────────────────────────────────
def prepare_factor_data(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    selected_factors: Optional[List[str]] = None,
    lag: int = 1,
    factor_lags: Optional[List[int]] = None,
    add_session_features: bool = True,          # ← 新增开关
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    在原有逻辑前注入两列会话特征（x_ 开头，与因子列一致）：
      x_is_night     : 0=白盘 / 1=夜盘  ── 捕捉两盘系统性差异
      x_session_sin  : sin(π × 盘内归一化进度) ── 捕捉开/收盘效应（非线性）

    棕榈油交易时段（含午休）：
      白盘  09:00-11:30 / 13:30-15:00  （共 240 min，午休直接线性插值跳过）
      夜盘  21:00-23:00                 （共 120 min）
    若 index 无时间分量（日线数据），两列均填 0，不影响后续逻辑。
    """

    # ── 0. 注入会话特征 ────────────────────────────────────────────────────
    if add_session_features:
        df = df.copy()
        idx      = pd.to_datetime(df.index)
        has_time = (idx.hour != 0).any() or (idx.minute != 0).any()

        if has_time:
            hm = idx.hour * 60 + idx.minute          # 距午夜分钟数

            # ── 夜盘哑变量 ────────────────────────────────────────────────
            # 21:00(1260) ≤ hm < 23:00(1380)
            is_night = ((hm >= 1260) & (hm < 1380)).astype(float)

            # ── 盘内归一化进度 [0, 1] ─────────────────────────────────────
            # 白盘：09:00(540)–15:00(900)，午休(690–810)视为连续拉伸
            #        progress = (hm - 540) / 360
            # 夜盘：21:00(1260)–23:00(1380)
            #        progress = (hm - 1260) / 120
            # 其余时段（数据极罕见，如结算时间）默认 0.5
            progress = pd.Series(0.5, index=idx, dtype=float)

            day_mask   = (hm >= 540)  & (hm < 900)
            night_mask = (hm >= 1260) & (hm < 1380)

            progress[day_mask]   = (hm[day_mask]   - 540)  / 360.0
            progress[night_mask] = (hm[night_mask]  - 1260) / 120.0
            progress = progress.clip(0.0, 1.0)

            # sin(π·p)：盘初≈0 → 盘中≈1 → 盘尾≈0，天然捕捉开/收盘对称效应
            session_sin = np.sin(np.pi * progress)

            df["x_is_night"]    = is_night
            df["x_session_sin"] = session_sin.values if hasattr(session_sin, 'values') else session_sin

            night_cnt = int(np.sum(is_night > 0))
            day_cnt   = len(is_night) - night_cnt
            print(f"[会话特征] 白盘={day_cnt} bars  夜盘={night_cnt} bars  "
                  f"sin均值={session_sin.mean():.3f}")
        else:
            # 日线数据：填 0，不引入噪声
            df["x_is_night"]    = 0.0
            df["x_session_sin"] = 0.0
            print("[会话特征] 日线数据，x_is_night/x_session_sin 均填 0")

    # ── 1. 以下与原版完全一致 ──────────────────────────────────────────────
    price_cols = [c for c in df.columns if c.lower().startswith("y_")]
    if not price_cols:
        raise ValueError("数据中无 y_ 价格列")

    avail = [c for c in df.columns if c.lower().startswith("x_")]
    if selected_factors:
        # 保留用户指定因子 + 新增会话特征（x_is_night / x_session_sin 自动包含）
        session_cols = [c for c in ["x_is_night", "x_session_sin"] if c in avail]
        base_sel     = [f for f in selected_factors if f in avail]
        factor_cols  = list(dict.fromkeys(base_sel + session_cols))   # 去重保序
        missing = [f for f in selected_factors if f not in avail]
        if missing:
            logging.warning(f"因子不存在: {missing}")
    else:
        factor_cols = avail

    open_col = next((c for c in price_cols if c.lower() == "y_open"), None)

    fd, pd_ = df[factor_cols].copy(), df[price_cols[0]].copy()
    open_ = df[open_col].copy() if open_col else None

    # 滞后特征（原逻辑不变）
    if factor_lags and len(factor_lags) == len(factor_cols):
        parts, names, max_lag = [], [], max(factor_lags)
        for col, cl in zip(factor_cols, factor_lags):
            parts.append(fd[col]); names.append(col)
            for i in range(1, cl):
                parts.append(fd[col].shift(i)); names.append(f"{col}_{i}")
        fd = pd.concat(parts, axis=1); fd.columns = names
        factor_cols = names
        fd, pd_ = fd.iloc[max_lag - 1:], pd_.iloc[max_lag - 1:]
        if open_ is not None:
            open_ = open_.iloc[max_lag - 1:]
    elif lag > 1:
        lagged = [fd]
        for i in range(1, lag):
            s = fd.shift(i); s.columns = [f"{c}_{i}" for c in factor_cols]; lagged.append(s)
        fd = pd.concat(lagged, axis=1); factor_cols = list(fd.columns)
        fd, pd_ = fd.iloc[lag - 1:], pd_.iloc[lag - 1:]
        if open_ is not None:
            open_ = open_.iloc[lag - 1:]

    if start_date:
        fd, pd_ = fd.loc[start_date:], pd_.loc[start_date:]
        if open_ is not None:
            open_ = open_.loc[start_date:]

    fd = fd.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    valid = ~(fd.isna().any(axis=1) | fd.isin([np.inf, -np.inf]).any(axis=1) | pd_.isna())
    if open_ is not None:
        valid = valid & open_.notna()

    fd_valid = fd[valid]
    pd_valid = pd_[valid].copy()
    if open_ is not None:
        pd_valid.attrs["open_price"] = open_[valid].copy()
    return fd_valid, pd_valid, factor_cols


# ── 标签：带反转检测（向量化）────────────────────────────────────────────────

def _compute_reversal_labels(price_data: pd.Series, fwd: int,
                              check_days: int = 0, multiplier: float = 1.2) -> pd.Series:
    """
    向量化反转标签：fwd天收益率，若后续出现更强反转则替换。
    过滤"假突破"，让模型学习持续性方向。
    """
    base = (price_data.shift(-fwd) - price_data) / price_data
    labels = base.copy()

    replaced = pd.Series(False, index=price_data.index)
    for k in range(1, check_days + 1):
        rk = (price_data.shift(-(fwd + k)) - price_data) / price_data
        valid   = base.notna() & rk.notna() & ~np.isinf(base) & ~np.isinf(rk) & (base != 0)
        replace = valid & ~replaced & (base * rk < 0) & (rk.abs() > multiplier * base.abs())
        labels  = labels.where(~replace, rk)
        replaced = replaced | replace

    return labels


# ── WLS 模型 ──────────────────────────────────────────────────────────────────

class _WLS:
    """
    加权最小二乘，支持三种权重估计方法：
      weight_method="park"    — Park Test（默认）：辅助回归估计异方差，适合截面/日线
      weight_method="rolling" — 滚动窗口方差：捕捉时变波动率，适合分钟线时序
      weight_method="ewma"    — EWMA方差估计：近期残差权重高，远期自然衰减，适合时间衰减场景
    """

    def __init__(self, alpha: float = 0.0, feature_names=None,
                 weight_method: str = "park", rolling_window: int = 60,
                 ewma_halflife: int = 21):
        self.alpha          = alpha
        self.feature_names  = feature_names
        self.weight_method  = weight_method   # "park" | "rolling" | "ewma"
        self.rolling_window = rolling_window  # 仅 rolling 模式使用
        self.ewma_halflife  = ewma_halflife   # 仅 ewma 模式使用
        self.coef_          = None
        self.intercept_     = 0.0

    # ── 方法 1：Park Test（辅助回归估计异方差）──────────────────────────────
    def _weights_park(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        log(ε²) = Xγ + v  → OLS 估计 γ → w = 1/√exp(Xγ)
        理论依据：Feasible GLS，适合方差由特征决定的场景
        """
        res  = y - LinearRegression().fit(X, y).predict(X)
        logv = np.log(res ** 2 + 1e-6)
        Xc   = np.hstack([np.ones((len(X), 1)), X])
        beta = np.linalg.lstsq(Xc, logv, rcond=None)[0]
        w    = 1.0 / np.sqrt(np.exp(Xc @ beta) + 1e-6)
        return w / w.mean()

    # ── 方法 2：滚动窗口方差（非参数，适合时序）─────────────────────────────
    def _weights_rolling(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        先用 OLS 算残差，再对残差做滚动标准差估计局部波动率。
        w = 1/σ_rolling，捕捉金融时序中的波动率聚集效应。
        """
        res = y - LinearRegression().fit(X, y).predict(X)
        s   = pd.Series(res)
        sigma = (s.rolling(self.rolling_window, min_periods=max(5, self.rolling_window // 4))
                  .std()
                  .fillna(s.expanding(min_periods=5).std())  # 头部不足窗口用扩展窗口补
                  .fillna(s.std())                           # 极端情况兜底
                  .values)
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)         # 避免除零
        w = 1.0 / sigma
        return w / w.mean()

    # ── 方法 3：EWMA方差估计（近期残差权重高，远期自然衰减）─────────────────
    def _weights_ewma(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        改进点：
        1. 用 expanding OLS 计算残差，消除前视偏差
        2. 权重截断（winsorize），防止极端值主导回归
        3. 量纲与 rolling/park 统一（1/σ）
        """
        n = len(y)
        resid = np.zeros(n)

        # ── 改进1：expanding OLS 消除前视偏差 ──────────────
        # 第 t 天的残差只用 [0, t) 的数据拟合的系数来算
        min_fit = max(X.shape[1] + 2, 30)  # 最少样本数才开始算

        for t in range(min_fit, n):
            m = LinearRegression().fit(X[:t], y[:t])
            resid[t] = y[t] - m.predict(X[t:t + 1])[0]

        # 前 min_fit 天没有历史，用全样本OLS的残差填充（无法避免）
        if min_fit > 0:
            m_full = LinearRegression().fit(X, y)
            resid[:min_fit] = y[:min_fit] - m_full.predict(X[:min_fit])

        # ── 改进2：EWMA 估计方差 ──────────────────────────
        resid_sq = pd.Series(resid ** 2)
        var_est = (resid_sq
                   .ewm(halflife=self.ewma_halflife, min_periods=5)
                   .mean()
                   .fillna(resid_sq.expanding(min_periods=1).mean())
                   .values)
        var_est = np.where(var_est < 1e-12, 1e-12, var_est)

        # ── 改进3：量纲统一（1/σ 而非 1/σ²）─────────────────
        # 与 rolling、park 保持一致
        sigma_est = np.sqrt(var_est)
        w = 1.0 / (sigma_est + 1e-8)

        # ── 改进4：Winsorize 截断极端权重 ────────────────────
        p_low = np.percentile(w, 1)
        p_high = np.percentile(w, 99)
        w = np.clip(w, p_low, p_high)

        return w / w.mean()
    # def _weights_ewma(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    #     """
    #     EWMA 方差估计：半衰期控制衰减速度，近期残差自动获得更高权重。
    #     w = 1 / var_ewma，与时间衰减思想一致。
    #     """
    #     resid = y - LinearRegression().fit(X, y).predict(X)
    #     resid_sq = pd.Series(resid ** 2)
    #     var_est = (resid_sq.ewm(halflife=self.ewma_halflife, min_periods=5)
    #                .mean()
    #                .fillna(resid_sq.expanding(min_periods=1).mean())
    #                .values)
    #     var_est = np.where(var_est < 1e-12, 1e-12, var_est)
    #     w = 1.0 / var_est
    #     return w / w.mean()

    def _weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.weight_method == "rolling":
            return self._weights_rolling(X, y)
        if self.weight_method == "ewma":
            return self._weights_ewma(X, y)
        return self._weights_park(X, y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).flatten()
        W  = np.diag(self._weights(X, y))
        Xd = np.hstack([np.ones((len(X), 1)), X])
        A  = Xd.T @ W @ Xd + np.eye(Xd.shape[1]) * max(self.alpha, 1e-8)
        A[0, 0] -= max(self.alpha, 1e-8)      # 截距不正则化
        try:
            beta = np.linalg.solve(A, Xd.T @ W @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(A) @ (Xd.T @ W @ y)
        self.intercept_, self.coef_ = beta[0], beta[1:]

        # print(f"    ===== WLS [{self.weight_method}] =====")
        # print(f"    截距: {self.intercept_:.3e}")
        # names = self.feature_names or [f"x{j}" for j in range(len(self.coef_))]
        # for n, c in sorted(zip(names, self.coef_), key=lambda t: abs(t[1]), reverse=True):
        #     print(f"    {n}: {c:.3e}")
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def _train_wls(X_train: pd.DataFrame, y_train: pd.Series,
               use_scaler: bool = True,
               weight_method: str = "park",
               rolling_window: int = 60,
               ewma_halflife: int = 21):
    """
    训练 WLS，返回 (model, scaler)。
    weight_method: "park"（默认）、"rolling" 或 "ewma"
    rolling_window: 仅 rolling 模式生效，默认 60
    ewma_halflife: 仅 ewma 模式生效，默认 21
    """
    scaler = None
    X = X_train.values
    if use_scaler:
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        X = scaler.fit_transform(X)
    model = _WLS(
        feature_names  = list(X_train.columns),
        weight_method  = weight_method,
        rolling_window = rolling_window,
        ewma_halflife  = ewma_halflife,
    )
    model.fit(X, y_train.values)
    return model, scaler


# ── 回测核心 ──────────────────────────────────────────────────────────────────

def _backtest(trade_df: pd.DataFrame, args) -> pd.DataFrame:
    """
    prediction → position → strategy_return
    支持 close_mode: fixed / threshold / hybrid / zscore_threshold / zscore_hybrid
    支持 open_zscore 过滤 + reg_threshold 开仓阈值
    """
    fwd             = max(1, int(getattr(args, "fwd", 1)))
    reg_threshold   = float(getattr(args, "reg_threshold", 0.0))
    close_threshold = getattr(args, "close_threshold", [0, 0])
    close_mode      = str(getattr(args, "close_mode", "fixed") or "fixed").lower()
    if close_mode == "zscore":
        close_mode = "zscore_threshold"

    long_close  = close_threshold[0] if isinstance(close_threshold, (list, tuple)) else 0
    short_close = close_threshold[1] if isinstance(close_threshold, (list, tuple)) and len(close_threshold) > 1 else 0

    n           = len(trade_df)
    predictions = trade_df["prediction"].values
    trade_open  = trade_df["open"] if "open" in trade_df.columns else trade_df["price"]

    # # ── 开仓 zscore 过滤（已禁用，改用 strength_filter）──
    # zscore_window    = int(getattr(args, "open_zscore_window", 0))
    # zscore_threshold = float(getattr(args, "open_zscore_threshold", 1.0))
    # if zscore_window > 1:
    #     ap   = trade_df["prediction"].abs()
    #     zmean = ap.rolling(zscore_window, min_periods=1).mean()
    #     zstd  = ap.rolling(zscore_window, min_periods=1).std().replace(0, np.nan)
    #     open_mask = ((ap - zmean) / zstd).abs().ge(zscore_threshold).fillna(False).values
    #     print(f"  [Backtest] zscore filter window={zscore_window} thr={zscore_threshold} "
    #           f"pass={open_mask.sum()}/{n}")
    # else:
    #     open_mask = np.ones(n, dtype=bool)
    open_mask = np.ones(n, dtype=bool)

    # ── 平仓 zscore ──
    use_pred_close  = close_mode in {"threshold", "hybrid"}
    use_zscore_close = close_mode in {"zscore_threshold", "zscore_hybrid"}
    threshold_only  = close_mode in {"threshold", "zscore_threshold"}

    close_zscore = np.zeros(n, dtype=float)
    if use_zscore_close:
        cz_win = int(getattr(args, "close_zscore_window", 50)) or 50
        cz_thr = float(getattr(args, "close_zscore_threshold", 0.0))
        ps = trade_df["prediction"]
        zm = ps.rolling(cz_win, min_periods=2).mean()
        zs = ps.rolling(cz_win, min_periods=2).std().replace(0, np.nan)
        close_zscore = ((ps - zm) / zs).ffill().fillna(0.0).values
    #     print(f"  [Backtest] close zscore win={cz_win} thr={cz_thr}")
    # elif use_pred_close:
    #     print(f"  [Backtest] close threshold long<={-long_close:.4f} short>={short_close:.4f}")
    # else:
    #     print(f"  [Backtest] close mode=fixed hold={fwd}bars")

    positions          = np.zeros(n)
    early_close_count  = fixed_close_count = 0
    i = 0

    while i < n:
        if open_mask[i]:
            pred = predictions[i]
            if reg_threshold > 0:
                signal = 1.0 if pred > reg_threshold else (-1.0 if pred < -reg_threshold else 0.0)
            else:
                signal = 1.0 if pred > 0 else (-1.0 if pred < 0 else 0.0)

            if signal != 0:
                positions[i] = signal
                j = i + 1
                max_hold = n if threshold_only else min(i + fwd, n)

                while j < max_hold:
                    should_close = False
                    if use_pred_close:
                        if signal > 0 and predictions[j] <= -long_close:
                            should_close = True
                        elif signal < 0 and predictions[j] >= short_close:
                            should_close = True
                    if not should_close and use_zscore_close:
                        z = close_zscore[j]
                        if signal > 0 and z <= cz_thr:
                            should_close = True
                        elif signal < 0 and z >= -cz_thr:
                            should_close = True

                    if should_close:
                        early_close_count += 1
                        i = j + 1
                        break
                    positions[j] = signal
                    j += 1
                else:
                    if not threshold_only:
                        fixed_close_count += 1
                    i = max_hold
                continue
        i += 1

    # shift(1): T日信号在T+1执行
    positions = np.roll(positions, 1); positions[0] = 0

    df = trade_df.copy()
    df["position"]     = positions
    df["price_return"] = trade_open.pct_change(fill_method=None).shift(-1).fillna(0)

    sw = set(pd.to_datetime(getattr(args, "contract_switch_dates", [])))
    df["is_switch"] = df.index.isin(sw)
    # if sw:
        # print(f"  [Backtest] 剔除切换日 {df['is_switch'].sum()} bars")

    df["strategy_return"]  = df["position"] * df["price_return"]
    df.loc[df["is_switch"], "strategy_return"] = 0.0
    df["actual_return"]    = df["price_return"].where(~df["is_switch"], 0.0)
    df["cumulative_value"] = 1 + df["strategy_return"].cumsum()
    df["daily_volume"]     = (df["position"].diff().fillna(0) != 0).astype(int)

    out_cols = ["prediction", "actual_return", "strategy_return",
                "daily_volume", "position", "cumulative_value", "price", "is_switch"]
    if "open" in df.columns:
        out_cols.append("open")
    return df[out_cols]


# ── 指标计算（聚合到日线）────────────────────────────────────────────────────

def calc_metrics_from_returns(returns, positions=None) -> dict:
    """从日收益率数组计算绩效（唯一实现）"""
    r = np.asarray(returns, dtype=float)
    empty = dict(sharpe_ratio=0, annual_return=0, calmar_ratio=0, sortino_ratio=0,
                 max_drawdown=0, volatility=0, win_rate=0, profit_loss_ratio=0,
                 total_return=0, trade_count=0)
    if len(r) == 0:
        return empty

    mu, sigma = r.mean(), r.std()
    cum  = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    mdd  = ((cum - peak) / peak).min()
    ann  = mu * 252
    vol  = sigma * np.sqrt(252)
    sr   = mu / sigma * np.sqrt(252) if sigma > 0 else 0
    neg  = r[r < 0]
    ds   = neg.std() if len(neg) else 0
    pos_r, neg_r = r[r > 0], r[r < 0]
    wr  = len(pos_r) / (len(pos_r) + len(neg_r)) if len(pos_r) + len(neg_r) > 0 else 0
    plr = pos_r.mean() / abs(neg_r.mean()) if len(pos_r) and len(neg_r) else 0
    return dict(
        sharpe_ratio=sr, annual_return=ann,
        calmar_ratio=ann / abs(mdd) if mdd else 0,
        sortino_ratio=mu / ds * np.sqrt(252) if ds > 0 else 0,
        max_drawdown=mdd, volatility=vol, win_rate=wr, profit_loss_ratio=plr,
        total_return=cum[-1] - 1,
        trade_count=int(np.sum(np.asarray(positions) != 0)) if positions is not None else 0,
    )


def recalc_performance(results_df: pd.DataFrame, args) -> dict:
    return _performance(results_df, args)


def _performance(results_df: pd.DataFrame, args) -> dict:
    """
    绩效统计：bar 级收益聚合到日线后计算，与 bar 频率无关。
    """
    split_point = getattr(args, "split_point", None)

    df    = results_df.copy()
    _key  = pd.to_datetime(df.index).normalize()   # 日期级别，避免列名冲突
    daily = df.groupby(_key).agg(
        strategy_return=("strategy_return", "sum"),
        actual_return=("actual_return",   "sum"),
        is_switch=("is_switch",           "any"),
        has_pos=("position",              lambda x: (x != 0).any()),
    )

    valid = daily[~daily["is_switch"]]
    r     = valid["strategy_return"].values
    br    = valid["actual_return"].values
    n_days = len(r)
    # print(f"  [Perf] bars={len(df)} 日线={len(daily)} 有效日={n_days}")

    if n_days == 0:
        m = calc_metrics_from_returns([])
        m["split_point"] = split_point
        return m

    # 日线指标
    mu, sigma = r.mean(), r.std()
    ann  = mu * 252
    vol  = sigma * np.sqrt(252)
    sr   = mu / sigma * np.sqrt(252) if sigma > 0 else 0
    cum  = 1.0 + np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    mdd  = ((cum - peak) / peak).min()
    neg  = r[r < 0]
    ds   = neg.std() if len(neg) else 0
    sortino = mu / ds * np.sqrt(252) if ds > 0 else 0

    active = valid[valid["has_pos"]]["strategy_return"].values
    pos_r, neg_r = active[active > 0], active[active < 0]
    wr  = len(pos_r) / len(active) if len(active) else 0
    plr = pos_r.mean() / abs(neg_r.mean()) if len(pos_r) and len(neg_r) else 0
    # print(f"  [Perf] 胜率={wr:.2%} 盈利{len(pos_r)}日/有仓{len(active)}日")

    # 基准
    bmu   = br.mean(); bstd = br.std()
    b_ann = bmu * 252
    b_sr  = bmu / bstd * np.sqrt(252) if bstd > 0 else 0

    # 信息比率
    excess    = r - br
    te        = excess.std() * np.sqrt(252)
    ir        = (ann - b_ann) / te if te > 0 else 0

    # 总收益（bar 级别累积）
    total_return = results_df["cumulative_value"].iloc[-1] - 1

    return dict(
        total_return=total_return,
        annual_return=ann, volatility=vol, sharpe_ratio=sr,
        max_drawdown=mdd, calmar_ratio=ann / abs(mdd) if mdd else 0,
        sortino_ratio=sortino, win_rate=wr, profit_loss_ratio=plr,
        benchmark_annual_return=b_ann, benchmark_sharpe=b_sr,
        information_ratio=ir, return_drawdown_ratio=ann / abs(mdd) if mdd else 0,
        trade_count=int(results_df["daily_volume"].sum()),
        split_point=split_point,
    )


# ── 滑动窗口回测（批量预测优化）──────────────────────────────────────────────

def _run_sliding_window_backtest(factor_data: pd.DataFrame,
                                 price_data: pd.Series, args):
    """返回 (results_df, performance)"""
    n, tw, fwd  = len(factor_data), args.train_window, args.fwd
    freq, mode  = args.retrain_freq, args.mode
    use_sc      = getattr(args, "use_scaler", True)

    predictions = np.full(n, np.nan)
    model = scaler = first_model = first_sc = None
    in_sample_done = False

    # 预计算（只算一次）
    all_labels  = _compute_reversal_labels(
        price_data, fwd,
        check_days=getattr(args, "check_days", 3),
        multiplier=getattr(args, "multiplier", 1.2),
    )
    factor_arr  = factor_data.values
    nan_mask    = np.isnan(factor_arr).any(axis=1) | np.isinf(factor_arr).any(axis=1)

    retrain_pts = [i for i in range(tw, n) if (i - tw) % freq == 0]

    # for rp_idx, rp in enumerate(tqdm(retrain_pts, desc=f"{mode}回测")):
    for rp_idx, rp in enumerate(retrain_pts):
        train_end   = rp - fwd - 5
        if train_end <= 0:
            continue
        train_start = max(0, train_end - tw) if mode == "rolling" else 0

        X_all = factor_data.iloc[train_start:train_end]
        y_all = all_labels.iloc[train_start:train_end]
        valid = ~(X_all.isna().any(axis=1)
                  | np.isinf(X_all.values).any(axis=1)
                  | y_all.isna()
                  | np.isinf(y_all))
        Xv, yv = X_all[valid], y_all[valid]

        if valid.sum() < 50:
            print(f"    样本不足({valid.sum()})，跳过"); continue

        # print(f"  [{rp}/{n}] {mode}[{train_start}:{train_end}] n={valid.sum()}")

        # 全量训练
        wm  = getattr(args, "weight_method",  "park")
        rw  = getattr(args, "rolling_window", 60)
        hl  = getattr(args, "ewma_halflife",  21)
        model, scaler = _train_wls(Xv, yv, use_sc, weight_method=wm, rolling_window=rw, ewma_halflife=hl)
        if first_model is None:
            first_model, first_sc = model, scaler

        # 回填样本内预测（首次）
        if not in_sample_done and first_model is not None:
            Xin  = factor_data.iloc[:tw]
            vin  = ~nan_mask[:tw]
            if vin.any():
                try:
                    Xin_v = Xin[vin].values
                    Xs    = first_sc.transform(Xin_v) if first_sc else Xin_v
                    predictions[:tw][vin] = first_model.predict(Xs)
                except Exception:
                    pass
            in_sample_done = True

        # 批量预测：当前 rp → 下一 rp
        pred_end = retrain_pts[rp_idx + 1] if rp_idx + 1 < len(retrain_pts) else n
        vp_mask  = ~nan_mask[rp:pred_end]
        if vp_mask.any() and model is not None:
            try:
                Xb = factor_data.iloc[rp:pred_end][vp_mask].values
                Xs = scaler.transform(Xb) if scaler else Xb
                predictions[rp:pred_end][vp_mask] = model.predict(Xs)
            except Exception:
                pass

    trade_df = pd.DataFrame({
        "prediction":    predictions,
        "price":         price_data.values,
        "actual_return": price_data.pct_change().values,
    }, index=price_data.index)
    open_price = price_data.attrs.get("open_price")
    if open_price is not None:
        trade_df["open"] = open_price.reindex(price_data.index).values

    results_df  = _backtest(trade_df, args)
    performance = _performance(results_df, args)
    return results_df, performance


def run_backtest_reg(factor_data: pd.DataFrame, price_data: pd.Series, args):
    """入口：返回 (results_df, performance)"""
    args.train_window = min(args.train_window, len(factor_data) - 1)
    args.split_point  = factor_data.index[args.train_window - 1]
    return _run_sliding_window_backtest(factor_data, price_data, args)



# ── 打印性能表 ────────────────────────────────────────────────────────────────

def print_performance_table(results_df: pd.DataFrame, args):
    split = getattr(args, "split_point", None)
    fwd   = getattr(args, "fwd", 1)

    def _row(df):
        if len(df) == 0:
            return {k: np.nan for k in ["Sharpe","年化收益","Calmar","Sortino","最大回撤","波动率","胜率","盈亏比","总收益"]}
        # 聚合到日线
        _key = pd.to_datetime(df.index).normalize()
        d = df.groupby(_key).agg(
            r=("strategy_return","sum"), sw=("is_switch","any"),
            hp=("position", lambda x: (x != 0).any())
        )
        r   = d[~d["sw"]]["r"].values
        m   = calc_metrics_from_returns(r)
        active = d[~d["sw"] & d["hp"]]["r"].values
        pos_r, neg_r = active[active > 0], active[active < 0]
        wr  = len(pos_r) / len(active) if len(active) else 0
        plr = pos_r.mean() / abs(neg_r.mean()) if len(pos_r) and len(neg_r) else 0
        return dict(Sharpe=m["sharpe_ratio"], 年化收益=m["annual_return"],
                    Calmar=m["calmar_ratio"], Sortino=m["sortino_ratio"],
                    最大回撤=m["max_drawdown"], 波动率=m["volatility"],
                    胜率=wr, 盈亏比=plr, 总收益=m["total_return"])

    in_df  = results_df[results_df.index < split] if split else results_df
    out_df = pd.DataFrame()
    if split and split in results_df.index:
        si = results_df.index.get_loc(split)
        if isinstance(si, slice):
            si = si.stop - 1 if si.stop is not None else len(results_df) - 1
        elif isinstance(si, np.ndarray):
            si = int(np.flatnonzero(si)[-1])
        out_df = results_df.iloc[si + fwd + 2:]

    rows     = [("全样本", _row(results_df)), ("样本内", _row(in_df)),
                ("样本外", _row(out_df) if len(out_df) else {k: np.nan for k in _row(in_df)})]
    pct_keys = {"年化收益","最大回撤","波动率","胜率","总收益"}
    cols     = ["Sharpe","年化收益","Calmar","Sortino","最大回撤","波动率","胜率","盈亏比","总收益"]
    widths   = [10, 14, 10, 10, 12, 12, 10, 10, 14]

    print("\n" + "="*110 + "\n回测性能指标汇总\n" + "="*110)
    print(f"{'样本':^10}" + "".join(f"{c:^{w}}" for c, w in zip(cols, widths)))
    print("-" * 110)
    for name, m in rows:
        line = f"{name:^10}"
        for c, w in zip(cols, widths):
            v = m[c]
            if pd.isna(v) or (isinstance(v, float) and np.isinf(v)):
                line += f"{'--':^{w}}"
            elif c in pct_keys:
                line += f"{v*100:^{w}.2f}%"
            else:
                line += f"{v:^{w}.2f}"
        print(line)
    print("=" * 110 + "\n")


# ── 保存结果 ──────────────────────────────────────────────────────────────────

def save_results(args, factor_cols, output_dir, performance, results_df):
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "results_df.csv"))

    with open(os.path.join(output_dir, "performance_summary.txt"), "w", encoding="utf-8") as f:
        f.write("棕榈油因子回测性能摘要\n" + "="*50 + "\n")
        for k, v in performance.items():
            if not isinstance(v, (list, dict)):
                f.write(f"{k}: {v}\n")

    params = {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
              for k, v in vars(args).items() if k not in ("contract_switch_dates",)}
    params["factor_cols"] = list(factor_cols)
    with open(os.path.join(output_dir, "backtest_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    _plot_results(results_df, performance, args, output_dir)
    # print(f"结果已保存: {output_dir}")


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def _plot_single(results_df, args, output_dir, mode="all"):
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

    # ── 日线聚合 ──
    _key = pd.to_datetime(df.index).normalize()
    d    = df.groupby(_key).agg(
        r=("strategy_return", "sum"),
        br=("actual_return",  "sum"),
        sw=("is_switch",      "any"),
        hp=("position",       lambda x: (x != 0).any()),
    )
    rv    = d[~d["sw"]]
    r     = rv["r"].values
    br    = rv["br"].values
    n_days = len(r)
    if n_days == 0:
        return

    cum_s = np.cumprod(1 + r)
    cum_b = np.cumprod(1 + br)

    # ── 指标（日线）──
    mu, sigma = r.mean(), r.std()
    ann  = mu * 252
    vol  = sigma * np.sqrt(252)
    sr   = mu / sigma * np.sqrt(252) if sigma > 0 else 0
    cum_arr = 1.0 + np.cumsum(r)
    peak    = np.maximum.accumulate(cum_arr)
    mdd     = ((cum_arr - peak) / peak).min()
    neg     = r[r < 0]
    ds      = neg.std() if len(neg) else 0
    sortino = mu / ds * np.sqrt(252) if ds > 0 else 0
    calmar  = ann / abs(mdd) if mdd else 0

    active   = rv[rv["hp"]]["r"].values
    pos_r, neg_r = active[active > 0], active[active < 0]
    wr  = len(pos_r) / len(active) if len(active) else 0
    plr = pos_r.mean() / abs(neg_r.mean()) if len(pos_r) and len(neg_r) else 0
    total_ret = cum_s[-1] - 1

    # ── 画布 ──
    fig = plt.figure(figsize=(16, 9))
    # 上方图区留 70%，下方表格留 25%，中间 5% 间隔
    ax  = fig.add_axes([0.07, 0.32, 0.91, 0.60])   # [left, bottom, width, height]

    fig.suptitle(f"回测结果{ttl}", fontsize=16, fontweight="bold", y=0.97)

    # ── 折线 ──
    ax.plot(cum_s, label="策略", linewidth=1.8, color="#2878B5")
    ax.plot(cum_b, label="价格", linewidth=1.8, color="#F28522")
    ax.axhline(1, color="#AAAAAA", linestyle="--", linewidth=0.9, alpha=0.8)

    # ── 样本分割线（在日线坐标里定位）──
    if mode == "all" and split is not None:
        try:
            split_day = pd.Timestamp(split).normalize()
            rv_idx    = pd.DatetimeIndex(rv.index)
            sp_pos    = int(rv_idx.get_indexer([split_day], method="nearest")[0])
            if 0 <= sp_pos < n_days:
                ax.axvline(sp_pos, color="#E84F4F", linestyle="--",
                           linewidth=1.8, alpha=0.85, label="样本分割")
        except Exception:
            pass

    # ── 坐标轴 ──
    ax.set_ylabel("累积收益（日线）", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # X 轴：约 10 个刻度，只显示日期
    ticks  = list(range(0, n_days, max(1, n_days // 10)))
    dates  = list(rv.index)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [pd.Timestamp(dates[i]).strftime("%Y-%m-%d") for i in ticks],
        rotation=30, ha="right", fontsize=10
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlim(-n_days * 0.01, n_days * 1.01)

    ax.legend(fontsize=11, loc="upper left", framealpha=0.85,
              edgecolor="#CCCCCC", fancybox=False)

    # ── 底部表格 ──
    col_names = ["Sharpe", "年化收益", "Calmar", "Sortino",
                 "最大回撤", "波动率",  "胜率",   "盈亏比",  "总收益",  "交易次数"]
    vals      = [
        f"{sr:.2f}",
        f"{ann*100:.1f}%",
        f"{calmar:.2f}",
        f"{sortino:.2f}",
        f"{mdd*100:.1f}%",
        f"{vol*100:.1f}%",
        f"{wr*100:.1f}%",
        f"{plr:.2f}",
        f"{total_ret*100:.1f}%",
        f"{int(d['hp'].sum())}",
    ]

    ncols = len(col_names)
    # 表格区域 axes（单独 axes，不挂在 ax 上，避免 bbox 干扰）
    ax_tbl = fig.add_axes([0.07, 0.04, 0.91, 0.20])
    ax_tbl.axis("off")

    tbl = ax_tbl.table(
        cellText=[col_names, vals],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    for j in range(ncols):
        # 表头行
        cell = tbl[(0, j)]
        cell.set_facecolor("#DDEEFF")
        cell.set_text_props(weight="bold", fontsize=11)
        cell.set_edgecolor("#AAAAAA")
        cell.set_linewidth(0.8)
        # 数值行
        cell2 = tbl[(1, j)]
        cell2.set_facecolor("#FFFFFF")
        cell2.set_edgecolor("#AAAAAA")
        cell2.set_linewidth(0.8)

    plt.savefig(
        os.path.join(output_dir, f"backtest_summary{sfx}.png"),
        dpi=200, bbox_inches="tight", facecolor="white"
    )
    plt.close()


def _plot_results(results_df, performance, args, output_dir):
    _plot_single(results_df, args, output_dir, "all")
    if performance.get("split_point"):
        _plot_single(results_df, args, output_dir, "in")
        _plot_single(results_df, args, output_dir, "out")


# ── 集成 PnL 对比图 ─────────────────────────────────────────────────────────

# 调色板：单组用浅灰系，集成方法用鲜明色
_ENSEMBLE_COLORS = ["#2878B5", "#E84F4F", "#9B59B6", "#27AE60"]
_SINGLE_COLORS   = ["#BBBBBB", "#CCCCCC", "#AAAAAA", "#C0C0C0", "#B5B5B5",
                     "#D0D0D0", "#A0A0A0", "#B8B8B8"]


def plot_ensemble_pnl(
    single_results: dict[str, pd.DataFrame],
    ensemble_results: dict[str, pd.DataFrame],
    split_point,
    output_dir: str,
    filename: str = "ensemble_pnl_comparison.png",
):
    """
    绘制集成方案 vs 单组模型的累积 PnL 对比图。

    Parameters
    ----------
    single_results   : {"单组1": results_df, ...}
    ensemble_results : {"方法A(信号等权)": results_df, ...}
    split_point      : 样本内/外分割点
    output_dir       : 输出目录
    filename         : 文件名
    """

    def _to_daily_cum(results_df):
        """bar 级 strategy_return 聚合到日线累积净值"""
        df  = results_df.copy()
        key = pd.to_datetime(df.index).normalize()
        d   = df.groupby(key).agg(
            r=("strategy_return", "sum"),
            sw=("is_switch", "any") if "is_switch" in df.columns else ("strategy_return", lambda _: False),
        )
        r = d[~d["sw"]]["r"].values
        cum = np.cumprod(1 + r)
        return pd.Series(cum, index=d[~d["sw"]].index)

    # ── 准备数据 ──────────────────────────────────────────────────────────
    single_cums   = {k: _to_daily_cum(v) for k, v in single_results.items()}
    ensemble_cums = {k: _to_daily_cum(v) for k, v in ensemble_results.items()}

    # 公共日期轴
    all_idx = sorted(set().union(*(s.index for s in [*single_cums.values(), *ensemble_cums.values()])))
    if len(all_idx) == 0:
        return
    n_days = len(all_idx)

    # ── 画布 ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    ax  = fig.add_axes([0.07, 0.32, 0.91, 0.58])
    fig.suptitle("集成方案 PnL 对比", fontsize=16, fontweight="bold", y=0.97)

    # 单组：细线 + 浅色
    for i, (name, cum) in enumerate(single_cums.items()):
        c = _SINGLE_COLORS[i % len(_SINGLE_COLORS)]
        ax.plot(cum.reindex(all_idx).values, label=name,
                linewidth=1.0, color=c, alpha=0.6, linestyle="--")

    # 集成方法：粗线 + 鲜明色
    for i, (name, cum) in enumerate(ensemble_cums.items()):
        c = _ENSEMBLE_COLORS[i % len(_ENSEMBLE_COLORS)]
        ax.plot(cum.reindex(all_idx).values, label=name,
                linewidth=2.2, color=c, alpha=0.9)

    ax.axhline(1, color="#AAAAAA", linestyle="--", linewidth=0.9, alpha=0.8)

    # 样本分割线
    if split_point is not None:
        try:
            split_day = pd.Timestamp(split_point).normalize()
            idx_arr   = pd.DatetimeIndex(all_idx)
            sp_pos    = int(idx_arr.get_indexer([split_day], method="nearest")[0])
            if 0 <= sp_pos < n_days:
                ax.axvline(sp_pos, color="#E84F4F", linestyle=":",
                           linewidth=1.8, alpha=0.7, label="样本分割")
        except Exception:
            pass

    # ── 坐标轴 ────────────────────────────────────────────────────────────
    ax.set_ylabel("累积净值（日线）", fontsize=12)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    ticks = list(range(0, n_days, max(1, n_days // 10)))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [pd.Timestamp(all_idx[i]).strftime("%Y-%m-%d") for i in ticks],
        rotation=30, ha="right", fontsize=10,
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlim(-n_days * 0.01, n_days * 1.01)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.85,
              edgecolor="#CCCCCC", fancybox=False, ncol=2)

    # ── 底部指标表格 ──────────────────────────────────────────────────────
    col_names = ["方案", "Sharpe", "年化收益", "最大回撤", "总收益"]
    table_rows = []
    for name, cum in {**single_cums, **ensemble_cums}.items():
        r = cum.pct_change().fillna(cum.iloc[0] - 1).values
        m = calc_metrics_from_returns(r)
        table_rows.append([
            name,
            f"{m['sharpe_ratio']:.2f}",
            f"{m['annual_return']*100:.1f}%",
            f"{m['max_drawdown']*100:.1f}%",
            f"{m['total_return']*100:.1f}%",
        ])

    ax_tbl = fig.add_axes([0.07, 0.02, 0.91, 0.20])
    ax_tbl.axis("off")

    tbl = ax_tbl.table(
        cellText=[col_names] + table_rows,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    ncols = len(col_names)
    nrows = 1 + len(table_rows)
    for j in range(ncols):
        cell = tbl[(0, j)]
        cell.set_facecolor("#DDEEFF")
        cell.set_text_props(weight="bold", fontsize=10)
        cell.set_edgecolor("#AAAAAA")
        cell.set_linewidth(0.8)
    for i in range(1, nrows):
        for j in range(ncols):
            cell = tbl[(i, j)]
            cell.set_facecolor("#FFFFFF")
            cell.set_edgecolor("#AAAAAA")
            cell.set_linewidth(0.8)

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=200, bbox_inches="tight", facecolor="white",
    )
    plt.close()
    print(f"  [图表] 集成PnL对比图已保存: {os.path.join(output_dir, filename)}")
