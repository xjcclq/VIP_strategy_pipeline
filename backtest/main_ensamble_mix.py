"""
main_ensamble_mix.py — 混合模型集成（WLS + XGBoost 任意组合）
═══════════════════════════════════════════════════════════════════════════
输入
  · 多个 backtest_params.json 路径（每个 JSON 自带模型类型 + 因子列表 + 训练参数）
  · JSON 中有 "model": "xgb" 的用 XGBoost，否则用 WLS

集成方式（三种）
  方法 A：信号等权平均
  方法 C：位置多数投票
  方法 S：线性 Stacking（IS 段训练 Ridge 回归，OOS 段应用，标签=未来fwd日收益）

强度映射（--strength_map）
  集成信号生成后统一应用强度映射，默认为空（不过滤）。
  格式示例："0.2-0.5:0.7,0.5-1:1"
    · 每段 <强度下界>-<强度上界>:<仓位比例>
    · 强度 = 当前信号绝对值的滚动百分位排名（0~1）
    · 方法C（投票）：强度 = 同向票数/总数
    · 不在任何区间内的信号视为强度不足 → 仓位为 0

用法
  python main_ensamble_mix.py --jsons path/to/params1.json path/to/params2.json ...
"""

from __future__ import annotations

import sys
import json
import ast
import copy
import warnings
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

script_path  = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(script_path.parent))

from utils2 import (
    load_palm_oil_data,
    prepare_factor_data,
    run_backtest_reg,
    save_results,
    get_timestamp,
    apply_strength_filter,
    recalc_performance,
    calc_metrics_from_returns,
    plot_ensemble_pnl,
)
from Utils_xgb import run_backtest_xgb


# ══════════════════════════════════════════════════════════════════════════════
def parse_arguments():
    p = argparse.ArgumentParser(description="混合模型集成：从 YAML 配置读取 JSON 路径")
    p.add_argument("--config", type=str,
                   default=str(Path(__file__).resolve().parent / "config" / "mix_ensamble.yaml"),
                   help="YAML 配置文件路径，每行一个 backtest_params.json 路径")
    p.add_argument("--vote_threshold", type=float, default=0.5,
                   help="方法C：投票比例阈值")
    p.add_argument("--min_agree",      type=int,   default=0,
                   help="方法C：最少同向票数，0 = 用 vote_threshold 计算")
    p.add_argument("--sharpe_floor",   type=float, default=0.0,
                   help="Sharpe 加权时，低于此值的组合权重截为 0")
    p.add_argument("--use_delta",      action="store_true", default=False,
                   help="为每个因子追加 delta 特征（当期 - 上一期），默认不启用")

    # ── 新增：集成后统一强度映射 ──────────────────────────────────────────
    p.add_argument(
        "--strength_map", type=str, default='',
        help=(
            "集成信号强度 → 仓位映射表，默认空（不过滤）。\n"
            "格式：'<低>-<高>:<仓位>,<低>-<高>:<仓位>,...'\n"
            "示例：'0.2-0.5:0.7,0.5-1:1'\n"
            "强度 = 信号绝对值的滚动百分位排名（方法C为投票比例），值域 [0,1]。\n"
            "不落在任何区间的信号 → 仓位=0（视为强度不足）。"
        ),
    )
    p.add_argument(
        "--strength_window", type=int, default=500,
        help="计算强度百分位排名时使用的滚动窗口长度（默认 500 bars）"
    )

    args = p.parse_args()

    # 从 YAML 读取 JSON 路径列表
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    # 清理每行：去掉 ^ 、空格、空行
    args.jsons = [line.strip().rstrip("^").strip() for line in lines
                  if line.strip() and not line.strip().startswith("#")]
    print(f"  从 {cfg_path.name} 读取 {len(args.jsons)} 个 JSON 路径")
    return args


# ══════════════════════════════════════════════════════════════════════════════
# 强度映射工具
# ══════════════════════════════════════════════════════════════════════════════

def _parse_strength_map(s: str) -> list[tuple[float, float, float]]:
    """
    解析强度映射字符串，返回有序区间列表。

    输入示例: "0.2-0.5:0.7,0.5-1:1"
    返回:     [(0.2, 0.5, 0.7), (0.5, 1.0, 1.0)]  — (下界, 上界, 仓位比例)

    规则
    ----
    · 区间按下界升序排列；最后一个区间右端点含闭。
    · 不在任何区间内的强度值 → 仓位 = 0。
    · 字符串为空或仅含空白 → 返回 []（不过滤，使用原始符号）。
    """
    s = s.strip()
    if not s:
        return []

    result: list[tuple[float, float, float]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            range_part, pos_str = part.split(":")
            lo_str, hi_str = range_part.split("-")
            lo, hi, pos = float(lo_str), float(hi_str), float(pos_str)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"strength_map 格式错误：'{part}'。"
                f"期望格式 '下界-上界:仓位'，例如 '0.2-0.5:0.7'。"
            ) from exc
        if lo >= hi:
            raise ValueError(f"strength_map 区间无效（下界 >= 上界）：'{part}'")
        if not (0.0 <= pos <= 1.0):
            raise ValueError(f"strength_map 仓位比例须在 [0,1]：'{part}'")
        result.append((lo, hi, pos))

    result.sort(key=lambda x: x[0])
    return result


def _signal_to_strength(
    signal_abs: pd.Series,
    window: int = 500,
) -> pd.Series:
    """
    将信号绝对值转换为 [0,1] 强度（滚动百分位排名）。

    Parameters
    ----------
    signal_abs : 信号绝对值序列
    window     : 滚动窗口大小

    Returns
    -------
    strength : pd.Series，值域 [0, 1]
    """
    strength = (
        signal_abs
        .rolling(window, min_periods=max(50, window // 10))
        .rank(pct=True)
        .fillna(0.0)
    )
    return strength


def _apply_strength_map(
    signal: pd.Series,
    strength_map: list[tuple[float, float, float]],
    strength: pd.Series | None = None,
    window: int = 500,
) -> pd.Series:
    """
    根据强度映射表将连续信号转为分数仓位。

    Parameters
    ----------
    signal       : 原始连续信号（可正可负）
    strength_map : _parse_strength_map 的输出；为空则退化为原始符号
    strength     : 预计算的强度序列（[0,1]）；None 则自动用滚动百分位计算
    window       : 滚动百分位窗口

    Returns
    -------
    frac_pos : pd.Series，值为 {-1, -0.7, 0, 0.7, 1, ...}（浮点分数仓位）

    逻辑
    ----
    · 强度 = 信号绝对值的滚动百分位排名（0~1）
    · 对每个 bar，查找强度落在哪个区间，取对应仓位比例
    · 仓位方向 = sign(signal)
    · 不在任何区间内 → 仓位 = 0
    """
    if not strength_map:
        # 无映射：直接返回方向（±1 或 0）
        return np.sign(signal).astype(float)

    if strength is None:
        strength = _signal_to_strength(signal.abs(), window=window)

    frac_pos = pd.Series(0.0, index=signal.index)

    for i, (lo, hi, size) in enumerate(strength_map):
        is_last = (i == len(strength_map) - 1)
        if is_last:
            # 最后一个区间右端点闭合（含等于 hi）
            mask = (strength >= lo) & (strength <= hi)
        else:
            mask = (strength >= lo) & (strength < hi)
        frac_pos[mask] = size

    # 乘以信号方向（保留 0：强度不足不开仓）
    frac_pos = frac_pos * np.sign(signal)
    return frac_pos


def _apply_strength_map_vote(
    long_votes: pd.Series,
    short_votes: pd.Series,
    n_total: int,
    strength_map: list[tuple[float, float, float]],
    min_agree: int,
) -> pd.Series:
    """
    方法 C（投票）的强度映射版本。

    强度 = 同向票数 / n_total（投票比例，天然在 [0,1]）

    Parameters
    ----------
    long_votes, short_votes : 各 bar 上多头/空头票数
    n_total    : 参与投票的子模型总数
    strength_map : 强度→仓位映射；为空则用原始 min_agree 逻辑（±1 或 0）
    min_agree  : 最少同向票数（强度映射为空时生效）

    Returns
    -------
    frac_pos : 分数仓位 pd.Series
    """
    idx = long_votes.index

    if not strength_map:
        # 原始逻辑：满足 min_agree 则满仓
        pos = pd.Series(0, index=idx)
        pos[long_votes  >= min_agree] = 1
        pos[short_votes >= min_agree] = -1
        return pos.astype(float)

    # 强度 = 票数比例
    long_strength  = long_votes  / n_total
    short_strength = short_votes / n_total

    frac_pos = pd.Series(0.0, index=idx)

    for i, (lo, hi, size) in enumerate(strength_map):
        is_last = (i == len(strength_map) - 1)
        if is_last:
            long_mask  = (long_strength  >= lo) & (long_strength  <= hi)
            short_mask = (short_votes >= 1) & (short_strength >= lo) & (short_strength <= hi)
        else:
            long_mask  = (long_strength  >= lo) & (long_strength  < hi)
            short_mask = (short_votes >= 1) & (short_strength >= lo) & (short_strength < hi)

        frac_pos[long_mask]  = size
        frac_pos[short_mask] = -size

    # 多空都满足时：取绝对值更大那个（极少发生，保险处理）
    conflict = (long_votes >= 1) & (short_votes >= 1)
    if conflict.any():
        ls = long_strength[conflict]
        ss = short_strength[conflict]
        frac_pos[conflict] = np.where(ls >= ss, frac_pos[conflict].abs(), -frac_pos[conflict].abs())

    return frac_pos


# ══════════════════════════════════════════════════════════════════════════════
# 其他工具函数（与原版相同）
# ══════════════════════════════════════════════════════════════════════════════

def _load_combo_config(json_path: str) -> SimpleNamespace:
    """读取 backtest_params.json，返回 SimpleNamespace（兼容 args 接口）"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    ct = raw.get("close_threshold", [0.0, 0.0])
    if isinstance(ct, str):
        try:
            ct = ast.literal_eval(ct)
        except (ValueError, SyntaxError):
            ct = [0.0, 0.0]

    ns = SimpleNamespace(
        json_path           = json_path,
        data_file           = raw.get("data_file", ""),
        start_date          = raw.get("start_date", "2018-04-17"),
        model_type          = "xgboost" if raw.get("model") == "xgb" else "wls",
        train_window        = int(raw.get("train_window", 4000)),
        mode                = raw.get("mode", "rolling"),
        retrain_freq        = int(raw.get("retrain_freq", 500)),
        fwd                 = int(raw.get("fwd", 7)),
        lag                 = int(raw.get("lag", 1)),
        factor_lags         = raw.get("factor_lags", ""),
        use_scaler          = bool(raw.get("use_scaler", True)),
        check_days          = int(raw.get("check_days", 3)),
        multiplier          = float(raw.get("multiplier", 2.0)),
        weight_method       = raw.get("weight_method", "rolling"),
        rolling_window      = int(raw.get("rolling_window", 1000)),
        reg_threshold       = float(raw.get("reg_threshold", 0.0)),
        close_threshold     = ct,
        close_mode          = raw.get("close_mode", "threshold"),
        # 单组 strength_filter 在集成模式下关闭，由集成后的 strength_map 统一处理
        use_strength_filter = False,
        entry_strength_pct  = float(raw.get("entry_strength_pct", 0.5)),
        threshold_window    = int(raw.get("threshold_window", 500)),
        factor_cols         = raw.get("factor_cols", []),
        split_point         = raw.get("split_point"),
        add_session_features = bool(raw.get("add_session_features", True)),
        # XGBoost 专用
        top_n_features      = int(raw.get("top_n_features", 0)),
        n_estimators        = int(raw.get("n_estimators", 60)),
        max_depth           = int(raw.get("max_depth", 3)),
        learning_rate       = float(raw.get("learning_rate", 0.03)),
        subsample           = float(raw.get("subsample", 0.7)),
        colsample_bytree    = float(raw.get("colsample_bytree", 0.5)),
        min_child_weight    = int(raw.get("min_child_weight", 50)),
        reg_alpha           = float(raw.get("reg_alpha", 1.0)),
        reg_lambda          = float(raw.get("reg_lambda", 8.0)),
        # 占位
        contract_switch_dates = [],
        output_dir          = "",
    )
    return ns


def _get_is_oos_sharpe(results_df: pd.DataFrame, args) -> tuple[float, float]:
    split = getattr(args, "split_point", None)
    fwd   = getattr(args, "fwd", 1)

    def _sh(df):
        if len(df) < 50:
            return np.nan
        key = pd.to_datetime(df.index).normalize()
        d   = df.groupby(key).agg(r=("strategy_return", "sum"), sw=("is_switch", "any"))
        r   = d[~d["sw"]]["r"].values
        return float(calc_metrics_from_returns(r)["sharpe_ratio"]) if len(r) >= 20 else np.nan

    if split is None or split not in results_df.index:
        return np.nan, np.nan
    si = results_df.index.get_loc(split)
    return _sh(results_df.iloc[:si]), _sh(results_df.iloc[si + fwd + 2:])


def _calc_metrics(results_df: pd.DataFrame, args) -> dict:
    split = getattr(args, "split_point", None)
    fwd   = getattr(args, "fwd", 1)

    def _m(df):
        if len(df) < 30:
            return {}
        key = pd.to_datetime(df.index).normalize()
        d   = df.groupby(key).agg(r=("strategy_return", "sum"), sw=("is_switch", "any"))
        r   = d[~d["sw"]]["r"].values
        return calc_metrics_from_returns(r) if len(r) >= 10 else {}

    all_m = _m(results_df)
    if split is not None and split in results_df.index:
        si    = results_df.index.get_loc(split)
        is_m  = _m(results_df.iloc[:si])
        oos_m = _m(results_df.iloc[si + fwd + 2:])
    else:
        is_m = oos_m = {}

    def _get(m, k):
        return round(m.get(k, np.nan), 4) if m else np.nan

    return {
        "all_sharpe":  _get(all_m,  "sharpe_ratio"),
        "all_annual":  _get(all_m,  "annual_return"),
        "all_maxdd":   _get(all_m,  "max_drawdown"),
        "is_sharpe":   _get(is_m,   "sharpe_ratio"),
        "is_annual":   _get(is_m,   "annual_return"),
        "oos_sharpe":  _get(oos_m,  "sharpe_ratio"),
        "oos_annual":  _get(oos_m,  "annual_return"),
        "oos_maxdd":   _get(oos_m,  "max_drawdown"),
        "is_oos_gap":  round(float(_get(is_m, "sharpe_ratio") or 0)
                             - float(_get(oos_m, "sharpe_ratio") or 0), 4),
    }


def _run_single(factor_cols, base_fd, price_data, combo_args):
    """根据 combo_args.model_type 分发到 WLS 或 XGBoost"""
    avail = [c for c in factor_cols if c in base_fd.columns]
    ta    = copy.copy(combo_args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if ta.model_type == "xgboost":
            res, perf = run_backtest_xgb(base_fd[avail].copy(), price_data, ta)
        else:
            res, perf = run_backtest_reg(base_fd[avail].copy(), price_data, ta)
    if res is None or perf is None:
        return None, None, ta
    ta.split_point = perf.get("split_point")
    # 注意：单组的 use_strength_filter 已在 _load_combo_config 中关闭，
    # 强度过滤统一在集成信号层处理。
    return res, perf, ta


def _zscore_series(s: pd.Series, window: int = 500) -> pd.Series:
    mu  = s.rolling(window, min_periods=50).mean()
    std = s.rolling(window, min_periods=50).std().clip(lower=1e-8)
    return (s - mu) / std


def _build_ensemble_results(
    combined_signal: pd.Series,
    base_results_df: pd.DataFrame,
    reg_threshold:   float,
    strength_map:    list[tuple[float, float, float]],
    strength_window: int = 500,
) -> pd.DataFrame:
    """
    用集成信号（连续值）构建回测结果 DataFrame。

    流程
    ----
    1. 将信号存入 df["prediction"]
    2. 若 strength_map 非空：信号强度（滚动百分位） → 分数仓位
       否则：沿用 reg_threshold 阈值逻辑（±1 整数仓位）
    3. 仓位滞后 1 bar（bar 收盘后才能建仓）
    4. 计算 strategy_return / cumulative_value
    """
    df = base_results_df.copy()
    df["prediction"] = combined_signal.reindex(df.index).fillna(0.0)

    if strength_map:
        # ── 强度映射路径 ──────────────────────────────────────────────────
        # 强度 = 信号绝对值的滚动百分位排名
        strength = _signal_to_strength(df["prediction"].abs(), window=strength_window)
        df["signal_strength"] = strength  # 留存诊断列

        frac_pos = _apply_strength_map(
            signal       = df["prediction"],
            strength_map = strength_map,
            strength     = strength,
        )
        df["position"] = frac_pos.shift(1).fillna(0.0)
    else:
        # ── 原始阈值路径（整数仓位）────────────────────────────────────
        if reg_threshold > 0:
            df["position"] = np.where(
                df["prediction"] >  reg_threshold,  1,
                np.where(df["prediction"] < -reg_threshold, -1, 0)
            )
        else:
            df["position"] = np.sign(df["prediction"]).astype(int)
        df["position"] = df["position"].shift(1).fillna(0).astype(int)

    # ── 收益计算 ──────────────────────────────────────────────────────────
    ret_col = next((c for c in ["price_return", "actual_return", "bar_return"]
                     if c in df.columns), None)
    if ret_col is None:
        df["price_return"] = df["price"].pct_change(fill_method=None).fillna(0)
        ret_col = "price_return"

    df["strategy_return"] = df["position"] * df[ret_col]
    if "is_switch" in df.columns:
        df.loc[df["is_switch"], "strategy_return"] = 0.0
    df["cumulative_value"] = 1 + df["strategy_return"].cumsum()
    return df


# ── 线性 Stacking ────────────────────────────────────────────────────────────

def _build_stacking_signal(
    zscored: list[pd.Series],
    base_res: pd.DataFrame,
    split_point,
    fwd: int,
    common_idx: pd.Index,
) -> pd.Series:
    """
    IS 段训练 Ridge 回归，OOS 段应用，返回合并信号。
    特征：各子模型 z-score 信号；标签：未来 fwd 根 bar 累计收益率。
    """
    from sklearn.linear_model import Ridge

    feat_df = pd.DataFrame(
        {f"s{i}": z.reindex(common_idx).fillna(0.0) for i, z in enumerate(zscored)},
        index=common_idx,
    )

    ret_col = next(
        (c for c in ["price_return", "actual_return", "bar_return"] if c in base_res.columns),
        None,
    )
    if ret_col is None:
        base_res = base_res.copy()
        base_res["price_return"] = base_res["price"].pct_change(fill_method=None).fillna(0)
        ret_col = "price_return"

    fwd_ret = (
        base_res[ret_col]
        .reindex(common_idx)
        .fillna(0.0)
        .rolling(fwd)
        .sum()
        .shift(-fwd)
    )

    if split_point is not None and split_point in common_idx:
        si = common_idx.get_loc(split_point)
        print(f"     Stacking IS 样本: {si} bars  OOS 样本: {len(common_idx) - si} bars")
    else:
        si = int(len(common_idx) * 0.7)
        print(f"  [Stacking] split_point 未找到，默认用前 70% ({si} bars) 作 IS")

    valid_mask = ~fwd_ret.iloc[:si].isna()
    X_is = feat_df.iloc[:si][valid_mask].values
    y_is = fwd_ret.iloc[:si][valid_mask].values

    if len(y_is) < 30:
        print("  [Stacking] IS 有效样本不足 30，退化为等权平均")
        stk = sum(z.reindex(common_idx).fillna(0.0) for z in zscored) / len(zscored)
        stk.name = "signal_stack"
        return stk

    mdl = Ridge(alpha=0.00001, fit_intercept=False)
    mdl.fit(X_is, y_is)

    coef_str = "  ".join(f"s{i}={v:+.4f}" for i, v in enumerate(mdl.coef_))
    print(f"     Ridge 系数: {coef_str}")

    stk = pd.Series(
        mdl.predict(feat_df.values),
        index=common_idx,
        name="signal_stack",
    )
    return stk


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_arguments()
    ts   = get_timestamp()
    out  = Path(project_root) / "results" / f"mix_ensemble_{ts}"
    out.mkdir(parents=True, exist_ok=True)

    SEP = "═" * 76
    print(f"\n{SEP}")
    print(f"  混合模型集成（WLS + XGBoost）")
    print(f"  输入 {len(args.jsons)} 个 JSON 配置")

    # ── 解析强度映射 ──────────────────────────────────────────────────────
    strength_map = _parse_strength_map(args.strength_map)
    if strength_map:
        print(f"  强度映射（集成后统一应用）：")
        for lo, hi, size in strength_map:
            print(f"    [{lo:.2f}, {hi:.2f}) → 仓位 {size:.2f}")
        print(f"  强度百分位窗口：{args.strength_window} bars")
        print(f"  * 单组 use_strength_filter 已自动关闭，由集成层统一处理")
    else:
        print(f"  强度映射：未设置（默认整数仓位，阈值逻辑）")
    print(SEP)

    # ── 加载各组配置 ──────────────────────────────────────────────────────
    combo_configs: list[SimpleNamespace] = []
    for jp in args.jsons:
        cfg = _load_combo_config(jp)
        combo_configs.append(cfg)
        print(f"\n  [{len(combo_configs)}] {Path(jp).parent.name}")
        print(f"      模型: {cfg.model_type.upper()}  因子数: {len(cfg.factor_cols)}")
        print(f"      tw={cfg.train_window}  fwd={cfg.fwd}  freq={cfg.retrain_freq}")

    # ── 加载原始数据 ──────────────────────────────────────────────────────
    data_file  = combo_configs[0].data_file
    start_date = combo_configs[0].start_date

    df = load_palm_oil_data(data_file)
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if "dominant_id" in df.columns:
        mask = df["dominant_id"] != df["dominant_id"].shift(1)
        mask.iloc[0] = False
        switch_dates = df.index[mask].tolist()
    else:
        switch_dates = []

    all_x_cols = sorted([c for c in df.columns if c.startswith("x_")])

    # ══════════════════════════════════════════════════════════════════════
    # Step 1：各组独立回测
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Step 1 / 3  ─  各组独立回测\n{SEP}")

    combos: list[dict] = []
    split_point = None

    for i, cfg in enumerate(combo_configs):
        rank = i + 1
        cfg.contract_switch_dates = switch_dates

        lag_val = cfg.lag
        factor_lags = None
        fl_str = getattr(cfg, "factor_lags", "")
        if fl_str and fl_str.strip():
            try:
                factor_lags = [int(x) for x in fl_str.split(",") if x.strip()]
            except ValueError:
                factor_lags = None

        need_session = getattr(cfg, "add_session_features", True)
        if not need_session:
            need_session = any(f.startswith("x_is_night") or f.startswith("x_session_sin")
                               for f in cfg.factor_cols)

        base_fd, price_data, _ = prepare_factor_data(
            df, selected_factors=all_x_cols, lag=lag_val, factor_lags=factor_lags,
            add_session_features=need_session,
        )

        if args.use_delta:
            x_cols_in_fd = [c for c in base_fd.columns if c.startswith("x_")]
            delta_df = base_fd[x_cols_in_fd].diff()
            delta_df.columns = [f"{c}_delta" for c in x_cols_in_fd]
            delta_cols = list(delta_df.columns)
            base_fd = pd.concat([base_fd, delta_df], axis=1)
            valid_mask = base_fd[delta_cols[0]].notna()
            base_fd = base_fd.loc[valid_mask].copy()
            price_data = price_data.reindex(base_fd.index)
            delta_set = set(delta_cols)
            extra = [f"{c}_delta" for c in cfg.factor_cols if f"{c}_delta" in delta_set]
            cfg.factor_cols = cfg.factor_cols + extra

        print(f"\n  ── 组合 {rank}（{cfg.model_type.upper()}, {len(cfg.factor_cols)} 因子）──")
        print(f"     lag={lag_val}  bars={len(base_fd)}  会话特征={'开启' if need_session else '关闭'}")
        print(f"     {cfg.factor_cols[:6]}{'...' if len(cfg.factor_cols) > 6 else ''}")

        res, perf, ta = _run_single(cfg.factor_cols, base_fd, price_data, cfg)
        if res is None:
            print(f"  [警告] 组合 {rank} 回测失败，跳过")
            continue

        if split_point is None:
            split_point = ta.split_point

        raw_signal = res["prediction"].fillna(0.0)
        is_s, oos_s = _get_is_oos_sharpe(res, ta)
        print(f"     IS={is_s:.3f}  OOS={oos_s:.3f}（单组，未经强度映射）")

        combos.append({
            "rank":       rank,
            "model_type": cfg.model_type,
            "factors":    cfg.factor_cols,
            "results_df": res,
            "perf":       perf,
            "ta":         ta,
            "raw_signal": raw_signal,
            "position":   res["position"].fillna(0).astype(int),
            "is_sharpe":  is_s,
            "oos_sharpe": oos_s,
        })

        sub = out / f"single_{rank}_{cfg.model_type}"
        sub.mkdir(exist_ok=True)
        ta2 = copy.copy(ta)
        ta2.output_dir = str(sub)
        save_results(ta2, cfg.factor_cols, str(sub), perf, res)

    if len(combos) < 2:
        print("[错误] 有效组合不足 2 个，无法集成"); return

    n = len(combos)
    equal_w = np.ones(n) / n

    print(f"\n  各组摘要：")
    print(f"  {'#':>3}  {'模型':>6}  {'OOS Sharpe':>10}  {'等权':>8}")
    print(f"  {'─'*34}")
    for i, c in enumerate(combos):
        print(f"  {c['rank']:3d}  {c['model_type']:>6}  {c['oos_sharpe']:10.3f}  {equal_w[i]:8.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2：构建集成信号
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Step 2 / 3  ─  构建集成信号\n{SEP}")

    common_idx = combos[0]["raw_signal"].index
    for c in combos[1:]:
        common_idx = common_idx.intersection(c["raw_signal"].index)

    zscored: list[pd.Series] = [
        _zscore_series(c["raw_signal"].reindex(common_idx).fillna(0.0))
        for c in combos
    ]
    positions: list[pd.Series] = [
        c["position"].reindex(common_idx).fillna(0).astype(int)
        for c in combos
    ]

    # 方法 A：信号等权平均
    sig_A = sum(w * z for w, z in zip(equal_w, zscored))
    sig_A.name = "signal_A_equal"
    print("  方法A（信号等权平均）：z-score 后各组等权平均")

    # 方法 C：投票（强度映射会覆盖 min_agree 逻辑）
    vote_mat   = pd.concat(positions, axis=1)
    min_agree  = (args.min_agree if args.min_agree > 0
                  else max(1, int(np.ceil(n * args.vote_threshold))))
    long_votes  = (vote_mat == 1).sum(axis=1)
    short_votes = (vote_mat == -1).sum(axis=1)

    if strength_map:
        pos_C = _apply_strength_map_vote(
            long_votes   = long_votes,
            short_votes  = short_votes,
            n_total      = n,
            strength_map = strength_map,
            min_agree    = min_agree,
        )
        long_pct  = (pos_C > 0).mean()
        short_pct = (pos_C < 0).mean()
        print(f"  方法C（投票+强度映射）：强度=票数比例，"
              f"多头={long_pct:.1%}，空头={short_pct:.1%}")
    else:
        pos_C = pd.Series(0, index=common_idx)
        pos_C[long_votes  >= min_agree] = 1
        pos_C[short_votes >= min_agree] = -1
        print(f"  方法C（多数投票）：≥{min_agree}/{n} 同向才开仓，"
              f"多头={(pos_C==1).mean():.1%}，空头={(pos_C==-1).mean():.1%}")

    # 方法 S：线性 Stacking
    print("  方法S（线性Stacking）：IS段训练Ridge，OOS段应用，标签=未来fwd日收益")
    base_res = combos[0]["results_df"]
    fwd_val  = combos[0]["ta"].fwd
    sig_S = _build_stacking_signal(
        zscored    = zscored,
        base_res   = base_res,
        split_point = split_point,
        fwd        = fwd_val,
        common_idx  = common_idx,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Step 3：集成方案回测评估
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Step 3 / 3  ─  集成方案回测评估\n{SEP}")

    if strength_map:
        map_desc = "+" + args.strength_map
    else:
        map_desc = ""

    method_configs = [
        # (键, 描述,                   类型,        信号/仓位)
        ("A", f"信号等权平均{map_desc}",     "signal",   sig_A, None),
        ("C", f"位置多数投票{map_desc}",     "position", None,  pos_C),
        ("S", f"线性Stacking{map_desc}",    "signal",   sig_S, None),
    ]

    all_results:  dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    # 单组基准行
    for c in combos:
        m = _calc_metrics(c["results_df"], c["ta"])
        summary_rows.append({
            "方案": f"单组{c['rank']}({c['model_type']})",
            "描述": f"{c['model_type']} combo{c['rank']}",
            **m,
        })

    for mkey, mdesc, mtype, msig, mpos in method_configs:
        print(f"\n  ── 方法 {mkey}：{mdesc} ──")
        sub_dir = out / f"method_{mkey}"
        sub_dir.mkdir(exist_ok=True)

        if mtype == "signal":
            ens_res = _build_ensemble_results(
                combined_signal = msig,
                base_results_df = base_res,
                reg_threshold   = 0.0,
                strength_map    = strength_map,
                strength_window = args.strength_window,
            )
        else:
            # 方法 C：pos_C 已是分数仓位（由 _apply_strength_map_vote 生成）
            ens_res = base_res.copy()
            ens_res["position"] = mpos.reindex(ens_res.index).fillna(0.0)
            # 仓位滞后（方法C 的位置已经是当期信号，需同样滞后 1 bar）
            ens_res["position"] = ens_res["position"].shift(1).fillna(0.0)

            ret_col = next((c for c in ["price_return", "actual_return"]
                            if c in ens_res.columns), None)
            if ret_col is None:
                ens_res["price_return"] = ens_res["price"].pct_change(fill_method=None).fillna(0)
                ret_col = "price_return"
            ens_res["strategy_return"] = ens_res["position"] * ens_res[ret_col]
            if "is_switch" in ens_res.columns:
                ens_res.loc[ens_res["is_switch"], "strategy_return"] = 0.0
            ens_res["cumulative_value"] = 1 + ens_res["strategy_return"].cumsum()

        ta0 = copy.copy(combos[0]["ta"])
        ta0.split_point = split_point
        ta0.output_dir  = str(sub_dir)

        all_results[mkey] = ens_res
        m = _calc_metrics(ens_res, ta0)

        is_note = "（含训练偏差）" if mkey == "S" else ""
        print(f"     IS  Sharpe={m.get('is_sharpe', float('nan')):.3f}{is_note}"
              f"  Ann={m.get('is_annual', float('nan')):.1%}")
        print(f"     OOS Sharpe={m.get('oos_sharpe', float('nan')):.3f}"
              f"  Ann={m.get('oos_annual', float('nan')):.1%}"
              f"  MaxDD={m.get('oos_maxdd', float('nan')):.1%}")
        print(f"     Gap(IS-OOS)={m.get('is_oos_gap', float('nan')):.3f}"
              + ("  ← IS 偏高属正常" if mkey == "S" else ""))

        # 输出强度分布统计（如果存在）
        if "signal_strength" in ens_res.columns:
            st = ens_res["signal_strength"].dropna()
            non_zero = (ens_res["position"].abs() > 0).mean()
            print(f"     强度百分位 均值={st.mean():.3f}  中位数={st.median():.3f}"
                  f"  开仓率={non_zero:.1%}")

        summary_rows.append({"方案": f"方法{mkey}", "描述": mdesc, **m})
        ens_res.to_csv(sub_dir / "results.csv", encoding="utf-8-sig")

    # ── 汇总表 ────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    col_order  = [
        "方案", "描述",
        "oos_sharpe", "oos_annual", "oos_maxdd",
        "is_sharpe", "is_annual",
        "is_oos_gap", "all_sharpe", "all_annual", "all_maxdd",
    ]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]
    summary_df.to_csv(out / "ensemble_summary.csv", index=False, encoding="utf-8-sig")

    # ── 信号相关矩阵 ──────────────────────────────────────────────────────
    ret_dict = {}
    for c in combos:
        ret_dict[f"单组{c['rank']}({c['model_type']})"] = (
            c["results_df"]["strategy_return"].fillna(0)
        )
    for mkey, ens_res in all_results.items():
        ret_dict[f"方法{mkey}"] = ens_res["strategy_return"].fillna(0)
    sig_corr_df = pd.DataFrame(ret_dict).corr()
    sig_corr_df.to_csv(out / "ensemble_signal_corr.csv", encoding="utf-8-sig")

    # ── 汇总打印 ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  完整汇总（OOS 指标优先排列）")
    print(f"\n  {'方案':<20}  {'描述':<32}  {'OOS-Sharpe':>10}  {'OOS-Ann':>8}  "
          f"{'OOS-MaxDD':>9}  {'IS-Sharpe':>9}  {'Gap':>7}")
    print(f"  {'─'*106}")

    for _, row in summary_df.sort_values("oos_sharpe", ascending=False).iterrows():
        print(
            f"  {str(row['方案']):<20}  {str(row['描述']):<32}  "
            f"{row.get('oos_sharpe', float('nan')):>10.3f}  "
            f"{row.get('oos_annual', float('nan')):>8.1%}  "
            f"{row.get('oos_maxdd',  float('nan')):>9.1%}  "
            f"{row.get('is_sharpe',  float('nan')):>9.3f}  "
            f"{row.get('is_oos_gap', float('nan')):>7.3f}"
        )

    print(f"\n  * 方法S 的 IS Sharpe 含 in-sample bias，请以 OOS Sharpe 为准")
    if strength_map:
        print(f"  * 强度映射已应用于全部集成方案，开仓仓位为分数值（非整数）")

    print(f"\n  信号相关矩阵：")
    print(sig_corr_df.round(3).to_string())

    # ── 集成 PnL 对比图 ───────────────────────────────────────────────────
    single_res_dict = {
        f"单组{c['rank']}({c['model_type']})": c["results_df"] for c in combos
    }
    method_labels = {
        "A": "信号等权平均",
        "C": "位置多数投票",
        "S": "线性Stacking",
    }
    ensemble_res_dict = {
        f"方法{k}({method_labels.get(k, k)})": v for k, v in all_results.items()
    }
    plot_ensemble_pnl(
        single_results   = single_res_dict,
        ensemble_results = ensemble_res_dict,
        split_point      = split_point,
        output_dir       = str(out),
    )

    print(f"\n{SEP}")
    print(f"  输出目录：{out}")
    print(f"  ensemble_summary.csv  ·  ensemble_signal_corr.csv")
    print(f"  ensemble_pnl_comparison.png")
    print(f"  method_A/ C/ S/  ·  single_1~{len(combos)}/")
    print(SEP)


if __name__ == "__main__":
    main()