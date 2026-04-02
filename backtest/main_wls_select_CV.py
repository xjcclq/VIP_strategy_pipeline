"""
factor_search_top5_v2_icir.py — 挑出 5 组相关性低的优质因子组合（含纯净测试集）
═══════════════════════════════════════════════════════════════════════════
相比 factor_search_top5_v2.py 的核心改动

  ★ 新增：ICIR 分层加权合成（移植自 factor_search_xgb_top5.py）
       · 将 x_ma_ret_1h ~ x_ma_ret_12h 按时间尺度分为 short/mid/long 三个子簇
       · 子簇内用滚动 ICIR 加权合成，子簇间再做一次 ICIR 加权
       · 生成合成因子 x_ma_ret_icir，替代原来 MUST_INCLUDE 中的 x_ma_ret_4h
       · 原始 x_ma_ret_1h ~ 12h 从候选池中删除，避免与合成因子重复

  1. 严格三段划分：Train / Val / Test
       Train : start_date ~ val_split
       Val   : val_split  ~ test_split   （Stage2 OOS Sharpe 打分窗口）
       Test  : test_split ~ 末尾          （全程不可见，搜索完成后只读一次）

  2. OOS Sharpe 打分改为 Purged Walk-Forward CV
       在 Val 段内均分 K 折，折间加 purge gap = fwd bars
       打分 = mean(Sharpe_k) − λ·std(Sharpe_k)

  3. Test 段一次性评估
       Stage4 选出 5 组后，对每组计算 Test Sharpe，只读一次，不回流决策

数据划分建议（2020-10-01 ~ 2026-03-31，约 5.5 年）
  Train pool : 2020-10-01 ~ 2023-04-01  （2.5 年）
  Val pool   : 2023-04-01 ~ 2024-10-01  （1.5 年，CV 打分用）
  Test       : 2024-10-01 ~ 末尾         （1.5 年，只读一次）

输出
  top5_combinations.json     5 组因子 + 所有复现参数（含 val/test 分界）
  top5_signal_corr.csv       5 组策略信号相关矩阵
  test_performance.csv       5 组 Test 段绩效（最终报告）
  backtest_combo_{1..5}/     每组完整回测结果
  stage2_all_trials.csv      Optuna 所有 trial 记录
  stage1_ranking.csv / stage1_clusters.csv
  icir_weights_x_ma_ret_icir.csv   ICIR 权重变化历史
"""

from __future__ import annotations

import sys
import json
import copy
import pickle
import warnings
import argparse
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

script_path  = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(script_path.parent))

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from utils2 import (
    load_palm_oil_data,
    prepare_factor_data,
    run_backtest_reg,
    save_results,
    get_timestamp,
    print_performance_table,
    apply_strength_filter,
    recalc_performance,
    calc_metrics_from_returns,
    _compute_reversal_labels,
)

# ══════════════════════════════════════════════════════════════════════════════
# ★ ICIR 分层加权合成配置（移植自 factor_search_xgb_top5.py）
# ══════════════════════════════════════════════════════════════════════════════
#
# key   = 合成后的新因子名（会被注入 select_fd / base_fd）
# value = dict:
#     "sub_clusters": 按时间尺度硬编码的子簇分组
#     "icir_min":     子簇内 ICIR 低于此阈值的因子权重置零（动态剔除）
#
# 合成方式：
#   1) 子簇内：滚动 ICIR 加权（窗口=icir_window，跟随 retrain_freq 更新）
#   2) 子簇间：再做一次 ICIR 加权，得到最终合成因子
#
# 原始族因子（x_ma_ret_1h ~ 12h）会从 select_fd / base_fd 中删除，
# 避免它们再次进入可选池与合成因子重复。
FACTOR_ICIR_GROUPS: dict[str, dict] = {
    "x_ma_ret_icir": {
        "sub_clusters": {
            "short":  ["x_ma_ret_1h", "x_ma_ret_2h"],
            "mid":    ["x_ma_ret_4h", "x_ma_ret_6h"],
            "long":   ["x_ma_ret_8h", "x_ma_ret_12h"],
        },
        "icir_min": 0.3,   # 子簇内 ICIR 低于此值的因子权重置零
    },
}

# ★ 核心改动：x_ma_ret_4h → x_ma_ret_icir（ICIR 分层加权合成因子）
MUST_INCLUDE: list[str] = [
    "x_vwap",
    "x_ma_ret_icir",   # ← 由 FACTOR_ICIR_GROUPS 的分层 ICIR 加权生成
]

N_COMBOS  = 5
POOL_SIZE = 10


# ══════════════════════════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    p = argparse.ArgumentParser(
        description="Top-5 多样性因子组合（含纯净测试集隔离 + ICIR 合成）"
    )

    # ── 数据 ──────────────────────────────────────────────────────────────
    p.add_argument("--data_file",  default=os.path.join(project_root, "data\\P_with_ma_features.csv"))
    p.add_argument("--start_date", default="2020-10-01")

    # ── 三段划分 ──────────────────────────────────────────────────────────
    p.add_argument("--val_split",  default="2023-04-01",
                   help="Train/Val 分界；Val 段用于 Optuna CV 打分")
    p.add_argument("--test_split", default="2024-10-01",
                   help="Val/Test 分界；Test 段搜索完成后只读一次")

    # ── 训练通用 ──────────────────────────────────────────────────────────
    p.add_argument("--train_window",   type=int,   default=3000)
    p.add_argument("--mode",           default="rolling")
    p.add_argument("--retrain_freq",   type=int,   default=1000)
    p.add_argument("--fwd",            type=int,   default=3)
    p.add_argument("--lag",            type=int,   default=2)
    p.add_argument("--factor_lags",    default="")
    p.add_argument("--use_scaler",     action="store_true",  default=True)
    p.add_argument("--no_scaler",      action="store_false", dest="use_scaler")
    p.add_argument("--check_days",     type=int,   default=3)
    p.add_argument("--multiplier",     type=float, default=2.0)
    p.add_argument("--weight_method",  default="rolling")
    p.add_argument("--rolling_window", type=int,   default=1000)

    # ── 回测信号 ──────────────────────────────────────────────────────────
    p.add_argument("--reg_threshold",   type=float, default=0.0)
    p.add_argument("--close_threshold", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--close_mode",      default="threshold")
    p.add_argument("--use_strength_filter", action="store_true",  default=True)
    p.add_argument("--no_strength_filter",  action="store_false",
                   dest="use_strength_filter")
    p.add_argument("--entry_strength_pct",  type=float, default=0.7)
    p.add_argument("--threshold_window",    type=int,   default=50)

    # ── Stage 1：残差 IC + Ward 聚类 ──────────────────────────────────────
    p.add_argument("--ic_window",       type=int,   default=500)
    p.add_argument("--ic_threshold",    type=float, default=0.02)
    p.add_argument("--n_clusters",      type=int,   default=0)
    p.add_argument("--corr_cluster_t",  type=float, default=0.55)
    p.add_argument("--top_per_cluster", type=int,   default=2)
    p.add_argument("--min_cluster_ic",  type=float, default=0.01)
    p.add_argument("--max_factors",     type=int,   default=50,
                   help="可选因子上限（按残差IC-IR截断）")

    # ── ★ ICIR 合成参数（新增）────────────────────────────────────────────
    p.add_argument("--icir_window",     type=int,   default=2000,
                   help="★ ICIR 计算滚动窗口（bar 数），建议 train_window 的 1/2~3/4")
    p.add_argument("--icir_min",        type=float, default=0.3,
                   help="ICIR 低于此值的因子权重置零，可被 FACTOR_ICIR_GROUPS 中的 icir_min 覆盖")

    # ── Stage 2：Optuna + Purged WF ICIR ─────────────────────────────────
    p.add_argument("--cv_folds",        type=int,   default=3)
    p.add_argument("--cv_lambda",       type=float, default=1.0)
    p.add_argument("--n_trials",        type=int,   default=40)
    p.add_argument("--n_startup_trials",type=int,   default=0)
    p.add_argument("--corr_penalty_w",  type=float, default=0.25)
    p.add_argument("--min_clusters_on", type=int,   default=1)
    p.add_argument("--max_clusters_on", type=int,   default=0)
    p.add_argument("--complexity_penalty", type=float, default=0.05)
    p.add_argument("--fold_std_cap",       type=float, default=1.5)
    p.add_argument("--gap_penalty",        type=float, default=0.3)

    # ── Stage 3：精修 ─────────────────────────────────────────────────────
    p.add_argument("--refine_iters",    type=int,   default=1)

    # ── Stage 4：多样性筛选 ───────────────────────────────────────────────
    p.add_argument("--diversity_w",     type=float, default=0.5)
    p.add_argument("--min_val_score",   type=float, default=-0.5)

    # ── 续跑 ──────────────────────────────────────────────────────────────
    p.add_argument("--resume",     action="store_true", default=False)
    p.add_argument("--study_path", default="")

    # ── 随机种子 ──────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数（与原版相同）
# ══════════════════════════════════════════════════════════════════════════════

def _rolling_ic(f: pd.Series, l: pd.Series, window: int) -> pd.Series:
    f = f.replace([np.inf, -np.inf], np.nan)
    l = l.replace([np.inf, -np.inf], np.nan)
    rf = f.rolling(window, min_periods=window // 2).rank()
    rl = l.rolling(window, min_periods=window // 2).rank()
    return rf.rolling(window, min_periods=window // 2).corr(rl).fillna(0.0)


def _residualize(target: pd.Series, regressors: pd.DataFrame) -> pd.Series:
    idx   = target.index.intersection(regressors.index)
    t_sub = target.loc[idx].replace([np.inf, -np.inf], np.nan)
    r_sub = regressors.loc[idx].replace([np.inf, -np.inf], np.nan)
    valid = t_sub.notna() & r_sub.notna().all(axis=1)
    if valid.sum() < 100:
        return target
    X = r_sub.loc[valid].values
    y = t_sub.loc[valid].values
    try:
        coef  = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ coef
        out   = target.copy().astype(float)
        out.loc[t_sub.loc[valid].index] = resid
        out.loc[~valid] = np.nan
        return out
    except Exception:
        return target


def _ic_ir(ic_s: pd.Series) -> float:
    v = ic_s.dropna()
    if len(v) < 10:
        return 0.0
    return abs(float(v.mean()) / (float(v.std()) + 1e-9))


def _mean_abs_corr(fd: pd.DataFrame) -> float:
    k = fd.shape[1]
    if k < 2:
        return 0.0
    c = fd.fillna(0).corr().values
    return float(np.abs(c[np.triu_indices(k, k=1)]).mean())


def _sharpe_from_returns(r: np.ndarray) -> float:
    if len(r) < 20:
        return -999.0
    m, s = float(np.mean(r)), float(np.std(r))
    return 0.0 if s < 1e-9 else m / s * np.sqrt(252)


def _daily_returns_in_window(results_df, t_start, t_end) -> np.ndarray:
    idx  = pd.to_datetime(results_df.index)
    mask = idx >= pd.to_datetime(t_start)
    if t_end is not None:
        mask &= idx < pd.to_datetime(t_end)
    sub = results_df.loc[mask]
    if len(sub) < 10:
        return np.array([])
    key = pd.to_datetime(sub.index).normalize()
    d   = sub.groupby(key).agg(r=("strategy_return", "sum"),
                                sw=("is_switch", "any"))
    return d[~d["sw"]]["r"].values


def _sharpe_in_window(results_df, t_start, t_end) -> float:
    return _sharpe_from_returns(_daily_returns_in_window(results_df, t_start, t_end))


def _rolling_icir_in_window(factor_vals, label_vals, window) -> float:
    n = len(factor_vals)
    if n < window // 2 or n < 30:
        return 0.0
    step    = max(1, window // 5)
    ic_list = []
    for start in range(0, n - window + 1, step):
        end   = start + window
        f_sub = factor_vals[start:end]
        l_sub = label_vals[start:end]
        mask  = np.isfinite(f_sub) & np.isfinite(l_sub)
        if mask.sum() < window // 2:
            continue
        corr, _ = spearmanr(f_sub[mask], l_sub[mask])
        if not np.isnan(corr):
            ic_list.append(float(corr))
    if len(ic_list) < 3:
        return 0.0
    arr = np.array(ic_list)
    return float(arr.mean() / (arr.std() + 1e-9))


def _composite_icir(factor_subset, base_fd, price_data, args, t_start, t_end, icir_window) -> float:
    avail = [c for c in factor_subset if c in base_fd.columns]
    if not avail:
        return -999.0
    idx  = pd.to_datetime(base_fd.index)
    mask = (idx >= pd.to_datetime(t_start)) & (idx < pd.to_datetime(t_end))
    if mask.sum() < icir_window // 2:
        return -999.0
    fd_win  = base_fd.loc[mask, avail].fillna(0)
    labels  = _compute_reversal_labels(
        price_data, fwd=args.fwd,
        check_days=getattr(args, "check_days", 3),
        multiplier=getattr(args, "multiplier", 2.0),
    )
    lb_win  = labels.loc[mask].fillna(0)
    if len(fd_win) < icir_window // 2:
        return -999.0
    f_arr = fd_win.values
    l_arr = lb_win.values
    signs = []
    for j in range(f_arr.shape[1]):
        icir_j = _rolling_icir_in_window(f_arr[:, j], l_arr, icir_window)
        signs.append(1.0 if icir_j >= 0 else -1.0)
    composite = f_arr @ np.array(signs) / len(signs)
    return _rolling_icir_in_window(composite, l_arr, icir_window)


def _wf_cv_icir(factor_subset, base_fd, price_data, args) -> tuple[float, float]:
    avail = [c for c in factor_subset if c in base_fd.columns]
    if not avail:
        return -999.0, 0.0
    mac        = _mean_abs_corr(base_fd[avail])
    val_start  = pd.to_datetime(args.val_split)
    test_start = pd.to_datetime(args.test_split)
    K          = args.cv_folds
    lam        = args.cv_lambda
    fwd        = args.fwd
    icir_win   = getattr(args, "icir_window", 120)
    idx      = pd.to_datetime(base_fd.index)
    val_mask = (idx >= val_start) & (idx < test_start)
    val_idx  = np.where(val_mask)[0]
    min_bars = max(icir_win * 2, K * 50)
    if len(val_idx) < min_bars:
        s = _composite_icir(avail, base_fd, price_data, args,
                             val_start, test_start, icir_win)
        return s, mac
    fold_len = len(val_idx) // K
    icirs    = []
    for k in range(K):
        bar_start = val_idx[k * fold_len]
        bar_end   = val_idx[(k + 1) * fold_len - 1] if k < K - 1 else val_idx[-1]
        bar_start_purged = min(bar_start + fwd, bar_end)
        t_s = base_fd.index[bar_start_purged]
        t_e = base_fd.index[bar_end]
        icir_k = _composite_icir(avail, base_fd, price_data, args,
                                  t_s, t_e, icir_win)
        icirs.append(icir_k)
    valid = [v for v in icirs if v > -900]
    if not valid:
        return -999.0, mac
    if len(valid) == 1:
        return valid[0], mac
    score = float(np.mean(valid) - lam * np.std(valid))
    return score, mac


def _score(cv_s, mac, pw) -> float:
    if cv_s <= -998.0:
        return -999.0
    return cv_s - pw * mac


# ══════════════════════════════════════════════════════════════════════════════
# ★ 新增：族内分层 ICIR 加权合成（移植自 factor_search_xgb_top5.py）
# ══════════════════════════════════════════════════════════════════════════════

def _compute_icir_in_window(factor_values: np.ndarray, label_values: np.ndarray) -> float:
    """
    在给定窗口内计算单因子的 IC-IR。
    将窗口切成若干子段（每段约 100 bar），每段算一个 rank IC，
    然后 ICIR = mean(ICs) / std(ICs)。
    """
    valid = ~(np.isnan(factor_values) | np.isnan(label_values))
    fv = factor_values[valid]
    lv = label_values[valid]

    if len(fv) < 200:
        if len(fv) < 30:
            return 0.0
        corr, _ = spearmanr(fv, lv)
        if np.isnan(corr):
            return 0.0
        pseudo_n = max(1, len(fv) // 100)
        return abs(corr) * np.sqrt(pseudo_n)

    seg_size = 100
    n_segs   = len(fv) // seg_size
    if n_segs < 3:
        seg_size = len(fv) // 3
        n_segs   = 3

    ics = []
    for s in range(n_segs):
        start = s * seg_size
        end   = start + seg_size
        if end > len(fv):
            break
        corr, _ = spearmanr(fv[start:end], lv[start:end])
        if not np.isnan(corr):
            ics.append(corr)

    if len(ics) < 3:
        return 0.0
    ic_mean = np.mean(ics)
    ic_std  = np.std(ics, ddof=1)
    if ic_std < 1e-9:
        return abs(ic_mean) * 10.0
    return abs(ic_mean) / ic_std


def apply_factor_icir_groups(
    fd:               pd.DataFrame,
    price_data:       pd.Series,
    icir_groups:      dict[str, dict],
    icir_window:      int,
    retrain_freq:     int,
    fwd:              int,
    check_days:       int,
    multiplier:       float,
    out_dir:          Path | None = None,
    default_icir_min: float = 0.3,
) -> tuple[pd.DataFrame, list[str]]:
    """
    对 FACTOR_ICIR_GROUPS 中的每组因子做分层 ICIR 加权合成。

    分层合成设计：
      第一层（子簇内）：同时间尺度的因子按 ICIR 加权合成为子簇信号
      第二层（子簇间）：三个子簇信号再按各自的 ICIR 加权合成为最终因子

    权重每 retrain_freq 个 bar 更新一次，无前视偏差：
    时间 t 的权重只用 [t-icir_window, t) 的数据计算。

    返回
    ----
    fd_new       : 注入了合成列、删除了原始族列的新 DataFrame
    removed_cols : 被删除的原始列名列表
    """
    labels = _compute_reversal_labels(
        price_data, fwd=fwd,
        check_days=check_days,
        multiplier=multiplier,
    )
    label_values_full = labels.reindex(fd.index).values.astype(float)

    fd_new       = fd.copy()
    removed_cols: list[str] = []

    for comp_name, group_cfg in icir_groups.items():
        sub_clusters = group_cfg["sub_clusters"]
        icir_min     = group_cfg.get("icir_min", default_icir_min)

        # 收集所有有效的原始因子
        all_raw_cols: list[str] = []
        valid_sub_clusters: dict[str, list[str]] = {}
        for sc_name, sc_cols in sub_clusters.items():
            avail = [c for c in sc_cols if c in fd.columns]
            if avail:
                valid_sub_clusters[sc_name] = avail
                all_raw_cols.extend(avail)
            else:
                print(f"  [ICIR合成] {comp_name}/{sc_name}: 列不存在，跳过")

        if not all_raw_cols:
            print(f"  [ICIR合成] {comp_name}: 无有效因子，跳过")
            continue

        if len(all_raw_cols) == 1:
            print(f"  [ICIR合成] {comp_name}: 仅 1 个有效因子，直接复制 {all_raw_cols[0]}")
            fd_new[comp_name] = fd[all_raw_cols[0]].values
            removed_cols.extend(all_raw_cols)
            fd_new.drop(columns=all_raw_cols, inplace=True, errors="ignore")
            continue

        n         = len(fd)
        composite = np.full(n, np.nan)
        raw_data  = {col: fd[col].values.astype(float) for col in all_raw_cols}

        # 确定 retrain 节点
        first_valid   = icir_window
        retrain_points = list(range(first_valid, n, retrain_freq))
        if not retrain_points:
            retrain_points = [first_valid]

        weight_history: list[dict] = []

        # 权重缓存（在 retrain 节点更新，中间保持不变）
        sub_weights:   dict[str, dict[str, float]] = {}
        inter_weights: dict[str, float] = {}
        current_retrain_idx = 0

        for t in range(first_valid, n):
            # 判断是否需要更新权重
            need_update = False
            if current_retrain_idx < len(retrain_points):
                if t >= retrain_points[current_retrain_idx]:
                    need_update = True
                    while (current_retrain_idx < len(retrain_points)
                           and retrain_points[current_retrain_idx] <= t):
                        current_retrain_idx += 1

            if need_update or not sub_weights:
                win_start  = max(0, t - icir_window)
                win_label  = label_values_full[win_start:t]
                sub_icir_values: dict[str, float] = {}

                # ── 第一层：子簇内 ICIR 加权 ─────────────────────────────
                for sc_name, sc_cols in valid_sub_clusters.items():
                    col_icirs: dict[str, float] = {}
                    for col in sc_cols:
                        icir_val = _compute_icir_in_window(
                            raw_data[col][win_start:t], win_label)
                        col_icirs[col] = icir_val

                    # 动态剔除 ICIR < icir_min 的因子
                    filtered = {c: v for c, v in col_icirs.items() if v >= icir_min}
                    if not filtered:
                        best_col = max(col_icirs, key=col_icirs.get)
                        filtered = {best_col: col_icirs[best_col]}

                    total = sum(filtered.values())
                    if total < 1e-9:
                        w = {c: 1.0 / len(filtered) for c in filtered}
                    else:
                        w = {c: v / total for c, v in filtered.items()}

                    full_w = {c: 0.0 for c in sc_cols}
                    full_w.update(w)
                    sub_weights[sc_name] = full_w

                    # 子簇 ICIR = 各因子 ICIR 的加权均值
                    sub_icir_values[sc_name] = sum(
                        v * col_icirs.get(c, 0.0) for c, v in w.items()
                    )

                # ── 第二层：子簇间 ICIR 加权 ─────────────────────────────
                total_inter = sum(sub_icir_values.values())
                if total_inter < 1e-9:
                    inter_weights = {sc: 1.0 / len(sub_icir_values)
                                     for sc in sub_icir_values}
                else:
                    inter_weights = {sc: v / total_inter
                                     for sc, v in sub_icir_values.items()}

                # 记录权重（用于诊断）
                record = {"bar_idx": t, "time": fd.index[t]}
                for sc_name in valid_sub_clusters:
                    iw = inter_weights.get(sc_name, 0.0)
                    record[f"inter_{sc_name}"] = round(iw, 4)
                    for col, w_val in sub_weights.get(sc_name, {}).items():
                        short_col = col.replace("x_ma_ret_", "")
                        record[f"w_{sc_name}_{short_col}"] = round(w_val * iw, 4)
                weight_history.append(record)

            # ── 用当前权重合成 ───────────────────────────────────────────
            val       = 0.0
            any_valid = False
            for sc_name, sc_cols in valid_sub_clusters.items():
                iw = inter_weights.get(sc_name, 0.0)
                if iw < 1e-9:
                    continue
                sw = sub_weights.get(sc_name, {})
                for col in sc_cols:
                    w_val = sw.get(col, 0.0)
                    if w_val < 1e-9:
                        continue
                    fv = raw_data[col][t]
                    if np.isnan(fv):
                        continue
                    val += iw * w_val * fv
                    any_valid = True

            if any_valid:
                composite[t] = val

        fd_new[comp_name] = composite

        n_valid = int(np.sum(~np.isnan(composite)))
        print(f"  [ICIR合成] {comp_name}: {len(all_raw_cols)} 个因子 → 分层 ICIR 加权合成")
        print(f"    有效 bar={n_valid}/{n}  子簇={list(valid_sub_clusters.keys())}")
        print(f"    icir_min={icir_min}  窗口={icir_window}  更新频率=每 {retrain_freq} bar")

        # 打印最新权重
        if weight_history:
            latest = weight_history[-1]
            print(f"  [ICIR合成] 最新权重（bar {latest['bar_idx']}）：")
            for sc_name in valid_sub_clusters:
                iw = latest.get(f"inter_{sc_name}", 0.0)
                col_details = []
                for col in valid_sub_clusters[sc_name]:
                    short = col.replace("x_ma_ret_", "")
                    final_w = latest.get(f"w_{sc_name}_{short}", 0.0)
                    col_details.append(f"{short}={final_w:.3f}")
                print(f"    {sc_name}（簇间权重={iw:.3f}）：{', '.join(col_details)}")

        # 保存权重历史
        if out_dir is not None and weight_history:
            wh_df = pd.DataFrame(weight_history)
            wh_df.to_csv(
                out_dir / f"icir_weights_{comp_name}.csv",
                index=False, encoding="utf-8-sig",
            )
            print(f"  [ICIR合成] 权重历史已保存 → icir_weights_{comp_name}.csv"
                  f"（{len(weight_history)} 条记录）")

        removed_cols.extend(all_raw_cols)
        fd_new.drop(columns=all_raw_cols, inplace=True, errors="ignore")

    return fd_new, removed_cols


# ══════════════════════════════════════════════════════════════════════════════
# 完整回测（与原版相同）
# ══════════════════════════════════════════════════════════════════════════════

def _run_backtest(factor_subset, base_fd, price_data, args):
    avail = [c for c in factor_subset if c in base_fd.columns]
    if not avail:
        return None, None
    ta = copy.copy(args)
    ta.contract_switch_dates = getattr(args, "contract_switch_dates", [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, perf = run_backtest_reg(base_fd[avail].copy(), price_data, ta)
    if res is None or perf is None:
        return None, None
    ta.split_point = perf.get("split_point")
    if getattr(ta, "use_strength_filter", False):
        res  = apply_strength_filter(res, ta)
        perf = recalc_performance(res, ta)
    return res, perf


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — 残差 IC + Ward 聚类（与原版相同）
# ══════════════════════════════════════════════════════════════════════════════

def stage1_cluster(factor_data, price_data, args):
    labels   = _compute_reversal_labels(
        price_data, fwd=args.fwd,
        check_days=getattr(args, "check_days", 3),
        multiplier=getattr(args, "multiplier", 2.0),
    )
    window   = args.ic_window
    opt_cols = [c for c in factor_data.columns if c not in MUST_INCLUDE]
    must_df  = factor_data[[c for c in MUST_INCLUDE if c in factor_data.columns]]
    M        = args.top_per_cluster
    max_f    = getattr(args, "max_factors", 50)
    total    = len(opt_cols)

    print(f"  计算 {total} 个可选因子残差 IC（窗口={window}）...")
    records, ic_series = {}, {}
    step = max(1, total // 10)

    for idx, col in enumerate(opt_cols, 1):
        if idx % step == 0 or idx == total:
            print(f"    {idx:4d}/{total}  {col[:55]}")
        raw_ic = _rolling_ic(factor_data[col], labels, window)
        res_f  = _residualize(factor_data[col], must_df)
        res_ic = _rolling_ic(res_f, labels, window)
        records[col]   = dict(factor=col, ic_ir=_ic_ir(raw_ic),
                              residual_ic_ir=_ic_ir(res_ic))
        ic_series[col] = raw_ic

    ranking_all = pd.DataFrame(list(records.values())).sort_values(
        "residual_ic_ir", ascending=False
    ).reset_index(drop=True)

    if max_f > 0 and len(opt_cols) > max_f:
        keep_cols = ranking_all.head(max_f)["factor"].tolist()
        print(f"\n  ★ 因子上限={max_f}，丢弃末尾 {len(opt_cols)-max_f} 个低 IC-IR 因子")
        opt_cols  = keep_cols
        ic_series = {c: ic_series[c] for c in keep_cols}

    ic_df    = pd.DataFrame({c: ic_series[c] for c in opt_cols}).fillna(0.0)
    corr_mat = ic_df.corr()
    dist_mat = (1.0 - corr_mat.abs().clip(0, 1)).clip(0)
    dist_arr = dist_mat.values.copy()
    np.fill_diagonal(dist_arr, 0.0)
    condensed = squareform(dist_arr, checks=False).clip(0)
    Z         = linkage(condensed, method="ward")

    if args.n_clusters > 0:
        cluster_ids = fcluster(Z, t=args.n_clusters, criterion="maxclust")
    else:
        cluster_ids = fcluster(Z, t=args.corr_cluster_t, criterion="distance")

    n_clusters = int(cluster_ids.max())
    for col, cid in zip(opt_cols, cluster_ids):
        records[col]["cluster"] = int(cid)

    ranking_df = (
        pd.DataFrame([records[c] for c in opt_cols])
        .sort_values("residual_ic_ir", ascending=False)
        .reset_index(drop=True)
    )
    ranking_df.index += 1

    cluster_candidates: list[list[str]] = []
    cluster_info:       list[dict]      = []

    for cid in range(1, n_clusters + 1):
        members = ranking_df[ranking_df["cluster"] == cid].sort_values(
            "residual_ic_ir", ascending=False
        )
        best_ir = members["residual_ic_ir"].iloc[0] if len(members) else 0.0
        if best_ir < args.min_cluster_ic:
            continue
        top_m = members.head(M)["factor"].tolist()
        cluster_candidates.append(top_m)
        cluster_info.append({
            "cluster_id":     cid,
            "size":           len(members),
            "top_factor":     top_m[0],
            "best_res_ic_ir": best_ir,
            "candidates":     " | ".join(top_m),
        })

    print(f"\n  聚类：{n_clusters} 簇 → 有效簇 {len(cluster_candidates)} 个"
          f"（每簇保留 top-{M}）")
    print(f"\n  {'簇':>3}  {'大小':>4}  {'最佳残差IC-IR':>13}  候选因子")
    print(f"  {'─'*72}")
    for info in cluster_info:
        print(f"  {info['cluster_id']:3d}  {info['size']:4d}  "
              f"{info['best_res_ic_ir']:13.4f}  {info['candidates']}")

    return cluster_candidates, ranking_all, pd.DataFrame(cluster_info)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — 跨簇 Optuna（与原版相同）
# ══════════════════════════════════════════════════════════════════════════════

def stage2_collect_pool(cluster_candidates, base_fd, price_data, args, out_dir):
    n_clusters       = len(cluster_candidates)
    min_on           = max(0, args.min_clusters_on)
    max_on           = args.max_clusters_on if args.max_clusters_on > 0 else n_clusters
    cw               = args.corr_penalty_w
    n_start          = (args.n_startup_trials if args.n_startup_trials > 0
                        else min(10, args.n_trials // 4))
    complexity_pen   = getattr(args, "complexity_penalty", 0.05)
    std_cap          = getattr(args, "fold_std_cap",       1.5)
    gap_pen          = getattr(args, "gap_penalty",        0.3)

    space_size = 1
    for cl in cluster_candidates:
        space_size *= (len(cl) + 1)
    print(f"  搜索空间：{space_size:,} 种  Optuna {args.n_trials} trials")
    print(f"  Val段：[{args.val_split}, {args.test_split})  CV {args.cv_folds}折  λ={args.cv_lambda}")

    trial_log        = []
    best_val         = [-999.0]
    result_cache: dict[str, pd.DataFrame | None] = {}

    def _score_with_guards(factor_subset):
        avail = [c for c in factor_subset if c in base_fd.columns]
        if not avail:
            return -999.0, -999.0, 0.0

        mac      = _mean_abs_corr(base_fd[avail])
        icir_win = getattr(args, "icir_window", 120)
        K        = args.cv_folds
        lam      = args.cv_lambda
        fwd      = args.fwd

        val_start  = pd.to_datetime(args.val_split)
        test_start = pd.to_datetime(args.test_split)
        idx        = pd.to_datetime(base_fd.index)
        val_mask   = (idx >= val_start) & (idx < test_start)
        val_idx    = np.where(val_mask)[0]

        min_bars = max(icir_win * 2, K * 50)
        if len(val_idx) < min_bars:
            raw_icir = _composite_icir(avail, base_fd, price_data, args,
                                        val_start, test_start, icir_win)
            fold_std = 0.0
        else:
            fold_len = len(val_idx) // K
            icirs    = []
            for k in range(K):
                bar_s = val_idx[k * fold_len]
                bar_e = val_idx[(k + 1) * fold_len - 1] if k < K - 1 else val_idx[-1]
                bar_s = min(bar_s + fwd, bar_e)
                icir_k = _composite_icir(
                    avail, base_fd, price_data, args,
                    base_fd.index[bar_s], base_fd.index[bar_e], icir_win,
                )
                icirs.append(icir_k)
            valid    = [v for v in icirs if v > -900]
            fold_std = float(np.std(valid)) if len(valid) > 1 else 0.0
            raw_icir = float(np.mean(valid) - lam * np.std(valid)) if valid else -999.0

        if raw_icir <= -998.0:
            return -999.0, -999.0, fold_std

        n_optional = len([f for f in factor_subset if f not in MUST_INCLUDE])
        n_pen      = complexity_pen * n_optional
        std_pen    = max(0.0, fold_std - std_cap) * lam
        train_icir = _composite_icir(
            avail, base_fd, price_data, args,
            args.start_date, args.val_split, icir_win,
        )
        gap_p = (gap_pen * max(0.0, train_icir - raw_icir)
                 if train_icir > -900 and raw_icir > -900 else 0.0)

        final = raw_icir - cw * mac - n_pen - std_pen - gap_p
        return final, raw_icir, fold_std

    def objective(trial):
        chosen = []
        for ci, cl in enumerate(cluster_candidates):
            pick = trial.suggest_categorical(f"cluster_{ci}", ["none"] + cl)
            if pick != "none":
                chosen.append(pick)

        if not (min_on <= len(chosen) <= max_on):
            raise optuna.exceptions.TrialPruned()

        factor_subset = MUST_INCLUDE + chosen
        key           = "|".join(sorted(factor_subset))
        if key not in result_cache:
            result_cache[key] = None

        score, icir_cv, f_std = _score_with_guards(factor_subset)

        is_best = score > best_val[0]
        if is_best:
            best_val[0] = score

        short  = lambda f: f.replace("x_","").replace("_60min","").replace("_240min","")
        t_num  = trial.number + 1
        if is_best or t_num % 10 == 0 or t_num == args.n_trials:
            print(
                f"  Trial {t_num:3d}{'★' if is_best else ' '}  "
                f"Score={score:+.3f}  ICIR={icir_cv:+.3f}  fStd={f_std:.2f}  "
                f"N={len(factor_subset)}  {[short(f) for f in chosen] or ['无']}"
            )

        mac = _mean_abs_corr(base_fd[[c for c in factor_subset if c in base_fd.columns]])
        trial_log.append({
            "trial":         t_num,
            "score":         score,
            "icir_cv":       icir_cv,
            "fold_std":      f_std,
            "mean_abs_corr": mac,
            "n_factors":     len(factor_subset),
            "optional":      "|".join(chosen),
            "all_factors":   "|".join(factor_subset),
        })
        return score

    if args.resume and args.study_path and Path(args.study_path).exists():
        with open(args.study_path, "rb") as f:
            study = pickle.load(f)
        print(f"  [续跑] 已加载 {len(study.trials)} 个 trial")
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=args.seed, n_startup_trials=n_start, multivariate=True),
            pruner=optuna.pruners.NopPruner(),
        )

    print(f"\n  {'─'*80}")
    print(f"  {'Trial':>6}  {'Score':>7}  {'CV':>7}  {'fStd':>5}  {'N':>3}  可选因子")
    print(f"  {'─'*80}")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1,
                   show_progress_bar=False)

    pd.DataFrame(trial_log).sort_values("score", ascending=False).to_csv(
        out_dir / "stage2_all_trials.csv", index=False, encoding="utf-8-sig"
    )
    with open(out_dir / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    seen: set[str] = set()
    pool: list[dict] = []
    for rec in sorted(trial_log, key=lambda r: r["score"], reverse=True):
        key = rec["all_factors"]
        if key not in seen:
            seen.add(key)
            pool.append(dict(rec))

    print(f"\n  Optuna 完成，为 top-{POOL_SIZE} 候选补跑完整回测...")
    for i, rec in enumerate(pool[:POOL_SIZE]):
        key = rec["all_factors"]
        if result_cache.get(key) is None:
            factors = key.split("|")
            res, _  = _run_backtest(factors, base_fd, price_data, args)
            result_cache[key] = res
        rec["results_df"] = result_cache.get(key)
        if (i + 1) % 5 == 0 or (i + 1) == min(POOL_SIZE, len(pool)):
            print(f"    [{i+1}/{min(POOL_SIZE,len(pool))}] 完成")

    print(f"\n  候选池（去重）：{len(pool)} 个，取 top-{POOL_SIZE} 进入 Stage3")
    return pool


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — 逐步精修（与原版相同）
# ══════════════════════════════════════════════════════════════════════════════

def stage3_refine_one(init_factors, cluster_candidates, base_fd, price_data, args):
    cw = args.corr_penalty_w
    factor_to_cluster: dict[str, list[str]] = {
        f: cl for cl in cluster_candidates for f in cl
    }

    current = list(init_factors)

    def _eval(factors):
        res, _ = _run_backtest(factors, base_fd, price_data, args)
        cv_s, mac = _wf_cv_icir(factors, base_fd, price_data, args)
        return _score(cv_s, mac, cw), res

    cur_score, cur_res = _eval(current)

    for it in range(args.refine_iters):
        best_delta, best_move, best_res = 0.0, None, None
        opt_cur = [f for f in current if f not in MUST_INCLUDE]
        covered = set(opt_cur)

        for f_old in opt_cur:
            cl = factor_to_cluster.get(f_old, [f_old])
            for f_new in [f for f in cl if f != f_old] + [None]:
                trial = [f for f in current if f != f_old]
                if f_new:
                    trial.append(f_new)
                s, r = _eval(trial)
                if s - cur_score > best_delta:
                    best_delta, best_move, best_res = s - cur_score, trial, r

        for cl in cluster_candidates:
            if not any(f in covered for f in cl):
                trial = current + [cl[0]]
                s, r = _eval(trial)
                if s - cur_score > best_delta:
                    best_delta, best_move, best_res = s - cur_score, trial, r

        if best_move is None or best_delta <= 1e-4:
            break
        current, cur_score, cur_res = best_move, cur_score + best_delta, best_res

    return current, cur_score, cur_res


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — 贪心多样性筛选（与原版相同）
# ══════════════════════════════════════════════════════════════════════════════

def stage4_diverse_top5(pool, base_fd, price_data, args, out_dir, n_pick=N_COMBOS):
    min_score = args.min_val_score
    dw        = args.diversity_w

    valid_pool = [r for r in pool if r["score"] >= min_score and r.get("results_df") is not None]
    if len(valid_pool) < n_pick:
        print(f"  [警告] 有效候选不足 {n_pick} 个（共 {len(valid_pool)} 个），降低 --min_val_score")
        valid_pool = [r for r in pool if r.get("results_df") is not None]

    scores       = np.array([c["score"] for c in valid_pool])
    s_min, s_max = scores.min(), scores.max()
    scores_norm  = (scores - s_min) / (s_max - s_min + 1e-9)

    val_start  = pd.to_datetime(args.val_split)
    test_start = pd.to_datetime(args.test_split)

    signal_returns = {}
    for i, c in enumerate(valid_pool):
        res = c["results_df"]
        idx = pd.to_datetime(res.index)
        val_mask = (idx >= val_start) & (idx < test_start)
        signal_returns[i] = res.loc[val_mask, "strategy_return"].fillna(0.0)

    selected_idx: list[int] = []

    for pick_n in range(min(n_pick, len(valid_pool))):
        if pick_n == 0:
            best_i = int(np.argmax(scores_norm))
        else:
            best_composite, best_i = -np.inf, -1
            already = [signal_returns[j] for j in selected_idx]
            for i in range(len(valid_pool)):
                if i in selected_idx:
                    continue
                sig_i    = signal_returns[i]
                max_corr = max(
                    abs(float(sig_i.corr(
                        already_s.reindex(sig_i.index).fillna(0.0)
                    )))
                    for already_s in already
                )
                composite = (1 - dw) * scores_norm[i] + dw * (1 - max_corr)
                if composite > best_composite:
                    best_composite, best_i = composite, i

        selected_idx.append(best_i)
        c = valid_pool[best_i]
        opt_short = [f.replace("x_","")[:20] for f in c["optional"].split("|") if f]
        print(f"  ✓ 第 {pick_n+1} 组  Score={c['score']:+.3f}  opt={opt_short}")

    selected = [valid_pool[i] for i in selected_idx]

    ret_dict = {}
    for i, c in enumerate(selected):
        opt_list = [f for f in c["optional"].split("|") if f]
        label    = (f"组合{i+1}（{','.join(f.replace('x_','')[:12] for f in opt_list[:2])}"
                    + ("..." if len(opt_list) > 2 else "") + "）")
        ret_dict[label] = signal_returns[selected_idx[i]]

    sig_corr = pd.DataFrame(ret_dict).corr()
    sig_corr.to_csv(out_dir / "top5_signal_corr.csv", encoding="utf-8-sig")
    print(f"\n  ── 5 组策略信号相关矩阵（Val 段）──")
    print(sig_corr.round(3).to_string())

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Test 段一次性评估（与原版相同）
# ══════════════════════════════════════════════════════════════════════════════

def stage5_test_eval(top5, out_dir, args):
    print(f"\n  ── Test 段评估（只读一次，结果仅供参考）──")
    rows         = []
    test_sharpes = []

    for i, combo in enumerate(top5):
        res = combo.get("results_df")
        if res is None:
            test_s, val_s = -999.0, -999.0
        else:
            val_s  = _sharpe_in_window(res, args.val_split, args.test_split)
            test_s = _sharpe_in_window(res, args.test_split, None)

        test_sharpes.append(test_s)
        factors  = combo["all_factors"].split("|") if isinstance(combo.get("all_factors"), str) \
                   else combo.get("factors", [])
        opt_list = [f for f in (combo.get("optional","").split("|")) if f]

        print(f"  组合{i+1}  Val={val_s:+.4f}  Test={test_s:+.4f}", end="")
        if val_s > -900 and test_s > -900:
            gap  = abs(val_s - test_s)
            flag = "  ⚠ 过拟合" if gap > 0.5 else "  ✓ 泛化好"
            print(f"  差距={gap:.3f}{flag}", end="")
        print()

        rows.append({
            "rank":        i + 1,
            "val_sharpe":  val_s,
            "test_sharpe": test_s,
            "cv_score":    combo.get("score", -999.0),
            "mac":         combo.get("mac", 0.0),
            "n_factors":   len(factors),
            "optional":    "|".join(opt_list),
        })

    pd.DataFrame(rows).to_csv(
        out_dir / "test_performance.csv", index=False, encoding="utf-8-sig"
    )
    return test_sharpes


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_arguments()
    if not HAS_OPTUNA:
        print("[错误] pip install optuna scipy scikit-learn"); return

    ts  = get_timestamp()
    out = Path(project_root) / "results" / f"top5_v2_icir_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out)

    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  Top-5 多样性因子组合（含纯净测试集 + ICIR 合成）")
    print(f"  必选: {MUST_INCLUDE}")
    print(f"    ★ x_ma_ret_icir = x_ma_ret_{{1h,2h}} / {{4h,6h}} / {{8h,12h}} 分层 ICIR 加权合成")
    print(f"  数据范围: {args.start_date} ~ 末尾")
    print(f"  三段划分:")
    print(f"    Train : {args.start_date} ~ {args.val_split}")
    print(f"    Val   : {args.val_split} ~ {args.test_split}  （CV打分用）")
    print(f"    Test  : {args.test_split} ~ 末尾              （只读一次）")
    print(f"  ICIR合成: icir_window={args.icir_window}  icir_min={args.icir_min}")
    print(f"  CV: folds={args.cv_folds}  λ={args.cv_lambda}  共线惩罚={args.corr_penalty_w}")
    print(SEP)

    # ── 加载数据 ──────────────────────────────────────────────────────────
    df = load_palm_oil_data(args.data_file)
    if args.start_date:
        df = df[df.index >= pd.to_datetime(args.start_date)]

    if "dominant_id" in df.columns:
        mask = df["dominant_id"] != df["dominant_id"].shift(1)
        mask.iloc[0] = False
        args.contract_switch_dates = df.index[mask].tolist()
    else:
        args.contract_switch_dates = []

    all_x_cols = sorted([c for c in df.columns if c.startswith("x_")])

    # ICIR 合成生成新列，不在原始数据里
    icir_generated = set(FACTOR_ICIR_GROUPS.keys())
    missing = [c for c in MUST_INCLUDE
               if c not in all_x_cols and c not in icir_generated]
    if missing:
        print(f"[错误] 必选因子不存在: {missing}"); return

    print(f"[因子] 共 {len(all_x_cols)} 个 x_ 因子")

    factor_lags = None
    if getattr(args, "factor_lags", "").strip():
        try:
            factor_lags = [int(x) for x in args.factor_lags.split(",") if x.strip()]
        except ValueError:
            pass

    base_fd, price_data, _ = prepare_factor_data(
        df, selected_factors=all_x_cols, lag=args.lag, factor_lags=factor_lags
    )
    print(f"[数据] bars={len(base_fd)}  {base_fd.index[0]} ~ {base_fd.index[-1]}")

    # 校验三段划分
    val_dt  = pd.to_datetime(args.val_split)
    test_dt = pd.to_datetime(args.test_split)
    end_dt  = pd.to_datetime(base_fd.index[-1])
    assert val_dt < test_dt <= end_dt, "val_split < test_split <= 数据末尾"
    n_train = (pd.to_datetime(base_fd.index) < val_dt).sum()
    n_val   = ((pd.to_datetime(base_fd.index) >= val_dt) &
               (pd.to_datetime(base_fd.index) < test_dt)).sum()
    n_test  = (pd.to_datetime(base_fd.index) >= test_dt).sum()
    print(f"[划分] Train={n_train}bars  Val={n_val}bars  Test={n_test}bars\n")

    # ══════════════════════════════════════════════════════════════════════
    # ★ ICIR 分层加权合成（Stage 0.5，在聚类之前执行）
    # ══════════════════════════════════════════════════════════════════════
    if FACTOR_ICIR_GROUPS:
        print(f"{SEP}\n  ★ 族内分层 ICIR 加权合成\n{SEP}")
        for comp_name, group_cfg in FACTOR_ICIR_GROUPS.items():
            subs     = group_cfg["sub_clusters"]
            icir_min = group_cfg.get("icir_min", args.icir_min)
            print(f"  {comp_name}  (icir_min={icir_min})")
            for sc_name, sc_cols in subs.items():
                print(f"    {sc_name}: {sc_cols}")
        print()

        base_fd, removed_cols = apply_factor_icir_groups(
            fd=base_fd,
            price_data=price_data,
            icir_groups=FACTOR_ICIR_GROUPS,
            icir_window=args.icir_window,
            retrain_freq=args.retrain_freq,
            fwd=args.fwd,
            check_days=args.check_days,
            multiplier=args.multiplier,
            out_dir=out,
            default_icir_min=args.icir_min,
        )

        # 更新 all_x_cols：删除原始族列，加入合成列
        removed_set = set(removed_cols)
        all_x_cols  = [c for c in all_x_cols if c not in removed_set]
        for comp_name in FACTOR_ICIR_GROUPS:
            if comp_name not in all_x_cols:
                all_x_cols.append(comp_name)

        # 验证 MUST_INCLUDE 均已存在
        missing_must = [c for c in MUST_INCLUDE if c not in base_fd.columns]
        if missing_must:
            print(f"[错误] ICIR 合成后 MUST_INCLUDE 中仍有列缺失: {missing_must}")
            return

        print(f"\n  合成完成：删除原始列 {len(removed_cols)} 个，"
              f"新增 ICIR 合成列 {len(FACTOR_ICIR_GROUPS)} 个\n")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1
    # ══════════════════════════════════════════════════════════════════════
    print(f"{SEP}\n  Stage 1 / 5  残差 IC + Ward 聚类（因子上限={args.max_factors}）\n{SEP}")
    cluster_candidates, ranking_df, cluster_info_df = stage1_cluster(
        base_fd, price_data, args
    )
    ranking_df.to_csv(out / "stage1_ranking.csv", index=True, encoding="utf-8-sig")
    cluster_info_df.to_csv(out / "stage1_clusters.csv", index=False, encoding="utf-8-sig")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 2 / 5  跨簇 Optuna（Purged WF CV 打分）\n{SEP}")
    pool = stage2_collect_pool(cluster_candidates, base_fd, price_data, args, out)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 3 / 5  逐步精修（top-{min(POOL_SIZE, len(pool))} 个候选）\n{SEP}")
    refined_pool: list[dict] = []
    for i, rec in enumerate(pool[:POOL_SIZE]):
        factors   = rec["all_factors"].split("|")
        opt_short = [f.replace("x_","")[:18] for f in factors if f not in MUST_INCLUDE]
        print(f"  精修 [{i+1:2d}/{min(POOL_SIZE,len(pool))}] {opt_short}")

        rf, rs, r_res = stage3_refine_one(
            factors, cluster_candidates, base_fd, price_data, args
        )
        if r_res is None:
            r_res, _ = _run_backtest(rf, base_fd, price_data, args)

        mac = _mean_abs_corr(base_fd[[c for c in rf if c in base_fd.columns]])
        refined_pool.append({
            "all_factors": "|".join(rf),
            "optional":    "|".join(f for f in rf if f not in MUST_INCLUDE),
            "score":       rs,
            "mac":         mac,
            "results_df":  r_res,
        })

    seen: set[str] = set()
    refined_dedup: list[dict] = []
    for r in sorted(refined_pool, key=lambda x: x["score"], reverse=True):
        if r["all_factors"] not in seen:
            seen.add(r["all_factors"])
            refined_dedup.append(r)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 4 / 5  贪心多样性筛选 {N_COMBOS} 组\n{SEP}")
    top5 = stage4_diverse_top5(
        refined_dedup, base_fd, price_data, args, out, n_pick=N_COMBOS
    )

    if not top5:
        print("[错误] Stage4 未能选出组合"); return

    # ══════════════════════════════════════════════════════════════════════
    # Stage 5 — Test 段一次性评估
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 5 / 5  Test 段评估（只读一次）\n{SEP}")
    test_sharpes = stage5_test_eval(top5, out, args)

    # ══════════════════════════════════════════════════════════════════════
    # 输出结果
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  最终 {len(top5)} 组因子组合")
    print(SEP)

    result_json = []
    for i, combo in enumerate(top5):
        factors  = combo["all_factors"].split("|")
        optional = [f for f in factors if f not in MUST_INCLUDE]

        print(f"\n  ── 第 {i+1} 组 ──────────────────────────────────────────")
        print(f"  Val CV Score = {combo['score']:+.4f}  "
              f"Test Sharpe = {test_sharpes[i]:+.4f}  "
              f"平均因子相关 = {combo['mac']:.3f}")
        print(f"  因子（{len(factors)} 个）：")
        for f in factors:
            tag = " ★必选" if f in MUST_INCLUDE else " ○选入"
            print(f"    {f:<50}{tag}")

        sub_dir = out / f"backtest_combo_{i+1}"
        sub_dir.mkdir(exist_ok=True)
        fa = copy.copy(args)
        fa.contract_switch_dates = getattr(args, "contract_switch_dates", [])
        fa.output_dir = str(sub_dir)

        res_df = combo.get("results_df")
        if res_df is not None:
            _, perf = _run_backtest(factors, base_fd, price_data, fa)
            if perf is not None:
                fa.split_point = perf.get("split_point")
                print_performance_table(res_df, fa)
                avail = [c for c in factors if c in base_fd.columns]
                save_results(fa, avail, fa.output_dir, perf, res_df)

        result_json.append({
            "rank":          i + 1,
            "model":         "wls",
            "factors":       factors,
            "factor_cols":   factors,
            "must_include":  MUST_INCLUDE,
            "optional":      optional,
            # ★ ICIR 合成配置（用于复现）
            "icir_groups":   {
                name: {
                    "sub_clusters": cfg["sub_clusters"],
                    "icir_min": cfg.get("icir_min", args.icir_min),
                }
                for name, cfg in FACTOR_ICIR_GROUPS.items()
            },
            "icir_window":   args.icir_window,
            # 数据划分
            "val_split":     args.val_split,
            "test_split":    args.test_split,
            # 训练标签
            "check_days":    args.check_days,
            "multiplier":    args.multiplier,
            # 特征工程
            "lag":           args.lag,
            "factor_lags":   args.factor_lags,
            # 训练通用
            "data_file":     args.data_file,
            "start_date":    args.start_date,
            "train_window":  args.train_window,
            "retrain_freq":  args.retrain_freq,
            "mode":          args.mode,
            "fwd":           args.fwd,
            "use_scaler":    args.use_scaler,
            # 信号
            "reg_threshold":       args.reg_threshold,
            "close_threshold":     list(args.close_threshold),
            "close_mode":          args.close_mode,
            "use_strength_filter": args.use_strength_filter,
            "entry_strength_pct":  args.entry_strength_pct,
            "threshold_window":    args.threshold_window,
            # 绩效
            "val_cv_score":  combo["score"],
            "test_sharpe":   test_sharpes[i],
            "mac":           combo["mac"],
            # CV 参数
            "cv_folds":      args.cv_folds,
            "cv_lambda":     args.cv_lambda,
        })

    with open(out / "top5_combinations.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"\n{SEP}")
    print(f"  输出目录：{out}")
    print(f"  top5_combinations.json  ·  top5_signal_corr.csv")
    print(f"  test_performance.csv    （Test 段绩效，只读一次）")
    print(f"  stage1_ranking.csv  ·  stage1_clusters.csv")
    print(f"  stage2_all_trials.csv")
    print(f"  icir_weights_x_ma_ret_icir.csv  （ICIR 权重变化历史）")
    print(f"  backtest_combo_1/ ~ backtest_combo_{len(top5)}/")
    print(SEP)


if __name__ == "__main__":
    main()