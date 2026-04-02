"""
factor_search_xgb_top5.py — XGBoost 后端，挑出 5 组相关性低的优质因子组合
═══════════════════════════════════════════════════════════════════════════
目标
  找出 5 组因子组合，满足：
    ① 每组自身因子低共线（簇间选取保证）
    ② 每组样本外 Sharpe 尽量高
    ③ 5 组之间的策略信号相关性尽量低（可用于后续集成）

流程
  Stage 1   残差 IC + Ward 聚类
              ★ 按残差 IC-IR 排序后截断，最多保留 --max_factors 个因子（默认 50）
  Stage 2   跨簇 Optuna，收集 top-POOL_SIZE 候选（默认 15，不只取最优）
  Stage 3   逐步精修每个候选（refine_iters=2），精修完成后顺带存 results_df 缓存
  Stage 4   贪心多样性筛选
              · 先取 Score 最高的组合作为第 1 组
              · 后续每组：在剩余候选中取与已选各组
                信号相关性最低 且 Score 足够好 的组合
              · "信号相关性" = 各组回测 strategy_return 的 Pearson 相关
              · 复用 Stage 3 缓存的 results_df（省去重复回测）

因子 ↔ PnL 一致性保证（★ 修复）
  · _backtest_quick 和 _backtest_full 均强制 top_n_features=0
    ── 精修和最终 PnL 使用完全相同的特征集，JSON 保存因子 = 模型实际使用因子
  · top5_combinations.json 保存所有影响复现的参数（check_days / multiplier 等）
  · 若需要 top_n_features > 0，请在 ensemble 阶段单独设置，不在搜索阶段使用

XGBoost 相关说明
  · 因子搜索阶段（Stage 2/3）全程强制 top_n_features=0
    ── 聚类已做结构化筛选，XGB 内部再次过滤会使各 trial 因子集不可比，
       且会导致保存因子与 PnL 曲线脱钩
  · 模型超参（n_estimators / max_depth / learning_rate 等）全程不变

输出
  top5_combinations.json     5 组因子 + 所有复现参数（含 check_days 等）
  top5_signal_corr.csv       5 组策略信号相关矩阵
  backtest_combo_{1..5}/     每组完整回测结果
  stage2_all_trials.csv      全部 trial 记录
  stage1_ranking.csv / stage1_clusters.csv
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

script_path  = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(script_path.parent))

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from Utils_xgb import run_backtest_xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    run_backtest_xgb = None

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



MUST_INCLUDE: list[str] = [
    "x_vwap",
    "x_ma_ret_4h",
]

N_COMBOS  = 5    # 目标组合数
POOL_SIZE = 15   # Stage2/3 候选池大小（XGBoost 较慢，缩小以加速）


# ══════════════════════════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    p = argparse.ArgumentParser(
        description="XGBoost 后端 Top-5 多样性因子组合选择"
    )

    # ── 数据 ──────────────────────────────────────────────────────────────
    p.add_argument("--data_file",  default=os.path.join(project_root,"data\\P_with_ma_features.csv"))
    # p.add_argument("--data_file",  default=r"G:\pail_oil_cta\data_process\data\output\Palm_oil.csv")
    p.add_argument("--start_date", default="2018-04-17")

    # ── 模型选择（保留 wls 作为备用，主流程默认 xgb）──────────────────────
    p.add_argument("--model", default="xgb", choices=["xgb", "wls"],
                   help="回测后端：xgb（XGBoost，默认）/ wls（加权最小二乘）")

    # ── 训练通用 ──────────────────────────────────────────────────────────
    p.add_argument("--train_window",  type=int,   default=4000)
    p.add_argument("--mode",          default="rolling", choices=["rolling", "expanding"])
    p.add_argument("--retrain_freq",  type=int,   default=1000)
    p.add_argument("--fwd",           type=int,   default=3)
    p.add_argument("--lag",           type=int,   default=1)
    p.add_argument("--factor_lags",   default="")
    p.add_argument("--use_scaler",    action="store_true",  default=True)
    p.add_argument("--no_scaler",     action="store_false", dest="use_scaler")
    p.add_argument("--check_days",    type=int,   default=3)
    p.add_argument("--multiplier",    type=float, default=2.0)

    # ── 会话特征 ──────────────────────────────────────────────────────────
    p.add_argument("--add_session_features", action="store_true",  default=True)
    p.add_argument("--no_session_features",  action="store_false",
                   dest="add_session_features")

    # ── WLS 专属 ──────────────────────────────────────────────────────────
    p.add_argument("--weight_method",  default="rolling", choices=["park", "rolling"])
    p.add_argument("--rolling_window", type=int, default=1000)

    # ── XGBoost 专属 ──────────────────────────────────────────────────────
    # ★ top_n_features 仅影响 ensemble 阶段，搜索阶段全程强制 0
    p.add_argument("--top_n_features",   type=int,   default=0,
                   help="[XGB] 仅供 ensemble 阶段参考；搜索/保存阶段固定为 0")
    p.add_argument("--n_estimators",     type=int,   default=300)
    p.add_argument("--max_depth",        type=int,   default=3)
    p.add_argument("--learning_rate",    type=float, default=0.03)
    p.add_argument("--subsample",        type=float, default=0.7)
    p.add_argument("--colsample_bytree", type=float, default=0.5)
    p.add_argument("--min_child_weight", type=int,   default=20)
    p.add_argument("--reg_alpha",        type=float, default=1.0)
    p.add_argument("--reg_lambda",       type=float, default=8.0)

    # ── 回测信号 ──────────────────────────────────────────────────────────
    p.add_argument("--reg_threshold",   type=float, default=0.0)
    p.add_argument("--close_threshold", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--close_mode",      default="threshold",
                   choices=["fixed", "threshold", "hybrid",
                            "zscore_threshold", "zscore_hybrid"])

    # ── 信号强度过滤 ──────────────────────────────────────────────────────
    p.add_argument("--use_strength_filter", action="store_true",  default=True)
    p.add_argument("--no_strength_filter",  action="store_false", dest="use_strength_filter")
    p.add_argument("--entry_strength_pct",  type=float, default=0.7)
    p.add_argument("--threshold_window",    type=int,   default=50)

    # ── Stage 1 ───────────────────────────────────────────────────────────
    p.add_argument("--ic_window",       type=int,   default=500)
    p.add_argument("--ic_threshold",    type=float, default=0.02)
    p.add_argument("--n_clusters",      type=int,   default=0,
                   help="目标簇数，0=自动（Ward 距离截断）")
    p.add_argument("--corr_cluster_t",  type=float, default=0.55)
    p.add_argument("--top_per_cluster", type=int,   default=6,
                   help="每簇保留 top-M 个候选")
    p.add_argument("--min_cluster_ic",  type=float, default=0.01)
    p.add_argument("--max_factors",     type=int,   default=50,
                   help="★ 进入聚类的因子上限（按残差IC-IR截断），0=不限。"
                        "XGBoost对共线性不敏感，推荐50")

    # ── Stage 2 ───────────────────────────────────────────────────────────
    p.add_argument("--n_trials",         type=int,   default=150,
                   help="Optuna trials 数（XGB较慢，建议60-100）")
    p.add_argument("--n_startup_trials", type=int,   default=0)
    p.add_argument("--corr_penalty_w",   type=float, default=0.25)
    p.add_argument("--min_clusters_on",  type=int,   default=1)
    p.add_argument("--max_clusters_on",  type=int,   default=0)

    # ── Stage 3 ───────────────────────────────────────────────────────────
    p.add_argument("--refine_iters",     type=int,   default=1,
                   help="精修轮数（XGB较慢，默认2轮）")

    # ── Stage 4 ───────────────────────────────────────────────────────────
    p.add_argument("--diversity_w",     type=float, default=0.5,
                   help="多样性权重：0=纯Score，1=纯多样性")
    p.add_argument("--min_oos_sharpe",  type=float, default=-0.5,
                   help="候选组合 OOS Sharpe 最低门槛")

    # ── Stage 0（自动 MUST_INCLUDE）─────────────────────────────────────
    p.add_argument("--skip_stage0", action="store_true", default=False,
                   help="跳过 Stage 0，直接使用代码中硬编码的 MUST_INCLUDE")
    p.add_argument("--s0_top_k",    type=int,   default=10,
                   help="Stage 0：IC 排名取 top-K 因子进入组合筛选")
    p.add_argument("--s0_noise",    type=float, default=0.1,
                   help="Stage 0：IC-IR 排名噪声幅度（0=确定性，>0 引入随机性）")

    # ── 随机种子 ──────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=0,
                   help="随机种子，0=每次不同（基于时间），>0=可复现")

    # ── 续跑 ──────────────────────────────────────────────────────────────
    p.add_argument("--resume",     action="store_true", default=False)
    p.add_argument("--study_path", default="")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
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


def _score(oos_sharpe: float, mac: float, penalty_w: float) -> float:
    if oos_sharpe <= -998.0:
        return -999.0
    return oos_sharpe - penalty_w * mac


def _get_oos_sharpe(results_df: pd.DataFrame, args) -> float:
    split = getattr(args, "split_point", None)
    fwd   = getattr(args, "fwd", 1)
    if split is None or split not in results_df.index:
        return -999.0
    si     = results_df.index.get_loc(split)
    if isinstance(si, slice):
        si = si.stop - 1 if si.stop is not None else len(results_df) - 1
    elif isinstance(si, np.ndarray):
        si = int(np.flatnonzero(si)[-1])
    out_df = results_df.iloc[si + fwd + 2:]
    if len(out_df) < 50:
        return -999.0
    key = pd.to_datetime(out_df.index).normalize()
    d   = out_df.groupby(key).agg(r=("strategy_return", "sum"),
                                   sw=("is_switch", "any"))
    r   = d[~d["sw"]]["r"].values
    return float(calc_metrics_from_returns(r)["sharpe_ratio"]) if len(r) >= 20 else -999.0


# ══════════════════════════════════════════════════════════════════════════════
# 核心：快速回测（Stage 2/3 搜索阶段）
# ══════════════════════════════════════════════════════════════════════════════

def _backtest_quick(
    factor_subset: list[str],
    base_fd:       pd.DataFrame,
    price_data:    pd.Series,
    args,
) -> tuple[float, float]:
    """
    Stage 2/3 搜索阶段快速回测（不缓存 results_df）。
    强制 top_n_features=0，保持各 trial 因子集可比。
    """
    avail = [c for c in factor_subset if c in base_fd.columns]
    if not avail:
        return -999.0, 0.0

    mac = _mean_abs_corr(base_fd[avail])
    ta  = copy.copy(args)
    ta.contract_switch_dates = getattr(args, "contract_switch_dates", [])

    # ★ 搜索阶段强制关闭 XGB 内部特征过滤
    if getattr(ta, "model", "xgb") == "xgb":
        ta.top_n_features = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if getattr(ta, "model", "xgb") == "xgb":
            res, perf = run_backtest_xgb(base_fd[avail].copy(), price_data, ta)
        else:
            res, perf = run_backtest_reg(base_fd[avail].copy(), price_data, ta)

    if res is None or perf is None:
        return -999.0, mac

    ta.split_point = perf.get("split_point")
    if getattr(ta, "use_strength_filter", False):
        res  = apply_strength_filter(res, ta)
        perf = recalc_performance(res, ta)

    return _get_oos_sharpe(res, ta), mac


# ══════════════════════════════════════════════════════════════════════════════
# 完整回测（Stage 3 末尾缓存 / Stage 4 备用）
# ══════════════════════════════════════════════════════════════════════════════

def _backtest_full(
    factor_subset: list[str],
    base_fd:       pd.DataFrame,
    price_data:    pd.Series,
    args,
) -> tuple[float, float, pd.DataFrame | None, dict | None]:
    """
    完整回测，返回 (oos_sharpe, mean_abs_corr, results_df, perf)。

    ★ 因子 ↔ PnL 一致性保证：
      同样强制 top_n_features=0，与 _backtest_quick 保持完全一致。
      这样保存到 JSON 的因子列表 = 模型实际使用的因子列表，
      File 1（集成脚本）重跑时可精确复现。

      若需要使用 top_n_features > 0，请在集成阶段（main_ensamble_mix.py）
      单独设置，而非在因子搜索阶段使用。
    """
    avail = [c for c in factor_subset if c in base_fd.columns]
    if not avail:
        return -999.0, 0.0, None, None

    mac = _mean_abs_corr(base_fd[avail])
    ta  = copy.copy(args)
    ta.contract_switch_dates = getattr(args, "contract_switch_dates", [])

    # ★ 与 _backtest_quick 保持一致：强制关闭 XGB 内部特征过滤
    #   确保 factor_subset（将被保存到 JSON）= 模型实际使用的特征集
    if getattr(ta, "model", "xgb") == "xgb":
        ta.top_n_features = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if getattr(ta, "model", "xgb") == "xgb":
            res, perf = run_backtest_xgb(base_fd[avail].copy(), price_data, ta)
        else:
            res, perf = run_backtest_reg(base_fd[avail].copy(), price_data, ta)

    if res is None or perf is None:
        return -999.0, mac, None, None

    ta.split_point = perf.get("split_point")
    if getattr(ta, "use_strength_filter", False):
        res  = apply_strength_filter(res, ta)
        perf = recalc_performance(res, ta)

    return _get_oos_sharpe(res, ta), mac, res, perf


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — 自动挑选 MUST_INCLUDE（WLS 快筛）
# ══════════════════════════════════════════════════════════════════════════════

def _wls_oos_sharpe(
    factor_subset: list[str],
    base_fd:       pd.DataFrame,
    price_data:    pd.Series,
    args,
) -> float:
    """用 WLS 快速回测，返回 OOS Sharpe（Stage 0 专用）。"""
    avail = [c for c in factor_subset if c in base_fd.columns]
    if not avail:
        return -999.0
    ta = copy.copy(args)
    ta.contract_switch_dates = getattr(args, "contract_switch_dates", [])
    ta.model = "wls"  # 强制 WLS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, perf = run_backtest_reg(base_fd[avail].copy(), price_data, ta)
    if res is None or perf is None:
        return -999.0
    ta.split_point = perf.get("split_point")
    if getattr(ta, "use_strength_filter", False):
        res  = apply_strength_filter(res, ta)
        perf = recalc_performance(res, ta)
    return _get_oos_sharpe(res, ta)


def stage0_auto_must_include(
    select_fd:  pd.DataFrame,
    price_data: pd.Series,
    args,
    out_dir:    Path,
    rng:        np.random.Generator,
) -> list[str]:
    """
    自动挑选 MUST_INCLUDE 因子（WLS 快筛）。

    流程
    ----
    1. 计算所有因子 IC-IR，加噪声后取 top-K
    2. 单因子 WLS 回测 → top 10（按 OOS Sharpe）
    3. top10 两两组合 → WLS 回测 → top 10 二因子组合
    4. top10 二因子 + 第三因子 → WLS 回测 → top 10 三因子组合
    5. 汇总 30 组，按 OOS Sharpe 取最优 1 组作为 MUST_INCLUDE
    """
    from itertools import combinations

    ic_window = args.ic_window
    noise_amp = getattr(args, "s0_noise", 0.1)
    top_k     = getattr(args, "s0_top_k", 10)

    labels = _compute_reversal_labels(
        price_data, fwd=args.fwd,
        check_days=getattr(args, "check_days", 3),
        multiplier=getattr(args, "multiplier", 2.0),
    )

    x_cols = [c for c in select_fd.columns if c.startswith("x_")
              and c not in ("x_is_night", "x_session_sin")]

    # ── Step 1: IC-IR 排名 + 噪声 → top-K ────────────────────────────────
    print(f"  计算 {len(x_cols)} 个因子 IC-IR（窗口={ic_window}）...")
    ic_ir_map: dict[str, float] = {}
    for col in x_cols:
        raw_ic = _rolling_ic(select_fd[col], labels, ic_window)
        ic_ir_map[col] = _ic_ir(raw_ic)

    # 加噪声引入不确定性
    if noise_amp > 0:
        noise_std = noise_amp * np.std(list(ic_ir_map.values()))
        noisy = {c: v + rng.normal(0, noise_std) for c, v in ic_ir_map.items()}
    else:
        noisy = ic_ir_map

    sorted_factors = sorted(noisy, key=lambda c: noisy[c], reverse=True)
    top_factors = sorted_factors[:top_k]
    print(f"  IC-IR top-{top_k}（噪声={noise_amp:.2f}）：")
    for i, f in enumerate(top_factors):
        print(f"    {i+1:2d}. {f:<50} IC-IR={ic_ir_map[f]:.4f}")

    # ── Step 2: 单因子 WLS 回测 → top 10 ─────────────────────────────────
    print(f"\n  单因子 WLS 回测...")
    single_results: list[tuple[float, list[str]]] = []
    for f in top_factors:
        oos_s = _wls_oos_sharpe([f], select_fd, price_data, args)
        single_results.append((oos_s, [f]))
        print(f"    {f:<50} OOS={oos_s:+.3f}")
    single_results.sort(key=lambda x: x[0], reverse=True)
    top10_single = single_results[:10]

    # ── Step 3: 两两组合 → WLS 回测 → top 10 ─────────────────────────────
    print(f"\n  二因子组合 WLS 回测（{len(list(combinations(top_factors, 2)))} 组）...")
    pair_results: list[tuple[float, list[str]]] = []
    for f1, f2 in combinations(top_factors, 2):
        oos_s = _wls_oos_sharpe([f1, f2], select_fd, price_data, args)
        pair_results.append((oos_s, [f1, f2]))
    pair_results.sort(key=lambda x: x[0], reverse=True)
    top10_pair = pair_results[:10]
    print(f"  二因子 top 3：")
    for oos_s, fs in top10_pair[:3]:
        short = [f.replace("x_","")[:25] for f in fs]
        print(f"    OOS={oos_s:+.3f}  {short}")

    # ── Step 4: 三因子组合 → WLS 回测 → top 10 ───────────────────────────
    print(f"\n  三因子组合 WLS 回测...")
    triple_results: list[tuple[float, list[str]]] = []
    tried: set[str] = set()
    for _, base_pair in top10_pair:
        for f3 in top_factors:
            if f3 in base_pair:
                continue
            key = "|".join(sorted(base_pair + [f3]))
            if key in tried:
                continue
            tried.add(key)
            oos_s = _wls_oos_sharpe(base_pair + [f3], select_fd, price_data, args)
            triple_results.append((oos_s, base_pair + [f3]))
    triple_results.sort(key=lambda x: x[0], reverse=True)
    top10_triple = triple_results[:10]
    print(f"  三因子 top 3：")
    for oos_s, fs in top10_triple[:3]:
        short = [f.replace("x_","")[:25] for f in fs]
        print(f"    OOS={oos_s:+.3f}  {short}")

    # ── Step 5: 汇总 30 组，取 OOS Sharpe 最高的 1 组 ─────────────────────
    all_candidates = top10_single + top10_pair + top10_triple
    all_candidates.sort(key=lambda x: x[0], reverse=True)

    # 保存完整记录
    s0_rows = []
    for oos_s, fs in all_candidates:
        s0_rows.append({
            "n_factors": len(fs),
            "oos_sharpe": oos_s,
            "factors": "|".join(fs),
        })
    pd.DataFrame(s0_rows).to_csv(
        out_dir / "stage0_candidates.csv", index=False, encoding="utf-8-sig"
    )

    best_oos, best_factors = all_candidates[0]
    print(f"\n  ★ Stage 0 最优 MUST_INCLUDE（OOS Sharpe={best_oos:+.3f}）：")
    for f in best_factors:
        print(f"    {f}")

    return best_factors


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — 残差 IC + Ward 聚类
# ══════════════════════════════════════════════════════════════════════════════

def stage1_cluster(
    factor_data: pd.DataFrame,
    price_data:  pd.Series,
    args,
) -> tuple[list[list[str]], pd.DataFrame, pd.DataFrame]:

    labels   = _compute_reversal_labels(
        price_data, fwd=args.fwd,
        check_days=getattr(args, "check_days", 3),
        multiplier=getattr(args, "multiplier", 2.0),
    )
    window   = args.ic_window
    mi = getattr(args, "_must_include", MUST_INCLUDE)
    opt_cols = [c for c in factor_data.columns if c not in mi]
    must_df  = factor_data[[c for c in mi if c in factor_data.columns]]
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
        dropped   = len(opt_cols) - max_f
        print(f"\n  ★ 因子上限={max_f}，丢弃末尾 {dropped} 个低 IC-IR 因子")
        opt_cols  = keep_cols
        ic_series = {c: ic_series[c] for c in keep_cols}

    ic_df    = pd.DataFrame({c: ic_series[c] for c in opt_cols}).fillna(0.0)
    corr_mat = ic_df.corr()
    dist_mat = (1.0 - corr_mat.abs().clip(0, 1)).clip(0)
    np.fill_diagonal(dist_mat.values, 0.0)
    condensed = squareform(dist_mat.values, checks=False).clip(0)
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
    cluster_info: list[dict]            = []

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

    print(f"\n  聚类：{n_clusters} 簇 -> 有效簇 {len(cluster_candidates)} 个"
          f"（每簇保留 top-{M}）")
    print(f"\n  {'簇':>3}  {'大小':>4}  {'最佳残差IC-IR':>13}  候选因子")
    print(f"  {'─'*80}")
    for info in cluster_info:
        print(f"  {info['cluster_id']:3d}  {info['size']:4d}  "
              f"{info['best_res_ic_ir']:13.4f}  {info['candidates']}")

    ranking_df["in_pool"] = ranking_df["factor"].isin(
        {f for cl in cluster_candidates for f in cl}
    )
    return cluster_candidates, ranking_all, pd.DataFrame(cluster_info)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — 跨簇 Optuna，收集 top-POOL_SIZE 候选
# ══════════════════════════════════════════════════════════════════════════════

def stage2_collect_pool(
    cluster_candidates: list[list[str]],
    base_fd:            pd.DataFrame,
    price_data:         pd.Series,
    args,
    out_dir:            Path,
) -> list[dict]:
    mi         = getattr(args, "_must_include", MUST_INCLUDE)
    n_clusters = len(cluster_candidates)
    min_on     = max(0, args.min_clusters_on)
    max_on     = args.max_clusters_on if args.max_clusters_on > 0 else n_clusters
    cw         = args.corr_penalty_w
    n_start    = (args.n_startup_trials if args.n_startup_trials > 0
                  else min(15, args.n_trials // 4))

    space_size = 1
    for cl in cluster_candidates:
        space_size *= (len(cl) + 1)
    print(f"  搜索空间：{space_size:,} 种  "
          f"Optuna {args.n_trials} trials  随机探索前 {n_start} 轮")
    print(f"  后端模型: {args.model.upper()}（全程 top_n_features=0）")

    trial_log   = []
    best_val    = [-999.0]
    PRINT_EVERY = 10

    def objective(trial: "optuna.Trial") -> float:
        chosen = []
        for ci, cl in enumerate(cluster_candidates):
            pick = trial.suggest_categorical(f"cluster_{ci}", ["none"] + cl)
            if pick != "none":
                chosen.append(pick)

        if not (min_on <= len(chosen) <= max_on):
            raise optuna.exceptions.TrialPruned()

        factor_subset = mi + chosen
        oos_s, mac    = _backtest_quick(factor_subset, base_fd, price_data, args)
        score         = _score(oos_s, mac, cw)

        is_best = score > best_val[0]
        if is_best:
            best_val[0] = score

        trial_num = trial.number + 1
        if is_best or trial_num % PRINT_EVERY == 0 or trial_num == args.n_trials:
            short = lambda f: f.replace("x_","").replace("_60min","").replace("_240min","")
            print(
                f"  Trial {trial_num:3d}{'★' if is_best else ' '}  "
                f"Score={score:+.3f}  OOS={oos_s:+.3f}  Corr={mac:.2f}  "
                f"N={len(factor_subset)}  {[short(f) for f in chosen] or ['无']}"
            )

        trial_log.append({
            "trial":         trial.number + 1,
            "score":         score,
            "oos_sharpe":    oos_s,
            "mean_abs_corr": mac,
            "n_factors":     len(factor_subset),
            "optional":      "|".join(chosen),
            "all_factors":   "|".join(factor_subset),
        })
        return score

    if args.resume and args.study_path and Path(args.study_path).exists():
        with open(args.study_path, "rb") as f:
            study = pickle.load(f)
        print(f"  [续跑] 已加载，已有 {len(study.trials)} 个 trial")
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=getattr(args, "_rng_seed", 42), n_startup_trials=n_start, multivariate=True),
            pruner=optuna.pruners.NopPruner(),
        )

    print(f"\n  {'─'*76}")
    print(f"  {'Trial':>6}  {'Score':>7}  {'OOS':>7}  {'Corr':>5}  {'N':>3}  选入可选因子")
    print(f"  {'─'*76}")
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
            pool.append(rec)

    print(f"\n  候选池（去重后）：{len(pool)} 个组合，取 top-{POOL_SIZE} 进入 Stage 3")
    return pool


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — 逐步精修每个候选，精修完成后顺带存 results_df 缓存
# ══════════════════════════════════════════════════════════════════════════════

def stage3_refine_pool(
    pool:               list[dict],
    cluster_candidates: list[list[str]],
    base_fd:            pd.DataFrame,
    price_data:         pd.Series,
    args,
) -> list[dict]:
    """
    对候选池（取前 POOL_SIZE 个）逐一精修。
    精修和最终完整回测均使用 top_n_features=0（由 _backtest_quick /
    _backtest_full 内部保证），确保因子列表 ↔ PnL 曲线严格对应。
    """
    cw = args.corr_penalty_w
    mi = getattr(args, "_must_include", MUST_INCLUDE)
    factor_to_cluster: dict[str, list[str]] = {
        f: cl for cl in cluster_candidates for f in cl
    }

    refined: list[dict] = []
    limit = min(POOL_SIZE, len(pool))

    for i, rec in enumerate(pool[:limit]):
        init_factors = rec["all_factors"].split("|")
        opt_short = [f.replace("x_","")[:18] for f in init_factors if f not in MUST_INCLUDE]

        current    = list(init_factors)
        oos_s, mac = _backtest_quick(current, base_fd, price_data, args)
        cur_score  = _score(oos_s, mac, cw)

        for it in range(args.refine_iters):
            best_delta = 0.0
            best_move  = None
            optional_cur = [f for f in current if f not in MUST_INCLUDE]
            covered = set(optional_cur)

            # (a)(b) swap / remove
            for f_old in optional_cur:
                cl = factor_to_cluster.get(f_old, [f_old])
                for f_new in [f for f in cl if f != f_old] + [None]:
                    trial = [f for f in current if f != f_old]
                    if f_new is not None:
                        trial.append(f_new)
                    s_oos, s_mac = _backtest_quick(trial, base_fd, price_data, args)
                    s_score = _score(s_oos, s_mac, cw)
                    if s_score - cur_score > best_delta:
                        best_delta = s_score - cur_score
                        best_move  = (trial, s_score, s_oos)

            # (c) add：为未选的簇加入 top-1
            for cl in cluster_candidates:
                if not any(f in covered for f in cl):
                    trial = current + [cl[0]]
                    s_oos, s_mac = _backtest_quick(trial, base_fd, price_data, args)
                    s_score = _score(s_oos, s_mac, cw)
                    if s_score - cur_score > best_delta:
                        best_delta = s_score - cur_score
                        best_move  = (trial, s_score, s_oos)

            if best_move is None or best_delta <= 1e-4:
                break
            current, cur_score, oos_s = best_move

        # ★ 精修完成后跑完整回测并缓存（top_n_features=0，由 _backtest_full 保证）
        full_oos, full_mac, res_df, perf = _backtest_full(
            current, base_fd, price_data, args
        )
        final_score = _score(full_oos, full_mac, cw)
        final_opt   = [f.replace("x_","")[:18] for f in current if f not in MUST_INCLUDE]
        print(f"  [{i+1:2d}/{limit}]  {opt_short} -> {final_opt}"
              f"  Score={final_score:+.4f}  OOS={full_oos:+.4f}")

        refined.append({
            "all_factors": "|".join(current),
            "optional":    "|".join(f for f in current if f not in MUST_INCLUDE),
            "oos_sharpe":  full_oos,
            "mean_abs_corr": full_mac,
            "score":       final_score,
            "results_df":  res_df,
            "perf":        perf,
        })

    seen: set[str] = set()
    dedup: list[dict] = []
    for r in sorted(refined, key=lambda x: x["score"], reverse=True):
        if r["all_factors"] not in seen:
            seen.add(r["all_factors"])
            dedup.append(r)

    return dedup


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — 贪心多样性筛选 5 组
# ══════════════════════════════════════════════════════════════════════════════

def stage4_diverse_top5(
    refined_pool: list[dict],
    out_dir:      Path,
    args,
    n_pick:       int = N_COMBOS,
) -> list[dict]:
    min_oos = args.min_oos_sharpe
    dw      = args.diversity_w

    valid = [r for r in refined_pool
             if r["results_df"] is not None and r["oos_sharpe"] >= min_oos]
    if len(valid) < n_pick:
        print(f"  [警告] 有效候选不足 {n_pick} 个（共 {len(valid)} 个），"
              f"尝试降低 --min_oos_sharpe")
        valid = [r for r in refined_pool if r["results_df"] is not None]
        valid = valid[:max(n_pick, len(valid))]

    if not valid:
        print("  [错误] 无有效候选（results_df 均为空）")
        return []

    print(f"  共 {len(valid)} 个有效候选，开始贪心多样性筛选...")

    scores     = np.array([c["score"] for c in valid])
    s_min, s_max = scores.min(), scores.max()
    scores_norm  = (scores - s_min) / (s_max - s_min + 1e-9)

    signal_returns = {
        i: c["results_df"]["strategy_return"].fillna(0.0)
        for i, c in enumerate(valid)
    }

    selected_idx: list[int] = []

    for pick_n in range(min(n_pick, len(valid))):
        if pick_n == 0:
            best_i = int(np.argmax(scores_norm))
        else:
            best_composite = -np.inf
            best_i = -1
            already = [signal_returns[j] for j in selected_idx]

            for i in range(len(valid)):
                if i in selected_idx:
                    continue
                sig_i = signal_returns[i]
                max_corr = max(
                    abs(float(sig_i.corr(
                        already_s.reindex(sig_i.index).fillna(0.0)
                    )))
                    for already_s in already
                )
                composite = (1 - dw) * scores_norm[i] + dw * (1 - max_corr)
                if composite > best_composite:
                    best_composite = composite
                    best_i = i

        selected_idx.append(best_i)
        c = valid[best_i]
        print(f"  [v] 第 {pick_n+1} 组  OOS={c['oos_sharpe']:+.3f}  "
              f"Score={c['score']:+.3f}  "
              f"opt={[f.replace('x_','')[:20] for f in c['optional'].split('|') if f]}")

    selected = [valid[i] for i in selected_idx]

    ret_dict = {}
    for i, c in enumerate(selected):
        opt_list = [f for f in c["optional"].split("|") if f]
        label = f"组合{i+1}（{','.join(f.replace('x_','')[:12] for f in opt_list[:2])}{'...' if len(opt_list)>2 else ''}）"
        ret_dict[label] = c["results_df"]["strategy_return"].fillna(0.0)

    sig_corr = pd.DataFrame(ret_dict).corr()
    sig_corr.to_csv(out_dir / "top5_signal_corr.csv", encoding="utf-8-sig")

    print(f"\n  ── 5 组策略信号相关矩阵 ──")
    print(sig_corr.round(3).to_string())

    return selected


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_arguments()

    if not HAS_OPTUNA:
        print("[错误] 请先安装：pip install optuna scipy scikit-learn"); return
    if args.model == "xgb" and not HAS_XGB:
        print("[错误] Utils_xgb.py 未找到或 xgboost 未安装。")
        print("       请执行：pip install xgboost，并确认 Utils_xgb.py 与本文件同目录。")
        return

    ts  = get_timestamp()
    out = Path(project_root) / "results" / f"xgb_top5_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out)

    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  XGBoost Top-5 多样性因子组合选择")
    print(f"  必选: {MUST_INCLUDE}")
    print(f"  ★ 可选因子上限: {args.max_factors}  每簇候选: {args.top_per_cluster}")
    if args.model == "xgb":
        print(f"  XGB: trees={args.n_estimators}  depth={args.max_depth}"
              f"  lr={args.learning_rate}  sub={args.subsample}"
              f"  col={args.colsample_bytree}")
        print(f"       mcw={args.min_child_weight}  α={args.reg_alpha}"
              f"  λ={args.reg_lambda}")
        print(f"  ★ top_n_features: 搜索/保存阶段固定=0（因子<->PnL严格对应）"
              f"  用户设置={args.top_n_features}（仅供 ensemble 参考）")
    else:
        print(f"  WLS: weight_method={args.weight_method}")
    print(f"  训练标签: check_days={args.check_days}  multiplier={args.multiplier}")
    print(f"  Stage2: {args.n_trials} trials  penalty={args.corr_penalty_w}"
          f"  候选池={POOL_SIZE}")
    print(f"  Stage3: refine_iters={args.refine_iters}")
    print(f"  Stage4: diversity_w={args.diversity_w}  min_oos={args.min_oos_sharpe}")
    print(SEP)

    # ── 加载数据 ──────────────────────────────────────────────────────────
    df = load_palm_oil_data(args.data_file)
    if args.start_date:
        df = df[df.index >= pd.to_datetime(args.start_date)]

    if "dominant_id" in df.columns and len(df) > 0:
        mask = df["dominant_id"] != df["dominant_id"].shift(1)
        mask.iloc[0] = False
        args.contract_switch_dates = df.index[mask].tolist()
    else:
        args.contract_switch_dates = []

    all_x_cols = sorted([c for c in df.columns if c.startswith("x_")])
    missing    = [c for c in MUST_INCLUDE if c not in all_x_cols]
    if missing:
        print(f"[错误] 必选因子不在数据中: {missing}"); return

    print(f"[因子发现] 共 {len(all_x_cols)} 个 x_ 因子")

    factor_lags = None
    if getattr(args, "factor_lags", "").strip():
        try:
            factor_lags = [int(x) for x in args.factor_lags.split(",") if x.strip()]
        except ValueError:
            factor_lags = None

    # ── Stage 1 选因子：用 lag=1（原始因子，不展开滞后列）──────────────
    select_fd, select_price, _ = prepare_factor_data(
        df,
        selected_factors=all_x_cols,
        lag=1,
        factor_lags=None,
        add_session_features=getattr(args, "add_session_features", True),
    )
    print(f"[选因子数据] lag=1  bars={len(select_fd)}  "
          f"{select_fd.index[0]} ~ {select_fd.index[-1]}")

    # ── Stage 2/3/4 回测：用 lag={args.lag}（展开滞后特征）────────────
    base_fd, price_data, _ = prepare_factor_data(
        df,
        selected_factors=all_x_cols,
        lag=args.lag,
        factor_lags=factor_lags,
        add_session_features=getattr(args, "add_session_features", True),
    )
    print(f"[回测数据]   lag={args.lag}  bars={len(base_fd)}  "
          f"{base_fd.index[0]} ~ {base_fd.index[-1]}\n")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1 — 残差 IC + Ward 聚类（用 lag=1 的原始因子）
    # ══════════════════════════════════════════════════════════════════════
    print(f"{SEP}\n  Stage 1 / 4  ─  残差 IC + Ward 聚类"
          f"（因子上限={args.max_factors}，选因子 lag=1）\n{SEP}")
    cluster_candidates, ranking_df, cluster_info_df = stage1_cluster(
        select_fd, select_price, args
    )
    ranking_df.to_csv(out / "stage1_ranking.csv", index=True, encoding="utf-8-sig")
    cluster_info_df.to_csv(out / "stage1_clusters.csv", index=False, encoding="utf-8-sig")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2 — 跨簇 Optuna，收集候选池（用 lag={args.lag} 的回测数据）
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 2 / 4  ─  跨簇 Optuna（收集候选池 top-{POOL_SIZE}）\n{SEP}")
    pool = stage2_collect_pool(cluster_candidates, base_fd, price_data, args, out)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3 — 逐步精修 + 缓存完整回测
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 3 / 4  ─  精修（{POOL_SIZE} 个候选 × {args.refine_iters} 轮）"
          f"  [{args.model.upper()}，top_n=0]\n{SEP}")
    refined_pool = stage3_refine_pool(
        pool, cluster_candidates, base_fd, price_data, args
    )

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4 — 贪心多样性筛选 5 组
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 4 / 4  ─  贪心多样性筛选 {N_COMBOS} 组\n{SEP}")
    top5 = stage4_diverse_top5(refined_pool, out, args, n_pick=N_COMBOS)

    if not top5:
        print("[错误] Stage 4 未能选出组合，请检查候选池和参数设置")
        return

    # ══════════════════════════════════════════════════════════════════════
    # 输出结果
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}")
    print(f"  最终 {len(top5)} 组因子组合  [{args.model.upper()}]")
    print(SEP)

    result_json = []
    for i, combo in enumerate(top5):
        factors  = combo["all_factors"].split("|")
        optional = [f for f in factors if f not in MUST_INCLUDE]

        print(f"\n  ── 第 {i+1} 组 ──────────────────────────────────────────")
        print(f"  OOS Sharpe = {combo['oos_sharpe']:+.4f}  "
              f"Score = {combo['score']:+.4f}  "
              f"平均因子相关 = {combo['mean_abs_corr']:.3f}")
        print(f"  因子（{len(factors)} 个）：")
        for f in factors:
            tag = " ★必选" if f in MUST_INCLUDE else " ○选入"
            print(f"    {f:<55}{tag}")

        sub_dir = out / f"backtest_combo_{i+1}"
        sub_dir.mkdir(exist_ok=True)
        fa = copy.copy(args)
        fa.contract_switch_dates = getattr(args, "contract_switch_dates", [])
        fa.output_dir = str(sub_dir)

        res_df = combo.get("results_df")
        perf   = combo.get("perf")

        if res_df is not None and perf is not None:
            fa.split_point = perf.get("split_point")
            print_performance_table(res_df, fa)
            avail = [c for c in factors if c in base_fd.columns]
            save_results(fa, avail, fa.output_dir, perf, res_df)
        else:
            print(f"  [警告] 第 {i+1} 组缓存为空，重新跑完整回测...")
            avail = [c for c in factors if c in base_fd.columns]
            full_oos, full_mac, res_df2, perf2 = _backtest_full(
                avail, base_fd, price_data, fa
            )
            if res_df2 is not None:
                fa.split_point = perf2.get("split_point")
                print_performance_table(res_df2, fa)
                save_results(fa, avail, fa.output_dir, perf2, res_df2)

        # ★ 保存所有影响复现的参数（含 check_days / multiplier 等）
        result_json.append({
            "rank":          i + 1,
            "model":         args.model,
            # ── 因子 ──────────────────────────────────────────────────
            "factors":       factors,
            "factor_cols":   factors,      # 与 File1 _load_combo_config 字段名对齐
            "must_include":  MUST_INCLUDE,
            "optional":      optional,
            # ── 训练标签（★ 关键，之前未保存）──────────────────────────
            "check_days":    args.check_days,
            "multiplier":    args.multiplier,
            # ── 特征工程 ──────────────────────────────────────────────
            "lag":                  args.lag,
            "factor_lags":          args.factor_lags,
            "add_session_features": args.add_session_features,
            # ── 训练通用 ──────────────────────────────────────────────
            "data_file":     args.data_file,
            "start_date":    args.start_date,
            "train_window":  args.train_window,
            "retrain_freq":  args.retrain_freq,
            "mode":          args.mode,
            "fwd":           args.fwd,
            "use_scaler":    args.use_scaler,
            # ── 信号 ──────────────────────────────────────────────────
            "reg_threshold":       args.reg_threshold,
            "close_threshold":     list(args.close_threshold),
            "close_mode":          args.close_mode,
            "use_strength_filter": args.use_strength_filter,
            "entry_strength_pct":  args.entry_strength_pct,
            "threshold_window":    args.threshold_window,
            # ── 绩效 ──────────────────────────────────────────────────
            "oos_sharpe":    combo["oos_sharpe"],
            "score":         combo["score"],
            "mean_abs_corr": combo["mean_abs_corr"],
            # ── XGBoost 超参 ──────────────────────────────────────────
            "xgb_params": {
                "n_estimators":     args.n_estimators,
                "max_depth":        args.max_depth,
                "learning_rate":    args.learning_rate,
                "subsample":        args.subsample,
                "colsample_bytree": args.colsample_bytree,
                "min_child_weight": args.min_child_weight,
                "reg_alpha":        args.reg_alpha,
                "reg_lambda":       args.reg_lambda,
                # ★ 搜索/保存阶段固定 0；集成阶段如需 top_n 可另行设置
                "top_n_features":   0,
            } if args.model == "xgb" else {},
        })

    with open(out / "top5_combinations.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    # Stage2 Top-10 速览
    try:
        tdf   = pd.read_csv(out / "stage2_all_trials.csv")
        short = lambda f: f.replace("x_","").replace("_60min","").replace("_240min","")
        print(f"\n  Stage2 Top-10（按 Score）：")
        print(f"  {'#':>3}  {'Score':>7}  {'OOS':>7}  {'Corr':>5}  可选因子")
        print(f"  {'─'*70}")
        for idx, row in tdf.head(10).iterrows():
            opt = [short(f) for f in str(row.get("optional","")).split("|") if f]
            print(f"  {idx+1:3d}  {row['score']:+7.3f}  {row['oos_sharpe']:+7.3f}  "
                  f"{row.get('mean_abs_corr',0):5.2f}  {opt or ['仅必选']}")
    except Exception:
        pass

    print(f"\n{SEP}")
    print(f"  输出目录：{out}")
    print(f"  top5_combinations.json  ·  top5_signal_corr.csv")
    print(f"  stage1_ranking.csv  ·  stage1_clusters.csv")
    print(f"  stage2_all_trials.csv  ·  optuna_study.pkl")
    print(f"  backtest_combo_1/ ~ backtest_combo_{len(top5)}/")
    print(SEP)


if __name__ == "__main__":
    main()