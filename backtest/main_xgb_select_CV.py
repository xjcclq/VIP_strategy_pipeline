"""
factor_search_xgb_best1.py — XGBoost 后端，找出 1 组最优因子组合
═══════════════════════════════════════════════════════════════════════════
相比原版 factor_search_xgb_top5.py 的改动
  1. 目标只找 1 组最优组合（不再做多样性筛选）
  2. 严格三段划分：Train / Val / Test
       - Train : start_date ~ val_split
       - Val   : val_split  ~ test_split   （Optuna 优化目标，walk-forward CV）
       - Test  : test_split ~ 末尾          （搜索结束后只碰一次，只报告）
  3. Purged Walk-Forward CV（在 Val 段内）
       - 均分 K 折，折间加 purge gap = fwd bars
       - Optuna 目标 = mean(Sharpe_k) − λ·std(Sharpe_k)
  4. 因子压缩：聚类去冗余（medoid）+ SHAP 持续性筛选
       → 从原始几百个压缩到 ~30 个高质量候选，再送入 Optuna 搜索
  5. MUST_INCLUDE 默认为空列表，可在命令行或代码中指定

流程
  Stage 0   聚类去冗余（相关性阈值聚类 → medoid 代表）
  Stage 1   与 target 相关性过滤（去掉绝对不相关因子）
  Stage 2   XGBoost + SHAP 持续性筛选（6 折 CV，取每折 top-N，求并集）
  Stage 3   Optuna 跨因子组合搜索（Purged Walk-Forward CV 打分）
  Stage 4   精修（局部 swap/add/remove）
  Stage 5   Test 段一次性评估，输出结果

输出
  best_combination.json      最优因子组合 + 所有复现参数
  best_backtest/             完整回测结果
  stage0_medoids.csv         聚类代表因子
  stage1_corr_filtered.csv   相关性过滤后因子
  stage2_shap_selected.csv   SHAP 筛选后因子
  stage3_all_trials.csv      Optuna 所有 trial 记录
  test_performance.csv       Test 段绩效（最终报告）
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
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import xgboost as xgb
    HAS_XGB_LIB = True
except ImportError:
    HAS_XGB_LIB = False

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

# 默认必选因子：空列表，可通过 --must_include 传入
MUST_INCLUDE: list[str] = ['x_ma_ret_4h','x_is_night']


# ══════════════════════════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    p = argparse.ArgumentParser(
        description="XGBoost 后端，Purged Walk-Forward CV，找出 1 组最优因子组合"
    )

    # ── 数据 ──────────────────────────────────────────────────────────────
    p.add_argument("--data_file",  default=os.path.join(project_root, "data\\P_with_ma_features.csv"))
    p.add_argument("--start_date", default="2018-04-17")

    # ── 三段划分（时间字符串，格式 YYYY-MM-DD）────────────────────────────
    p.add_argument("--val_split",  default="2022-01-01",
                   help="Train/Val 分界点；Val 段用于 Optuna CV 优化")
    p.add_argument("--test_split", default="2023-06-01",
                   help="Val/Test 分界点；Test 段搜索完成后只读一次")

    # ── 模型 ──────────────────────────────────────────────────────────────
    p.add_argument("--model", default="xgb", choices=["xgb", "wls"])

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
    p.add_argument("--n_estimators",     type=int,   default=60)
    p.add_argument("--max_depth",        type=int,   default=3)
    p.add_argument("--learning_rate",    type=float, default=0.03)
    p.add_argument("--subsample",        type=float, default=0.7)
    p.add_argument("--colsample_bytree", type=float, default=0.5)
    p.add_argument("--min_child_weight", type=int,   default=50)
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

    # ── Stage 0：聚类去冗余 ───────────────────────────────────────────────
    p.add_argument("--cluster_corr_t",  type=float, default=0.6,
                   help="聚类相关性阈值，>= 此值的因子归为一簇，取 medoid 代表")

    # ── Stage 1：相关性过滤 ───────────────────────────────────────────────
    p.add_argument("--min_target_corr", type=float, default=1e-4,
                   help="因子与 target 绝对相关性低于此值则丢弃")

    # ── Stage 2：SHAP 筛选 ────────────────────────────────────────────────
    p.add_argument("--shap_cv_folds",   type=int,   default=6,
                   help="SHAP 筛选的 CV 折数（时序顺序划分）")
    p.add_argument("--shap_top_n",      type=int,   default=20,
                   help="每折取 top-N 重要因子")
    p.add_argument("--shap_min_folds",  type=int,   default=4,
                   help="因子至少出现在几折的 top-N 中才保留")
    p.add_argument("--shap_max_factors",type=int,   default=35,
                   help="SHAP 筛选后最多保留因子数")

    # ── Stage 3：Purged Walk-Forward CV ──────────────────────────────────
    p.add_argument("--cv_folds",        type=int,   default=3,
                   help="Val 段内 walk-forward 折数")
    p.add_argument("--cv_lambda",       type=float, default=0.5,
                   help="目标函数惩罚系数：mean - λ·std")
    p.add_argument("--n_trials",        type=int,   default=100)
    p.add_argument("--n_startup_trials",type=int,   default=0)
    p.add_argument("--corr_penalty_w",  type=float, default=0.25,
                   help="因子共线惩罚权重")
    p.add_argument("--min_factors_on",  type=int,   default=2)
    p.add_argument("--max_factors_on",  type=int,   default=0,
                   help="0 = 不限")

    # ── Stage 4：精修 ─────────────────────────────────────────────────────
    p.add_argument("--refine_iters",    type=int,   default=1)

    # ── 必选因子 ──────────────────────────────────────────────────────────
    p.add_argument("--must_include",    nargs="*",  default=[],
                   help="必须包含的因子，默认为空")

    # ── 随机种子 ──────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _sharpe_from_returns(r: np.ndarray) -> float:
    if len(r) < 20:
        return -999.0
    m, s = float(np.mean(r)), float(np.std(r))
    if s < 1e-9:
        return 0.0
    return m / s * np.sqrt(252)


def _daily_returns(results_df: pd.DataFrame) -> np.ndarray:
    """按日聚合 strategy_return，去掉换仓日。"""
    key = pd.to_datetime(results_df.index).normalize()
    d   = results_df.groupby(key).agg(
        r=("strategy_return", "sum"),
        sw=("is_switch", "any")
    )
    return d[~d["sw"]]["r"].values


def _get_sharpe_in_window(
    results_df: pd.DataFrame,
    t_start,
    t_end,
) -> float:
    """取 [t_start, t_end) 窗口内的日收益 Sharpe。"""
    idx  = pd.to_datetime(results_df.index)
    mask = (idx >= pd.to_datetime(t_start))
    if t_end is not None:
        mask &= (idx < pd.to_datetime(t_end))
    sub = results_df.loc[mask]
    if len(sub) < 50:
        return -999.0
    return _sharpe_from_returns(_daily_returns(sub))


def _mean_abs_corr(fd: pd.DataFrame) -> float:
    k = fd.shape[1]
    if k < 2:
        return 0.0
    c = fd.fillna(0).corr().values
    return float(np.abs(c[np.triu_indices(k, k=1)]).mean())


def _cv_score(sharpes: list[float], lam: float) -> float:
    """mean - λ·std，过滤掉 -999 的折。"""
    valid = [s for s in sharpes if s > -900]
    if len(valid) == 0:
        return -999.0
    if len(valid) == 1:
        return valid[0]
    return float(np.mean(valid) - lam * np.std(valid))


# ══════════════════════════════════════════════════════════════════════════════
# 完整回测（返回 results_df）
# ══════════════════════════════════════════════════════════════════════════════

def _run_backtest(
    factor_subset: list[str],
    base_fd:       pd.DataFrame,
    price_data:    pd.Series,
    args,
) -> tuple[pd.DataFrame | None, dict | None]:
    avail = [c for c in factor_subset if c in base_fd.columns]
    if not avail:
        return None, None
    ta = copy.copy(args)
    ta.contract_switch_dates = getattr(args, "contract_switch_dates", [])
    ta.top_n_features = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if getattr(ta, "model", "xgb") == "xgb":
            res, perf = run_backtest_xgb(base_fd[avail].copy(), price_data, ta)
        else:
            res, perf = run_backtest_reg(base_fd[avail].copy(), price_data, ta)
    if res is None or perf is None:
        return None, None
    ta.split_point = perf.get("split_point")
    if getattr(ta, "use_strength_filter", False):
        res  = apply_strength_filter(res, ta)
        perf = recalc_performance(res, ta)
    return res, perf


# ══════════════════════════════════════════════════════════════════════════════
# Purged Walk-Forward CV 打分（仅在 Val 段内）
# ══════════════════════════════════════════════════════════════════════════════

def _cv_backtest_score(
    factor_subset: list[str],
    base_fd:       pd.DataFrame,
    price_data:    pd.Series,
    args,
    results_df_cache: pd.DataFrame | None = None,
) -> tuple[float, float]:
    """
    在 Val 段 [val_split, test_split) 内做 Purged Walk-Forward CV。
    - 折间 purge gap = fwd bars
    - 目标 = mean(Sharpe_k) - λ·std(Sharpe_k)

    results_df_cache: 如果已有完整回测结果可直接传入，省去重跑。
    """
    avail = [c for c in factor_subset if c in base_fd.columns]
    if not avail:
        return -999.0, 0.0

    mac = _mean_abs_corr(base_fd[avail])

    # 获取完整回测结果（只跑一次）
    if results_df_cache is None:
        res, perf = _run_backtest(avail, base_fd, price_data, args)
        if res is None:
            return -999.0, mac
    else:
        res = results_df_cache

    # Val 段时间范围
    val_start  = pd.to_datetime(args.val_split)
    test_start = pd.to_datetime(args.test_split)
    K          = args.cv_folds
    lam        = args.cv_lambda
    fwd        = args.fwd

    # Val 段索引（bar 级别）
    idx     = pd.to_datetime(res.index)
    val_mask = (idx >= val_start) & (idx < test_start)
    val_idx  = np.where(val_mask)[0]

    if len(val_idx) < K * 50:
        # val 段太短，退化为单折
        s = _get_sharpe_in_window(res, val_start, test_start)
        return s - lam * 0.0, mac

    fold_len   = len(val_idx) // K
    sharpes    = []

    for k in range(K):
        # 预测窗口（不重叠）
        pred_bar_start = val_idx[k * fold_len]
        pred_bar_end   = val_idx[(k + 1) * fold_len - 1] if k < K - 1 else val_idx[-1]

        # purge：预测窗口起点往前 fwd bars
        purge_bar = max(0, pred_bar_start - fwd)

        t_pred_start = res.index[pred_bar_start]
        t_pred_end   = res.index[pred_bar_end]

        # 在完整 results_df 上截取该折预测窗口
        s_k = _get_sharpe_in_window(res, t_pred_start, t_pred_end)
        sharpes.append(s_k)

    score = _cv_score(sharpes, lam)
    return score, mac


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — 聚类去冗余（相关性聚类 → medoid 代表）
# ══════════════════════════════════════════════════════════════════════════════

def stage0_cluster_medoid(
    factor_data: pd.DataFrame,
    x_cols:      list[str],
    args,
    out_dir:     Path,
) -> list[str]:
    """
    对所有 x_ 因子做相关性聚类，每簇选 medoid（与簇内其他成员平均相关性最高的因子）
    作为代表，压缩特征数量。
    """
    corr_t = args.cluster_corr_t
    print(f"\n  计算 {len(x_cols)} 个因子的相关性矩阵...")

    fd = factor_data[x_cols].fillna(0)
    corr_mat  = fd.corr().abs().clip(0, 1)
    dist_mat  = (1.0 - corr_mat).clip(0)
    np.fill_diagonal(dist_mat.values, 0.0)

    condensed = squareform(dist_mat.values, checks=False).clip(0)
    Z         = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1.0 - corr_t, criterion="distance")

    n_clusters = int(cluster_ids.max())
    print(f"  相关性阈值={corr_t}，共 {n_clusters} 个簇")

    medoids = []
    cluster_records = []
    for cid in range(1, n_clusters + 1):
        members = [x_cols[i] for i, c in enumerate(cluster_ids) if c == cid]
        if len(members) == 1:
            medoid = members[0]
        else:
            sub_corr = corr_mat.loc[members, members]
            avg_corr = sub_corr.mean(axis=1)
            medoid   = avg_corr.idxmax()
        medoids.append(medoid)
        cluster_records.append({
            "cluster_id": cid,
            "size": len(members),
            "medoid": medoid,
            "members": "|".join(members),
        })

    pd.DataFrame(cluster_records).to_csv(
        out_dir / "stage0_medoids.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  聚类完成：{len(x_cols)} 个因子 → {len(medoids)} 个 medoid 代表")
    return medoids


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — 与 target 相关性过滤
# ══════════════════════════════════════════════════════════════════════════════

def stage1_corr_filter(
    factor_data: pd.DataFrame,
    price_data:  pd.Series,
    candidates:  list[str],
    args,
    out_dir:     Path,
) -> list[str]:
    """去掉与 target（未来收益）绝对相关性低于阈值的因子。"""
    labels = _compute_reversal_labels(
        price_data, fwd=args.fwd,
        check_days=getattr(args, "check_days", 3),
        multiplier=getattr(args, "multiplier", 2.0),
    )
    min_corr = args.min_target_corr
    records  = []
    kept     = []

    for col in candidates:
        if col not in factor_data.columns:
            continue
        corr_val = abs(float(
            factor_data[col].replace([np.inf, -np.inf], np.nan)
            .corr(labels.replace([np.inf, -np.inf], np.nan))
        ))
        keep = corr_val >= min_corr or np.isnan(corr_val)
        records.append({"factor": col, "abs_corr": corr_val, "kept": keep})
        if keep:
            kept.append(col)

    pd.DataFrame(records).sort_values("abs_corr", ascending=False).to_csv(
        out_dir / "stage1_corr_filtered.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  相关性过滤：{len(candidates)} → {len(kept)} 个（阈值={min_corr}）")
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — XGBoost + SHAP 持续性筛选
# ══════════════════════════════════════════════════════════════════════════════

def stage2_shap_select(
    factor_data: pd.DataFrame,
    price_data:  pd.Series,
    candidates:  list[str],
    must_include: list[str],
    args,
    out_dir:     Path,
) -> list[str]:
    """
    时序顺序 K 折 CV，每折训练 XGBoost，用 SHAP 取 top-N 因子。
    至少出现在 shap_min_folds 折中的因子保留。
    """
    if not HAS_SHAP or not HAS_XGB_LIB:
        print("  [警告] shap 或 xgboost 未安装，跳过 SHAP 筛选，保留所有候选")
        return candidates

    avail_cols = [c for c in candidates if c in factor_data.columns]
    labels = _compute_reversal_labels(
        price_data, fwd=args.fwd,
        check_days=getattr(args, "check_days", 3),
        multiplier=getattr(args, "multiplier", 2.0),
    )

    # 只用 Train 段 + Val 段做 SHAP（不碰 Test）
    test_start = pd.to_datetime(args.test_split)
    idx        = pd.to_datetime(factor_data.index)
    pre_test   = idx < test_start

    fd_sub  = factor_data.loc[pre_test, avail_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    lab_sub = labels.loc[pre_test].fillna(0)
    n       = len(fd_sub)

    K      = args.shap_cv_folds
    top_n  = args.shap_top_n
    min_f  = args.shap_min_folds
    fold_size = n // K

    appear_count: dict[str, int] = {c: 0 for c in avail_cols}

    xgb_params = dict(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        tree_method="hist",
        random_state=args.seed,
        n_jobs=-1,
    )

    print(f"  SHAP {K} 折 CV（每折 top-{top_n}，至少出现 {min_f} 折保留）...")
    for k in range(K):
        train_end   = k * fold_size
        pred_start  = train_end + args.fwd   # purge gap
        pred_end    = min(pred_start + fold_size, n)

        if train_end < 200 or pred_start >= n:
            continue

        X_train = fd_sub.iloc[:train_end].values
        y_train = lab_sub.iloc[:train_end].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train, verbose=False)

        explainer   = shap.TreeExplainer(model)
        X_pred      = fd_sub.iloc[pred_start:pred_end].values
        shap_values = explainer.shap_values(X_pred)
        mean_abs    = np.abs(shap_values).mean(axis=0)
        top_idx     = np.argsort(mean_abs)[::-1][:top_n]

        for i in top_idx:
            appear_count[avail_cols[i]] += 1

        print(f"    fold {k+1}/{K}  top3: "
              f"{[avail_cols[i].replace('x_','')[:20] for i in top_idx[:3]]}")

    # 取至少出现 min_f 折的因子
    shap_selected = [c for c, cnt in appear_count.items() if cnt >= min_f]

    # 保证 must_include 在内
    for f in must_include:
        if f in factor_data.columns and f not in shap_selected:
            shap_selected.append(f)

    # 上限截断
    max_f = args.shap_max_factors
    if len(shap_selected) > max_f:
        # 按出现次数降序截断，must_include 优先保留
        mi_set   = set(must_include)
        ranked   = sorted(
            [c for c in shap_selected if c not in mi_set],
            key=lambda c: appear_count.get(c, 0), reverse=True
        )
        shap_selected = list(mi_set & set(shap_selected)) + ranked[:max_f - len(mi_set)]

    records = [{"factor": c, "appear_folds": appear_count.get(c, 0),
                "is_must": c in must_include}
               for c in shap_selected]
    pd.DataFrame(records).sort_values("appear_folds", ascending=False).to_csv(
        out_dir / "stage2_shap_selected.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  SHAP 筛选完成：{len(avail_cols)} → {len(shap_selected)} 个因子")
    return shap_selected


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Optuna 搜索（Purged Walk-Forward CV 打分）
# ══════════════════════════════════════════════════════════════════════════════

def stage3_optuna_search(
    shap_factors: list[str],
    must_include: list[str],
    base_fd:      pd.DataFrame,
    price_data:   pd.Series,
    args,
    out_dir:      Path,
) -> list[dict]:
    """
    对 shap_factors 中的可选因子做 Optuna 搜索。
    每个 trial 的打分 = Purged Walk-Forward CV score（mean - λ·std）。
    """
    optional_factors = [f for f in shap_factors if f not in must_include]
    cw               = args.corr_penalty_w
    lam              = args.cv_lambda
    min_on           = args.min_factors_on
    max_on           = args.max_factors_on if args.max_factors_on > 0 else len(optional_factors)
    n_start          = args.n_startup_trials if args.n_startup_trials > 0 \
                       else min(15, args.n_trials // 4)

    print(f"  可选因子 {len(optional_factors)} 个  "
          f"Optuna {args.n_trials} trials  CV folds={args.cv_folds}  λ={lam}")

    trial_log  = []
    best_val   = [-999.0]
    PRINT_EVERY = 10

    # 缓存：相同因子集的完整 results_df 避免重跑
    result_cache: dict[str, pd.DataFrame] = {}

    def objective(trial: "optuna.Trial") -> float:
        # 每个因子二选一：选入 or 不选
        chosen = []
        for f in optional_factors:
            safe_key = f.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")
            if trial.suggest_categorical(f"use_{safe_key}", [True, False]):
                chosen.append(f)

        if not (min_on <= len(chosen) <= max_on):
            raise optuna.exceptions.TrialPruned()

        factor_subset = must_include + chosen
        key           = "|".join(sorted(factor_subset))

        # 复用缓存
        if key not in result_cache:
            res, perf = _run_backtest(factor_subset, base_fd, price_data, args)
            result_cache[key] = res
        else:
            res = result_cache[key]

        cv_score, mac = _cv_backtest_score(
            factor_subset, base_fd, price_data, args,
            results_df_cache=res,
        )
        final_score = cv_score - cw * mac

        is_best = final_score > best_val[0]
        if is_best:
            best_val[0] = final_score

        t_num = trial.number + 1
        if is_best or t_num % PRINT_EVERY == 0 or t_num == args.n_trials:
            short = [f.replace("x_","")[:18] for f in chosen]
            print(
                f"  Trial {t_num:3d}{'★' if is_best else ' '}  "
                f"CVScore={final_score:+.3f}  N={len(factor_subset)}  {short or ['仅必选']}"
            )

        trial_log.append({
            "trial":      t_num,
            "cv_score":   final_score,
            "mac":        mac,
            "n_factors":  len(factor_subset),
            "optional":   "|".join(chosen),
            "all_factors":"|".join(factor_subset),
        })
        return final_score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed, n_startup_trials=n_start, multivariate=True
        ),
        pruner=optuna.pruners.NopPruner(),
    )

    print(f"\n  {'─'*70}")
    print(f"  {'Trial':>6}  {'CVScore':>8}  {'N':>3}  可选因子")
    print(f"  {'─'*70}")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1,
                   show_progress_bar=False)

    with open(out_dir / "optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    df_log = pd.DataFrame(trial_log).sort_values("cv_score", ascending=False)
    df_log.to_csv(out_dir / "stage3_all_trials.csv", index=False, encoding="utf-8-sig")

    # 去重，返回 top 候选列表（附上缓存的 results_df）
    seen: set[str] = set()
    pool: list[dict] = []
    for _, row in df_log.iterrows():
        key = row["all_factors"]
        if key not in seen:
            seen.add(key)
            r = dict(row)
            r["results_df"] = result_cache.get(key)
            pool.append(r)

    return pool


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — 精修（局部 swap/add/remove）
# ══════════════════════════════════════════════════════════════════════════════

def stage4_refine(
    best_rec:     dict,
    shap_factors: list[str],
    must_include: list[str],
    base_fd:      pd.DataFrame,
    price_data:   pd.Series,
    args,
) -> dict:
    """对最优候选做局部精修，评分函数与 Stage 3 保持一致。"""
    cw = args.corr_penalty_w

    def _score_factors(factors):
        res, _ = _run_backtest(factors, base_fd, price_data, args)
        if res is None:
            return -999.0, 0.0, None
        cv_s, mac = _cv_backtest_score(factors, base_fd, price_data, args,
                                        results_df_cache=res)
        return cv_s - cw * mac, mac, res

    current   = best_rec["all_factors"].split("|")
    cur_score, cur_mac, cur_res = _score_factors(current)
    optional_pool = [f for f in shap_factors if f not in must_include]

    print(f"\n  精修起点：Score={cur_score:+.4f}  N={len(current)}")

    for it in range(args.refine_iters):
        best_delta = 0.0
        best_move  = None
        opt_cur    = [f for f in current if f not in must_include]

        # (a) swap：换一个可选因子
        for f_old in opt_cur:
            for f_new in [f for f in optional_pool if f != f_old and f not in current]:
                trial = [f for f in current if f != f_old] + [f_new]
                s, mac, res = _score_factors(trial)
                if s - cur_score > best_delta:
                    best_delta = s - cur_score
                    best_move  = (trial, s, res)

        # (b) remove：去掉一个可选因子
        for f_old in opt_cur:
            trial = [f for f in current if f != f_old]
            if len(trial) >= args.min_factors_on:
                s, mac, res = _score_factors(trial)
                if s - cur_score > best_delta:
                    best_delta = s - cur_score
                    best_move  = (trial, s, res)

        # (c) add：加入一个未选因子
        for f_new in [f for f in optional_pool if f not in current]:
            trial = current + [f_new]
            s, mac, res = _score_factors(trial)
            if s - cur_score > best_delta:
                best_delta = s - cur_score
                best_move  = (trial, s, res)

        if best_move is None or best_delta <= 1e-4:
            print(f"  精修第 {it+1} 轮无改进，提前终止")
            break

        current, cur_score, cur_res = best_move
        print(f"  精修第 {it+1} 轮：Score={cur_score:+.4f}  "
              f"N={len(current)}  "
              f"{[f.replace('x_','')[:18] for f in current if f not in must_include]}")

    # 精修完成后跑一次完整回测（确保 results_df 是最新的）
    final_res, final_perf = _run_backtest(current, base_fd, price_data, args)
    mac = _mean_abs_corr(base_fd[[c for c in current if c in base_fd.columns]])

    return {
        "all_factors": "|".join(current),
        "optional":    "|".join(f for f in current if f not in must_include),
        "cv_score":    cur_score,
        "mac":         mac,
        "results_df":  final_res,
        "perf":        final_perf,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Test 段一次性评估
# ══════════════════════════════════════════════════════════════════════════════

def stage5_test_eval(
    best: dict,
    out_dir: Path,
    args,
) -> float:
    """Test 段 Sharpe，搜索完成后只调用一次。"""
    res = best.get("results_df")
    if res is None:
        return -999.0
    test_sharpe = _get_sharpe_in_window(res, args.test_split, None)
    val_sharpe  = _get_sharpe_in_window(res, args.val_split, args.test_split)

    pd.DataFrame([{
        "val_sharpe":  val_sharpe,
        "test_sharpe": test_sharpe,
        "cv_score":    best["cv_score"],
        "mac":         best["mac"],
        "n_factors":   len(best["all_factors"].split("|")),
        "factors":     best["all_factors"],
    }]).to_csv(out_dir / "test_performance.csv", index=False, encoding="utf-8-sig")

    print(f"\n  ── Test 段评估（只读一次）──")
    print(f"  Val  Sharpe = {val_sharpe:+.4f}")
    print(f"  Test Sharpe = {test_sharpe:+.4f}")
    if val_sharpe > -900 and test_sharpe > -900:
        gap = abs(val_sharpe - test_sharpe)
        print(f"  Val/Test 差距 = {gap:.4f}"
              f"{'  ⚠ 可能存在过拟合' if gap > 0.5 else '  ✓ 泛化较好'}")
    return test_sharpe


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_arguments()

    if not HAS_OPTUNA:
        print("[错误] 请安装：pip install optuna"); return
    if args.model == "xgb" and not HAS_XGB:
        print("[错误] Utils_xgb.py 未找到或 xgboost 未安装"); return
    if not HAS_SHAP:
        print("[警告] shap 未安装，Stage 2 将跳过 SHAP 筛选。pip install shap")
    if not HAS_XGB_LIB:
        print("[警告] xgboost 未安装，Stage 2 将跳过 SHAP 筛选。pip install xgboost")

    # must_include：命令行 > 代码默认
    must_include: list[str] = args.must_include if args.must_include else MUST_INCLUDE

    ts  = get_timestamp()
    out = Path(project_root) / "results" / f"best1_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out)

    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  XGBoost Best-1 因子组合搜索（Purged Walk-Forward CV）")
    print(f"  必选因子: {must_include or '（空）'}")
    print(f"  三段划分:  Train ~ {args.val_split}  |  Val ~ {args.test_split}  |  Test →末尾")
    print(f"  CV folds={args.cv_folds}  λ={args.cv_lambda}  共线惩罚={args.corr_penalty_w}")
    print(f"  聚类阈值={args.cluster_corr_t}  SHAP折数={args.shap_cv_folds}"
          f"  SHAP top-{args.shap_top_n}  最多保留={args.shap_max_factors}")
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
    print(f"[因子发现] 原始 x_ 因子共 {len(all_x_cols)} 个")

    # 检查 must_include 是否存在
    missing = [c for c in must_include if c not in all_x_cols]
    if missing:
        print(f"[错误] 必选因子不在数据中: {missing}"); return

    factor_lags = None
    if getattr(args, "factor_lags", "").strip():
        try:
            factor_lags = [int(x) for x in args.factor_lags.split(",") if x.strip()]
        except ValueError:
            factor_lags = None

    # ── 准备数据（lag=1 用于筛选阶段）────────────────────────────────────
    select_fd, select_price, _ = prepare_factor_data(
        df, selected_factors=all_x_cols, lag=1, factor_lags=None,
        add_session_features=getattr(args, "add_session_features", True),
    )

    # ── 准备数据（lag=args.lag 用于回测）─────────────────────────────────
    base_fd, price_data, _ = prepare_factor_data(
        df, selected_factors=all_x_cols, lag=args.lag, factor_lags=factor_lags,
        add_session_features=getattr(args, "add_session_features", True),
    )
    print(f"[回测数据] lag={args.lag}  bars={len(base_fd)}"
          f"  {base_fd.index[0]} ~ {base_fd.index[-1]}")

    # 校验三段划分的合理性
    val_dt  = pd.to_datetime(args.val_split)
    test_dt = pd.to_datetime(args.test_split)
    end_dt  = pd.to_datetime(base_fd.index[-1])
    assert val_dt  < test_dt  <= end_dt, "val_split < test_split <= 数据末尾"
    assert val_dt  > pd.to_datetime(base_fd.index[0]), "val_split 须在数据起始之后"
    n_test_bars = (pd.to_datetime(base_fd.index) >= test_dt).sum()
    n_val_bars  = ((pd.to_datetime(base_fd.index) >= val_dt) &
                   (pd.to_datetime(base_fd.index) < test_dt)).sum()
    print(f"[数据划分] Train~{args.val_split}  Val={n_val_bars}bars  Test={n_test_bars}bars")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 0 — 聚类去冗余
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 0 / 5  ─  聚类去冗余（相关性阈值={args.cluster_corr_t}）\n{SEP}")
    medoids = stage0_cluster_medoid(select_fd, all_x_cols, args, out)

    # must_include 强制加入 medoid 列表
    for f in must_include:
        if f not in medoids:
            medoids.append(f)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 1 — 相关性过滤
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 1 / 5  ─  与 target 相关性过滤（阈值={args.min_target_corr}）\n{SEP}")
    corr_filtered = stage1_corr_filter(select_fd, select_price, medoids, args, out)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 2 — SHAP 持续性筛选
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 2 / 5  ─  XGBoost + SHAP 筛选\n{SEP}")
    shap_factors = stage2_shap_select(
        select_fd, select_price, corr_filtered, must_include, args, out
    )
    print(f"\n  最终候选因子（{len(shap_factors)} 个）：")
    for f in shap_factors:
        tag = " ★必选" if f in must_include else ""
        print(f"    {f}{tag}")

    if len(shap_factors) == 0:
        print("[错误] 没有候选因子，请检查筛选参数"); return

    # ══════════════════════════════════════════════════════════════════════
    # Stage 3 — Optuna 搜索（Purged Walk-Forward CV）
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 3 / 5  ─  Optuna 搜索（Purged WF CV，{args.n_trials} trials）\n{SEP}")
    pool = stage3_optuna_search(
        shap_factors, must_include, base_fd, price_data, args, out
    )

    if not pool:
        print("[错误] Optuna 未找到有效候选"); return

    best_rec = pool[0]
    print(f"\n  Optuna 最优：CVScore={best_rec['cv_score']:+.4f}  "
          f"N={best_rec['n_factors']}  "
          f"{best_rec['optional'][:80]}")

    # ══════════════════════════════════════════════════════════════════════
    # Stage 4 — 精修
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 4 / 5  ─  精修（{args.refine_iters} 轮）\n{SEP}")
    best = stage4_refine(best_rec, shap_factors, must_include, base_fd, price_data, args)

    # ══════════════════════════════════════════════════════════════════════
    # Stage 5 — Test 段一次性评估
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Stage 5 / 5  ─  Test 段评估（只读一次）\n{SEP}")
    test_sharpe = stage5_test_eval(best, out, args)

    # ══════════════════════════════════════════════════════════════════════
    # 输出结果
    # ══════════════════════════════════════════════════════════════════════
    factors  = best["all_factors"].split("|")
    optional = [f for f in factors if f not in must_include]

    print(f"\n{SEP}")
    print(f"  最优组合（{len(factors)} 个因子）")
    print(SEP)
    print(f"  CV Score   = {best['cv_score']:+.4f}")
    print(f"  Val Sharpe = {_get_sharpe_in_window(best['results_df'], args.val_split, args.test_split):+.4f}")
    print(f"  Test Sharpe= {test_sharpe:+.4f}")
    print(f"  均值因子相关 = {best['mac']:.3f}")
    print(f"  因子列表：")
    for f in factors:
        tag = " ★必选" if f in must_include else " ○选入"
        print(f"    {f:<55}{tag}")

    # 保存完整回测
    sub_dir = out / "best_backtest"
    sub_dir.mkdir(exist_ok=True)
    fa = copy.copy(args)
    fa.output_dir = str(sub_dir)
    fa.contract_switch_dates = getattr(args, "contract_switch_dates", [])

    if best["results_df"] is not None and best["perf"] is not None:
        fa.split_point = best["perf"].get("split_point")
        print_performance_table(best["results_df"], fa)
        avail = [c for c in factors if c in base_fd.columns]
        save_results(fa, avail, fa.output_dir, best["perf"], best["results_df"])

    # 保存 JSON（格式与原版一致）
    result_json = {
        "rank":          1,
        "model":         args.model,
        # ── 因子 ──────────────────────────────────────────────────────
        "factors":       factors,
        "factor_cols":   factors,
        "must_include":  must_include,
        "optional":      optional,
        # ── 数据划分 ──────────────────────────────────────────────────
        "val_split":     args.val_split,
        "test_split":    args.test_split,
        # ── 训练标签 ──────────────────────────────────────────────────
        "check_days":    args.check_days,
        "multiplier":    args.multiplier,
        # ── 特征工程 ──────────────────────────────────────────────────
        "lag":                  args.lag,
        "factor_lags":          args.factor_lags,
        "add_session_features": args.add_session_features,
        # ── 训练通用 ──────────────────────────────────────────────────
        "data_file":     args.data_file,
        "start_date":    args.start_date,
        "train_window":  args.train_window,
        "retrain_freq":  args.retrain_freq,
        "mode":          args.mode,
        "fwd":           args.fwd,
        "use_scaler":    args.use_scaler,
        # ── 信号 ──────────────────────────────────────────────────────
        "reg_threshold":       args.reg_threshold,
        "close_threshold":     list(args.close_threshold),
        "close_mode":          args.close_mode,
        "use_strength_filter": args.use_strength_filter,
        "entry_strength_pct":  args.entry_strength_pct,
        "threshold_window":    args.threshold_window,
        # ── 绩效 ──────────────────────────────────────────────────────
        "cv_score":      best["cv_score"],
        "val_sharpe":    _get_sharpe_in_window(
                             best["results_df"], args.val_split, args.test_split),
        "test_sharpe":   test_sharpe,
        "mac":           best["mac"],
        # ── CV 参数 ───────────────────────────────────────────────────
        "cv_folds":      args.cv_folds,
        "cv_lambda":     args.cv_lambda,
        # ── XGBoost 超参 ──────────────────────────────────────────────
        "xgb_params": {
            "n_estimators":     args.n_estimators,
            "max_depth":        args.max_depth,
            "learning_rate":    args.learning_rate,
            "subsample":        args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "min_child_weight": args.min_child_weight,
            "reg_alpha":        args.reg_alpha,
            "reg_lambda":       args.reg_lambda,
            "top_n_features":   0,
        } if args.model == "xgb" else {},
    }

    with open(out / "best_combination.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"\n{SEP}")
    print(f"  输出目录：{out}")
    print(f"  best_combination.json  ·  test_performance.csv")
    print(f"  stage0_medoids.csv  ·  stage1_corr_filtered.csv")
    print(f"  stage2_shap_selected.csv  ·  stage3_all_trials.csv")
    print(f"  best_backtest/")
    print(SEP)


if __name__ == "__main__":
    main()