"""
factor_search_top5.py — 挑出 5 组相关性低的优质因子组合
═══════════════════════════════════════════════════════════════════════════
目标
  找出 5 组因子组合，满足：
    ① 每组自身因子低共线（簇间选取保证）
    ② 每组样本外 Sharpe 尽量高
    ③ 5 组之间的策略信号相关性尽量低（可用于后续集成）

流程
  Stage 1   残差 IC + Ward 聚类（同 v2）
              ★ 新增：按残差 IC-IR 排序后截断，最多保留 --max_factors 个因子
  Stage 2   跨簇 Optuna，收集 top-30 候选（不只取最优）
  Stage 3   逐步精修每个候选
  Stage 4   贪心多样性筛选
              · 先取 Score 最高的组合作为第 1 组
              · 后续每组：在剩余候选中取与已选各组
                信号相关性最低 且 Score 足够好 的组合
              · "信号相关性" = 各组回测 strategy_return 的 Pearson 相关

输出
  top5_combinations.json     5 组因子 + 各项指标
  top5_signal_corr.csv       5 组策略信号相关矩阵
  backtest_combo_{1..5}/     每组完整回测结果
  stage2_all_trials.csv      全部 trial 记录
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
MUST_INCLUDE: list[str] = [
    "x_vwap_8slot",
    "x_ma_ret_4h",
    # "x_ma_afternoon_ret",
    # 'x_马来RBD棕榈油离岸价',
    # 'x_美国1年通胀预期',
# 'x_印尼RBD棕榈油离岸价'
# 'x_粉矿：62%Fe：澳大利亚产：远期现货均价指数（日）_x',
# 'x_PB粉：即期合约：现货落地利润（日）_x',
# 'x_巴拿马型：运费价格：理查德兹/萨尔达尼亚→青岛港（日）_x',

]

N_COMBOS   = 5      # 目标组合数
POOL_SIZE  = 30     # Stage2/3 候选池大小（缩小以加速）


# ══════════════════════════════════════════════════════════════════════════════
def parse_arguments():
    p = argparse.ArgumentParser()
    # p.add_argument("--data_file",  default=r"G:\pail_oil_cta\data_process\data\output\P_60min_with_ma_features_from_scratch.csv")
    p.add_argument("--data_file",  default=os.path.join(project_root,"data\\P_60min_with_ma_features_from_scratch.csv"))
    # p.add_argument("--data_file",  default=r"G:\pail_oil_cta\data_process\data\output\Palm_oil.csv")
    p.add_argument("--start_date", default="2018-04-17")
    p.add_argument("--train_window",   type=int,   default=4000)
    p.add_argument("--mode",           default="rolling")
    p.add_argument("--retrain_freq",   type=int,   default=1000)
    p.add_argument("--fwd",            type=int,   default=3)
    p.add_argument("--lag",            type=int,   default=2)
    p.add_argument("--factor_lags",    default="")
    p.add_argument("--use_scaler",     action="store_true", default=True)
    p.add_argument("--no_scaler",      action="store_false", dest="use_scaler")
    p.add_argument("--check_days",     type=int,   default=3)
    p.add_argument("--multiplier",     type=float, default=2)
    p.add_argument("--weight_method",  default="rolling")
    p.add_argument("--rolling_window", type=int,   default=1000)
    p.add_argument("--reg_threshold",  type=float, default=0.0)
    p.add_argument("--close_threshold",type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--close_mode",     default="threshold")
    p.add_argument("--use_strength_filter", action="store_true",  default=True)
    p.add_argument("--no_strength_filter",  action="store_false", dest="use_strength_filter")
    p.add_argument("--entry_strength_pct",  type=float, default=0.7)
    # Stage 1
    p.add_argument("--ic_window",       type=int,   default=500)
    p.add_argument("--ic_threshold",    type=float, default=0.02)
    p.add_argument("--n_clusters",      type=int,   default=0)
    p.add_argument("--corr_cluster_t",  type=float, default=0.55)
    p.add_argument("--top_per_cluster", type=int,   default=2)     # ★ 2（原3）
    p.add_argument("--min_cluster_ic",  type=float, default=0.01)
    p.add_argument("--max_factors",     type=int,   default=50,    # ★ 新增：因子上限
                   help="可选因子总数上限（按残差IC-IR截断），0=不限")
    # Stage 2
    p.add_argument("--n_trials",        type=int,   default=60)    # ★ 60（原150）
    p.add_argument("--n_startup_trials",type=int,   default=0)
    p.add_argument("--corr_penalty_w",  type=float, default=0.25)
    p.add_argument("--min_clusters_on", type=int,   default=1)
    p.add_argument("--max_clusters_on", type=int,   default=0)
    # Stage 3
    p.add_argument("--refine_iters",    type=int,   default=1)     # ★ 1（原3）
    # Stage 4
    p.add_argument("--diversity_w",     type=float, default=0.5,
                   help="多样性权重：0=纯 Score 排名，1=纯多样性")
    p.add_argument("--min_oos_sharpe",  type=float, default=-0.5,
                   help="候选组合 OOS Sharpe 最低门槛（过滤垃圾组合）")
    p.add_argument("--oos_recent_days", type=int, default=252,
                   help="OOS Sharpe 只取最近 N 个交易日（默认252≈1年），0=全部OOS")
    p.add_argument("--resume",    action="store_true", default=False)
    p.add_argument("--study_path",default="")
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


def _score(oos: float, mac: float, pw: float) -> float:
    return -999.0 if oos <= -998.0 else oos - pw * mac


def _get_oos_sharpe(results_df: pd.DataFrame, args) -> float:
    """近一年 OOS 夏普（优化目标）。

    只取 OOS 区间中最近 252 个交易日的收益来计算 Sharpe，
    让优化器聚焦于近期表现而非历史均值。
    """
    split = getattr(args, "split_point", None)
    fwd   = getattr(args, "fwd", 1)
    if split is None or split not in results_df.index:
        return -999.0
    si     = results_df.index.get_loc(split)
    out_df = results_df.iloc[si + fwd + 2:]
    if len(out_df) < 50:
        return -999.0
    key = pd.to_datetime(out_df.index).normalize()
    d   = out_df.groupby(key).agg(r=("strategy_return","sum"), sw=("is_switch","any"))
    d   = d[~d["sw"]]
    # ★ 只取最近 252 个交易日（约一年）
    recent_days = getattr(args, "oos_recent_days", 252)
    if recent_days > 0 and len(d) > recent_days:
        d = d.iloc[-recent_days:]
    r = d["r"].values
    return float(calc_metrics_from_returns(r)["sharpe_ratio"]) if len(r) >= 20 else -999.0


def _backtest_full(
    factor_subset: list[str],
    base_fd:       pd.DataFrame,
    price_data:    pd.Series,
    args,
) -> tuple[float, float, pd.DataFrame | None, dict | None]:
    """完整回测，返回 (oos_sharpe, mean_abs_corr, results_df, performance)"""
    avail = [c for c in factor_subset if c in base_fd.columns]
    if not avail:
        return -999.0, 0.0, None, None

    mac = _mean_abs_corr(base_fd[avail])
    ta  = copy.copy(args)
    ta.contract_switch_dates = getattr(args, "contract_switch_dates", [])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, perf = run_backtest_reg(base_fd[avail].copy(), price_data, ta)

    if res is None or perf is None:
        return -999.0, mac, None, None

    ta.split_point = perf.get("split_point")
    if getattr(ta, "use_strength_filter", False):
        res  = apply_strength_filter(res, ta)
        perf = recalc_performance(res, ta)

    return _get_oos_sharpe(res, ta), mac, res, perf


def _backtest_quick(
    factor_subset: list[str],
    base_fd:       pd.DataFrame,
    price_data:    pd.Series,
    args,
) -> tuple[float, float]:
    """快速回测（只返回 oos_sharpe 和 mac）"""
    oos, mac, _, _ = _backtest_full(factor_subset, base_fd, price_data, args)
    return oos, mac


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — 残差 IC + Ward 聚类  ★ 新增因子上限截断
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
    opt_cols = [c for c in factor_data.columns if c not in MUST_INCLUDE]
    must_df  = factor_data[[c for c in MUST_INCLUDE if c in factor_data.columns]]
    M        = args.top_per_cluster
    max_f    = getattr(args, "max_factors", 20)   # ★
    total    = len(opt_cols)

    print(f"  计算 {total} 个可选因子残差 IC（窗口={window}）...")
    records   = {}
    ic_series = {}
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

    # ★ 按残差 IC-IR 排序后截断，最多保留 max_factors 个可选因子
    ranking_all = pd.DataFrame(list(records.values())).sort_values(
        "residual_ic_ir", ascending=False
    ).reset_index(drop=True)

    if max_f > 0 and len(opt_cols) > max_f:
        keep_cols = ranking_all.head(max_f)["factor"].tolist()
        dropped   = len(opt_cols) - max_f
        print(f"\n  ★ 因子上限={max_f}，丢弃末尾 {dropped} 个低 IC-IR 因子")
        opt_cols  = keep_cols
        ic_series = {c: ic_series[c] for c in keep_cols}

    # 聚类（仅对截断后因子）
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
    cluster_info: list[dict] = []

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
# Stage 2 — 跨簇 Optuna，收集 top-POOL_SIZE 候选
# ══════════════════════════════════════════════════════════════════════════════

def stage2_collect_pool(
    cluster_candidates: list[list[str]],
    base_fd:            pd.DataFrame,
    price_data:         pd.Series,
    args,
    out_dir:            Path,
) -> list[dict]:
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

    trial_log  = []
    best_val   = [-999.0]

    def objective(trial: "optuna.Trial") -> float:
        chosen = []
        for ci, cl in enumerate(cluster_candidates):
            pick = trial.suggest_categorical(f"cluster_{ci}", ["none"] + cl)
            if pick != "none":
                chosen.append(pick)

        if not (min_on <= len(chosen) <= max_on):
            raise optuna.exceptions.TrialPruned()

        factor_subset = MUST_INCLUDE + chosen
        oos_s, mac    = _backtest_quick(factor_subset, base_fd, price_data, args)
        score         = _score(oos_s, mac, cw)

        is_best = score > best_val[0]
        if is_best:
            best_val[0] = score

        short = lambda f: f.replace("x_","").replace("_60min","").replace("_240min","")
        print(
            f"  Trial {trial.number+1:3d}{'★' if is_best else ' '}  "
            f"Score={score:+.3f}  OOS={oos_s:+.3f}  Corr={mac:.2f}  "
            f"N={len(factor_subset)}  {[short(f) for f in chosen] or ['无']}"
        )
        trial_log.append({
            "trial":        trial.number + 1,
            "score":        score,
            "oos_sharpe":   oos_s,
            "mean_abs_corr":mac,
            "n_factors":    len(factor_subset),
            "optional":     "|".join(chosen),
            "all_factors":  "|".join(factor_subset),
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
                seed=42, n_startup_trials=n_start, multivariate=True),
            pruner=optuna.pruners.NopPruner(),
        )

    print(f"\n  {'─'*72}")
    print(f"  {'Trial':>6}  {'Score':>7}  {'OOS':>7}  {'Corr':>5}  {'N':>3}  可选因子")
    print(f"  {'─'*72}")
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

    return pool


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — 逐步精修每个候选
# ══════════════════════════════════════════════════════════════════════════════

def stage3_refine_one(
    init_factors:       list[str],
    cluster_candidates: list[list[str]],
    base_fd:            pd.DataFrame,
    price_data:         pd.Series,
    args,
    label:              str = "",
) -> tuple[list[str], float, float]:
    cw = args.corr_penalty_w
    factor_to_cluster: dict[str, list[str]] = {
        f: cl for cl in cluster_candidates for f in cl
    }
    current  = list(init_factors)
    oos_s, mac = _backtest_quick(current, base_fd, price_data, args)
    cur_score  = _score(oos_s, mac, cw)

    for it in range(args.refine_iters):
        best_delta = 0.0
        best_move  = None
        optional_cur = [f for f in current if f not in MUST_INCLUDE]
        covered = set(optional_cur)

        for f_old in optional_cur:
            cl = factor_to_cluster.get(f_old, [f_old])
            for f_new in [f for f in cl if f != f_old] + [None]:
                trial = [f for f in current if f != f_old]
                if f_new:
                    trial.append(f_new)
                s_oos, s_mac = _backtest_quick(trial, base_fd, price_data, args)
                s_score = _score(s_oos, s_mac, cw)
                if s_score - cur_score > best_delta:
                    best_delta = s_score - cur_score
                    best_move  = (trial, s_score, s_oos)

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

    return current, cur_score, oos_s


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — 贪心多样性筛选 5 组
# ══════════════════════════════════════════════════════════════════════════════

def stage4_diverse_top5(
    pool:       list[dict],
    base_fd:    pd.DataFrame,
    price_data: pd.Series,
    args,
    out_dir:    Path,
    n_pick:     int = N_COMBOS,
) -> list[dict]:
    min_oos = args.min_oos_sharpe
    dw      = args.diversity_w

    valid_pool = [r for r in pool if r["oos_sharpe"] >= min_oos]
    if len(valid_pool) < n_pick:
        print(f"  [警告] 有效候选不足 {n_pick} 个（共 {len(valid_pool)} 个），"
              f"尝试降低 --min_oos_sharpe")
        valid_pool = pool[:max(n_pick, len(pool))]

    print(f"\n  对 top-{min(len(valid_pool), POOL_SIZE)} 个候选跑完整回测获取信号序列...")

    candidates_full: list[dict] = []
    limit = min(len(valid_pool), POOL_SIZE)

    for i, rec in enumerate(valid_pool[:limit]):
        factors  = rec["all_factors"].split("|")
        avail    = [c for c in factors if c in base_fd.columns]
        oos_s, mac, res_df, perf = _backtest_full(avail, base_fd, price_data, args)

        if res_df is None:
            continue

        candidates_full.append({
            "factors":    avail,
            "optional":   [f for f in avail if f not in MUST_INCLUDE],
            "score":      rec["score"],
            "oos_sharpe": oos_s,
            "mac":        mac,
            "returns":    res_df["strategy_return"].fillna(0.0),
            "perf":       perf,
            "results_df": res_df,
        })
        print(f"  [{i+1:3d}/{limit}] OOS={oos_s:+.3f}  "
              f"opt={[f.replace('x_','')[:20] for f in avail if f not in MUST_INCLUDE]}")

    if not candidates_full:
        print("  [错误] 无有效候选")
        return []

    scores = np.array([c["score"] for c in candidates_full])
    s_min, s_max = scores.min(), scores.max()
    scores_norm  = (scores - s_min) / (s_max - s_min + 1e-9)

    selected_idx: list[int] = []
    signal_corrs: dict[int, pd.Series] = {
        i: c["returns"] for i, c in enumerate(candidates_full)
    }

    for pick_n in range(n_pick):
        if pick_n == 0:
            best_i = int(np.argmax(scores_norm))
        else:
            best_composite = -np.inf
            best_i = -1
            already = [signal_corrs[j] for j in selected_idx]

            for i, c in enumerate(candidates_full):
                if i in selected_idx:
                    continue
                max_corr = max(
                    abs(float(signal_corrs[i]
                              .corr(already_s.reindex(signal_corrs[i].index).fillna(0.0))))
                    for already_s in already
                )
                composite = (1 - dw) * scores_norm[i] + dw * (1 - max_corr)
                if composite > best_composite:
                    best_composite = composite
                    best_i = i

        selected_idx.append(best_i)
        c = candidates_full[best_i]
        print(f"  ✓ 第 {pick_n+1} 组  OOS={c['oos_sharpe']:+.3f}  "
              f"Score={c['score']:+.3f}  "
              f"opt={[f.replace('x_','')[:20] for f in c['optional']]}")

    selected = [candidates_full[i] for i in selected_idx]

    ret_dict = {
        f"组合{i+1}（{','.join(c['optional'][:2])+'...' if len(c['optional'])>2 else ','.join(c['optional'])}）":
        c["returns"]
        for i, c in enumerate(selected)
    }
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
        print("[错误] pip install optuna scipy scikit-learn"); return

    ts  = get_timestamp()
    out = Path(project_root) / "results" / f"top5_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out)

    SEP = "═" * 72
    print(f"\n{SEP}")
    print(f"  Top-5 多样性因子组合选择")
    print(f"  必选: {MUST_INCLUDE}")
    print(f"  ★ 可选因子上限: {args.max_factors}  每簇候选: {args.top_per_cluster}")
    print(f"  Stage2: {args.n_trials} trials  penalty={args.corr_penalty_w}")
    print(f"  Stage3: refine_iters={args.refine_iters}")
    print(f"  Stage4: diversity_w={args.diversity_w}  min_oos={args.min_oos_sharpe}")
    print(f"  ★ OOS优化目标: 近{args.oos_recent_days}交易日夏普" if args.oos_recent_days > 0
          else f"  ★ OOS优化目标: 全部OOS夏普")
    print(SEP)

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
    missing = [c for c in MUST_INCLUDE if c not in all_x_cols]
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
    print(f"[数据] bars={len(base_fd)}  {base_fd.index[0]} ~ {base_fd.index[-1]}\n")

    # Stage 1
    print(f"{SEP}\n  Stage 1 / 4  ─  残差 IC + Ward 聚类（因子上限={args.max_factors}）\n{SEP}")
    cluster_candidates, ranking_df, cluster_info_df = stage1_cluster(
        base_fd, price_data, args
    )
    ranking_df.to_csv(out / "stage1_ranking.csv", index=True, encoding="utf-8-sig")
    cluster_info_df.to_csv(out / "stage1_clusters.csv", index=False, encoding="utf-8-sig")

    # Stage 2
    print(f"\n{SEP}\n  Stage 2 / 4  ─  跨簇 Optuna（收集候选池）\n{SEP}")
    pool = stage2_collect_pool(cluster_candidates, base_fd, price_data, args, out)
    print(f"\n  候选池（去重后）：{len(pool)} 个组合")

    # Stage 3
    print(f"\n{SEP}\n  Stage 3 / 4  ─  逐步精修（top {min(POOL_SIZE, len(pool))} 个候选）\n{SEP}")
    refined_pool: list[dict] = []
    for i, rec in enumerate(pool[:POOL_SIZE]):
        factors = rec["all_factors"].split("|")
        print(f"  精修第 {i+1:3d}/{min(POOL_SIZE,len(pool))} 个：{factors}")
        rf, rs, ro = stage3_refine_one(
            factors, cluster_candidates, base_fd, price_data, args, label=str(i+1)
        )
        refined_pool.append({
            "all_factors": "|".join(rf),
            "oos_sharpe":  ro,
            "score":       rs,
            "optional":    "|".join(f for f in rf if f not in MUST_INCLUDE),
        })

    seen: set[str] = set()
    refined_pool_dedup: list[dict] = []
    for r in sorted(refined_pool, key=lambda x: x["score"], reverse=True):
        if r["all_factors"] not in seen:
            seen.add(r["all_factors"])
            refined_pool_dedup.append(r)

    # Stage 4
    print(f"\n{SEP}\n  Stage 4 / 4  ─  贪心多样性筛选 {N_COMBOS} 组\n{SEP}")
    top5 = stage4_diverse_top5(
        refined_pool_dedup, base_fd, price_data, args, out, n_pick=N_COMBOS
    )

    # 输出结果
    print(f"\n{SEP}")
    print(f"  最终 {len(top5)} 组因子组合")
    print(SEP)

    result_json = []
    for i, combo in enumerate(top5):
        print(f"\n  ── 第 {i+1} 组 ──────────────────────────────────────────")
        print(f"  OOS Sharpe = {combo['oos_sharpe']:+.4f}  "
              f"Score = {combo['score']:+.4f}  "
              f"平均因子相关 = {combo['mac']:.3f}")
        print(f"  因子（{len(combo['factors'])} 个）：")
        for f in combo["factors"]:
            tag = " ★必选" if f in MUST_INCLUDE else " ○选入"
            print(f"    {f:<50}{tag}")

        sub_dir = out / f"backtest_combo_{i+1}"
        sub_dir.mkdir(exist_ok=True)
        fa = copy.copy(args)
        fa.contract_switch_dates = getattr(args, "contract_switch_dates", [])
        fa.output_dir = str(sub_dir)
        avail = [c for c in combo["factors"] if c in base_fd.columns]
        res_df = combo.get("results_df")
        perf   = combo.get("perf")
        if res_df is not None and perf is not None:
            fa.split_point = perf.get("split_point")
            print_performance_table(res_df, fa)
            save_results(fa, avail, fa.output_dir, perf, res_df)
        else:
            _, _, res_df2, perf2 = _backtest_full(avail, base_fd, price_data, fa)
            if res_df2 is not None:
                fa.split_point = perf2.get("split_point")
                print_performance_table(res_df2, fa)
                save_results(fa, avail, fa.output_dir, perf2, res_df2)

        result_json.append({
            "rank":         i + 1,
            "oos_sharpe":   combo["oos_sharpe"],
            "score":        combo["score"],
            "mean_abs_corr":combo["mac"],
            "factors":      combo["factors"],
            "must_include": MUST_INCLUDE,
            "optional":     combo["optional"],
        })

    with open(out / "top5_combinations.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"\n{SEP}")
    print(f"  输出目录：{out}")
    print(f"  top5_combinations.json  ·  top5_signal_corr.csv")
    print(f"  stage1_ranking.csv  ·  stage1_clusters.csv")
    print(f"  stage2_all_trials.csv")
    print(f"  backtest_combo_1/ ~ backtest_combo_{len(top5)}/")
    print(SEP)


if __name__ == "__main__":
    main()