"""
ensemble_top5.py — 5 组因子组合模型集成
═══════════════════════════════════════════════════════════════════════════
输入
  · top5_combinations.json（由 factor_search_top5.py 生成）
  · 同样的原始数据和回测参数

两种集成方式（结果并排输出）
  ┌──────────────────────────────────────────────────────────────────┐
  │  方法一：信号级加权平均（Signal-Level Ensemble）                 │
  │    · 每个因子组合跑 WLS 得到原始预测值（regression score）      │
  │    · 对各组合的预测值做加权平均后统一阈值判断方向               │
  │    · 权重方案 A：等权（1/N）                                    │
  │    · 权重方案 B：OOS Sharpe 加权（负值截为 0）                  │
  ├──────────────────────────────────────────────────────────────────┤
  │  方法二：位置级投票（Position-Level Vote Ensemble）              │
  │    · 每个组合独立产生 +1/0/-1 仓位决策                          │
  │    · 投票方案 C：多数投票（超半数同向才开仓）                   │
  │    · 投票方案 D：加权投票（Sharpe 加权，超阈值才开仓）          │
  └──────────────────────────────────────────────────────────────────┘

输出
  ensemble_<ts>/
    ensemble_results.csv        四种方案逐日收益对比
    ensemble_summary.csv        各方案 IS/OOS 指标汇总（含单组基准）
    ensemble_signal_corr.csv    各方案策略信号相关矩阵
    method_A_B_C_D/             各方案完整回测目录

用法
  python ensemble_top5.py --top5_json results/top5_xxx/top5_combinations.json
  python ensemble_top5.py --top5_json results/top5_xxx/top5_combinations.json \\
      --vote_threshold 0.5 --min_agree 3
"""

from __future__ import annotations

import sys
import json
import copy
import warnings
import argparse
from pathlib import Path

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
    print_performance_table,
    apply_strength_filter,
    recalc_performance,
    calc_metrics_from_returns,
    plot_ensemble_pnl,
)
from Utils_xgb import run_backtest_xgb

# ══════════════════════════════════════════════════════════════════════════════
MUST_INCLUDE: list[str] = [
    "x_vwap_60min",
    "x_ma_ret_6h",
    "x_ma_afternoon_ret",
]


# ══════════════════════════════════════════════════════════════════════════════
def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--top5_json",
                   default=str(Path(__file__).resolve().parents[1] / "results" / "xgb_top5_20260320_113951" / "top5_combinations.json"),
                   help="factor_search_top5.py 输出的 top5_combinations.json 路径")
    p.add_argument("--data_file",  default=r"G:\bond\data\P_60min_with_ma_features_from_scratch.csv")
    p.add_argument("--start_date", default="2018-04-17")
    p.add_argument("--model_type", default="xgboost", choices=["wls", "xgboost"],
                   help="模型类型：wls（加权最小二乘）或 xgboost")

    # 训练/回测（与原始保持完全一致）
    p.add_argument("--train_window",    type=int,   default=4000)
    p.add_argument("--mode",            default="rolling")
    p.add_argument("--retrain_freq",    type=int,   default=500)
    p.add_argument("--fwd",             type=int,   default=7)
    p.add_argument("--lag",             type=int,   default=2)
    p.add_argument("--factor_lags",     default="")
    p.add_argument("--use_scaler",      action="store_true",  default=True)
    p.add_argument("--no_scaler",       action="store_false", dest="use_scaler")
    p.add_argument("--check_days",      type=int,   default=0)
    p.add_argument("--multiplier",      type=float, default=2)
    p.add_argument("--weight_method",   default="rolling")
    p.add_argument("--rolling_window",  type=int,   default=1000)
    p.add_argument("--reg_threshold",   type=float, default=0.0)
    p.add_argument("--close_threshold", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--close_mode",      default="threshold")
    p.add_argument("--use_strength_filter", action="store_true",  default=True)
    p.add_argument("--no_strength_filter",  action="store_false", dest="use_strength_filter")
    p.add_argument("--entry_strength_pct",  type=float, default=0.7)

    # XGBoost 专用参数（仅 --model_type xgboost 时生效）
    p.add_argument("--top_n_features",    type=int,   default=0,
                   help="XGBoost 特征选择：选前 N 个重要特征，0=全部")
    p.add_argument("--n_estimators",      type=int,   default=60)
    p.add_argument("--max_depth",         type=int,   default=3)
    p.add_argument("--learning_rate",     type=float, default=0.03)
    p.add_argument("--subsample",         type=float, default=0.7)
    p.add_argument("--colsample_bytree",  type=float, default=0.5)
    p.add_argument("--min_child_weight",  type=int,   default=50)
    p.add_argument("--reg_alpha",         type=float, default=1.0)
    p.add_argument("--reg_lambda",        type=float, default=8.0)

    # 集成专用参数
    p.add_argument("--vote_threshold", type=float, default=0.6,
                   help="方法C：投票比例阈值（默认 0.5 = 超半数）")
    p.add_argument("--min_agree",      type=int,   default=0,
                   help="方法C：最少同向票数，0 = 用 vote_threshold 计算")
    p.add_argument("--sharpe_floor",   type=float, default=0.0,
                   help="Sharpe 加权时，低于此值的组合权重截为 0")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════════════════

def _get_is_oos_sharpe(results_df: pd.DataFrame, args) -> tuple[float, float]:
    split = getattr(args, "split_point", None)
    fwd   = getattr(args, "fwd", 1)

    def _sh(df):
        if len(df) < 50:
            return np.nan
        key = pd.to_datetime(df.index).normalize()
        d   = df.groupby(key).agg(r=("strategy_return","sum"), sw=("is_switch","any"))
        r   = d[~d["sw"]]["r"].values
        return float(calc_metrics_from_returns(r)["sharpe_ratio"]) if len(r) >= 20 else np.nan

    if split is None or split not in results_df.index:
        return np.nan, np.nan
    si = results_df.index.get_loc(split)
    return _sh(results_df.iloc[:si]), _sh(results_df.iloc[si + fwd + 2:])


def _calc_metrics(results_df: pd.DataFrame, args) -> dict:
    """计算完整指标字典（IS + OOS + 全样本）"""
    split = getattr(args, "split_point", None)
    fwd   = getattr(args, "fwd", 1)

    def _m(df):
        if len(df) < 30:
            return {}
        key = pd.to_datetime(df.index).normalize()
        d   = df.groupby(key).agg(r=("strategy_return","sum"), sw=("is_switch","any"))
        r   = d[~d["sw"]]["r"].values
        if len(r) < 10:
            return {}
        return calc_metrics_from_returns(r)

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
        "is_oos_gap":  round(float(_get(is_m,"sharpe_ratio") or 0)
                             - float(_get(oos_m,"sharpe_ratio") or 0), 4),
    }


def _run_single(factor_subset, base_fd, price_data, args):
    """运行单组回测，返回 (results_df, perf, ta)"""
    avail = [c for c in factor_subset if c in base_fd.columns]
    ta    = copy.copy(args)
    ta.contract_switch_dates = getattr(args, "contract_switch_dates", [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if getattr(args, "model_type", "wls") == "xgboost":
            res, perf = run_backtest_xgb(base_fd[avail].copy(), price_data, ta)
        else:
            res, perf = run_backtest_reg(base_fd[avail].copy(), price_data, ta)
    if res is None or perf is None:
        return None, None, ta
    ta.split_point = perf.get("split_point")
    if getattr(ta, "use_strength_filter", False):
        res  = apply_strength_filter(res, ta)
        perf = recalc_performance(res, ta)
    return res, perf, ta


def _zscore_series(s: pd.Series, window: int = 500) -> pd.Series:
    """滚动 z-score 标准化信号，便于跨模型平均"""
    mu  = s.rolling(window, min_periods=50).mean()
    std = s.rolling(window, min_periods=50).std().clip(lower=1e-8)
    return (s - mu) / std


def _build_ensemble_results(
    combined_signal: pd.Series,
    base_results_df: pd.DataFrame,
    reg_threshold:   float,
    close_threshold: list,
    close_mode:      str,
) -> pd.DataFrame:
    """
    用合并后的信号替换单模型预测，重新生成 position 和 strategy_return。
    """
    df = base_results_df.copy()

    # 用集成信号替换 prediction
    df["prediction"] = combined_signal.reindex(df.index).fillna(0.0)

    # 重新判断方向
    if reg_threshold > 0:
        df["position"] = np.where(
            df["prediction"] > reg_threshold, 1,
            np.where(df["prediction"] < -reg_threshold, -1, 0)
        )
    else:
        df["position"] = np.sign(df["prediction"]).astype(int)

    # shift(1): T日信号在T+1执行
    df["position"] = df["position"].shift(1).fillna(0).astype(int)

    # 重算 strategy_return
    ret_col = next((c for c in ["price_return", "actual_return", "bar_return", "close_return"]
                     if c in df.columns), None)
    if ret_col is None:
        df["price_return"] = df["price"].pct_change(fill_method=None).fillna(0)
        ret_col = "price_return"
    df["strategy_return"] = df["position"] * df[ret_col]

    # 剔除合约切换日
    if "is_switch" in df.columns:
        df.loc[df["is_switch"], "strategy_return"] = 0.0

    df["cumulative_value"] = 1 + df["strategy_return"].cumsum()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_arguments()
    ts   = get_timestamp()
    out  = Path(project_root) / "results" / f"ensemble_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out)

    SEP = "═" * 76
    print(f"\n{SEP}")
    print(f"  5 组因子组合模型集成  [模型: {args.model_type.upper()}]")
    print(f"  top5_json: {args.top5_json}")
    print(SEP)

    # ── 加载 top5 配置 ────────────────────────────────────────────────────
    with open(args.top5_json, "r", encoding="utf-8") as f:
        top5_cfg = json.load(f)

    print(f"\n  读入 {len(top5_cfg)} 组因子组合：")
    for cfg in top5_cfg:
        oos_s = cfg.get('oos_sharpe')
        is_s  = cfg.get('is_sharpe')
        print(f"    组合{cfg['rank']}  OOS={f'{oos_s:.3f}' if oos_s is not None else '?'}  "
              f"IS={f'{is_s:.3f}' if is_s is not None else '?'}  "
              f"score={cfg.get('score', 0):.3f}  "
              f"可选：{cfg.get('optional', [])}")

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
    factor_lags = None
    if getattr(args, "factor_lags", "").strip():
        try:
            factor_lags = [int(x) for x in args.factor_lags.split(",") if x.strip()]
        except ValueError:
            pass

    # 自动检测 JSON 组合中是否用到了会话特征（x_is_night / x_session_sin）
    all_combo_factors = set()
    for cfg in top5_cfg:
        all_combo_factors.update(cfg["factors"])
    need_session = any(f.startswith("x_is_night") or f.startswith("x_session_sin")
                       for f in all_combo_factors)

    base_fd, price_data, _ = prepare_factor_data(
        df, selected_factors=all_x_cols, lag=args.lag, factor_lags=factor_lags,
        add_session_features=need_session,
    )
    print(f"\n[数据] bars={len(base_fd)}  {base_fd.index[0]} ~ {base_fd.index[-1]}"
          f"  会话特征={'开启' if need_session else '关闭'}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 1：对 5 组因子分别跑完整回测，收集信号序列
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Step 1 / 3  ─  各组独立回测（获取信号序列）\n{SEP}")

    combos: list[dict] = []           # 存每组结果
    split_point = None                # 用第一组确定的 split_point

    for cfg in top5_cfg:
        factors = cfg["factors"]
        avail   = [c for c in factors if c in base_fd.columns]
        rank    = cfg["rank"]

        print(f"\n  ── 组合 {rank}（{len(avail)} 个因子）──")
        print(f"     {avail}")

        res, perf, ta = _run_single(avail, base_fd, price_data, args)
        if res is None:
            print(f"  [警告] 组合 {rank} 回测失败，跳过")
            continue

        if split_point is None:
            split_point = ta.split_point

        # 提取信号列（原始预测分数，未经阈值）
        raw_signal = res["prediction"].fillna(0.0) if "prediction" in res.columns else res["position"].fillna(0.0).astype(float)

        is_s, oos_s = _get_is_oos_sharpe(res, ta)
        print(f"     IS={is_s:.3f}  OOS={oos_s:.3f}  "
              f"split={ta.split_point}")

        combos.append({
            "rank":       rank,
            "factors":    avail,
            "cfg":        cfg,
            "results_df": res,
            "perf":       perf,
            "ta":         ta,
            "raw_signal": raw_signal,     # 原始预测值
            "position":   res["position"].fillna(0).astype(int),
            "is_sharpe":  is_s,
            "oos_sharpe": oos_s,
        })

        # 保存单组回测
        sub = out / f"single_combo_{rank}"
        sub.mkdir(exist_ok=True)
        ta2 = copy.copy(ta)
        ta2.output_dir = str(sub)
        save_results(ta2, avail, str(sub), perf, res)

    if len(combos) < 2:
        print("[错误] 有效组合不足 2 个，无法集成"); return

    args.split_point = split_point
    n = len(combos)

    # ── Sharpe 权重 ───────────────────────────────────────────────────────
    oos_sharpes = np.array([max(c["oos_sharpe"], args.sharpe_floor) for c in combos])
    sharpe_w    = np.where(oos_sharpes > 0, oos_sharpes, 0.0)
    if sharpe_w.sum() > 0:
        sharpe_w = sharpe_w / sharpe_w.sum()
    else:
        sharpe_w = np.ones(n) / n  # fallback 等权

    equal_w = np.ones(n) / n

    print(f"\n  权重汇总：")
    print(f"  {'组合':>4}  {'OOS Sharpe':>10}  {'等权(A/C)':>10}  {'Sharpe权(B/D)':>13}")
    print(f"  {'─'*48}")
    for i, c in enumerate(combos):
        print(f"  {c['rank']:4d}  {c['oos_sharpe']:10.3f}  "
              f"{equal_w[i]:10.4f}  {sharpe_w[i]:13.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2：构建 4 种集成信号
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Step 2 / 3  ─  构建集成信号\n{SEP}")

    # 对齐 index
    common_idx = combos[0]["raw_signal"].index
    for c in combos[1:]:
        common_idx = common_idx.intersection(c["raw_signal"].index)

    # 对每组信号做滚动 z-score 标准化（消除量纲差异）
    zscored: list[pd.Series] = [
        _zscore_series(c["raw_signal"].reindex(common_idx).fillna(0.0))
        for c in combos
    ]
    positions: list[pd.Series] = [
        c["position"].reindex(common_idx).fillna(0).astype(int)
        for c in combos
    ]

    # ── 方法 A：信号等权平均 ─────────────────────────────────────────────
    sig_A = sum(w * z for w, z in zip(equal_w, zscored))
    sig_A.name = "signal_A_equal"
    print("  方法A（信号等权）：z-score 后各组等权平均")

    # ── 方法 B：信号 Sharpe 加权 ─────────────────────────────────────────
    sig_B = sum(w * z for w, z in zip(sharpe_w, zscored))
    sig_B.name = "signal_B_sharpe"
    print("  方法B（信号Sharpe加权）：z-score 后 OOS Sharpe 加权")

    # ── 方法 C：位置多数投票 ─────────────────────────────────────────────
    vote_mat   = pd.concat(positions, axis=1)  # N × K
    min_agree  = (args.min_agree if args.min_agree > 0
                  else max(1, int(np.ceil(n * args.vote_threshold))))
    long_votes  = (vote_mat == 1).sum(axis=1)
    short_votes = (vote_mat == -1).sum(axis=1)
    pos_C = pd.Series(0, index=common_idx)
    pos_C[long_votes  >= min_agree] = 1
    pos_C[short_votes >= min_agree] = -1
    print(f"  方法C（多数投票）：≥{min_agree}/{n} 同向才开仓，"
          f"多头占比={( pos_C==1).mean():.1%}，空头占比={(pos_C==-1).mean():.1%}")

    # ── 方法 D：位置 Sharpe 加权投票 ─────────────────────────────────────
    weighted_vote = sum(w * p for w, p in zip(sharpe_w, positions))
    # 用 0.3 作为最终开仓阈值（相当于 Sharpe 加权后需一定程度共识）
    vote_thr = 0.3
    pos_D = pd.Series(0, index=common_idx, dtype=int)
    pos_D[weighted_vote >  vote_thr] = 1
    pos_D[weighted_vote < -vote_thr] = -1
    print(f"  方法D（Sharpe加权投票）：Sharpe加权合票 > ±{vote_thr} 才开仓，"
          f"多头占比={(pos_D==1).mean():.1%}，空头占比={(pos_D==-1).mean():.1%}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 3：对 4 种方案做完整回测评估
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Step 3 / 3  ─  集成方案回测评估\n{SEP}")

    base_res = combos[0]["results_df"]  # 用第一组的基础 DataFrame 提供 bar_return

    method_configs = [
        ("A", "信号等权平均",      "signal", sig_A,  None,  equal_w),
        ("B", "信号Sharpe加权",    "signal", sig_B,  None,  sharpe_w),
        ("C", f"位置多数投票(≥{min_agree}/{n})", "position", None, pos_C, equal_w),
        ("D", "位置Sharpe加权投票","position", None, pos_D, sharpe_w),
    ]

    all_results:   dict[str, pd.DataFrame] = {}
    summary_rows:  list[dict] = []

    # 先填入各单组基准行
    for c in combos:
        m = _calc_metrics(c["results_df"], c["ta"])
        summary_rows.append({
            "方案":      f"单组{c['rank']}",
            "描述":      f"单独回测 combo{c['rank']}",
            **m
        })

    for mkey, mdesc, mtype, msig, mpos, mw in method_configs:
        print(f"\n  ── 方法 {mkey}：{mdesc} ──")

        sub_dir = out / f"method_{mkey}"
        sub_dir.mkdir(exist_ok=True)

        if mtype == "signal":
            # 用合并信号替换原始预测，重新生成 position + strategy_return
            ens_res = _build_ensemble_results(
                combined_signal = msig,
                base_results_df = base_res,
                reg_threshold   = args.reg_threshold,
                close_threshold = args.close_threshold,
                close_mode      = args.close_mode,
            )
        else:
            # 直接用合并 position
            ens_res = base_res.copy()
            ens_res["position"] = mpos.reindex(ens_res.index).fillna(0).astype(int)
            # 用 price 列重算收益
            ret_col = next((c for c in ["price_return", "actual_return"] if c in ens_res.columns), None)
            if ret_col is None:
                ens_res["price_return"] = ens_res["price"].pct_change(fill_method=None).fillna(0)
                ret_col = "price_return"
            ens_res["strategy_return"] = ens_res["position"] * ens_res[ret_col]
            if "is_switch" in ens_res.columns:
                ens_res.loc[ens_res["is_switch"], "strategy_return"] = 0.0
            ens_res["cumulative_value"] = 1 + ens_res["strategy_return"].cumsum()

        # 集成后不再做 strength_filter（它会用 prediction 列重新生成信号，覆盖集成结果）
        ta0 = copy.copy(combos[0]["ta"])
        ta0.split_point = split_point
        ta0.output_dir  = str(sub_dir)
        perf_ens = {}

        all_results[mkey] = ens_res
        m = _calc_metrics(ens_res, ta0)

        print(f"     IS  Sharpe={m.get('is_sharpe','?'):.3f}  "
              f"Ann={m.get('is_annual','?'):.1%}")
        print(f"     OOS Sharpe={m.get('oos_sharpe','?'):.3f}  "
              f"Ann={m.get('oos_annual','?'):.1%}  "
              f"MaxDD={m.get('oos_maxdd','?'):.1%}")
        print(f"     Gap(IS-OOS)={m.get('is_oos_gap','?'):.3f}")

        summary_rows.append({
            "方案":  f"方法{mkey}",
            "描述":  mdesc,
            **m
        })

        # 保存
        ens_res.to_csv(sub_dir / "results.csv", encoding="utf-8-sig")
        if perf_ens:
            pd.Series(perf_ens).to_csv(
                sub_dir / "performance.csv", encoding="utf-8-sig"
            )

    # ── 汇总表 ────────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    col_order  = [
        "方案","描述",
        "oos_sharpe","oos_annual","oos_maxdd",
        "is_sharpe","is_annual",
        "is_oos_gap","all_sharpe","all_annual","all_maxdd",
    ]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]
    summary_df.to_csv(out / "ensemble_summary.csv",
                      index=False, encoding="utf-8-sig")

    # ── 各方案信号相关矩阵 ────────────────────────────────────────────────
    ret_dict = {}
    for c in combos:
        ret_dict[f"单组{c['rank']}"] = c["results_df"]["strategy_return"].fillna(0)
    for mkey, ens_res in all_results.items():
        ret_dict[f"方法{mkey}"] = ens_res["strategy_return"].fillna(0)
    sig_corr_df = pd.DataFrame(ret_dict).corr()
    sig_corr_df.to_csv(out / "ensemble_signal_corr.csv", encoding="utf-8-sig")

    # ── 并排汇总打印 ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  完整汇总（OOS 指标优先排列）")
    print(f"\n  {'方案':<12}  {'描述':<22}  {'OOS-Sharpe':>10}  {'OOS-Ann':>8}  "
          f"{'OOS-MaxDD':>9}  {'IS-Sharpe':>9}  {'Gap':>7}")
    print(f"  {'─'*88}")

    for _, row in summary_df.sort_values("oos_sharpe", ascending=False).iterrows():
        print(
            f"  {str(row['方案']):<12}  {str(row['描述']):<22}  "
            f"{row.get('oos_sharpe', np.nan):>10.3f}  "
            f"{row.get('oos_annual', np.nan):>8.1%}  "
            f"{row.get('oos_maxdd', np.nan):>9.1%}  "
            f"{row.get('is_sharpe', np.nan):>9.3f}  "
            f"{row.get('is_oos_gap', np.nan):>7.3f}"
        )

    print(f"\n  信号相关矩阵：")
    print(sig_corr_df.round(3).to_string())

    # ── 集成 PnL 对比图 ───────────────────────────────────────────────────
    single_res_dict = {f"单组{c['rank']}": c["results_df"] for c in combos}
    method_labels = {"A": "信号等权", "B": "信号Sharpe加权",
                     "C": "位置多数投票", "D": "位置Sharpe加权投票"}
    ensemble_res_dict = {f"方法{k}({method_labels.get(k, k)})": v
                         for k, v in all_results.items()}
    plot_ensemble_pnl(
        single_results=single_res_dict,
        ensemble_results=ensemble_res_dict,
        split_point=split_point,
        output_dir=str(out),
    )

    print(f"\n{SEP}")
    print(f"  输出目录：{out}")
    print(f"  ensemble_summary.csv  ·  ensemble_signal_corr.csv")
    print(f"  ensemble_pnl_comparison.png")
    print(f"  method_A/ B/ C/ D/    ·  single_combo_1~{len(combos)}/")
    print(SEP)


if __name__ == "__main__":
    main()