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
    p.add_argument("--vote_threshold", type=float, default=0.6,
                   help="方法C：投票比例阈值")
    p.add_argument("--min_agree",      type=int,   default=0,
                   help="方法C：最少同向票数，0 = 用 vote_threshold 计算")
    p.add_argument("--sharpe_floor",   type=float, default=0.0,
                   help="Sharpe 加权时，低于此值的组合权重截为 0")
    p.add_argument("--use_delta",      action="store_true", default=False,
                   help="为每个因子追加 delta 特征（当期 - 上一期），默认不启用")
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
# 工具
# ══════════════════════════════════════════════════════════════════════════════

def _load_combo_config(json_path: str) -> SimpleNamespace:
    """读取 backtest_params.json，返回 SimpleNamespace（兼容 args 接口）"""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # close_threshold 存的是字符串 "[0.0, 0.0]"，需要解析
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
        use_strength_filter = bool(raw.get("use_strength_filter", True)),
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
    if getattr(ta, "use_strength_filter", False):
        res  = apply_strength_filter(res, ta)
        perf = recalc_performance(res, ta)
    return res, perf, ta


def _zscore_series(s: pd.Series, window: int = 500) -> pd.Series:
    mu  = s.rolling(window, min_periods=50).mean()
    std = s.rolling(window, min_periods=50).std().clip(lower=1e-8)
    return (s - mu) / std


def _build_ensemble_results(
    combined_signal: pd.Series,
    base_results_df: pd.DataFrame,
    reg_threshold:   float,
) -> pd.DataFrame:
    df = base_results_df.copy()
    df["prediction"] = combined_signal.reindex(df.index).fillna(0.0)
    if reg_threshold > 0:
        df["position"] = np.where(
            df["prediction"] > reg_threshold, 1,
            np.where(df["prediction"] < -reg_threshold, -1, 0)
        )
    else:
        df["position"] = np.sign(df["prediction"]).astype(int)
    df["position"] = df["position"].shift(1).fillna(0).astype(int)

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


# ── 新增：线性 Stacking ────────────────────────────────────────────────────

def _build_stacking_signal(
    zscored: list[pd.Series],
    base_res: pd.DataFrame,
    split_point,
    fwd: int,
    common_idx: pd.Index,
) -> pd.Series:
    """
    IS 段训练 Ridge 回归，OOS 段应用，返回合并信号。

    特征：各子模型 z-score 信号（对齐到 common_idx）
    标签：未来 fwd 根 bar 的累计收益率
    注意：IS 段输出为模型拟合值（有 in-sample bias），OOS 段为真实外推。
    """
    from sklearn.linear_model import Ridge

    # 特征矩阵
    feat_df = pd.DataFrame(
        {f"s{i}": z.reindex(common_idx).fillna(0.0) for i, z in enumerate(zscored)},
        index=common_idx,
    )

    # 标签：未来 fwd 根 bar 累计收益率
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
        .rolling(fwd)          # 滚动求和（每根 bar 的收益先做滚动窗口）
        .sum()
        .shift(-fwd)           # 向前移 fwd 步，使每行对应"未来 fwd 根的累计收益"
    )

    # IS / OOS 切分位置
    if split_point is not None and split_point in common_idx:
        si = common_idx.get_loc(split_point)
        print(f"     Stacking IS 样本: {si} bars  OOS 样本: {len(common_idx) - si} bars")
    else:
        si = int(len(common_idx) * 0.7)
        print(f"  [Stacking] split_point 未找到，默认用前 70% ({si} bars) 作 IS")

    # IS 段有效行（排除 fwd_ret 末尾 NaN）
    valid_mask = ~fwd_ret.iloc[:si].isna()
    X_is = feat_df.iloc[:si][valid_mask].values
    y_is = fwd_ret.iloc[:si][valid_mask].values

    if len(y_is) < 30:
        print("  [Stacking] IS 有效样本不足 30，退化为等权平均")
        stk = sum(z.reindex(common_idx).fillna(0.0) for z in zscored) / len(zscored)
        stk.name = "signal_stack"
        return stk

    # 训练 Ridge（无截距，信号已经 z-score 对称）
    mdl = Ridge(alpha=0.00001, fit_intercept=False)
    mdl.fit(X_is, y_is)

    coef_str = "  ".join(f"s{i}={v:+.4f}" for i, v in enumerate(mdl.coef_))
    print(f"     Ridge 系数: {coef_str}")

    # 全段预测并返回
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
    print(SEP)

    # ── 加载各组配置 ──────────────────────────────────────────────────────
    combo_configs: list[SimpleNamespace] = []
    for jp in args.jsons:
        cfg = _load_combo_config(jp)
        combo_configs.append(cfg)
        print(f"\n  [{len(combo_configs)}] {Path(jp).parent.name}")
        print(f"      模型: {cfg.model_type.upper()}  因子数: {len(cfg.factor_cols)}")
        print(f"      tw={cfg.train_window}  fwd={cfg.fwd}  freq={cfg.retrain_freq}")

    # ── 加载原始数据（取第一个 JSON 的 data_file / start_date）──────────
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
    # Step 1：各组独立回测（每组用自己的 lag 独立 prepare_factor_data）
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Step 1 / 3  ─  各组独立回测\n{SEP}")

    combos: list[dict] = []
    split_point = None

    for i, cfg in enumerate(combo_configs):
        rank = i + 1
        cfg.contract_switch_dates = switch_dates

        # ── 每组独立 prepare_factor_data，用各自的 lag ──────────────────
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

        # ── delta 特征（当期 - 上一期）────────────────────────────────
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
        print(f"     IS={is_s:.3f}  OOS={oos_s:.3f}")

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

        # 保存单组
        sub = out / f"single_{rank}_{cfg.model_type}"
        sub.mkdir(exist_ok=True)
        ta2 = copy.copy(ta)
        ta2.output_dir = str(sub)
        save_results(ta2, cfg.factor_cols, str(sub), perf, res)

    if len(combos) < 2:
        print("[错误] 有效组合不足 2 个，无法集成"); return

    n = len(combos)

    # ── 各组权重一览（等权，供参考）────────────────────────────────────────
    equal_w = np.ones(n) / n

    print(f"\n  各组摘要：")
    print(f"  {'#':>3}  {'模型':>6}  {'OOS Sharpe':>10}  {'等权':>8}")
    print(f"  {'─'*34}")
    for i, c in enumerate(combos):
        print(f"  {c['rank']:3d}  {c['model_type']:>6}  {c['oos_sharpe']:10.3f}  {equal_w[i]:8.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 2：构建 3 种集成信号
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

    # 方法 C：位置多数投票
    vote_mat   = pd.concat(positions, axis=1)
    min_agree  = (args.min_agree if args.min_agree > 0
                  else max(1, int(np.ceil(n * args.vote_threshold))))
    long_votes  = (vote_mat == 1).sum(axis=1)
    short_votes = (vote_mat == -1).sum(axis=1)
    pos_C = pd.Series(0, index=common_idx)
    pos_C[long_votes  >= min_agree] = 1
    pos_C[short_votes >= min_agree] = -1
    print(f"  方法C（多数投票）：≥{min_agree}/{n} 同向才开仓，"
          f"多头={(pos_C==1).mean():.1%}，空头={(pos_C==-1).mean():.1%}")

    # 方法 S：线性 Stacking（IS 训练 Ridge，OOS 外推）
    print("  方法S（线性Stacking）：IS段训练Ridge，OOS段应用，标签=未来fwd日收益")
    base_res = combos[0]["results_df"]
    fwd_val  = combos[0]["ta"].fwd
    sig_S = _build_stacking_signal(
        zscored   = zscored,
        base_res  = base_res,
        split_point = split_point,
        fwd       = fwd_val,
        common_idx = common_idx,
    )

    # ══════════════════════════════════════════════════════════════════════
    # Step 3：集成方案回测评估
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{SEP}\n  Step 3 / 3  ─  集成方案回测评估\n{SEP}")

    method_configs = [
        ("A", "信号等权平均",          "signal",   sig_A, None),
        ("C", f"位置多数投票(≥{min_agree}/{n})", "position", None, pos_C),
        ("S", "线性Stacking(Ridge)",   "signal",   sig_S, None),
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
            ens_res = _build_ensemble_results(msig, base_res, reg_threshold=0.0)
        else:
            ens_res = base_res.copy()
            ens_res["position"] = mpos.reindex(ens_res.index).fillna(0).astype(int)
            ret_col = next((c for c in ["price_return", "actual_return"] if c in ens_res.columns), None)
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

        # 方法 S 的 IS Sharpe 有 in-sample bias，加注说明
        is_note = "（含训练偏差）" if mkey == "S" else ""
        print(f"     IS  Sharpe={m.get('is_sharpe','?'):.3f}{is_note}  Ann={m.get('is_annual','?'):.1%}")
        print(f"     OOS Sharpe={m.get('oos_sharpe','?'):.3f}  Ann={m.get('oos_annual','?'):.1%}  "
              f"MaxDD={m.get('oos_maxdd','?'):.1%}")
        print(f"     Gap(IS-OOS)={m.get('is_oos_gap','?'):.3f}"
              + ("  ← IS 偏高属正常" if mkey == "S" else ""))

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
        ret_dict[f"单组{c['rank']}({c['model_type']})"] = c["results_df"]["strategy_return"].fillna(0)
    for mkey, ens_res in all_results.items():
        ret_dict[f"方法{mkey}"] = ens_res["strategy_return"].fillna(0)
    sig_corr_df = pd.DataFrame(ret_dict).corr()
    sig_corr_df.to_csv(out / "ensemble_signal_corr.csv", encoding="utf-8-sig")

    # ── 汇总打印 ──────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  完整汇总（OOS 指标优先排列）")
    print(f"\n  {'方案':<20}  {'描述':<28}  {'OOS-Sharpe':>10}  {'OOS-Ann':>8}  "
          f"{'OOS-MaxDD':>9}  {'IS-Sharpe':>9}  {'Gap':>7}")
    print(f"  {'─'*102}")

    for _, row in summary_df.sort_values("oos_sharpe", ascending=False).iterrows():
        print(
            f"  {str(row['方案']):<20}  {str(row['描述']):<28}  "
            f"{row.get('oos_sharpe', np.nan):>10.3f}  "
            f"{row.get('oos_annual', np.nan):>8.1%}  "
            f"{row.get('oos_maxdd', np.nan):>9.1%}  "
            f"{row.get('is_sharpe', np.nan):>9.3f}  "
            f"{row.get('is_oos_gap', np.nan):>7.3f}"
        )

    print(f"\n  * 方法S 的 IS Sharpe 含 in-sample bias，请以 OOS Sharpe 为准")

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
        single_results=single_res_dict,
        ensemble_results=ensemble_res_dict,
        split_point=split_point,
        output_dir=str(out),
    )

    print(f"\n{SEP}")
    print(f"  输出目录：{out}")
    print(f"  ensemble_summary.csv  ·  ensemble_signal_corr.csv")
    print(f"  ensemble_pnl_comparison.png")
    print(f"  method_A/ C/ S/  ·  single_1~{len(combos)}/")
    print(SEP)


if __name__ == "__main__":
    main()