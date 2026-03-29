"""
main_xgboost.py — XGBoost 单次回测
═══════════════════════════════════════════════════════════════════════════
用法
  python backtest/main_xgboost.py
  python backtest/main_xgboost.py --json path/to/backtest_params.json
"""

import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os

script_path  = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(script_path.parent))

from utils2 import (
    load_palm_oil_data,
    prepare_factor_data,
    save_results,
    create_output_directories,
    get_timestamp,
    print_performance_table,
    apply_strength_filter,
    recalc_performance,
)
from Utils_xgb import run_backtest_xgb

# ══════════════════════════════════════════════════════════════════════════════
# 默认因子（与 backtest_params.json 一致）
# ══════════════════════════════════════════════════════════════════════════════

FACTORS = [
    "x_vwap_60min",
    "x_ma_ret_4h",
    "x_obv_60min",
    "x_intraday_return_240min",
    "x_chaikin_osc_60min",
    "x_ad_line_240min",
    "x_overnight_return_60min",
    "x_absorption_60min",
    "x_price_jump_60min",
    "x_cvd_60min",
    "x_session_sin",
    "x_price_delta_volume_60min",
    # "x_vpin_zscore_filtered_60min",
    "x_rel_vol_240min",
    "x_volume_profile_60min",
]


# ══════════════════════════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    p = argparse.ArgumentParser(description="XGBoost 单次回测")

    p.add_argument("--json", type=str, default="",
                   help="从 backtest_params.json 加载参数（覆盖默认值）")

    p.add_argument("--data_file",  default=os.path.join(project_root, "data", "P_60min_with_ma_features_from_scratch.csv"))
    p.add_argument("--start_date", default="2018-04-17")

    # 训练
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
    p.add_argument("--add_session_features", action="store_true",  default=True)
    p.add_argument("--no_session_features",  action="store_false", dest="add_session_features")

    # XGBoost
    p.add_argument("--top_n_features",   type=int,   default=0)
    p.add_argument("--n_estimators",     type=int,   default=300)
    p.add_argument("--max_depth",        type=int,   default=3)
    p.add_argument("--learning_rate",    type=float, default=0.03)
    p.add_argument("--subsample",        type=float, default=0.7)
    p.add_argument("--colsample_bytree", type=float, default=0.5)
    p.add_argument("--min_child_weight", type=int,   default=20)
    p.add_argument("--reg_alpha",        type=float, default=1.0)
    p.add_argument("--reg_lambda",       type=float, default=8.0)

    # 信号
    p.add_argument("--reg_threshold",   type=float, default=0.0)
    p.add_argument("--close_threshold", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--close_mode",      default="threshold",
                   choices=["fixed", "threshold", "hybrid",
                            "zscore_threshold", "zscore_hybrid"])

    # 强度过滤
    p.add_argument("--use_strength_filter", action="store_true",  default=True)
    p.add_argument("--no_strength_filter",  action="store_false", dest="use_strength_filter")
    p.add_argument("--entry_strength_pct",  type=float, default=0.7)
    p.add_argument("--threshold_window",    type=int,   default=50)

    args = p.parse_args()

    # 从 JSON 覆盖参数
    if args.json and Path(args.json).exists():
        with open(args.json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if k in ("output_dir", "split_point", "factor_cols",
                     "model", "weight_method", "rolling_window"):
                continue
            if hasattr(args, k):
                if k == "close_threshold" and isinstance(v, str):
                    import ast
                    v = ast.literal_eval(v)
                setattr(args, k, v)
        if "factor_cols" in cfg:
            args._factor_cols_from_json = cfg["factor_cols"]
        print(f"[JSON] 已加载参数: {args.json}")

    return args


# ══════════════════════════════════════════════════════════════════════════════
# 辅助
# ══════════════════════════════════════════════════════════════════════════════

def get_contract_switch_dates(df: pd.DataFrame):
    if "dominant_id" not in df.columns:
        return []
    mask = df["dominant_id"] != df["dominant_id"].shift(1)
    mask.iloc[0] = False
    return df.index[mask].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_arguments()
    args.model = "xgb"

    sep = "─" * 56
    print(f"\n{sep}")
    print(f"  XGBoost 回测  trees={args.n_estimators}  depth={args.max_depth}  "
          f"lr={args.learning_rate}  top_n={args.top_n_features}")
    print(f"  fwd={args.fwd}  lag={args.lag}  check_days={args.check_days}  "
          f"multiplier={args.multiplier}")
    print(sep)

    df = load_palm_oil_data(args.data_file)
    if args.start_date:
        df = df[df.index >= pd.to_datetime(args.start_date)]

    args.contract_switch_dates = get_contract_switch_dates(df)

    factor_lags = None
    if getattr(args, "factor_lags", "").strip():
        try:
            factor_lags = [int(x) for x in args.factor_lags.split(",") if x.strip()]
        except ValueError:
            factor_lags = None

    factors = getattr(args, "_factor_cols_from_json", FACTORS)
    factor_data, price_data, factor_cols = prepare_factor_data(
        df, selected_factors=factors, lag=args.lag, factor_lags=factor_lags,
        add_session_features=args.add_session_features,
    )
    print(f"[因子] {len(factor_cols)} 个: {factor_cols[:5]}{'...' if len(factor_cols) > 5 else ''}")

    args.output_dir = create_output_directories(project_root, get_timestamp(), "backtest_xgb")

    results_df, performance = run_backtest_xgb(factor_data, price_data, args)
    if performance is None:
        print("回测失败"); return

    args.split_point = performance.get("split_point")

    if getattr(args, "use_strength_filter", False):
        results_df  = apply_strength_filter(results_df, args)
        performance = recalc_performance(results_df, args)

    print_performance_table(results_df, args)
    save_results(args, factor_cols, args.output_dir, performance, results_df)


if __name__ == "__main__":
    main()
