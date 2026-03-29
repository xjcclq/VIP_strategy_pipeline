"""
main_min.py — 棕榈油/商品分钟线因子回测
  - WLS 权重：滚动窗口方差（rolling），适合分钟线时序
  - 无 YAML / IC / R² / MSE 依赖
  - 可选信号强度过滤 (--use_strength_filter)
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os

script_path  = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(script_path.parent))

from utils2 import (
    load_palm_oil_data, prepare_factor_data, run_backtest_reg,
    save_results, create_output_directories, get_timestamp,
    print_performance_table, apply_strength_filter, recalc_performance,
)

# ── 因子列表 ──────────────────────────────────────────────────────────────────

FACTORS = [
    "x_chaikin_osc_60min",
    "x_vwap_60min",
    # "x_vpin_zscore_filtered_60min",
    "x_volume_profile_60min",
    "x_ease_of_movement_60min",
    "x_ma_ret_6h",
    "x_ma_afternoon_ret",
    "x_price_accel_240min",
    "x_vol_ma_ratio_240min",
]


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_arguments():
    p = argparse.ArgumentParser(description="分钟线因子回测")

    # 数据
    # p.add_argument("--data_file",  default=r"G:\bond\data\P_60min_with_ma_features_from_scratch.csv")
    p.add_argument("--data_file",  default=os.path.join(project_root,"data\\P_60min_with_ma_features_from_scratch.csv"))
    p.add_argument("--start_date", default="2018-04-17")

    # 训练
    p.add_argument("--train_window",  type=int, default=4000)
    p.add_argument("--mode",          default="rolling", choices=["rolling", "expanding"])
    p.add_argument("--retrain_freq",  type=int, default=1000)
    p.add_argument("--fwd",           type=int, default=7)
    p.add_argument("--lag",           type=int, default=2)
    p.add_argument("--factor_lags",   default="")
    p.add_argument("--use_scaler",    action="store_true", default=True)
    p.add_argument("--no_scaler",     action="store_false", dest="use_scaler")
    # 标签参数
    p.add_argument("--check_days",    type=int,   default=3)
    p.add_argument("--multiplier",    type=float, default=2)

    # WLS 权重方法
    # "park"    — Park Test（默认，适合日线/截面）
    # "rolling" — 滚动窗口方差（适合分钟线时序，捕捉波动率聚集）
    p.add_argument("--weight_method",  default="Park Test", choices=["park", "rolling"],
                   help="WLS 权重估计方法：park（Park Test）或 rolling（滚动窗口）")
    p.add_argument("--rolling_window", type=int, default=1000,
                   help="rolling 模式下残差滚动标准差的窗口大小（单位：bar）")

    # 回测
    p.add_argument("--reg_threshold",         type=float, default=0.0)
    # p.add_argument("--open_zscore_window",    type=int,   default=20)
    # p.add_argument("--open_zscore_threshold", type=float, default=0)
    p.add_argument("--close_threshold",       type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--close_mode",            default="threshold",
                   choices=["fixed", "threshold", "hybrid", "zscore_threshold", "zscore_hybrid"])

    # 时间编码（会话特征：x_is_night / x_session_sin）
    p.add_argument("--add_session_features", action="store_true",  default=True)
    p.add_argument("--no_session_features",  action="store_true", dest="add_session_features")

    # 信号强度过滤
    p.add_argument("--use_strength_filter", action="store_true",  default=True)
    p.add_argument("--no_strength_filter",  action="store_false", dest="use_strength_filter")
    p.add_argument("--entry_strength_pct",  type=float, default=0.7)

    return p.parse_args()


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def get_contract_switch_dates(df: pd.DataFrame):
    if "dominant_id" not in df.columns:
        return []
    mask       = df["dominant_id"] != df["dominant_id"].shift(1)
    mask.iloc[0] = False
    dates      = df.index[mask].tolist()
    if dates:
        print(f"主力合约切换: {dates[:5]}{'...' if len(dates) > 5 else ''}")
    return dates


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()

    print(f"[WLS] 权重方法={args.weight_method}"
          + (f"  窗口={args.rolling_window} bars" if args.weight_method == "rolling" else ""))

    # 加载数据
    df = load_palm_oil_data(args.data_file)
    if args.start_date:
        df = df[df.index >= pd.to_datetime(args.start_date)]

    args.contract_switch_dates = get_contract_switch_dates(df)

    # 解析 factor_lags
    factor_lags = None
    if getattr(args, "factor_lags", "").strip():
        try:
            factor_lags = [int(x) for x in args.factor_lags.split(",") if x.strip()]
        except ValueError:
            factor_lags = None

    # 准备因子数据
    use_session = args.add_session_features and not getattr(args, "no_session_features", False)
    factor_data, price_data, factor_cols = prepare_factor_data(
        df, selected_factors=FACTORS, lag=args.lag, factor_lags=factor_lags,
        add_session_features=use_session,
    )
    print(f"[因子] {len(factor_cols)} 个: {factor_cols[:5]}{'...' if len(factor_cols) > 5 else ''}")

    # 输出目录
    args.output_dir = create_output_directories(project_root, get_timestamp(), "backtest")

    # 回测（weight_method / rolling_window 由 args 透传给 _train_wls）
    results_df, performance = run_backtest_reg(factor_data, price_data, args)
    if performance is None:
        print("回测失败"); return

    args.split_point = performance.get("split_point")

    # 信号强度过滤（可选）
    if getattr(args, "use_strength_filter", False):
        results_df  = apply_strength_filter(results_df, args)
        performance = recalc_performance(results_df, args)
        print(f"[过滤后] Sharpe={performance['sharpe_ratio']:.4f}  "
              f"Ann={performance['annual_return']:.4f}  "
              f"DD={performance['max_drawdown']:.4f}  "
              f"WR={performance['win_rate']:.4f}")

    print_performance_table(results_df, args)
    save_results(args, factor_cols, args.output_dir, performance, results_df)


if __name__ == "__main__":
    main()