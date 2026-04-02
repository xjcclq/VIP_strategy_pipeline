"""
Linear-model backtest entrypoint.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
import os

script_path = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(script_path.parent))

from Utils_linear import (
    load_palm_oil_data,
    prepare_factor_data,
    run_backtest_reg,
    save_results,
    create_output_directories,
    get_timestamp,
    print_performance_table,
    apply_strength_filter,
    recalc_performance,
)


FACTORS = [
    # "x_chaikin_osc_60min",
    "x_vwap_60min",
    # # "x_vpin_zscore_filtered_60min",
    # "x_volume_profile_60min",
    # "x_ease_of_movement_60min",
    "x_ma_ret_1h",
    "x_ma_ret_2h",
    "x_ma_ret_4h",
    "x_ma_ret_6h",
    "x_ma_ret_8h",
    "x_ma_ret_12h",
    # "x_ma_afternoon_ret",
    # "x_price_accel_240min",
    # "x_vol_ma_ratio_240min",
]


def parse_arguments():
    p = argparse.ArgumentParser(description="线性模型回测")

    p.add_argument(
        "--data_file",
        default=os.path.join(project_root, "data\\P_60min_with_ma_features_from_scratch.csv"),
    )
    p.add_argument("--start_date", default="2018-04-17")

    p.add_argument("--train_window", type=int, default=4000)
    p.add_argument("--mode", default="rolling", choices=["rolling", "expanding"])
    p.add_argument("--retrain_freq", type=int, default=1000)
    p.add_argument("--fwd", type=int, default=3)
    p.add_argument("--lag", type=int, default=2)
    p.add_argument("--factor_lags", default="")
    p.add_argument("--use_scaler", action="store_true", default=True)
    p.add_argument("--no_scaler", action="store_false", dest="use_scaler")
    p.add_argument("--check_days", type=int, default=3)
    p.add_argument("--multiplier", type=float, default=2)

    p.add_argument(
        "--model_type",
        default="pls",
        choices=["wls", "lasso", "ridge", "elastic_net", "pls"],
        help="选择线性模型类型",
    )
    p.add_argument("--lasso_alpha", type=float, default=1.0)
    p.add_argument("--ridge_alpha", type=float, default=0)
    p.add_argument("--elastic_net_alpha", type=float, default=1.0)
    p.add_argument("--elastic_net_l1_ratio", type=float, default=0.5)
    p.add_argument("--pls_n_components", type=int, default=1, help="PLS 潜变量个数")
    p.add_argument("--linear_max_iter", type=int, default=10000)

    p.add_argument(
        "--weight_method",
        default="Park Test",
        choices=["park", "rolling"],
        help="WLS 权重方法",
    )
    p.add_argument("--rolling_window", type=int, default=1000, help="WLS rolling 窗口")

    p.add_argument("--reg_threshold", type=float, default=0.0)
    p.add_argument("--close_threshold", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument(
        "--close_mode",
        default="threshold",
        choices=["fixed", "threshold", "hybrid", "zscore_threshold", "zscore_hybrid"],
    )

    p.add_argument("--add_session_features", action="store_true", default=True)
    p.add_argument("--no_session_features", action="store_true", dest="add_session_features")

    p.add_argument("--use_strength_filter", action="store_true", default=True)
    p.add_argument("--no_strength_filter", action="store_false", dest="use_strength_filter")
    p.add_argument("--entry_strength_pct", type=float, default=0.7)

    return p.parse_args()


def get_contract_switch_dates(df: pd.DataFrame):
    if "dominant_id" not in df.columns:
        return []
    mask = df["dominant_id"] != df["dominant_id"].shift(1)
    mask.iloc[0] = False
    dates = df.index[mask].tolist()
    if dates:
        print(f"检测到换月日期: {dates[:5]}{'...' if len(dates) > 5 else ''}")
    return dates


def main():
    args = parse_arguments()

    model_type = str(getattr(args, "model_type", "wls") or "wls").lower()
    if model_type == "wls":
        print(f"[MODEL] WLS  method={args.weight_method}"
              + (f"  rolling_window={args.rolling_window} bars" if args.weight_method == "rolling" else ""))
    elif model_type == "lasso":
        print(f"[MODEL] LASSO  alpha={args.lasso_alpha}  max_iter={args.linear_max_iter}")
    elif model_type == "ridge":
        print(f"[MODEL] RIDGE  alpha={args.ridge_alpha}")
    elif model_type == "elastic_net":
        print(f"[MODEL] ELASTIC_NET  alpha={args.elastic_net_alpha}  "
              f"l1_ratio={args.elastic_net_l1_ratio}  max_iter={args.linear_max_iter}")
    elif model_type == "pls":
        print(f"[MODEL] PLS  n_components={args.pls_n_components}  max_iter={args.linear_max_iter}")

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

    use_session = args.add_session_features and not getattr(args, "no_session_features", False)
    factor_data, price_data, factor_cols = prepare_factor_data(
        df,
        selected_factors=FACTORS,
        lag=args.lag,
        factor_lags=factor_lags,
        add_session_features=use_session,
    )
    print(f"[FACTORS] {len(factor_cols)} cols  {factor_cols[:5]}{'...' if len(factor_cols) > 5 else ''}")

    args.output_dir = create_output_directories(project_root, get_timestamp(), "backtest")

    results_df, performance = run_backtest_reg(factor_data, price_data, args)
    if performance is None:
        print("回测失败")
        return

    args.split_point = performance.get("split_point")

    if getattr(args, "use_strength_filter", False):
        results_df = apply_strength_filter(results_df, args)
        performance = recalc_performance(results_df, args)
        print(
            f"[FILTER] Sharpe={performance['sharpe_ratio']:.4f}  "
            f"Ann={performance['annual_return']:.4f}  "
            f"DD={performance['max_drawdown']:.4f}  "
            f"WR={performance['win_rate']:.4f}"
        )

    print_performance_table(results_df, args)
    save_results(args, factor_cols, args.output_dir, performance, results_df)


if __name__ == "__main__":
    main()
