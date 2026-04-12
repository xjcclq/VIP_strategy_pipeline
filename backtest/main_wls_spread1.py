"""
main_min_abs.py — 棕榈油/商品分钟线因子回测（绝对价差版）
  - 训练标签：price[t+fwd] - price[t]，单位：价格点，不除以分母
  - PnL     ：position * (price[t] - price[t-1])，绝对点数
  - 绘图    ：cumsum(pnl)，Y轴为价格点，无收益率概念
  - WLS 权重：滚动窗口方差（rolling），适合分钟线时序
  - 可选信号强度过滤 (--use_strength_filter)

【新增改动】（以 ★新增 标注）
  ① --check_days / --reversal_multiplier  反转标签参数
  ② --stop_loss_pts                       固定止损参数
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

from utils_spread1 import (
    load_palm_oil_data, prepare_factor_data, run_backtest_reg,
    save_results, create_output_directories, get_timestamp,
    print_performance_table, apply_strength_filter, recalc_performance,
)

# ── 因子列表 ──────────────────────────────────────────────────────────────────

FACTORS = [
    # 'x_卷螺基差率',
    # 'x_卷螺利润差',
    'x_bolling_position_5',
    'x_momentum1',
    # 'x_close'
    # 'x_log_return_1d',
    # "x_rsi_14d",
    # "x_trend_strength",
    # "x_volatility_ratio",
    # "x_obv",
    # "x_volume_ma_ratio",
    # "x_hurst_exponent",
    # 'x_momentum3'
    # 'x_盘面利润率差',
    # 'x_基差差',
    # 'x_price_ma_deviation',
    # 'x_bolling_position',
    # 'x_adx'
]


# ── 参数 ──────────────────────────────────────────────────────────────────────

def parse_arguments():
    p = argparse.ArgumentParser(description="分钟线因子回测（绝对价差版）")

    # 数据
    p.add_argument("--data_file",  default=os.path.join(project_root, "data\\jlspread_15m.csv"))
    p.add_argument("--start_date", default="2021-01-17")

    # 训练
    p.add_argument("--train_window",  type=int, default=2500)
    p.add_argument("--mode",          default="rolling", choices=["rolling", "expanding"])
    p.add_argument("--retrain_freq",  type=int, default=500)
    p.add_argument("--fwd",           type=int, default=3,
                   help="标签：price[t+fwd] - price[t]")
    p.add_argument("--lag",           type=int, default=2)
    p.add_argument("--factor_lags",   default="")
    p.add_argument("--use_scaler",    action="store_true", default=True)
    p.add_argument("--no_scaler",     action="store_false", dest="use_scaler")

    # WLS 权重方法
    p.add_argument("--weight_method",  default="Park Test",
                   choices=["park", "rolling"],
                   help="WLS 权重：park（Park Test）或 rolling（滚动窗口）")
    p.add_argument("--rolling_window", type=int, default=1000,
                   help="rolling 模式下残差滚动标准差的窗口大小（bar）")

    # 回测
    p.add_argument("--reg_threshold",   type=float, default=0.0,
                   help="开仓阈值（单位：价格点，prediction 绝对值需超过此值）")
    p.add_argument("--close_threshold", type=float, nargs=2, default=[0, 0])
    p.add_argument("--close_mode",      default="threshold",
                   choices=["fixed", "threshold", "hybrid",
                             "zscore_threshold", "zscore_hybrid"])

    # 时间编码：改 default=True/False 控制是否启用
    p.add_argument("--add_session_features", type=lambda x: x.lower() != "false", default=False,
                   metavar="True/False")

    # 信号强度过滤
    p.add_argument("--use_strength_filter", action="store_true",  default=True)
    p.add_argument("--no_strength_filter",  action="store_false", dest="use_strength_filter")
    p.add_argument("--entry_strength_pct",  type=float, default=0.3)

    # 资金账户参数
    p.add_argument("--init_capital",   type=float, default=3000.0,  help="初始资金（元）")
    p.add_argument("--multiplier",     type=float, default=10.0,    help="合约乘数")
    p.add_argument("--commission",     type=float, default=1e-4,   help="手续费率（万2=0.0002）")
    p.add_argument("--rebalance_days", type=int,   default=30,      help="rebalance 间隔交易日数")

    # ★新增①：反转标签参数
    p.add_argument("--check_days",          type=int,   default=3,
                   help="反转检查额外bar数：在fwd之后继续向前看check_days根bar"
                        "检测反转（0=不启用，与原逻辑一致）")
    p.add_argument("--reversal_multiplier", type=float, default=1.2,
                   help="反转强度倍数：反转幅度需超过基础标签的此倍数才替换"
                        "（默认1.2，即反转需比原信号强20%%）")

    # ★新增②：固定止损参数
    p.add_argument("--stop_loss_pts",  type=float, default=10.0,
                   help="固定止损点数：浮亏超过此值强制平仓（0=不启用）")

    return p.parse_args()


# ── 辅助 ──────────────────────────────────────────────────────────────────────

def get_contract_switch_dates(df: pd.DataFrame):
    if "dominant_id" not in df.columns:
        return []
    mask = df["dominant_id"] != df["dominant_id"].shift(1)
    mask.iloc[0] = False
    dates = df.index[mask].tolist()
    if dates:
        print(f"主力合约切换: {dates[:5]}{'...' if len(dates) > 5 else ''}")
    return dates


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()

    print(f"[绝对价差版] 标签=price[t+{args.fwd}]-price[t]  "
          f"WLS={args.weight_method}"
          + (f"  窗口={args.rolling_window}bars" if args.weight_method == "rolling" else ""))
    # ★新增：打印新参数
    print(f"[新增参数] 反转检查={args.check_days}bars  "
          f"反转倍数={args.reversal_multiplier:.1f}  "
          f"固定止损={args.stop_loss_pts}点")

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
    factor_data, price_data, factor_cols = prepare_factor_data(
        df, selected_factors=FACTORS, lag=args.lag, factor_lags=factor_lags,
        add_session_features=args.add_session_features,
    )
    print(f"[因子] {len(factor_cols)} 个: {factor_cols[:5]}"
          f"{'...' if len(factor_cols) > 5 else ''}")
    print(f"[价格] 样本数={len(price_data)}  "
          f"min={price_data.min():.2f}  max={price_data.max():.2f}  "
          f"mean={price_data.mean():.2f}")

    # 输出目录
    args.output_dir = create_output_directories(project_root, get_timestamp(), "backtest_abs")

    # 回测
    results_df, performance = run_backtest_reg(factor_data, price_data, args)
    if performance is None:
        print("回测失败")
        return

    args.split_point = performance.get("split_point")

    # 信号强度过滤（可选）
    if getattr(args, "use_strength_filter", False):
        results_df  = apply_strength_filter(results_df, args)
        performance = recalc_performance(results_df, args)
        print(f"[过滤后] Sharpe(元)={performance['sharpe_cny']:.4f}  "
              f"净PnL={performance['total_net_cny']:.2f}元  "
              f"最大回撤={performance['max_drawdown_cny']:.2f}元  "
              f"胜率={performance['win_rate']:.4f}")

    print_performance_table(results_df, args)
    save_results(args, factor_cols, args.output_dir, performance, results_df)


if __name__ == "__main__":
    main()