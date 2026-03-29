"""
main_min.py — 棕榈油/商品分钟线因子回测
═══════════════════════════════════════════════════════════════════════════
模型选择（--model）
  wls   加权最小二乘（原始行为，默认）
  xgb   XGBoost 回归，支持 --top_n_features 自动选特征

XGBoost 专属参数
  --top_n_features   选 feature_importances_ 前 N 个特征（0 = 全部）
  --n_estimators     树的数量           (default: 60)
  --max_depth        最大深度           (default: 3)
  --learning_rate    学习率             (default: 0.03)
  --subsample        行采样比           (default: 0.7)
  --colsample_bytree 列采样比           (default: 0.5)
  --min_child_weight 最小叶节点样本数   (default: 50)
  --reg_alpha        L1 正则系数        (default: 1.0)
  --reg_lambda       L2 正则系数        (default: 8.0)

WLS 专属参数（--model wls 时有效）
  --weight_method    park / rolling     (default: rolling)
  --rolling_window   残差滚动窗口 bars  (default: 1000)

共用参数与逻辑与原版完全相同，utils2.py 不做任何修改。
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os

script_path  = Path(__file__).resolve()
project_root = script_path.parents[1]
print(project_root)
sys.path.insert(0, str(script_path.parent))

# ── utils2.py（不修改）────────────────────────────────────────────────────────
from utils2 import (
    load_palm_oil_data,
    prepare_factor_data,
    run_backtest_reg,           # WLS 回测入口
    save_results,
    create_output_directories,
    get_timestamp,
    print_performance_table,
    apply_strength_filter,
    recalc_performance,
)

# ── utils_xgb.py（新建，不改动 utils2.py）────────────────────────────────────
try:
    from Utils_xgb import run_backtest_xgb   # XGBoost 回测入口
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    run_backtest_xgb = None
# ── utils_gru.py ──────────────────────────────────────────────────────────────
try:
    from Utils_gru import run_backtest_gru
    HAS_GRU = True
except ImportError:
    HAS_GRU = False
    run_backtest_gru = None

# ══════════════════════════════════════════════════════════════════════════════
# 因子列表
# ══════════════════════════════════════════════════════════════════════════════

MUST_HAVE_FACTORS = [
    # "x_chaikin_osc_60min",
    # "x_vwap_60min",
    # "x_volume_profile_60min",
    # "x_ease_of_movement_60min",
    # "x_ma_ret_6h",
    # "x_ma_afternoon_ret",
    # "x_price_accel_240min",
    # "x_vol_ma_ratio_240min",
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
    "x_vpin_zscore_filtered_60min",
    "x_rel_vol_240min",
    "x_volume_profile_60min"
]

# FACTORS = [] 时使用全部 x_ 因子，但 MUST_HAVE_FACTORS 始终保留
FACTORS = []

# ══════════════════════════════════════════════════════════════════════════════
# 参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    p = argparse.ArgumentParser(description="分钟线因子回测（WLS / XGBoost）")

    # ── 数据 ──────────────────────────────────────────────────────────────
    # p.add_argument("--data_file",  default=r"G:\bond\data\P_60min_with_ma_features_from_scratch.csv")
    p.add_argument("--data_file",  default=os.path.join(project_root,"data\\P_60min_with_ma_features_from_scratch.csv"))
    # p.add_argument("--start_date", default="2018-04-17")
    p.add_argument("--start_date", default="2020-09-17")

    # ── 模型选择 ──────────────────────────────────────────────────────────
    # 见下方 GRU 段统一定义

    # ── 训练通用 ──────────────────────────────────────────────────────────
    p.add_argument("--train_window",  type=int,   default=4000)
    p.add_argument("--mode",          default="rolling", choices=["rolling", "expanding"])
    p.add_argument("--retrain_freq",  type=int,   default=20)
    p.add_argument("--fwd",           type=int,   default=7)
    p.add_argument("--lag",           type=int,   default=2)
    p.add_argument("--factor_lags",   default="")
    p.add_argument("--use_scaler",    action="store_true",  default=True)
    p.add_argument("--no_scaler",     action="store_false", dest="use_scaler")
    # 标签参数
    p.add_argument("--check_days",    type=int,   default=3)
    p.add_argument("--multiplier",    type=float, default=2.0)

    # ── WLS 专属 ──────────────────────────────────────────────────────────
    p.add_argument("--weight_method",  default="rolling", choices=["park", "rolling"],
                   help="[WLS] 权重估计：park（Park Test）/ rolling（滚动窗口）")
    p.add_argument("--rolling_window", type=int, default=1000,
                   help="[WLS] rolling 模式残差窗口大小（bars）")

    # ── XGBoost 专属 ──────────────────────────────────────────────────────
    p.add_argument("--top_n_features",   type=int,   default=25,
                   help="[XGB] 按 feature_importances_ 选前 N 个特征（0 = 全部）")
    p.add_argument("--n_estimators",     type=int,   default=60,
                   help="[XGB] 树的数量")
    p.add_argument("--max_depth",        type=int,   default=3,
                   help="[XGB] 最大树深")
    p.add_argument("--learning_rate",    type=float, default=0.03,
                   help="[XGB] 学习率")
    p.add_argument("--subsample",        type=float, default=0.7,
                   help="[XGB] 行采样比")
    p.add_argument("--colsample_bytree", type=float, default=0.5,
                   help="[XGB] 列采样比")
    p.add_argument("--min_child_weight", type=int,   default=50,
                   help="[XGB] 最小叶节点样本数（防过拟合）")
    p.add_argument("--reg_alpha",        type=float, default=1.0,
                   help="[XGB] L1 正则系数")
    p.add_argument("--reg_lambda",       type=float, default=8.0,
                   help="[XGB] L2 正则系数")

    # ── 模型选择：加入 gru ────────────────────────────────────────────────────
    p.add_argument("--model", default="xgb", choices=["wls", "xgb", "gru"],
                   help="回归模型：wls / xgb / gru")

    # ── GRU 专属参数 ──────────────────────────────────────────────────────────
    p.add_argument("--seq_len", type=int, default=10,
                   help="[GRU] 时序回看窗口（bar 数）")
    p.add_argument("--hidden_size", type=int, default=32,
                   help="[GRU] 隐藏层维度")
    p.add_argument("--num_layers", type=int, default=1,
                   help="[GRU] 堆叠层数")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="[GRU] Dropout 比例")
    p.add_argument("--gru_epochs", type=int, default=50,
                   help="[GRU] 最大训练 epoch")
    p.add_argument("--gru_batch", type=int, default=512,
                   help="[GRU] mini-batch 大小")
    p.add_argument("--gru_patience", type=int, default=8,
                   help="[GRU] 早停耐心（val_loss 不改善的 epoch 数）")
    p.add_argument("--gru_lr", type=float, default=5e-4,
                   help="[GRU] Adam 学习率")
    p.add_argument("--gru_wd", type=float, default=1e-3,
                   help="[GRU] Adam 权重衰减（L2）")
    p.add_argument("--val_ratio", type=float, default=0.2,
                   help="[GRU] 验证集比例（从训练窗口末尾切出，用于早停）")
    p.add_argument("--perm_repeats", type=int, default=3,
                   help="[GRU] 置换重要性重复次数")

    # ── 回测 ──────────────────────────────────────────────────────────────
    p.add_argument("--reg_threshold",   type=float, default=0.0)
    p.add_argument("--close_threshold", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--close_mode",      default="threshold",
                   choices=["fixed", "threshold", "hybrid",
                            "zscore_threshold", "zscore_hybrid"])

    # ── 信号强度过滤 ──────────────────────────────────────────────────────
    p.add_argument("--use_strength_filter", action="store_true",  default=True)
    p.add_argument("--no_strength_filter",  action="store_false", dest="use_strength_filter")
    p.add_argument("--entry_strength_pct",  type=float, default=0.7)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 辅助
# ══════════════════════════════════════════════════════════════════════════════

def get_contract_switch_dates(df: pd.DataFrame):
    if "dominant_id" not in df.columns:
        return []
    mask = df["dominant_id"] != df["dominant_id"].shift(1)
    mask.iloc[0] = False
    dates = df.index[mask].tolist()
    if dates:
        print(f"主力合约切换: {dates[:5]}{'...' if len(dates) > 5 else ''}")
    return dates


def _print_model_banner(args):
    sep = "─" * 56
    print(f"\n{sep}")
    if args.model == "xgb":
        n = args.top_n_features
        feat_str = f"top {n}" if n > 0 else "全部"
        print(f"  模型: XGBoost  特征选择: {feat_str}  "
              f"trees={args.n_estimators}  depth={args.max_depth}  "
              f"lr={args.learning_rate}")
    else:
        print(f"  模型: WLS  weight_method={args.weight_method}  "
              f"rolling_window={args.rolling_window}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_arguments()

    # ── 检查 XGBoost 可用性 ───────────────────────────────────────────────
    if args.model == "xgb":
        if not HAS_XGB:
            print("[错误] utils_xgb.py 未找到或 xgboost 未安装。")
            print("       请执行: pip install xgboost")
            print("       并确保 utils_xgb.py 与 main_min.py 同目录。")
            return

    _print_model_banner(args)

    # ── 加载数据 ──────────────────────────────────────────────────────────
    df = load_palm_oil_data(args.data_file)
    if args.start_date:
        df = df[df.index >= pd.to_datetime(args.start_date)]

    args.contract_switch_dates = get_contract_switch_dates(df)

    # ── 解析 factor_lags ──────────────────────────────────────────────────
    factor_lags = None
    if getattr(args, "factor_lags", "").strip():
        try:
            factor_lags = [int(x) for x in args.factor_lags.split(",") if x.strip()]
        except ValueError:
            factor_lags = None

    # ── 准备因子数据 ──────────────────────────────────────────────────────
    selected = FACTORS if FACTORS else MUST_HAVE_FACTORS
    factor_data, price_data, factor_cols = prepare_factor_data(
        df, selected_factors=selected, lag=args.lag, factor_lags=factor_lags
    )

    # 确保 MUST_HAVE_FACTORS 始终包含（lag 展开后的列名）
    missing = [f for f in MUST_HAVE_FACTORS if not any(c.startswith(f) for c in factor_cols)]
    if missing:
        print(f"[警告] 必选因子缺失: {missing}")

    n_show = min(5, len(factor_cols))
    print(f"[因子] {len(factor_cols)} 个: "
          f"{factor_cols[:n_show]}{'...' if len(factor_cols) > n_show else ''}")

    # ── XGBoost：提示有效特征选择范围 ────────────────────────────────────
    if args.model == "xgb" and args.top_n_features > 0:
        if args.top_n_features >= len(factor_cols):
            print(f"[提示] top_n_features={args.top_n_features} >= "
                  f"总特征数 {len(factor_cols)}，将使用全部特征。")
        else:
            print(f"[特征选择] 将从 {len(factor_cols)} 个因子中保留 "
                  f"top {args.top_n_features} 个（按 feature_importances_）")

    # ── 输出目录 ──────────────────────────────────────────────────────────
    args.output_dir = create_output_directories(
        project_root, get_timestamp(),
        f"backtest_{args.model}"
    )

    # ══════════════════════════════════════════════════════════════════════
    # 回测（模型分支）
    # ══════════════════════════════════════════════════════════════════════
    if args.model == "xgb":
        results_df, performance = run_backtest_xgb(factor_data, price_data, args)
    elif args.model == "gru":  # ← 新增
        if not HAS_GRU:  # ← 新增
            print("[错误] utils_gru.py 未找到或 PyTorch 未安装。")  # ← 新增
            print("       请执行: pip install torch")  # ← 新增
            return  # ← 新增
        results_df, performance = run_backtest_gru(factor_data, price_data, args)
    else:
        # WLS（原始逻辑，完全不变）
        print(f"[WLS] weight_method={args.weight_method}"
              + (f"  rolling_window={args.rolling_window} bars"
                 if args.weight_method == "rolling" else ""))
        results_df, performance = run_backtest_reg(factor_data, price_data, args)

    if performance is None:
        print("回测失败"); return

    args.split_point = performance.get("split_point")

    # ── 信号强度过滤（可选，两种模型均支持）─────────────────────────────
    if getattr(args, "use_strength_filter", False):
        results_df  = apply_strength_filter(results_df, args)
        performance = recalc_performance(results_df, args)
        print(f"[过滤后] Sharpe={performance['sharpe_ratio']:+.4f}  "
              f"Ann={performance['annual_return']:+.2%}  "
              f"DD={performance['max_drawdown']:.2%}  "
              f"WR={performance['win_rate']:.2%}")

    # ── 绩效输出 ──────────────────────────────────────────────────────────
    print_performance_table(results_df, args)
    save_results(args, factor_cols, args.output_dir, performance, results_df)


if __name__ == "__main__":
    main()