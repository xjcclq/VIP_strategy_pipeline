"""
tune_gru.py — GRU 防过拟合调参网格搜索
运行: python backtest/tune_gru.py
"""

import sys
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import io

script_path  = Path(__file__).resolve()
project_root = script_path.parents[1]
sys.path.insert(0, str(script_path.parent))

from utils2 import (
    load_palm_oil_data, prepare_factor_data,
    apply_strength_filter, recalc_performance,
)
from Utils_gru import run_backtest_gru

MUST_HAVE_FACTORS = [
    "x_chaikin_osc_60min",
    "x_vwap_60min",
    "x_volume_profile_60min",
    "x_ease_of_movement_60min",
    "x_ma_ret_6h",
    "x_ma_afternoon_ret",
    "x_price_accel_240min",
    "x_vol_ma_ratio_240min",
]

# ── 搜索空间 ──────────────────────────────────────────────────────────────────
GRID = {
    "seq_len":      [5, 10, 20],
    "hidden_size":  [16, 32],
    "num_layers":   [1],
    "dropout":      [0.3, 0.5],
    "gru_lr":       [3e-4, 1e-3],
    "gru_wd":       [1e-3, 5e-3],
}
# 固定不搜的维度
FIXED = dict(
    gru_epochs=50,
    gru_batch=512,
    gru_patience=8,
    val_ratio=0.2,
    perm_repeats=2,
    top_n_features=0,   # 因子已手选，不再二次筛选
)


def make_args(combo):
    return SimpleNamespace(
        data_file=r"G:\bond\data\P_60min_with_ma_features_from_scratch.csv",
        start_date="2018-04-17",
        model="gru",
        train_window=4000,
        mode="rolling",
        retrain_freq=1000,
        fwd=7,
        lag=2,
        factor_lags="",
        use_scaler=True,
        check_days=3,
        multiplier=2.0,
        reg_threshold=0.0,
        close_threshold=[0.0, 0.0],
        close_mode="hybrid",
        use_strength_filter=True,
        entry_strength_pct=0.7,
        contract_switch_dates=[],
        output_dir="",
        **FIXED,
        **combo,
    )


def calc_oos(results_df, performance, args):
    split = performance.get("split_point")
    fwd = args.fwd
    if split and split in results_df.index:
        si = results_df.index.get_loc(split)
        oos = results_df.iloc[si + fwd + 2:]
    else:
        oos = pd.DataFrame()

    if len(oos) == 0:
        return 0, 0, 0

    _key = pd.to_datetime(oos.index).normalize()
    daily = oos.groupby(_key).agg(r=("strategy_return", "sum"))
    dr = daily["r"].values
    mu, sigma = dr.mean(), dr.std()
    oos_sharpe = mu / sigma * np.sqrt(252) if sigma > 0 else 0
    oos_ann = mu * 252
    cum = 1.0 + np.cumsum(dr)
    peak = np.maximum.accumulate(cum)
    oos_mdd = ((cum - peak) / peak).min()
    return oos_sharpe, oos_ann, oos_mdd


def run_one(factor_data, price_data, combo, idx, total):
    args = make_args(combo)
    args.train_window = min(args.train_window, len(factor_data) - 1)
    args.split_point  = factor_data.index[args.train_window - 1]

    tag = " | ".join(f"{k}={v}" for k, v in combo.items())
    print(f"\n[{idx}/{total}] {tag}", flush=True)

    try:
        f = io.StringIO()
        with redirect_stdout(f):
            results_df, performance = run_backtest_gru(factor_data, price_data, args)

        if performance is None:
            return None

        args.split_point = performance.get("split_point")

        if args.use_strength_filter:
            with redirect_stdout(f):
                results_df  = apply_strength_filter(results_df, args)
                performance = recalc_performance(results_df, args)

        all_sharpe = performance.get("sharpe_ratio", 0)
        all_ann = performance.get("annual_return", 0)
        all_mdd = performance.get("max_drawdown", 0)
        oos_sharpe, oos_ann, oos_mdd = calc_oos(results_df, performance, args)

        result = {
            **combo,
            "all_sharpe": round(all_sharpe, 3),
            "all_ann%": round(all_ann * 100, 1),
            "all_mdd%": round(all_mdd * 100, 1),
            "oos_sharpe": round(oos_sharpe, 3),
            "oos_ann%": round(oos_ann * 100, 1),
            "oos_mdd%": round(oos_mdd * 100, 1),
            "gap": round(all_sharpe - oos_sharpe, 3),
        }

        print(f"  ALL={all_sharpe:.2f}  OOS={oos_sharpe:.2f}  gap={all_sharpe-oos_sharpe:.2f}  "
              f"OOS_ann={oos_ann*100:.1f}%  OOS_mdd={oos_mdd*100:.1f}%", flush=True)
        return result

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def main():
    df = load_palm_oil_data(r"G:\bond\data\P_60min_with_ma_features_from_scratch.csv")
    df = df[df.index >= pd.to_datetime("2018-04-17")]

    factor_data, price_data, factor_cols = prepare_factor_data(
        df, selected_factors=MUST_HAVE_FACTORS, lag=2, factor_lags=None
    )
    print(f"[数据] {factor_data.shape}  因子: {len(factor_cols)}")
    print(f"[必选因子] {MUST_HAVE_FACTORS}")

    keys = list(GRID.keys())
    vals = list(GRID.values())
    combos = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    total = len(combos)
    print(f"共 {total} 组参数\n")

    results = []
    for idx, combo in enumerate(combos, 1):
        r = run_one(factor_data, price_data, combo, idx, total)
        if r is not None:
            results.append(r)

    if results:
        rdf = pd.DataFrame(results)
        rdf = rdf.sort_values("oos_sharpe", ascending=False)
        out_path = project_root / "output" / "gru_tune_results.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rdf.to_csv(out_path, index=False)

        print(f"\n{'='*120}")
        print(f"Top 15（按样本外 Sharpe）")
        print(f"{'='*120}")
        print(rdf.head(15).to_string(index=False))
        print(f"\n结果已保存: {out_path}")
    else:
        print("所有组合均失败")


if __name__ == "__main__":
    main()
