"""
utils_xgb.py — XGBoost 回归回测工具
═══════════════════════════════════════════════════════════════════════════
核心能力
  · _train_xgb()        两阶段训练：先全特征跑一遍获得 feature_importances_，
                        选 top-N 后重新训练最终模型（scaler 独立拟合）
  · run_backtest_xgb()  滑动/扩展窗口回测，与 run_backtest_reg() 接口完全一致
                        可直接替换 main_min.py 中的调用

与 utils2.py 的关系
  · 不修改 utils2.py；从中 import 私有底层函数 _compute_reversal_labels /
    _backtest / _performance（Python 允许显式 import，无需修改源文件）
  · 所有绩效计算路径与 WLS 版本完全相同
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Tuple

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("xgboost 未安装，请执行: pip install xgboost", ImportWarning, stacklevel=2)

from sklearn.preprocessing import RobustScaler

# ── 复用 utils2.py 底层函数（不改动 utils2.py）────────────────────────────────
from utils2 import (
    _compute_reversal_labels,   # 向量化反转标签
    _backtest,                  # prediction → position → strategy_return
    _performance,               # bar 聚合到日线后计算指标
    calc_metrics_from_returns,  # 从收益数组直接计算指标
)


# ══════════════════════════════════════════════════════════════════════════════
# 超参默认值（可通过 args 覆盖）
# ══════════════════════════════════════════════════════════════════════════════

# XGB_DEFAULTS = dict(
#     n_estimators     = 60,
#     max_depth        = 3,
#     learning_rate    = 0.03,
#     subsample        = 0.7,
#     colsample_bytree = 0.5,
#     min_child_weight = 50,
#     reg_alpha        = 1.0,
#     reg_lambda       = 8.0,
#     random_state     = 42,
#     n_jobs           = -1,
#     tree_method      = "hist",   # 快速直方图算法
# )

XGB_DEFAULTS = dict(
    n_estimators     = 300,
    max_depth        = 3,
    learning_rate    = 0.03,
    subsample        = 0.7,
    colsample_bytree = 0.5,
    min_child_weight = 20,
    reg_alpha        = 1.0,
    reg_lambda       = 8.0,
    random_state     = 42,
    n_jobs           = -1,
    tree_method      = "hist",   # 快速直方图算法
)

# ── 无论重要性排名如何，这些特征始终保留 ────────────────────────────────────
# MUST_INCLUDE_FEATURES: List[str] = [
#     # "x_chaikin_osc_60min",
#     "x_vwap_60min",
#     # "x_volume_profile_60min",
#     # "x_ease_of_movement_60min",
#     "x_ma_ret_6h",
#     # "x_ma_afternoon_ret",
#     # "x_price_accel_240min",
#     # "x_vol_ma_ratio_240min",
# ]
MUST_INCLUDE_FEATURES: list[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# 核心训练函数
# ══════════════════════════════════════════════════════════════════════════════

def _train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_scaler:      bool = True,
    top_n_features:  int  = 0,
    must_include:    Optional[List[str]] = None,   # ← 新增：强制保留的特征名
    **xgb_kwargs,
) -> Tuple[XGBRegressor, Optional[RobustScaler], List[str]]:
    """
    两阶段 XGBoost 训练
    ─────────────────────────────────────────────────────
    Phase 1  全特征训练 → feature_importances_（gain 重要度）
    Phase 2  选 top-N 特征，再强制并入 must_include → 独立 scaler + 最终模型

    must_include 中不在训练集列名里的条目会被自动跳过（静默忽略）。

    参数
    ─────
    X_train          训练特征（pd.DataFrame）
    y_train          标签（pd.Series）
    use_scaler       是否用 RobustScaler 标准化
    top_n_features   选前 N 个重要特征；0 或 >= 总特征数 = 使用全部特征
    must_include     强制保留的特征名列表（在 top-N 基础上追加，去重）
    **xgb_kwargs     XGBRegressor 超参，覆盖 XGB_DEFAULTS

    返回
    ─────
    (model, scaler, selected_cols)
    """
    if not HAS_XGB:
        raise RuntimeError("xgboost 未安装，请 pip install xgboost")

    all_cols = list(X_train.columns)
    params   = {**XGB_DEFAULTS, **xgb_kwargs}

    # ── Phase 1: 全特征 → 获取 feature_importances_ ──────────────────────
    X_arr = X_train.values.astype(float)
    y_arr = y_train.values.astype(float)

    scaler_all = None
    if use_scaler:
        scaler_all = RobustScaler()
        X_arr = scaler_all.fit_transform(X_arr)

    model_all = XGBRegressor(**params, verbosity=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_all.fit(X_arr, y_arr)

    imp_series = (
        pd.Series(model_all.feature_importances_, index=all_cols)
        .sort_values(ascending=False)
    )

    do_select = (top_n_features > 0) and (top_n_features < len(all_cols))

    # ── Phase 2: 特征选择 + 强制保留 + 最终模型 ──────────────────────────
    if do_select:
        selected_cols = imp_series.head(top_n_features).index.tolist()

        # ── 强制并入 must_include ────────────────────────────────────────
        if must_include:
            forced_in = []
            for col in must_include:
                if col in all_cols and col not in selected_cols:
                    selected_cols.append(col)
                    forced_in.append(col)
            if forced_in:
                print(f"    [强制保留] {forced_in}  "
                      f"（最终特征数 {len(selected_cols)}，"
                      f"其中 top-{top_n_features} + 强制 {len(forced_in)}）")
        # ────────────────────────────────────────────────────────────────

        X_sel = X_train[selected_cols].values.astype(float)
        scaler = None
        if use_scaler:
            scaler = RobustScaler()
            X_sel = scaler.fit_transform(X_sel)

        model = XGBRegressor(**params, verbosity=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_sel, y_arr)

        final_imp = (
            pd.Series(model.feature_importances_, index=selected_cols)
            .sort_values(ascending=False)
        )
        tag = f"top {top_n_features} + 强制 {len(must_include or [])} / 共 {len(all_cols)}"
    else:
        selected_cols = all_cols
        scaler        = scaler_all
        model         = model_all
        final_imp     = imp_series
        # tag           = f"全部 {len(all_cols)}"tqdm

    # ── 打印 ──────────────────────────────────────────────────────────────
    # print(f"    ===== XGBoost 特征重要性 ({tag}) =====")
    # top_imp = final_imp.iloc[0] if len(final_imp) > 0 else 0
    # must_set = set(must_include or [])
    # for feat, imp in final_imp.head(min(8, len(final_imp))).items():
    #     bar    = "█" * int(imp / top_imp * 20) if top_imp > 0 else ""
    #     marker = " ★" if feat in must_set else ""          # ★ 标记强制保留特征
    #     print(f"    {feat:<40s}  {imp:.4f}  {bar}{marker}")
    # if len(final_imp) > 8:
    #     print(f"    ... 共 {len(final_imp)} 个特征")

    return model, scaler, selected_cols


# ══════════════════════════════════════════════════════════════════════════════
# 滑动窗口回测（XGBoost 版）
# ══════════════════════════════════════════════════════════════════════════════

def _run_xgb_sliding_window(
    factor_data: pd.DataFrame,
    price_data:  pd.Series,
    args,
) -> Tuple[pd.DataFrame, dict]:
    """
    与 utils2.py::_run_sliding_window_backtest 结构相同，
    将 _train_wls 替换为 _train_xgb，其余逻辑保持一致。

    注意
    ────
    · predictions[:tw][vin] = ... 是链式索引（可能不回写），
      此处改用 np.where(vin) 绝对索引写回，行为明确。
    · selected_cols 随再训练点更新，批量预测时自动对齐到当前窗口的特征子集。
    """
    n        = len(factor_data)
    tw       = args.train_window
    fwd      = args.fwd
    freq     = args.retrain_freq
    mode     = args.mode
    use_sc   = getattr(args, "use_scaler",     True)
    top_n    = getattr(args, "top_n_features", 0)

    # ── 强制保留特征：优先读 args，fallback 到模块级常量 ─────────────────
    must = list(getattr(args, "must_include_features", None) or MUST_INCLUDE_FEATURES)
    must_valid = [c for c in must if c in factor_data.columns]
    if len(must_valid) < len(must):
        missing_must = [c for c in must if c not in factor_data.columns]
        print(f"  [警告] must_include 中以下特征不在数据列中，已忽略: {missing_must}")

    must_final = [c for c in must_valid if c in factor_data.columns]
    # print(f"[强制保留] {must_final}")

    # XGBoost 超参（从 args 读取，缺省使用 XGB_DEFAULTS）
    xgb_kw = {k: getattr(args, k, v) for k, v in XGB_DEFAULTS.items()}

    predictions   = np.full(n, np.nan)
    model = scaler = None
    selected_cols  = list(factor_data.columns)

    first_model = first_scaler = first_cols = None
    in_sample_done = False

    # ── 预计算标签（一次性） ──────────────────────────────────────────────
    all_labels = _compute_reversal_labels(
        price_data, fwd,
        check_days = getattr(args, "check_days", 5),
        multiplier = getattr(args, "multiplier", 1.2),
    )
    factor_arr = factor_data.values.astype(float)
    nan_mask   = (
        np.isnan(factor_arr).any(axis=1)
        | np.isinf(factor_arr).any(axis=1)
    )

    retrain_pts = [i for i in range(tw, n) if (i - tw) % freq == 0]

    # for rp_idx, rp in enumerate(tqdm(retrain_pts, desc=f"XGBoost {mode}回测")):
    for rp_idx, rp in enumerate(retrain_pts):
        train_end   = rp - fwd - 5
        if train_end <= 0:
            continue
        train_start = max(0, train_end - tw) if mode == "rolling" else 0

        X_all = factor_data.iloc[train_start:train_end]
        y_all = all_labels.iloc[train_start:train_end]
        valid = ~(
            X_all.isna().any(axis=1)
            | np.isinf(X_all.values).any(axis=1)
            | y_all.isna()
            | np.isinf(y_all)
        )
        Xv, yv = X_all[valid], y_all[valid]

        if valid.sum() < 50:
            print(f"    样本不足 ({valid.sum()})，跳过"); continue

        # print(f"  [{rp}/{n}] {mode}[{train_start}:{train_end}]  n={valid.sum()}")

        # ── 训练（传入 must_final）────────────────────────────────────────
        model, scaler, selected_cols = _train_xgb(
            Xv, yv, use_sc, top_n,
            must_include=must_final,   # ← 每折都传入
            **xgb_kw
        )
        if first_model is None:
            first_model  = model
            first_scaler = scaler
            first_cols   = selected_cols

        # ── 回填样本内预测（仅首次）──────────────────────────────────────
        if not in_sample_done and first_model is not None:
            Xin    = factor_data.iloc[:tw][first_cols]
            vin    = ~nan_mask[:tw]
            if vin.any():
                try:
                    Xin_v  = Xin.values[vin].astype(float)
                    Xs     = first_scaler.transform(Xin_v) if first_scaler else Xin_v
                    preds  = first_model.predict(Xs)
                    valid_idx = np.where(vin)[0]
                    predictions[valid_idx] = preds
                except Exception as e:
                    print(f"    [警告] 样本内回填失败: {e}")
            in_sample_done = True

        # ── 批量样本外预测（当前 rp → 下一 rp）──────────────────────────
        pred_end = retrain_pts[rp_idx + 1] if rp_idx + 1 < len(retrain_pts) else n
        vp_mask  = ~nan_mask[rp:pred_end]
        if vp_mask.any() and model is not None:
            try:
                Xb = factor_data.iloc[rp:pred_end][selected_cols].values[vp_mask].astype(float)
                Xs = scaler.transform(Xb) if scaler else Xb
                preds    = model.predict(Xs)
                abs_idx  = rp + np.where(vp_mask)[0]
                predictions[abs_idx] = preds
            except Exception as e:
                print(f"    [警告] 批量预测失败: {e}")

    # ── 构建 trade_df → _backtest ─────────────────────────────────────────
    trade_df = pd.DataFrame(
        {
            "prediction":    predictions,
            "price":         price_data.values,
            "actual_return": price_data.pct_change(fill_method=None).values,
        },
        index=price_data.index,
    )

    results_df  = _backtest(trade_df, args)
    performance = _performance(results_df, args)
    return results_df, performance


# ══════════════════════════════════════════════════════════════════════════════
# 公开入口（与 run_backtest_reg 接口一致）
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_xgb(
    factor_data: pd.DataFrame,
    price_data:  pd.Series,
    args,
) -> Tuple[pd.DataFrame, dict]:
    """
    XGBoost 版回测入口。
    接口与 utils.run_backtest_reg() 完全相同，可直接替换。

    args 额外读取字段（WLS 不需要）
    ─────────────────────────────
    top_n_features        int        选前 N 重要特征（0 = 全部）
    must_include_features list[str]  强制保留的特征名；None → 使用 MUST_INCLUDE_FEATURES
    n_estimators          int        树的数量         (default: 60)
    max_depth             int        最大深度         (default: 3)
    learning_rate         float      学习率           (default: 0.03)
    subsample             float      行采样比         (default: 0.7)
    colsample_bytree      float      列采样比         (default: 0.5)
    min_child_weight      int        最小叶节点样本   (default: 50)
    reg_alpha             float      L1 正则          (default: 1.0)
    reg_lambda            float      L2 正则          (default: 8.0)
    """
    args.train_window = min(args.train_window, len(factor_data) - 1)
    args.split_point  = factor_data.index[args.train_window - 1]
    return _run_xgb_sliding_window(factor_data, price_data, args)