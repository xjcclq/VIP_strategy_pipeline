"""
utils_gru.py — GRU（门控循环单元）回归回测工具
═══════════════════════════════════════════════════════════════════════════
核心能力
  · _GRUNet            PyTorch GRU 回归网络（多层 GRU + 全连接头）
  · _build_sequences   将平坦因子矩阵转换为 (N, seq_len, F) 时序样本
  · _perm_importance   置换重要性：打乱第 f 列 → 预测误差增量 → 特征排名
  · _train_gru         两阶段训练（与 utils_xgb._train_xgb 结构相同）：
                         Phase1 全特征 → 置换重要性排名
                         Phase2 选 top-N → 独立 scaler + 最终模型
  · run_backtest_gru   公开入口，接口与 run_backtest_reg / run_backtest_xgb 一致

与 utils2.py / utils_xgb.py 的关系
  · 不修改任何已有文件
  · 复用 utils2.py 私有底层：_compute_reversal_labels / _backtest / _performance
  · GRU 独有超参见文末 GRU_DEFAULTS
"""

from __future__ import annotations

import warnings
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch 未安装，请执行: pip install torch", ImportWarning, stacklevel=2)

from sklearn.preprocessing import RobustScaler

from utils2 import (
    _compute_reversal_labels,
    _backtest,
    _performance,
)


# ══════════════════════════════════════════════════════════════════════════════
# 超参默认值
# ══════════════════════════════════════════════════════════════════════════════

GRU_DEFAULTS = dict(
    seq_len       = 20,     # 每个样本回看的 bar 数（时间步）
    hidden_size   = 64,     # GRU 隐藏层维度
    num_layers    = 2,      # 堆叠 GRU 层数
    dropout       = 0.2,    # Dropout（num_layers>1 时 GRU 内部也使用）
    gru_epochs    = 100,    # 最大训练 epoch
    gru_batch     = 256,    # mini-batch 大小
    gru_patience  = 10,     # 早停耐心（val_loss 不改善的 epoch 数）
    gru_lr        = 1e-3,   # Adam 学习率
    gru_wd        = 1e-4,   # Adam 权重衰减（L2 正则）
    val_ratio     = 0.15,   # 验证集比例（从训练窗口末尾切出，用于早停）
    perm_repeats  = 5,      # 置换重要性重复次数（越多越稳定，越慢）
)


# ══════════════════════════════════════════════════════════════════════════════
# GRU 网络定义
# ══════════════════════════════════════════════════════════════════════════════

class _GRUNet(nn.Module):
    """
    多层 GRU + 两层全连接回归头

    输入:  (batch, seq_len, n_features)
    输出:  (batch,)  ← 回归预测值（无激活）

    结构
    ────
    GRU(n_features → hidden_size, num_layers, dropout)
         ↓ 取最后时刻隐状态 h[-1]  shape=(batch, hidden_size)
    Dropout → Linear(hidden_size → hidden_size//2) → ReLU
         ↓
    Dropout → Linear(hidden_size//2 → 1)
    """

    def __init__(self, n_features: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        # num_layers==1 时 GRU 内部 dropout 必须为 0，否则警告
        gru_drop = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size   = n_features,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = gru_drop,
        )
        half = max(hidden_size // 2, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, half),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(half, 1),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # out: (batch, seq_len, hidden); h_n: (num_layers, batch, hidden)
        _, h_n = self.gru(x)
        last_h = h_n[-1]          # 取最顶层最后时刻：(batch, hidden)
        return self.head(last_h).squeeze(-1)   # (batch,)


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _get_device() -> "torch.device":
    """优先使用 CUDA，其次 MPS（Apple Silicon），最后 CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将平坦因子矩阵转换为时序样本。

    规则：第 i 个样本（i >= seq_len-1）
      · 输入  = X[i-seq_len+1 : i+1]  shape (seq_len, n_features)
      · 标签  = y[i]

    返回
    ────
    X_seq  (N, seq_len, n_features)  float32
    y_seq  (N,)                      float32
    N = len(X) - seq_len + 1
    """
    n = len(X)
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.empty(0, dtype=np.float32)

    n_seq  = n - seq_len + 1
    n_feat = X.shape[1]
    X_seq  = np.lib.stride_tricks.sliding_window_view(
        X, (seq_len, n_feat)
    ).reshape(n_seq, seq_len, n_feat).astype(np.float32)
    y_seq  = y[seq_len - 1:].astype(np.float32)
    return X_seq, y_seq


def _perm_importance(
    model:    "_GRUNet",
    X_seq:    np.ndarray,
    y:        np.ndarray,
    device:   "torch.device",
    n_repeats: int = 5,
) -> np.ndarray:
    """
    置换重要性：对每个特征 f，在 batch 维度随机打乱 X_seq[:, :, f]，
    计算预测均方误差相对基准的增量。增量越大 → 该特征越重要。

    参数
    ────
    model      已训练的 _GRUNet，eval 模式
    X_seq      (N, seq_len, F) 验证序列（numpy float32）
    y          (N,) 标签（numpy float32）
    device     torch device
    n_repeats  每个特征重复打乱次数（降低随机方差）

    返回
    ────
    importances  (F,) float64，值 >= 0
    """
    model.eval()
    n_features = X_seq.shape[2]
    rng = np.random.RandomState(42)

    with torch.no_grad():
        base_pred = (model(torch.from_numpy(X_seq).to(device))
                     .cpu().numpy().astype(np.float64))
    baseline_mse = np.mean((base_pred - y.astype(np.float64)) ** 2)

    importances = np.zeros(n_features, dtype=np.float64)
    for f in range(n_features):
        scores = []
        for _ in range(n_repeats):
            X_perm = X_seq.copy()
            # 在 batch 维度打乱第 f 个特征的所有时间步 → 破坏特征与标签映射
            perm = rng.permutation(len(X_seq))
            X_perm[:, :, f] = X_seq[perm, :, f]
            with torch.no_grad():
                perm_pred = (model(torch.from_numpy(X_perm).to(device))
                             .cpu().numpy().astype(np.float64))
            perm_mse = np.mean((perm_pred - y.astype(np.float64)) ** 2)
            scores.append(max(0.0, perm_mse - baseline_mse))
        importances[f] = float(np.mean(scores))

    return importances


def _fit_gru(
    X_seq_tr:  np.ndarray,
    y_tr:      np.ndarray,
    X_seq_val: np.ndarray,
    y_val:     np.ndarray,
    n_features: int,
    hp:         dict,
    device:    "torch.device",
) -> "_GRUNet":
    """
    单次 GRU 训练（含早停）。

    参数
    ────
    X_seq_tr / y_tr    训练集序列与标签
    X_seq_val / y_val  验证集序列与标签（用于早停）
    n_features         特征维度（已经过特征选择）
    hp                 超参字典（从 GRU_DEFAULTS + args 合并）
    device             torch device

    返回
    ────
    best_model  val_loss 最低时的模型（深拷贝权重）
    """
    import copy

    model = _GRUNet(
        n_features  = n_features,
        hidden_size = hp["hidden_size"],
        num_layers  = hp["num_layers"],
        dropout     = hp["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["gru_lr"], weight_decay=hp["gru_wd"]
    )
    # 余弦退火，让学习率在 epochs 结束时降到 gru_lr/10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hp["gru_epochs"], eta_min=hp["gru_lr"] / 10
    )
    loss_fn   = nn.HuberLoss(delta=1.0)   # 比 MSE 更鲁棒（对异常标签）

    # DataLoader
    ds_tr  = TensorDataset(
        torch.from_numpy(X_seq_tr).to(device),
        torch.from_numpy(y_tr).to(device),
    )
    loader = DataLoader(ds_tr, batch_size=hp["gru_batch"], shuffle=True, drop_last=False)

    X_val_t = torch.from_numpy(X_seq_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_val   = math.inf
    best_state = copy.deepcopy(model.state_dict())
    patience   = hp["gru_patience"]
    no_imp     = 0

    for epoch in range(1, hp["gru_epochs"] + 1):
        model.train()
        tr_loss = 0.0
        for Xb, yb in loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            tr_loss += loss.item() * len(Xb)
        tr_loss /= len(ds_tr)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()

        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_imp     = 0
        else:
            no_imp += 1

        if epoch % max(1, hp["gru_epochs"] // 5) == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"      epoch {epoch:3d}/{hp['gru_epochs']}  "
                  f"tr={tr_loss:.5f}  val={val_loss:.5f}  "
                  f"best={best_val:.5f}  lr={lr_now:.2e}  "
                  f"no_imp={no_imp}/{patience}")

        if no_imp >= patience:
            print(f"      早停 @ epoch {epoch}  best_val={best_val:.5f}")
            break

    model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 核心训练函数（两阶段，镜像 utils_xgb._train_xgb）
# ══════════════════════════════════════════════════════════════════════════════

def _train_gru(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_scaler:     bool = True,
    top_n_features: int  = 0,
    device:         Optional["torch.device"] = None,
    **gru_kwargs,
) -> Tuple["_GRUNet", Optional[RobustScaler], List[str]]:
    """
    两阶段 GRU 训练（与 _train_xgb 结构完全对应）

    Phase 1  全特征序列 → 训练完整模型 → 置换重要性排名
    Phase 2  选 top-N 特征 → 独立 scaler + 最终模型（含早停）

    参数
    ────
    X_train          训练特征（pd.DataFrame，平坦，行=bar）
    y_train          标签（pd.Series）
    use_scaler       是否 RobustScaler 标准化（强烈建议开启）
    top_n_features   选前 N 重要特征；0 或 >= 总特征数 = 使用全部
    device           torch.device；None 则自动检测
    **gru_kwargs     覆盖 GRU_DEFAULTS 中的超参

    返回
    ────
    (model, scaler, selected_cols)
    model          _GRUNet，eval 模式，已加载最优权重
    scaler         RobustScaler 或 None
    selected_cols  最终使用的特征列名列表
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch 未安装，请 pip install torch")

    device = device or _get_device()
    hp     = {**GRU_DEFAULTS, **gru_kwargs}

    all_cols = list(X_train.columns)
    n_total  = len(X_train)
    seq_len  = hp["seq_len"]
    val_n    = max(seq_len + 1, int(n_total * hp["val_ratio"]))

    # ── 标准化（Phase 1 用）────────────────────────────────────────────────
    X_arr = X_train.values.astype(np.float32)
    y_arr = y_train.values.astype(np.float32)

    scaler_all: Optional[RobustScaler] = None
    if use_scaler:
        scaler_all = RobustScaler()
        X_arr = scaler_all.fit_transform(X_arr).astype(np.float32)

    # ── 构建序列 ────────────────────────────────────────────────────────────
    X_seq, y_seq = _build_sequences(X_arr, y_arr, seq_len)
    if len(X_seq) < 30:
        raise ValueError(f"序列样本不足（{len(X_seq)} < 30），请增大训练窗口或减小 seq_len")

    # 切分训练/验证（验证集取末尾 val_ratio）
    val_seq   = max(seq_len + 1, int(len(X_seq) * hp["val_ratio"]))
    tr_seq    = len(X_seq) - val_seq
    if tr_seq < 10:
        raise ValueError(f"训练序列太少（{tr_seq}），请增大训练窗口")

    X_tr, y_tr   = X_seq[:tr_seq],  y_seq[:tr_seq]
    X_val, y_val = X_seq[tr_seq:],  y_seq[tr_seq:]

    # ════════════════════════════════════════════════════════════════════════
    # Phase 1: 全特征 → 置换重要性
    # ════════════════════════════════════════════════════════════════════════
    do_select = (top_n_features > 0) and (top_n_features < len(all_cols))

    print(f"    ===== GRU Phase1: 全特征训练 ({len(all_cols)} 个) =====")
    model_all = _fit_gru(X_tr, y_tr, X_val, y_val, len(all_cols), hp, device)
    model_all.eval()

    if do_select:
        print(f"    [置换重要性] 使用验证集 {len(X_val)} 个序列，重复 {hp['perm_repeats']} 次 ...")
        imps = _perm_importance(model_all, X_val, y_val, device, hp["perm_repeats"])
        imp_series = pd.Series(imps, index=all_cols).sort_values(ascending=False)

        print(f"    ===== GRU 置换重要性（全特征）=====")
        for feat, imp in imp_series.head(min(8, len(all_cols))).items():
            norm_imp = imp / (imp_series.iloc[0] + 1e-12)
            bar = "█" * int(norm_imp * 20)
            print(f"    {feat:<40s}  {imp:.6f}  {bar}")
        if len(all_cols) > 8:
            print(f"    ... 共 {len(all_cols)} 个特征")
    else:
        imp_series = pd.Series(np.ones(len(all_cols)), index=all_cols)

    # ════════════════════════════════════════════════════════════════════════
    # Phase 2: 特征选择 + 最终模型
    # ════════════════════════════════════════════════════════════════════════
    if do_select:
        selected_cols = imp_series.head(top_n_features).index.tolist()
        tag = f"top {len(selected_cols)}/{len(all_cols)}"

        # 重新用选定列标准化
        col_idx  = [all_cols.index(c) for c in selected_cols]
        X_sel    = X_train[selected_cols].values.astype(np.float32)
        scaler: Optional[RobustScaler] = None
        if use_scaler:
            scaler = RobustScaler()
            X_sel  = scaler.fit_transform(X_sel).astype(np.float32)

        X_seq2, y_seq2 = _build_sequences(X_sel, y_arr, seq_len)
        val_seq2  = max(seq_len + 1, int(len(X_seq2) * hp["val_ratio"]))
        tr_seq2   = len(X_seq2) - val_seq2

        X_tr2, y_tr2   = X_seq2[:tr_seq2],  y_seq2[:tr_seq2]
        X_val2, y_val2 = X_seq2[tr_seq2:],  y_seq2[tr_seq2:]

        print(f"    ===== GRU Phase2: 最终训练 ({tag}) =====")
        model = _fit_gru(X_tr2, y_tr2, X_val2, y_val2, len(selected_cols), hp, device)

        # 打印最终模型置换重要性（Phase2 子集）
        model.eval()
        final_imps  = _perm_importance(model, X_val2, y_val2, device, hp["perm_repeats"])
        final_imp_s = pd.Series(final_imps, index=selected_cols).sort_values(ascending=False)
        print(f"    ===== GRU 最终特征重要性 ({tag}) =====")
        for feat, imp in final_imp_s.items():
            norm_imp = imp / (final_imp_s.iloc[0] + 1e-12)
            bar = "█" * int(norm_imp * 20)
            print(f"    {feat:<40s}  {imp:.6f}  {bar}")
    else:
        selected_cols = all_cols
        scaler        = scaler_all
        model         = model_all
        tag           = f"全部 {len(all_cols)}"
        print(f"    ===== GRU 最终特征重要性 (全部 {len(all_cols)}) =====")
        imp_normed = imp_series / (imp_series.iloc[0] + 1e-12)
        for feat, imp in imp_series.head(min(8, len(all_cols))).items():
            bar = "█" * int(imp_normed[feat] * 20)
            print(f"    {feat:<40s}  {imp:.6f}  {bar}")

    model.eval()
    return model, scaler, selected_cols


# ══════════════════════════════════════════════════════════════════════════════
# 滑动窗口回测（GRU 版，镜像 utils_xgb._run_xgb_sliding_window）
# ══════════════════════════════════════════════════════════════════════════════

def _run_gru_sliding_window(
    factor_data: pd.DataFrame,
    price_data:  pd.Series,
    args,
) -> Tuple[pd.DataFrame, dict]:
    """
    与 _run_xgb_sliding_window 结构完全对应。

    GRU 独有逻辑
    ────────────
    · 每次预测时需要 seq_len 根历史 bar 的因子矩阵。
      预测 bar abs_i 需要 factor_data.iloc[abs_i-seq_len+1 : abs_i+1]。
    · 样本内回填：bar abs_i in [seq_len-1, tw-1]（前 seq_len-1 根无法建序列）
    · 批量样本外：[rp, pred_end)，由于 rp >= tw >> seq_len，历史始终充足。
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch 未安装，请 pip install torch")

    n        = len(factor_data)
    tw       = args.train_window
    fwd      = args.fwd
    freq     = args.retrain_freq
    mode     = args.mode
    use_sc   = getattr(args, "use_scaler",     True)
    top_n    = getattr(args, "top_n_features", 0)
    seq_len  = getattr(args, "seq_len",        GRU_DEFAULTS["seq_len"])

    device = _get_device()
    print(f"    [GRU] device={device}  seq_len={seq_len}  top_n={top_n or '全部'}")

    # GRU 超参（从 args 读，缺省用 GRU_DEFAULTS）
    gru_kw = {k: getattr(args, k, v) for k, v in GRU_DEFAULTS.items()}

    predictions   = np.full(n, np.nan)
    model = scaler = None
    selected_cols  = list(factor_data.columns)

    first_model = first_scaler = first_cols = None
    in_sample_done = False

    # ── 预计算标签（一次性）────────────────────────────────────────────────
    all_labels = _compute_reversal_labels(
        price_data, fwd,
        check_days = getattr(args, "check_days", 5),
        multiplier = getattr(args, "multiplier", 1.2),
    )
    factor_arr = factor_data.values.astype(np.float32)
    nan_mask   = (
        np.isnan(factor_arr).any(axis=1)
        | np.isinf(factor_arr).any(axis=1)
    )

    retrain_pts = [i for i in range(tw, n) if (i - tw) % freq == 0]

    for rp_idx, rp in enumerate(tqdm(retrain_pts, desc=f"GRU {mode}回测")):
        train_end   = rp - fwd - 5
        if train_end <= seq_len:
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

        min_samples = seq_len * 3 + 50
        if valid.sum() < min_samples:
            print(f"    样本不足 ({valid.sum()} < {min_samples})，跳过"); continue

        print(f"  [{rp}/{n}] {mode}[{train_start}:{train_end}]  n={valid.sum()}")

        # ── 训练 ──────────────────────────────────────────────────────────
        try:
            model, scaler, selected_cols = _train_gru(
                Xv, yv, use_sc, top_n, device, **gru_kw
            )
        except Exception as e:
            print(f"    [警告] GRU 训练失败: {e}，跳过本折"); continue

        if first_model is None:
            first_model  = model
            first_scaler = scaler
            first_cols   = selected_cols

        # ── 回填样本内预测（仅首次）──────────────────────────────────────
        if not in_sample_done and first_model is not None:
            print(f"    [回填] 样本内 bars [{seq_len - 1}, {tw})")
            first_model.eval()
            for abs_i in range(seq_len - 1, tw):
                if nan_mask[abs_i]:
                    continue
                try:
                    # 取历史 seq_len 根 bar（可能包含 NaN 前的 bars，跳过）
                    hist_start = abs_i - seq_len + 1
                    x_hist = factor_data.iloc[hist_start:abs_i + 1][first_cols].values
                    if np.isnan(x_hist).any() or np.isinf(x_hist).any():
                        continue
                    x_hist = x_hist.astype(np.float32)
                    if first_scaler:
                        x_hist = first_scaler.transform(x_hist).astype(np.float32)
                    x_t = torch.from_numpy(x_hist[np.newaxis]).to(device)  # (1, seq, F)
                    with torch.no_grad():
                        predictions[abs_i] = first_model(x_t).item()
                except Exception:
                    pass
            in_sample_done = True

        # ── 批量样本外预测（当前 rp → 下一 rp）──────────────────────────
        pred_end = retrain_pts[rp_idx + 1] if rp_idx + 1 < len(retrain_pts) else n
        print(f"    [批量预测] bars [{rp}, {pred_end})")
        model.eval()

        valid_abs  = [i for i in range(rp, pred_end)
                      if not nan_mask[i] and i >= seq_len - 1]
        if not valid_abs:
            continue

        # 一次性组装整批序列，避免逐 bar 循环
        try:
            X_batch_list = []
            for abs_i in valid_abs:
                hist_start = abs_i - seq_len + 1
                x_hist = factor_data.iloc[hist_start:abs_i + 1][selected_cols].values.astype(np.float32)
                if scaler:
                    x_hist = scaler.transform(x_hist).astype(np.float32)
                X_batch_list.append(x_hist)

            X_batch = np.stack(X_batch_list, axis=0)   # (N_valid, seq_len, F)
            X_t     = torch.from_numpy(X_batch).to(device)

            # 分 mini-batch 预测（大批量防显存溢出）
            chunk  = 1024
            preds_list = []
            for s in range(0, len(X_t), chunk):
                with torch.no_grad():
                    preds_list.append(model(X_t[s:s + chunk]).cpu().numpy())
            preds_arr = np.concatenate(preds_list)

            for local_i, abs_i in enumerate(valid_abs):
                predictions[abs_i] = preds_arr[local_i]

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
# 公开入口（接口与 run_backtest_reg / run_backtest_xgb 完全一致）
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest_gru(
    factor_data: pd.DataFrame,
    price_data:  pd.Series,
    args,
) -> Tuple[pd.DataFrame, dict]:
    """
    GRU 版回测入口，接口与 run_backtest_reg / run_backtest_xgb 完全相同。

    args 额外读取字段（WLS / XGB 不需要）
    ──────────────────────────────────────
    seq_len          int    GRU 时间步（回看 bar 数）     default: 20
    hidden_size      int    GRU 隐藏层维度               default: 64
    num_layers       int    堆叠 GRU 层数                default: 2
    dropout          float  Dropout 比例                 default: 0.2
    gru_epochs       int    最大训练 epoch               default: 100
    gru_batch        int    mini-batch 大小              default: 256
    gru_patience     int    早停耐心 epoch 数            default: 10
    gru_lr           float  Adam 学习率                  default: 1e-3
    gru_wd           float  Adam 权重衰减（L2）          default: 1e-4
    val_ratio        float  验证集比例                   default: 0.15
    perm_repeats     int    置换重要性重复次数            default: 5
    top_n_features   int    选前 N 重要特征（0=全部）    default: 0
    """
    args.train_window = min(args.train_window, len(factor_data) - 1)
    args.split_point  = factor_data.index[args.train_window - 1]
    return _run_gru_sliding_window(factor_data, price_data, args)