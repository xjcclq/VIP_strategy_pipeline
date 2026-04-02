"""
滚动 Spearman IC —— 向量化加速版
原版：190万次 Python 循环，约需 10~20 分钟
本版：矩阵运算，预计 10~30 秒

核心思路：
  Spearman IC = Pearson(rank(factor), rank(ret))
  在每个窗口内对因子矩阵和收益率同时做 rank，
  然后用矩阵乘法一次算出所有因子的相关系数。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# ── 配置 ──────────────────────────────────────────────────────────────
CSV_FILE    = r"F:\VIP\data\P_60min_with_ma_features_from_scratch.csv"
OUTPUT_DIR  = r"F:\VIP\output"
WINDOW      = 4000
N_PER_GROUP = 3
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 核心函数：向量化滚动 Spearman IC ─────────────────────────────────
def vectorized_rolling_ic(factor_arr, ret_arr, window):
    """
    factor_arr : shape (T, N)  —— T个时间点，N个因子
    ret_arr    : shape (T,)    —— T个收益率
    window     : 滚动窗口大小

    返回 ic_matrix: shape (T-window, N)
    每行 = 该时间点用过去 window 个样本算出的 IC 向量
    """
    T, N = factor_arr.shape
    steps = T - window
    ic_matrix = np.zeros((steps, N), dtype=np.float32)

    for i in range(steps):
        # 取窗口内数据
        F = factor_arr[i: i + window]   # (window, N)
        r = ret_arr[i: i + window]       # (window,)

        # ── 对每列做 rank（沿 axis=0）──
        # argsort两次 = rank，复杂度 O(window * N * log(window))
        F_rank = np.argsort(np.argsort(F, axis=0), axis=0).astype(np.float32)
        r_rank = np.argsort(np.argsort(r)).astype(np.float32)  # (window,)

        # ── 中心化 ──
        F_rank -= F_rank.mean(axis=0)          # (window, N)
        r_rank -= r_rank.mean()                 # (window,)

        # ── 计算相关系数（向量化，一次算所有N个因子）──
        # 分子：r_rank @ F_rank = shape (N,)
        numerator   = r_rank @ F_rank           # (N,)
        # 分母：每个因子列的标准差 × r_rank 的标准差
        denom_F     = np.sqrt((F_rank ** 2).sum(axis=0))   # (N,)
        denom_r     = np.sqrt((r_rank ** 2).sum())          # scalar
        denom       = denom_F * denom_r + 1e-9

        ic_matrix[i] = numerator / denom

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{steps}")

    return ic_matrix


# ── 1. 加载 & 预处理 ──────────────────────────────────────────────────
df = pd.read_csv(CSV_FILE)
print(f"读取数据: {df.shape}")

x_cols = [c for c in df.columns if c.startswith('x_')]
print(f"因子数量: {len(x_cols)}")

# df['ret_fwd'] = df['y_open'].shift(-2) / df['y_open'].shift(-1) - 1
df['ret_fwd'] = df['y_open'].shift(-4) / df['y_open'].shift(-1) - 1
df = df.iloc[:-2].reset_index(drop=True)
print(f"有效样本: {df.shape[0]}")

df[x_cols] = df[x_cols].ffill().fillna(0)

factor_arr = df[x_cols].values.astype(np.float32)   # (T, N)
ret_arr    = df['ret_fwd'].values.astype(np.float32) # (T,)
time_index = df['datetime'].iloc[WINDOW:].values


# ── 2. 计算滚动 IC ────────────────────────────────────────────────────
import time
print(f"\n开始计算滚动 IC（向量化版），window={WINDOW}...")
t0 = time.time()

ic_matrix = vectorized_rolling_ic(factor_arr, ret_arr, WINDOW)

elapsed = time.time() - t0
print(f"完成！耗时 {elapsed:.1f} 秒")

ic_df = pd.DataFrame(ic_matrix, index=time_index, columns=x_cols)
ic_df.index.name = 'datetime'


# ── 3. 输出1：IC 排名表 ───────────────────────────────────────────────
summary = pd.DataFrame({
    'IC_mean':     ic_df.mean(),
    'IC_std':      ic_df.std(),
    'IC_abs_mean': ic_df.abs().mean(),
    'ICIR':        ic_df.mean() / (ic_df.std() + 1e-9),
    'valid_rate':  (ic_df.abs() > 0.02).mean(),
}).sort_values('IC_mean', ascending=False)

out_csv = os.path.join(OUTPUT_DIR, "ic_summary_rank.csv")
summary.to_csv(out_csv)
print(f"\nTop 10 稳定因子：")
print(summary.head(10).to_string())
print(f"排名表已保存: {out_csv}")


# ── 4. 输出2：分组 IC 曲线图 ──────────────────────────────────────────
all_factors = summary.index.tolist()
n_f = len(all_factors)
q   = n_f // 4

groups = {
    'G1 强正向 (Top 25%)':    all_factors[:q],
    'G2 弱正向 (25%~50%)':    all_factors[q: 2*q],
    'G3 弱负向 (50%~75%)':    all_factors[2*q: 3*q],
    'G4 强负向 (Bottom 25%)': all_factors[3*q:],
}
group_colors = {
    'G1 强正向 (Top 25%)':    ['#1a6e1a', '#3ab53a', '#7de07d'],
    'G2 弱正向 (25%~50%)':    ['#1a4a9e', '#4a7fd4', '#90b8f0'],
    'G3 弱负向 (50%~75%)':    ['#c07000', '#e8a020', '#f8d080'],
    'G4 强负向 (Bottom 25%)': ['#a01010', '#d04040', '#f08080'],
}

group_reps = {}
for gname, members in groups.items():
    sub  = summary.loc[members].copy()
    sub['ICIR_abs'] = sub['ICIR'].abs()
    reps = sub.sort_values('ICIR_abs', ascending=False).head(N_PER_GROUP).index.tolist()
    group_reps[gname] = reps

total_steps = len(ic_df)
tick_step   = max(1, total_steps // 8)
tick_pos    = list(range(0, total_steps, tick_step))
tick_label  = [str(time_index[p])[:13] for p in tick_pos]

fig, axes = plt.subplots(4, N_PER_GROUP, figsize=(8 * N_PER_GROUP, 5 * 4))
for row_idx, (gname, reps) in enumerate(group_reps.items()):
    colors = group_colors[gname]
    for col_idx, factor in enumerate(reps):
        ax  = axes[row_idx][col_idx]
        ser = ic_df[factor].values
        c   = colors[col_idx]

        ax.plot(ser, linewidth=0.9, color=c, alpha=0.9)
        ax.axhline(0,     color='black', linewidth=0.5)
        ax.axhline( 0.02, color='grey',  linewidth=0.7, linestyle='--', alpha=0.6)
        ax.axhline(-0.02, color='grey',  linewidth=0.7, linestyle='--', alpha=0.6)
        ax.fill_between(range(len(ser)),  0.02, ser,
                        where=ser >  0.02, color='green', alpha=0.15)
        ax.fill_between(range(len(ser)), -0.02, ser,
                        where=ser < -0.02, color='red',   alpha=0.15)

        ic_m  = summary.loc[factor, 'IC_mean']
        icir  = summary.loc[factor, 'ICIR']
        vrate = summary.loc[factor, 'valid_rate']
        ax.set_title(
            f"[{gname[:2]}] {factor}\n"
            f"IC_mean={ic_m:.4f}  ICIR={icir:.2f}  有效率={vrate:.1%}",
            fontsize=8, pad=4
        )
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_label, rotation=30, ha='right', fontsize=6)
        ax.set_ylabel("IC", fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.3)

plt.suptitle(
    f"因子 IC 分组曲线（window={WINDOW}，每组取 ICIR 最稳定的 {N_PER_GROUP} 个）",
    fontsize=13, y=1.003
)
plt.tight_layout()
out_curves = os.path.join(OUTPUT_DIR, "ic_grouped_curves.png")
plt.savefig(out_curves, dpi=130, bbox_inches='tight')
plt.close()
print(f"分组曲线已保存: {out_curves}")


# ── 5. 输出3：Top 20 IC 热力图 ────────────────────────────────────────
top20    = summary.sort_values('IC_abs_mean', ascending=False).head(20).index.tolist()
ic_top20 = ic_df[top20]

sample_step = max(1, total_steps // 200)
ic_sampled  = ic_top20.iloc[::sample_step, :]

fig, ax = plt.subplots(figsize=(20, 8))
im = ax.imshow(
    ic_sampled.T.values,
    aspect='auto', cmap='RdYlGn',
    vmin=-0.08, vmax=0.08,
    interpolation='nearest'
)
plt.colorbar(im, ax=ax, label='Spearman IC', fraction=0.015, pad=0.01)

ax.set_yticks(range(20))
ax.set_yticklabels(top20, fontsize=8)

n_t      = ic_sampled.shape[0]
x_step   = max(1, n_t // 10)
x_ticks  = list(range(0, n_t, x_step))
x_labels = [str(ic_sampled.index[p])[:13] for p in x_ticks]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=7)

for yi, factor in enumerate(top20):
    ic_m = summary.loc[factor, 'IC_mean']
    ax.text(-0.5, yi, f" {ic_m:+.4f}", va='center', ha='right',
            fontsize=6.5, color='navy')

ax.set_title(
    f"IC 热力图 — Top 20 因子（|IC_mean| 排序，window={WINDOW}）\n"
    f"深绿=持续正向有效  深红=持续负向有效  白/黄=无效期",
    fontsize=10
)
plt.tight_layout()
out_heatmap = os.path.join(OUTPUT_DIR, "ic_heatmap_top20.png")
plt.savefig(out_heatmap, dpi=150, bbox_inches='tight')
plt.close()
print(f"热力图已保存: {out_heatmap}")


# ── 完成 ──────────────────────────────────────────────────────────────
print("\n========== 全部完成 ==========")
print(f"  {out_csv}")
print(f"  {out_curves}")
print(f"  {out_heatmap}")
print("\n结果解读：")
print("  ic_summary_rank.csv  → ICIR > 0.3 的因子值得重点关注")
print("  ic_grouped_curves    → 曲线近期下降 = 因子正在衰减")
print("  ic_heatmap_top20     → 出现横向色块断层 = 该因子某段时间集体失效")