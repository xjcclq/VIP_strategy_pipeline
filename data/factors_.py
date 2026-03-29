"""
因子计算模块
============
所有因子函数通用于任意周期，自动在列名末尾附加周期标识。
命名规则: x_{因子名}_{周期}  例如 x_rsi14_10min, x_vwap_60min

Tick级因子（输入tick原始数据 + freq）:
  - calc_tick_delta           从tick合成delta/买卖量
  - calc_tick_vwap            从tick合成真实VWAP及衍生
  - calc_tick_er              从tick计算效率比率(Efficiency Ratio)
  - calc_tick_run             从tick计算价格连续同向变动比例(Tick Run)

Bar级因子（输入bar数据 + freq）:
  - calc_ma_factors          均线类
  - calc_momentum_factors    动量类
  - calc_volatility_factors  波动率类
  - calc_technical_factors   技术指标 (RSI/MACD/Bollinger/ADX...)
  - calc_volume_factors      成交量类 (OBV/MFI/量价相关...)
  - calc_advanced_factors    高级因子 (偏度/Hurst/分形维数...)
  - calc_delta_factors       Delta订单流因子 (需x_delta列)
  - calc_breakout_factors    突破类因子
  - calc_vpin_factors        VPIN毒性因子 (Volume-Synchronized Probability of Informed Trading)

  - calc_all_factors         一键全算bar级因子
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from bar_bucket_utils import (
    BAR_PERIOD_COL,
    build_trading_blocks,
    build_period_table,
    filter_to_trading_time,
    attach_period_info,
)

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


# ============================================================
# 工具
# ============================================================

def _col(name: str, freq: str) -> str:
    return f"x_{name}_{freq}"


def _f64(s: pd.Series) -> np.ndarray:
    return s.astype('float64').values


def zscore(s: pd.Series, window: int) -> pd.Series:
    return (s - s.rolling(window).mean()) / s.rolling(window).std().clip(lower=1e-9)


# ============================================================
# Tick级：Delta合成
# ============================================================

def _empty_tick_delta_frame() -> pd.DataFrame:
    cols = [
        'x_delta', 'x_volume', 'x_turnover', 'x_ofi',
        'x_open_interest', 'x_buy_vol', 'x_sell_vol',
        BAR_PERIOD_COL,
    ]
    return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name='datetime'))


def _calc_tick_delta_one_day(day_df: pd.DataFrame,
                             freq: str,
                             trading_day: pd.Timestamp) -> pd.DataFrame:
    """单交易日tick -> bar delta聚合。"""
    df = day_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')

    blocks = build_trading_blocks(pd.Timestamp(trading_day), source_ts=df.index.to_series())
    periods = build_period_table(blocks, freq)
    df = filter_to_trading_time(df, blocks)
    if df.empty or periods.empty:
        return _empty_tick_delta_frame()

    # 每tick增量
    df['vol_diff'] = df['volume'].diff().clip(lower=0).fillna(0)
    df['turnover_diff'] = df['total_turnover'].diff().clip(lower=0).fillna(0)

    # ── 修改：基于last价格与上一tick挂单价格比较 ─────────────────────────────
    # 获取上一tick的各档位价格和挂单量
    prev_b1 = df['b1'].shift(1)
    prev_a1 = df['a1'].shift(1)
    prev_b1_v = df['b1_v'].shift(1)
    prev_a1_v = df['a1_v'].shift(1)
    prev_b2 = df['b2'].shift(1)
    prev_a2 = df['a2'].shift(1)
    prev_b2_v = df['b2_v'].shift(1)
    prev_a2_v = df['a2_v'].shift(1)
    prev_b3 = df['b3'].shift(1)
    prev_a3 = df['a3'].shift(1)
    prev_b3_v = df['b3_v'].shift(1)
    prev_a3_v = df['a3_v'].shift(1)

    # 1. 买一卖一（L1）计算
    df['raw_sell'] = 0.0
    # 情形A：成交价下跌，完全吃掉买一档
    mask = df['last'] < prev_b1
    df.loc[mask, 'raw_sell'] = prev_b1_v[mask]
    # 情形B：成交价持平，部分吃掉买一档
    mask = (df['last'] == prev_b1) & (df['b1_v'] < prev_b1_v)
    df.loc[mask, 'raw_sell'] = (prev_b1_v - df['b1_v'])[mask]

    df['raw_buy'] = 0.0
    # 成交价上涨，完全吃掉卖一档
    mask = df['last'] > prev_a1
    df.loc[mask, 'raw_buy'] = prev_a1_v[mask]
    # 成交价持平，部分吃掉卖一档
    mask = (df['last'] == prev_a1) & (df['a1_v'] < prev_a1_v)
    df.loc[mask, 'raw_buy'] = (prev_a1_v - df['a1_v'])[mask]

    # 2. 买二卖二计算
    df['raw_sell_b2'] = 0.0
    mask = df['last'] < prev_b2
    df.loc[mask, 'raw_sell_b2'] = prev_b2_v[mask]
    mask = (df['last'] == prev_b2) & (df['b2_v'] < prev_b2_v)
    df.loc[mask, 'raw_sell_b2'] = (prev_b2_v - df['b2_v'])[mask]

    df['raw_buy_b2'] = 0.0
    mask = df['last'] > prev_a2
    df.loc[mask, 'raw_buy_b2'] = prev_a2_v[mask]
    mask = (df['last'] == prev_a2) & (df['a2_v'] < prev_a2_v)
    df.loc[mask, 'raw_buy_b2'] = (prev_a2_v - df['a2_v'])[mask]

    # 3. 买三卖三计算
    df['raw_sell_b3'] = 0.0
    mask = df['last'] < prev_b3
    df.loc[mask, 'raw_sell_b3'] = prev_b3_v[mask]
    mask = (df['last'] == prev_b3) & (df['b3_v'] < prev_b3_v)
    df.loc[mask, 'raw_sell_b3'] = (prev_b3_v - df['b3_v'])[mask]

    df['raw_buy_b3'] = 0.0
    mask = df['last'] > prev_a3
    df.loc[mask, 'raw_buy_b3'] = prev_a3_v[mask]
    mask = (df['last'] == prev_a3) & (df['a3_v'] < prev_a3_v)
    df.loc[mask, 'raw_buy_b3'] = (prev_a3_v - df['a3_v'])[mask]

    # ── 五档 L2 估算（保持不变，已使用last价格）───────────────────────────
    LEVELS = range(1, 6)
    prev_b = {i: df[f'b{i}'].shift(1) for i in LEVELS}
    prev_bv = {i: df[f'b{i}_v'].shift(1) for i in LEVELS}
    prev_a = {i: df[f'a{i}'].shift(1) for i in LEVELS}
    prev_av = {i: df[f'a{i}_v'].shift(1) for i in LEVELS}

    raw_sell_l2 = pd.Series(0.0, index=df.index)
    raw_buy_l2 = pd.Series(0.0, index=df.index)

    for i in LEVELS:
        mask_full_sell = df['last'] < prev_b[i]
        raw_sell_l2[mask_full_sell] += prev_bv[i][mask_full_sell]

        mask_part_sell = (df['last'] == prev_b[i]) & (df[f'b{i}_v'] < prev_bv[i])
        raw_sell_l2[mask_part_sell] += (prev_bv[i] - df[f'b{i}_v'])[mask_part_sell].clip(lower=0)

        mask_full_buy = df['last'] > prev_a[i]
        raw_buy_l2[mask_full_buy] += prev_av[i][mask_full_buy]

        mask_part_buy = (df['last'] == prev_a[i]) & (df[f'a{i}_v'] < prev_av[i])
        raw_buy_l2[mask_part_buy] += (prev_av[i] - df[f'a{i}_v'])[mask_part_buy].clip(lower=0)

    df['raw_sell_l2'] = raw_sell_l2
    df['raw_buy_l2'] = raw_buy_l2

    # 等比例缩放到实际成交量
    # L1（买一卖一）缩放
    total_raw = df['raw_buy'] + df['raw_sell']
    df['buy_vol'] = np.where(total_raw > 0, df['vol_diff'] * df['raw_buy'] / total_raw, 0.0)
    df['sell_vol'] = np.where(total_raw > 0, df['vol_diff'] * df['raw_sell'] / total_raw, 0.0)
    df['delta_contrib'] = df['buy_vol'] - df['sell_vol']

    # 买二卖二缩放
    total_raw_b2 = df['raw_buy_b2'] + df['raw_sell_b2']
    df['buy_vol_b2'] = np.where(total_raw_b2 > 0,
                                df['vol_diff'] * df['raw_buy_b2'] / total_raw_b2, 0.0)
    df['sell_vol_b2'] = np.where(total_raw_b2 > 0,
                                 df['vol_diff'] * df['raw_sell_b2'] / total_raw_b2, 0.0)
    df['delta_contrib_b2'] = df['buy_vol_b2'] - df['sell_vol_b2']

    # 买三卖三缩放
    total_raw_b3 = df['raw_buy_b3'] + df['raw_sell_b3']
    df['buy_vol_b3'] = np.where(total_raw_b3 > 0,
                                df['vol_diff'] * df['raw_buy_b3'] / total_raw_b3, 0.0)
    df['sell_vol_b3'] = np.where(total_raw_b3 > 0,
                                 df['vol_diff'] * df['raw_sell_b3'] / total_raw_b3, 0.0)
    df['delta_contrib_b3'] = df['buy_vol_b3'] - df['sell_vol_b3']

    # L2（五档）缩放
    total_raw_l2 = df['raw_buy_l2'] + df['raw_sell_l2']
    df['buy_vol_l2'] = np.where(total_raw_l2 > 0,
                                df['vol_diff'] * df['raw_buy_l2'] / total_raw_l2, 0.0)
    df['sell_vol_l2'] = np.where(total_raw_l2 > 0,
                                 df['vol_diff'] * df['raw_sell_l2'] / total_raw_l2, 0.0)
    df['delta_contrib_l2'] = df['buy_vol_l2'] - df['sell_vol_l2']

    assigned = attach_period_info(df, periods, closed='left')
    if assigned.empty:
        return _empty_tick_delta_frame()

    grouped = assigned.groupby('bar_end', sort=True)
    result = pd.DataFrame(index=grouped.size().index)
    result.index.name = 'datetime'

    # 基础字段
    result['x_volume'] = grouped['vol_diff'].sum()
    result['x_turnover'] = grouped['turnover_diff'].sum()
    result['x_open_interest'] = grouped['open_interest'].last()

    # 四种版本的delta和ofi
    result['x_delta_l1'] = grouped['delta_contrib'].sum()  # 买一卖一
    result['x_ofi_l1'] = (result['x_delta_l1'] / result['x_volume']).fillna(0)

    result['x_delta_l2'] = grouped['delta_contrib_l2'].sum()  # 五档
    result['x_ofi_l2'] = (result['x_delta_l2'] / result['x_volume']).fillna(0)

    result['x_delta_b2'] = grouped['delta_contrib_b2'].sum()  # 买二卖二
    result['x_ofi_b2'] = (result['x_delta_b2'] / result['x_volume']).fillna(0)

    result['x_delta_b3'] = grouped['delta_contrib_b3'].sum()  # 买三卖三
    result['x_ofi_b3'] = (result['x_delta_b3'] / result['x_volume']).fillna(0)

    result[BAR_PERIOD_COL] = grouped[BAR_PERIOD_COL].first()
    return result


# def _calc_tick_delta_one_day(day_df: pd.DataFrame,
#                              freq: str,
#                              trading_day: pd.Timestamp) -> pd.DataFrame:
#     """单交易日tick -> bar delta聚合。"""
#     df = day_df.copy()
#     df['datetime'] = pd.to_datetime(df['datetime'])
#     df = df.sort_values('datetime').set_index('datetime')
#
#     blocks = build_trading_blocks(pd.Timestamp(trading_day), source_ts=df.index.to_series())
#     periods = build_period_table(blocks, freq)
#     df = filter_to_trading_time(df, blocks)
#     if df.empty or periods.empty:
#         return _empty_tick_delta_frame()
#
#     # 每tick增量
#     df['vol_diff'] = df['volume'].diff().clip(lower=0).fillna(0)
#     df['turnover_diff'] = df['total_turnover'].diff().clip(lower=0).fillna(0)
#
#     # ── 原有逻辑：仅用买一 / 卖一 ──────────────────────────────────────────
#     prev_b1 = df['b1'].shift(1)
#     prev_a1 = df['a1'].shift(1)
#     prev_b1_v = df['b1_v'].shift(1)
#     prev_a1_v = df['a1_v'].shift(1)
#
#     df['raw_sell'] = 0.0
#     mask = df['b1'] < prev_b1
#     df.loc[mask, 'raw_sell'] = prev_b1_v[mask]
#     mask = (df['b1'] == prev_b1) & (df['b1_v'] < prev_b1_v)
#     df.loc[mask, 'raw_sell'] = (prev_b1_v - df['b1_v'])[mask]
#
#     df['raw_buy'] = 0.0
#     mask = df['a1'] > prev_a1
#     df.loc[mask, 'raw_buy'] = prev_a1_v[mask]
#     mask = (df['a1'] == prev_a1) & (df['a1_v'] < prev_a1_v)
#     df.loc[mask, 'raw_buy'] = (prev_a1_v - df['a1_v'])[mask]
#
#     # ── 新增：五档 L2 估算（中金"买五卖五"方法）───────────────────────────
#     #
#     # 核心规则（以主动卖出为例）：
#     #   - 当 last_t < prev_b_i  →  b_i 档被完全穿越，加入 prev_b_i_v
#     #   - 当 last_t == prev_b_i →  b_i 档被部分成交，加入 (prev_b_i_v - b_i_v).clip(0)
#     #   - 两个条件互斥（<  vs ==），五档独立判断后求和，不会重复计数
#     #   - 主动买入对称处理卖档 a1~a5
#     #
#     # 注意：bid 档位 b1 > b2 > b3 > b4 > b5，ask 档位 a1 < a2 < a3 < a4 < a5
#     # 因此 last < b2 时必然也 last < b1，逐档独立加和天然包含穿越效果。
#
#     LEVELS = range(1, 6)
#
#     # 预先缓存各档前值，避免在循环中重复 shift
#     prev_b = {i: df[f'b{i}'].shift(1) for i in LEVELS}
#     prev_bv = {i: df[f'b{i}_v'].shift(1) for i in LEVELS}
#     prev_a = {i: df[f'a{i}'].shift(1) for i in LEVELS}
#     prev_av = {i: df[f'a{i}_v'].shift(1) for i in LEVELS}
#
#     raw_sell_l2 = pd.Series(0.0, index=df.index)
#     raw_buy_l2  = pd.Series(0.0, index=df.index)
#
#     for i in LEVELS:
#         # ---- 主动卖出：last 穿越或停在第 i 买档 ----
#         # 完全穿越：last 严格低于第 i 买档价，该档挂单全部成交
#         mask_full_sell = df['last'] < prev_b[i]
#         raw_sell_l2[mask_full_sell] += prev_bv[i][mask_full_sell]
#
#         # 停在第 i 买档价：同价位挂单量减少部分为主动成交量
#         mask_part_sell = (df['last'] == prev_b[i]) & (df[f'b{i}_v'] < prev_bv[i])
#         raw_sell_l2[mask_part_sell] += (prev_bv[i] - df[f'b{i}_v'])[mask_part_sell].clip(lower=0)
#
#         # ---- 主动买入：last 穿越或停在第 i 卖档 ----
#         # 完全穿越：last 严格高于第 i 卖档价
#         mask_full_buy = df['last'] > prev_a[i]
#         raw_buy_l2[mask_full_buy] += prev_av[i][mask_full_buy]
#
#         # 停在第 i 卖档价：同价位挂单量减少部分
#         mask_part_buy = (df['last'] == prev_a[i]) & (df[f'a{i}_v'] < prev_av[i])
#         raw_buy_l2[mask_part_buy] += (prev_av[i] - df[f'a{i}_v'])[mask_part_buy].clip(lower=0)
#
#     df['raw_sell_l2'] = raw_sell_l2
#     df['raw_buy_l2']  = raw_buy_l2
#
#     # 等比例缩放到实际成交量（与原有 L1 方法保持一致）
#     total_raw_l2 = df['raw_buy_l2'] + df['raw_sell_l2']
#     df['buy_vol_l2']  = np.where(total_raw_l2 > 0,
#                                   df['vol_diff'] * df['raw_buy_l2']  / total_raw_l2, 0.0)
#     df['sell_vol_l2'] = np.where(total_raw_l2 > 0,
#                                   df['vol_diff'] * df['raw_sell_l2'] / total_raw_l2, 0.0)
#     df['delta_contrib_l2'] = df['buy_vol_l2'] - df['sell_vol_l2']
#     # ── L2 新增结束 ────────────────────────────────────────────────────────
#
#     # 原有 L1 缩放（保持不变）
#     total_raw = df['raw_buy'] + df['raw_sell']
#     df['buy_vol']  = np.where(total_raw > 0, df['vol_diff'] * df['raw_buy']  / total_raw, 0.0)
#     df['sell_vol'] = np.where(total_raw > 0, df['vol_diff'] * df['raw_sell'] / total_raw, 0.0)
#     df['delta_contrib'] = df['buy_vol'] - df['sell_vol']
#
#     assigned = attach_period_info(df, periods, closed='left')
#     if assigned.empty:
#         return _empty_tick_delta_frame()
#
#     grouped = assigned.groupby('bar_end', sort=True)
#     result = pd.DataFrame(index=grouped.size().index)
#     result.index.name = 'datetime'
#
#     # 原有字段
#     result['x_delta']         = grouped['delta_contrib'].sum()
#     result['x_volume']        = grouped['vol_diff'].sum()
#     result['x_turnover']      = grouped['turnover_diff'].sum()
#     result['x_open_interest'] = grouped['open_interest'].last()
#     result['x_buy_vol']       = grouped['buy_vol'].sum()
#     result['x_sell_vol']      = grouped['sell_vol'].sum()
#     result['x_ofi']           = (result['x_buy_vol'] - result['x_sell_vol']) / (
#                                     result['x_buy_vol'] + result['x_sell_vol'])
#
#     # ── 新增 L2 字段 ───────────────────────────────────────────────────────
#     result['x_buy_vol_l2']  = grouped['buy_vol_l2'].sum()
#     result['x_sell_vol_l2'] = grouped['sell_vol_l2'].sum()
#     result['x_delta_l2']    = grouped['delta_contrib_l2'].sum()   # ← 主角
#     result['x_ofi_l2']      = (result['x_buy_vol_l2'] - result['x_sell_vol_l2']) / (
#                                     result['x_buy_vol_l2'] + result['x_sell_vol_l2'])
#     # ── 新增结束 ────────────────────────────────────────────────────────────
#
#     result[BAR_PERIOD_COL] = grouped[BAR_PERIOD_COL].first()
#     return result


def _calc_tick_delta_legacy(tick_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    从tick数据合成指定周期的 delta / 买卖量 / 成交量 / 成交额 / 持仓量。
    规则与主流程一致: 先过滤非交易时段，再按交易时长分桶（扣除休市时段）。
    """
    df = tick_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    if 'trading_date' in df.columns:
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        groups = df.groupby(df['trading_date'].dt.date)
    else:
        groups = df.groupby(df['datetime'].dt.date)

    bars = []
    for date, day_df in groups:
        day_bar = _calc_tick_delta_one_day(day_df, freq, pd.Timestamp(date))
        if day_bar.empty:
            continue
        bars.append(day_bar)

    if not bars:
        return _empty_tick_delta_frame()

    return pd.concat(bars).sort_index()


def _empty_tick_factor_bar_frame() -> pd.DataFrame:
    cols = [
        'open', 'high', 'low', 'y_close',
        'x_volume', 'x_turnover', 'x_open_interest',
        'x_vwap_pv', 'x_vwap_v', 'x_bvwap_pv', 'x_bvwap_v',
        'x_delta', 'x_ofi',
        'x_delta_l1', 'x_ofi_l1',
        'x_delta_l2', 'x_ofi_l2',
        'x_delta_b2', 'x_ofi_b2',
        'x_delta_b3', 'x_ofi_b3',
        BAR_PERIOD_COL,
    ]
    return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name='datetime'))


def _empty_tick_delta_frame() -> pd.DataFrame:
    cols = [
        'x_volume', 'x_turnover', 'x_open_interest',
        'x_delta', 'x_ofi',
        'x_delta_l1', 'x_ofi_l1',
        'x_delta_l2', 'x_ofi_l2',
        'x_delta_b2', 'x_ofi_b2',
        'x_delta_b3', 'x_ofi_b3',
        BAR_PERIOD_COL,
    ]
    return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name='datetime'))


def _estimate_raw_sell_by_queue(curr_price: pd.Series,
                                curr_volume: pd.Series,
                                prev_price: pd.Series,
                                prev_volume: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=curr_price.index, dtype='float64')
    mask = curr_price < prev_price
    out.loc[mask] = prev_volume.loc[mask]
    mask = (curr_price == prev_price) & (curr_volume < prev_volume)
    out.loc[mask] = (prev_volume - curr_volume).loc[mask]
    return out.fillna(0.0)


def _estimate_raw_buy_by_queue(curr_price: pd.Series,
                               curr_volume: pd.Series,
                               prev_price: pd.Series,
                               prev_volume: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=curr_price.index, dtype='float64')
    mask = curr_price > prev_price
    out.loc[mask] = prev_volume.loc[mask]
    mask = (curr_price == prev_price) & (curr_volume < prev_volume)
    out.loc[mask] = (prev_volume - curr_volume).loc[mask]
    return out.fillna(0.0)


def _allocate_delta_by_volume(vol_diff: pd.Series,
                              raw_buy: pd.Series,
                              raw_sell: pd.Series) -> pd.Series:
    total_raw = raw_buy + raw_sell
    buy = np.divide(
        vol_diff.values * raw_buy.values,
        total_raw.values,
        out=np.zeros(len(vol_diff), dtype='float64'),
        where=total_raw.values > 0,
    )
    sell = np.divide(
        vol_diff.values * raw_sell.values,
        total_raw.values,
        out=np.zeros(len(vol_diff), dtype='float64'),
        where=total_raw.values > 0,
    )
    return pd.Series(buy - sell, index=vol_diff.index, dtype='float64')


def _has_nonzero_book_level(df: pd.DataFrame, level: int) -> bool:
    cols = [f'b{level}', f'a{level}', f'b{level}_v', f'a{level}_v']
    if not all(col in df.columns for col in cols):
        return False
    return any(df[col].fillna(0).ne(0).any() for col in cols)


def _calc_tick_factor_bar_one_day(day_df: pd.DataFrame,
                                  freq: str,
                                  trading_day: pd.Timestamp) -> pd.DataFrame:
    df = day_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')

    blocks = build_trading_blocks(pd.Timestamp(trading_day), source_ts=df.index.to_series())
    periods = build_period_table(blocks, freq)
    df = filter_to_trading_time(df, blocks)
    if df.empty or periods.empty:
        return _empty_tick_factor_bar_frame()

    vol_diff = df['volume'].astype('float64').diff().clip(lower=0)
    vol_diff.iloc[0] = max(float(df['volume'].iloc[0]), 0.0)
    df['vol_diff'] = vol_diff

    turnover = df['total_turnover'].astype('float64')
    turnover_diff = turnover.diff().clip(lower=0)
    turnover_diff.iloc[0] = max(float(turnover.iloc[0]), 0.0)
    df['turnover_diff'] = turnover_diff

    raw_sell_l1 = _estimate_raw_sell_by_queue(df['b1'], df['b1_v'], df['b1'].shift(1), df['b1_v'].shift(1))
    raw_buy_l1 = _estimate_raw_buy_by_queue(df['a1'], df['a1_v'], df['a1'].shift(1), df['a1_v'].shift(1))
    delta_l1 = _allocate_delta_by_volume(df['vol_diff'], raw_buy_l1, raw_sell_l1)

    has_b2 = _has_nonzero_book_level(df, 2)
    if has_b2:
        raw_sell_b2 = _estimate_raw_sell_by_queue(df['b2'], df['b2_v'], df['b2'].shift(1), df['b2_v'].shift(1))
        raw_buy_b2 = _estimate_raw_buy_by_queue(df['a2'], df['a2_v'], df['a2'].shift(1), df['a2_v'].shift(1))
        delta_b2 = _allocate_delta_by_volume(df['vol_diff'], raw_buy_b2, raw_sell_b2)
    else:
        delta_b2 = pd.Series(np.nan, index=df.index, dtype='float64')

    has_b3 = _has_nonzero_book_level(df, 3)
    if has_b3:
        raw_sell_b3 = _estimate_raw_sell_by_queue(df['b3'], df['b3_v'], df['b3'].shift(1), df['b3_v'].shift(1))
        raw_buy_b3 = _estimate_raw_buy_by_queue(df['a3'], df['a3_v'], df['a3'].shift(1), df['a3_v'].shift(1))
        delta_b3 = _allocate_delta_by_volume(df['vol_diff'], raw_buy_b3, raw_sell_b3)
    else:
        delta_b3 = pd.Series(np.nan, index=df.index, dtype='float64')

    has_l2 = all(f'b{i}' in df.columns and f'b{i}_v' in df.columns for i in range(1, 6))
    has_l2 = has_l2 and all(f'a{i}' in df.columns and f'a{i}_v' in df.columns for i in range(1, 6))
    has_l2 = has_l2 and any(_has_nonzero_book_level(df, i) for i in range(2, 6))
    if has_l2:
        raw_sell_l2 = pd.Series(0.0, index=df.index, dtype='float64')
        raw_buy_l2 = pd.Series(0.0, index=df.index, dtype='float64')
        for i in range(1, 6):
            raw_sell_l2 += _estimate_raw_sell_by_queue(
                df[f'b{i}'], df[f'b{i}_v'], df[f'b{i}'].shift(1), df[f'b{i}_v'].shift(1)
            )
            raw_buy_l2 += _estimate_raw_buy_by_queue(
                df[f'a{i}'], df[f'a{i}_v'], df[f'a{i}'].shift(1), df[f'a{i}_v'].shift(1)
            )
        delta_l2 = _allocate_delta_by_volume(df['vol_diff'], raw_buy_l2, raw_sell_l2)
    else:
        delta_l2 = pd.Series(np.nan, index=df.index, dtype='float64')

    df['delta_contrib_l1'] = delta_l1
    df['delta_contrib_b2'] = delta_b2
    df['delta_contrib_b3'] = delta_b3
    df['delta_contrib_l2'] = delta_l2
    df['_vwap_pv'] = df['last'].astype('float64') * df['vol_diff']

    bid_ok = (df['b1'] > 0) & (df['b1_v'] > 0)
    ask_ok = (df['a1'] > 0) & (df['a1_v'] > 0)
    df['_bvwap_pv'] = (
        np.where(bid_ok, df['b1'].astype('float64') * df['b1_v'].astype('float64'), 0.0)
        + np.where(ask_ok, df['a1'].astype('float64') * df['a1_v'].astype('float64'), 0.0)
    )
    df['_bvwap_v'] = (
        np.where(bid_ok, df['b1_v'].astype('float64'), 0.0)
        + np.where(ask_ok, df['a1_v'].astype('float64'), 0.0)
    )

    assigned = attach_period_info(df, periods, closed='left')
    if assigned.empty:
        return _empty_tick_factor_bar_frame()

    grouped = assigned.groupby('bar_end', sort=True)
    result = grouped['last'].agg(open='first', high='max', low='min', y_close='last')
    result['x_volume'] = grouped['vol_diff'].sum()
    result['x_turnover'] = grouped['turnover_diff'].sum()
    result['x_open_interest'] = grouped['open_interest'].last()
    result['x_vwap_pv'] = grouped['_vwap_pv'].sum()
    result['x_vwap_v'] = grouped['vol_diff'].sum()
    result['x_bvwap_pv'] = grouped['_bvwap_pv'].sum()
    result['x_bvwap_v'] = grouped['_bvwap_v'].sum()
    result['x_delta_l1'] = grouped['delta_contrib_l1'].sum()
    result['x_delta_l2'] = grouped['delta_contrib_l2'].sum() if has_l2 else np.nan
    result['x_delta_b2'] = grouped['delta_contrib_b2'].sum() if has_b2 else np.nan
    result['x_delta_b3'] = grouped['delta_contrib_b3'].sum() if has_b3 else np.nan

    volume_safe = result['x_volume'].replace(0, np.nan)
    result['x_ofi_l1'] = (result['x_delta_l1'] / volume_safe).fillna(0.0)
    result['x_ofi_l2'] = (result['x_delta_l2'] / volume_safe).fillna(0.0) if has_l2 else np.nan
    result['x_ofi_b2'] = (result['x_delta_b2'] / volume_safe).fillna(0.0) if has_b2 else np.nan
    result['x_ofi_b3'] = (result['x_delta_b3'] / volume_safe).fillna(0.0) if has_b3 else np.nan
    result['x_delta'] = result['x_delta_l1']
    result['x_ofi'] = result['x_ofi_l1']
    result[BAR_PERIOD_COL] = grouped[BAR_PERIOD_COL].first()
    result.index.name = 'datetime'

    for col in ('dominant_id', 'order_book_id'):
        if col in assigned.columns:
            result[col] = grouped[col].last()

    return result


def calc_tick_factor_bar(tick_df: pd.DataFrame,
                         freq: str,
                         trading_day: pd.Timestamp | None = None) -> pd.DataFrame:
    df = tick_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    if trading_day is not None:
        return _calc_tick_factor_bar_one_day(df, freq, pd.Timestamp(trading_day))

    if 'trading_date' in df.columns:
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        group_key = df['trading_date'].dt.normalize()
    else:
        group_key = df['datetime'].dt.normalize()

    bars = []
    for date, day_df in df.groupby(group_key):
        day_bar = _calc_tick_factor_bar_one_day(day_df, freq, pd.Timestamp(date))
        if day_bar.empty:
            continue
        bars.append(day_bar)

    if not bars:
        return _empty_tick_factor_bar_frame()

    return pd.concat(bars).sort_index()


def calc_tick_delta(tick_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    result = calc_tick_factor_bar(tick_df, freq)
    if result.empty:
        return _empty_tick_delta_frame()

    cols = [
        'x_volume', 'x_turnover', 'x_open_interest',
        'x_delta', 'x_ofi',
        'x_delta_l1', 'x_ofi_l1',
        'x_delta_l2', 'x_ofi_l2',
        'x_delta_b2', 'x_ofi_b2',
        'x_delta_b3', 'x_ofi_b3',
        BAR_PERIOD_COL,
    ]
    for col in ('dominant_id', 'order_book_id'):
        if col in result.columns:
            cols.append(col)
    return result[cols].copy()


# ============================================================
# Tick级：Efficiency Ratio (ER)
# ============================================================

def _calc_tick_er_one_day(day_df: pd.DataFrame,
                          freq: str,
                          trading_day: pd.Timestamp) -> pd.DataFrame:
    """
    单交易日tick -> bar ER聚合。

    ER = 净位移 / 总位移
       = abs(最后一个价格 - 第一个价格) / sum(abs(相邻价格差值))

    价格使用中间价: (a1 + b1) / 2
    """
    df = day_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')

    blocks = build_trading_blocks(pd.Timestamp(trading_day), source_ts=df.index.to_series())
    periods = build_period_table(blocks, freq)
    df = filter_to_trading_time(df, blocks)
    if df.empty or periods.empty:
        return pd.DataFrame(columns=['x_er', BAR_PERIOD_COL],
                          index=pd.DatetimeIndex([], name='datetime'))

    # 计算中间价
    df['mid_price'] = (df['a1'] + df['b1']) / 2

    assigned = attach_period_info(df, periods, closed='left')
    if assigned.empty:
        return pd.DataFrame(columns=['x_er', BAR_PERIOD_COL],
                          index=pd.DatetimeIndex([], name='datetime'))

    def calc_er(group):
        """计算单个bar的ER"""
        prices = group['mid_price'].values
        if len(prices) < 2:
            return np.nan

        # 净位移
        net_displacement = abs(prices[-1] - prices[0])

        # 总位移
        price_diffs = np.abs(np.diff(prices))
        total_displacement = np.sum(price_diffs)

        if total_displacement == 0:
            return 0.0

        return net_displacement / total_displacement

    grouped = assigned.groupby('bar_end', sort=True)
    result = pd.DataFrame(index=grouped.size().index)
    result.index.name = 'datetime'
    result['x_er'] = grouped.apply(calc_er)
    result[BAR_PERIOD_COL] = grouped[BAR_PERIOD_COL].first()

    return result


def calc_tick_er(tick_df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    从tick数据计算指定周期的效率比率(Efficiency Ratio)。

    ER = 净位移 / 总位移
       = abs(最后一个价格 - 第一个价格) / sum(abs(相邻价格差值))

    价格使用中间价: (a1 + b1) / 2

    注意：避免未来函数，时间戳9:10表示9:00-9:10分钟内的数据。

    Args:
        tick_df: tick数据，需包含 a1, b1, datetime 列
        freq: 周期标识（如 '3min', '10min'）

    Returns:
        DataFrame with columns: x_er, bar_period
    """
    df = tick_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])

    if 'trading_date' in df.columns:
        df['trading_date'] = pd.to_datetime(df['trading_date'])
        group_key = df['trading_date'].dt.normalize()
    else:
        group_key = df['datetime'].dt.normalize()

    bars = []
    for date, day_df in df.groupby(group_key):
        day_bar = _calc_tick_er_one_day(day_df, freq, pd.Timestamp(date))
        if day_bar.empty:
            continue
        bars.append(day_bar)

    if not bars:
        return pd.DataFrame(columns=['x_er', BAR_PERIOD_COL],
                          index=pd.DatetimeIndex([], name='datetime'))

    return pd.concat(bars).sort_index()



def _rolling_slope(s: pd.Series, window: int) -> pd.Series:
    """向量化滚动线性回归斜率，替代 rolling().apply(polyfit)。"""
    # slope = (n*sum(i*y) - sum(i)*sum(y)) / (n*sum(i^2) - sum(i)^2)
    n = window
    i_sum = n * (n - 1) / 2                    # sum(0..n-1)
    i2_sum = n * (n - 1) * (2 * n - 1) / 6     # sum(i^2)
    denom = n * i2_sum - i_sum ** 2

    # sum(i * y_i) via convolution: weight = [0, 1, 2, ..., n-1] reversed
    weights = np.arange(n, dtype=float)
    iy_sum = s.rolling(n).apply(lambda x: np.dot(weights, x), raw=True)
    y_sum = s.rolling(n).sum()

    return (n * iy_sum - i_sum * y_sum) / denom


def calc_vwap_factors(df: pd.DataFrame, freq: str,
                      close_col: str = 'y_close',
                      slope_window: int = 5,
                      dev_window: int = 20) -> pd.DataFrame:
    """
    x_vwap, x_vwap_dev, x_vwap_slope, x_bvwap, x_bvwap_dev, x_bvwap_slope

    VWAP / BVWAP 均为"开盘至今"累计值：
      tick_to_bar 输出每根bar的分子分母 (x_vwap_pv/x_vwap_v, x_bvwap_pv/x_bvwap_v)，
      这里按 trading_date 分组 cumsum 后再算比值。
    若无分子分母列，回退到 x_turnover / x_volume 近似（单bar级别）。
    """
    df = df.copy()
    close = df[close_col]

    has_td = 'trading_date' in df.columns

    # ── VWAP: 开盘至今 ───────────────────────────────────────
    if 'x_vwap_pv' in df.columns and 'x_vwap_v' in df.columns and has_td:
        cum_pv = df.groupby('trading_date')['x_vwap_pv'].cumsum()
        cum_v  = df.groupby('trading_date')['x_vwap_v'].cumsum()
        vwap = cum_pv / cum_v.replace(0, np.nan)
    elif 'x_turnover' in df.columns and 'x_volume' in df.columns:
        vwap = df['x_turnover'] / df['x_volume'].replace(0, np.nan)
    else:
        return df

    df[_col('vwap', freq)] = vwap
    vwap_ma = vwap.rolling(dev_window).mean()
    df[_col('vwap_dev', freq)] = (close - vwap_ma) / vwap_ma
    df[_col('vwap_slope', freq)] = _rolling_slope(vwap, slope_window)

    # ── BVWAP: 开盘至今 ──────────────────────────────────────
    if 'x_bvwap_pv' in df.columns and 'x_bvwap_v' in df.columns and has_td:
        cum_bpv = df.groupby('trading_date')['x_bvwap_pv'].cumsum()
        cum_bv  = df.groupby('trading_date')['x_bvwap_v'].cumsum()
        bvwap = cum_bpv / cum_bv.replace(0, np.nan)
        df[_col('bvwap', freq)] = bvwap
        bvwap_ma = bvwap.rolling(dev_window).mean()
        df[_col('bvwap_dev', freq)] = (close - bvwap_ma) / bvwap_ma
        df[_col('bvwap_slope', freq)] = _rolling_slope(bvwap, slope_window)

    return df


# ============================================================
# 均线类因子
# ============================================================

def calc_ma_factors(df: pd.DataFrame, freq: str,
                    close_col: str = 'y_close') -> pd.DataFrame:
    """
    x_ma5, x_ma20, x_ma60, x_ma5_ma20, x_trend_strength,
    x_price_ma20_dev, x_ma5_slope, x_price_ma250_dev
    """
    df = df.copy()
    close = df[close_col]

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    df[_col('ma5', freq)] = ma5
    df[_col('ma20', freq)] = ma20
    df[_col('ma60', freq)] = ma60
    df[_col('ma5_ma20', freq)] = ma5 / ma20
    df[_col('ma20_ma60', freq)] = ma20 / ma60

    # dev_5_20 = (ma5 - ma20) / ma20
    # dev_20_60 = (ma20 - ma60) / ma60
    # df[_col('trend_strength', freq)] = 0.6 * dev_5_20 + 0.4 * dev_20_60

    # df[_col('price_ma20_dev', freq)] = (close - ma20) / ma20
    # df[_col('ma5_slope', freq)] = (ma5 - ma5.shift(5)) / ma5.shift(5)

    # --- 新增: 价格相对MA250偏离度 ---
    ma250 = close.rolling(window=250, min_periods=1).mean()
    df[_col('price_ma250_dev', freq)] = (close - ma250) / ma250 * 100

    return df


# ============================================================
# 动量类因子
# ============================================================

def calc_momentum_factors(df: pd.DataFrame, freq: str,
                          close_col: str = 'y_close') -> pd.DataFrame:
    """
    x_mom5, x_mom20, x_snr, x_price_accel, x_intraday_return, x_gap_ratio,
    x_price_reversal, x_overnight_return
    """
    df = df.copy()
    close = df[close_col]

    # --- 新增: 收益率 / 对数收益率 ---
    df[_col('return_1', freq)] = close.pct_change(1)
    df[_col('log_return_1', freq)] = np.log(close / close.shift(1))

    df[_col('mom5', freq)] = close.pct_change(5)
    df[_col('mom20', freq)] = close.pct_change(20)

    price_change = close - close.shift(20)
    price_vol = close.rolling(20).std()
    df[_col('snr', freq)] = price_change / (price_vol + 1e-8)

    ret = close.pct_change()
    df[_col('price_accel', freq)] = ret.diff(5)

    # --- 新增: 价格反转因子 (负收益率) ---
    df[_col('price_reversal', freq)] = -ret

    if 'open' in df.columns:
        df[_col('intraday_return', freq)] = (close - df['open']) / df['open']
        df[_col('gap_ratio', freq)] = (df['open'] - close.shift(1)) / close.shift(1)
        # --- 新增: 隔夜收益率 (与gap_ratio同义，保留独立语义) ---
        df[_col('overnight_return', freq)] = (df['open'] - close.shift(1)) / close.shift(1)

    return df


# ============================================================
# 波动率类因子
# ============================================================

def calc_volatility_factors(df: pd.DataFrame, freq: str,
                            close_col: str = 'y_close') -> pd.DataFrame:
    """
    x_atr, x_volatility, x_realized_vol, x_intraday_vol,
    x_parkinson_vol, x_vol_ratio_10_30,
    x_overnight_vol, x_price_jump
    """
    df = df.copy()
    close = df[close_col]
    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))

    df[_col('volatility', freq)] = ret.rolling(20).std()
    df[_col('realized_vol', freq)] = log_ret.rolling(20).std()

    if all(c in df.columns for c in ['high', 'low']):
        prev_close = close.shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df[_col('atr', freq)] = tr.rolling(14).mean()

        df[_col('intraday_vol', freq)] = np.log(df['high'] / df['low'])

        log_hl2 = np.log(df['high'] / df['low']) ** 2
        df[_col('parkinson_vol', freq)] = np.sqrt(
            (1 / (4 * np.log(2))) * log_hl2.rolling(20).mean()
        )

    vol_short = log_ret.rolling(10).std()
    vol_long = log_ret.rolling(30).std()
    df[_col('vol_ratio_10_30', freq)] = vol_short / vol_long

    # --- 新增: 隔夜波动率 ---
    if 'open' in df.columns:
        df[_col('overnight_vol', freq)] = np.log(df['open'] / close.shift(1))

    # --- 新增: 价格跳跃 (|return| > threshold 则为1) ---
    df[_col('price_jump', freq)] = (ret.abs() > 0.02).astype(int)

    return df


# ============================================================
# 技术指标类
# ============================================================

def calc_technical_factors(df: pd.DataFrame, freq: str,
                           close_col: str = 'y_close') -> pd.DataFrame:
    """
    x_rsi14, x_macd, x_macd_signal, x_macd_hist,
    x_boll_pos, x_boll_upper, x_boll_lower,
    x_stoch_k, x_stoch_d, x_cci, x_adx,
    x_aroon_osc, x_williams_r, x_trix, x_ultosc,
    x_sar_diff, x_roc, x_keltner_pos, x_donchian_pos,
    x_close_pos_ratio, x_price_channel_breakout
    """
    df = df.copy()
    close = df[close_col]
    has_hl = all(c in df.columns for c in ['high', 'low'])

    if HAS_TALIB and has_hl:
        h, l, c = _f64(df['high']), _f64(df['low']), _f64(close)

        df[_col('rsi14', freq)] = pd.Series(talib.RSI(c, timeperiod=14), index=df.index)

        macd, sig, hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        df[_col('macd', freq)] = pd.Series(macd, index=df.index)
        df[_col('macd_signal', freq)] = pd.Series(sig, index=df.index)
        df[_col('macd_hist', freq)] = pd.Series(hist, index=df.index)

        slowk, slowd = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)
        df[_col('stoch_k', freq)] = pd.Series(slowk, index=df.index)
        df[_col('stoch_d', freq)] = pd.Series(slowd, index=df.index)

        df[_col('cci', freq)] = pd.Series(talib.CCI(h, l, c, timeperiod=20), index=df.index)
        df[_col('adx', freq)] = pd.Series(talib.ADX(h, l, c, timeperiod=14), index=df.index)
        df[_col('williams_r', freq)] = pd.Series(talib.WILLR(h, l, c, timeperiod=14), index=df.index)

        aroon_up, aroon_down = talib.AROON(h, l, timeperiod=14)
        df[_col('aroon_osc', freq)] = pd.Series(aroon_up - aroon_down, index=df.index)

        df[_col('trix', freq)] = pd.Series(talib.TRIX(c, timeperiod=14), index=df.index)
        df[_col('ultosc', freq)] = pd.Series(talib.ULTOSC(h, l, c), index=df.index)

        sar = pd.Series(talib.SAR(h, l, acceleration=0.02, maximum=0.2), index=df.index)
        df[_col('sar_diff', freq)] = (close - sar) / close

    elif HAS_TALIB:
        c = _f64(close)
        df[_col('rsi14', freq)] = pd.Series(talib.RSI(c, timeperiod=14), index=df.index)
        macd, sig, hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        df[_col('macd', freq)] = pd.Series(macd, index=df.index)
        df[_col('macd_signal', freq)] = pd.Series(sig, index=df.index)
        df[_col('macd_hist', freq)] = pd.Series(hist, index=df.index)
        df[_col('trix', freq)] = pd.Series(talib.TRIX(c, timeperiod=14), index=df.index)

    # 无需talib
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    df[_col('boll_pos', freq)] = (close - bb_lower) / (bb_upper - bb_lower)

    # --- 新增: 布林带上下轨绝对值 ---
    df[_col('boll_upper', freq)] = bb_upper
    df[_col('boll_lower', freq)] = bb_lower

    df[_col('roc', freq)] = (close - close.shift(10)) / close.shift(10) * 100

    if has_hl:
        ema20 = close.ewm(span=20).mean()
        atr_col = _col('atr', freq)
        if atr_col in df.columns:
            kc_upper = ema20 + 2 * df[atr_col]
            kc_lower = ema20 - 2 * df[atr_col]
            df[_col('keltner_pos', freq)] = (close - kc_lower) / (kc_upper - kc_lower)

        dc_high = df['high'].rolling(20).max()
        dc_low = df['low'].rolling(20).min()
        df[_col('donchian_pos', freq)] = (close - dc_low) / (dc_high - dc_low)

        df[_col('close_pos_ratio', freq)] = (close - df['low']) / (df['high'] - df['low'])

        # --- 新增: 价格通道突破 (+1=突破上轨, -1=突破下轨, 0=区间内) ---
        high_max_prev = df['high'].rolling(20).max().shift(1)
        low_min_prev = df['low'].rolling(20).min().shift(1)
        breakout_up = (close > high_max_prev).astype(int)
        breakout_down = (close < low_min_prev).astype(int) * -1
        df[_col('price_channel_breakout', freq)] = breakout_up + breakout_down

    return df


# ============================================================
# 成交量类因子
# ============================================================

def calc_volume_factors(df: pd.DataFrame, freq: str,
                        close_col: str = 'y_close',
                        volume_col: str = 'x_volume') -> pd.DataFrame:
    """
    x_vol_mom, x_rel_vol, x_vol_ma_ratio, x_vol_accel,
    x_vol_change_rate, x_vol_weighted_return,
    x_obv, x_mfi, x_chaikin_osc, x_ad_line,
    x_pvt, x_vol_price_corr, x_money_flow_ratio,
    x_ease_of_movement, x_volume_profile
    """
    df = df.copy()
    close = df[close_col]

    if volume_col not in df.columns:
        return df

    vol = df[volume_col]

    df[_col('vol_mom', freq)] = vol / vol.shift(20) - 1
    df[_col('rel_vol', freq)] = vol / vol.rolling(20).mean()
    vol_short = vol.rolling(5).mean()
    vol_long = vol.rolling(20).mean()
    df[_col('vol_ma_ratio', freq)] = vol_short / vol_long
    df[_col('vol_accel', freq)] = vol.pct_change().diff(5)

    # --- 新增: 成交量变化率 ---
    df[_col('vol_change_rate', freq)] = vol.pct_change()

    ret = close.pct_change()
    log_ret = np.log(close / close.shift(1))
    vol_chg = vol.pct_change()
    df[_col('vol_price_corr', freq)] = ret.rolling(20).corr(vol_chg)

    df[_col('pvt', freq)] = (ret * vol).cumsum()

    # --- 新增: 成交量加权收益率 ---
    total_vol_20 = vol.rolling(20).sum()
    df[_col('vol_weighted_return', freq)] = (log_ret * vol).rolling(20).sum() / total_vol_20

    has_hl = all(c in df.columns for c in ['high', 'low'])

    if has_hl:
        tp = (df['high'] + df['low'] + close) / 3
        mf = tp * vol
        pos_flow = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg_flow = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
        df[_col('money_flow_ratio', freq)] = pos_flow / (pos_flow + neg_flow.abs())

        # --- 新增: 简易波动指标 (Ease of Movement) ---
        distance = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
        box_ratio = vol / 10000 / (df['high'] - df['low']).replace(0, np.nan)
        emv = distance / box_ratio
        df[_col('ease_of_movement', freq)] = emv.rolling(14).mean()

    if HAS_TALIB and has_hl:
        h, l, c, v = _f64(df['high']), _f64(df['low']), _f64(close), _f64(vol)
        df[_col('obv', freq)] = pd.Series(talib.OBV(c, v), index=df.index)
        df[_col('mfi', freq)] = pd.Series(talib.MFI(h, l, c, v, timeperiod=14), index=df.index)
        df[_col('chaikin_osc', freq)] = pd.Series(
            talib.ADOSC(h, l, c, v, fastperiod=3, slowperiod=10), index=df.index
        )
        df[_col('ad_line', freq)] = pd.Series(talib.AD(h, l, c, v), index=df.index)

    # --- 新增: 成交量分布指标 (Volume Profile Indicator) ---
    def _calc_volume_profile(close_arr, vol_arr, window=20, bins=10):
        n = len(close_arr)
        results = np.full(n, np.nan)
        for i in range(window - 1, n):
            prices = close_arr[i - window + 1: i + 1]
            volumes = vol_arr[i - window + 1: i + 1]
            pmin, pmax = prices.min(), prices.max()
            if pmax == pmin or np.isnan(pmin):
                continue
            # 向量化 digitize 替代内层 for 循环
            bin_idx = np.digitize(prices, np.linspace(pmin, pmax, bins + 1)) - 1
            bin_idx = np.clip(bin_idx, 0, bins - 1)
            vol_profile = np.bincount(bin_idx, weights=volumes, minlength=bins)
            results[i] = np.argmax(vol_profile) / bins
        return results

    close_arr = close.values.astype(float)
    vol_arr = vol.values.astype(float)
    df[_col('volume_profile', freq)] = pd.Series(
        _calc_volume_profile(close_arr, vol_arr), index=df.index
    )

    return df


# ============================================================
# 高级因子
# ============================================================

def calc_advanced_factors(df: pd.DataFrame, freq: str,
                          close_col: str = 'y_close') -> pd.DataFrame:
    """
    x_skew, x_kurtosis, x_hurst, x_fractal_dim,
    x_high_low_ratio, x_bid_ask_proxy, x_liquidity,
    x_price_impact, x_market_depth_imbalance
    """
    df = df.copy()
    close = df[close_col]
    log_ret = np.log(close / close.shift(1))

    df[_col('skew', freq)] = log_ret.rolling(20).skew()
    df[_col('kurtosis', freq)] = log_ret.rolling(20).kurt()

    # --- Hurst指数: 完全向量化实现 ---
    hurst_window = 100
    log_price = np.log(close).values.astype(float)
    n = len(log_price)
    lags = np.arange(2, 20)
    log_lags = np.log(lags)
    n_lags = len(lags)
    lags_mean = log_lags.mean()
    lags_var = ((log_lags - lags_mean) ** 2).sum()

    # 预计算每个lag的rolling std
    log_tau_matrix = np.full((n, n_lags), np.nan)
    for j, lag in enumerate(lags):
        lag = int(lag)
        diff_k = log_price[lag:] - log_price[:-lag]
        diff_s = pd.Series(diff_k)
        win = hurst_window - lag
        rolling_std = diff_s.rolling(win, min_periods=win).std(ddof=0).values
        sqrt_std = np.sqrt(np.maximum(rolling_std, 0))
        valid_mask = sqrt_std > 0
        log_vals = np.where(valid_mask, np.log(sqrt_std), np.nan)
        # 映射回log_price位置
        for_pos = np.arange(len(log_vals)) + lag
        mask = (for_pos >= hurst_window - 1) & (for_pos < n) & valid_mask
        log_tau_matrix[for_pos[mask], j] = log_vals[mask]

    # 计算每个位置的slope
    hurst_arr = np.full(n, np.nan)
    for i in range(hurst_window - 1, n):
        lt = log_tau_matrix[i]
        if not np.any(np.isnan(lt)):
            slope = ((lt - lt.mean()) * (log_lags - lags_mean)).sum() / lags_var
            hurst_arr[i] = slope * 2.0

    df[_col('hurst', freq)] = hurst_arr

    # --- 分形维数: 向量化 ---
    fd_window = 20
    close_arr = close.values.astype(float)
    fd_arr = np.full(n, np.nan)
    log_2N_minus_2 = np.log(2 * (fd_window - 1))
    for i in range(fd_window - 1, n):
        prices = close_arr[i - fd_window + 1: i + 1]
        if np.isnan(prices).any():
            continue
        L = np.sum(np.abs(np.diff(prices)))
        if L > 0:
            fd_arr[i] = 1 + np.log(L) / log_2N_minus_2

    df[_col('fractal_dim', freq)] = fd_arr

    has_hl = all(c in df.columns for c in ['high', 'low'])

    if has_hl:
        hl_ratio = df['high'] / df['low']
        df[_col('high_low_ratio', freq)] = hl_ratio.rolling(20).mean()
        spread = (df['high'] - df['low']) / close
        df[_col('bid_ask_proxy', freq)] = spread.rolling(20).mean()

    vol_col = 'x_volume' if 'x_volume' in df.columns else ('volume' if 'volume' in df.columns else None)

    if vol_col and vol_col in df.columns:
        vol = df[vol_col]
        turnover = vol * close
        df[_col('liquidity', freq)] = (turnover / (log_ret.abs() + 1e-8)).rolling(20).mean()

        # --- 新增: 价格冲击 (|return| 与 vol_change 的滚动相关) ---
        ret_abs = close.pct_change().abs()
        vol_chg = vol.pct_change()
        df[_col('price_impact', freq)] = ret_abs.rolling(5).corr(vol_chg)

        # --- 新增: 市场深度不平衡 ---
        if has_hl:
            high_vol = (close == df['high']).astype(int) * vol
            low_vol = (close == df['low']).astype(int) * vol
            high_vol_sum = high_vol.rolling(20).sum()
            low_vol_sum = low_vol.rolling(20).sum()
            total_vol_sum = vol.rolling(20).sum()
            df[_col('market_depth_imbalance', freq)] = (high_vol_sum - low_vol_sum) / total_vol_sum

    return df


# ============================================================
# Delta / 订单流因子（需要 x_delta 列）
# ============================================================

def calc_delta_factors(df: pd.DataFrame, freq: str,
                       close_col: str = 'y_close') -> pd.DataFrame:
    """
    x_cvd, x_price_delta, x_price_delta_volume, x_price_delta_trend,
    x_effort_vs_result, x_distribute, x_trend, x_trend1, x_absorption,
    x_ofi
    """
    df = df.copy()
    price_ret = df[close_col].pct_change()

    df[_col('cvd', freq)] = df['x_delta'].cumsum()
    df[_col('price_delta', freq)] = price_ret * df['x_delta']
    df[_col('price_delta_volume', freq)] = price_ret * df['x_delta'] * df['x_volume']
    df[_col('price_delta_trend', freq)] = df[_col('price_delta', freq)].cumsum()

    # OFI: (buy_vol - sell_vol) / (buy_vol + sell_vol)
    if 'x_buy_vol' in df.columns and 'x_sell_vol' in df.columns:
        total = df['x_buy_vol'] + df['x_sell_vol']
        df[_col('ofi', freq)] = (df['x_buy_vol'] - df['x_sell_vol']) / total.replace(0, np.nan)

    _dz_win, _prz_win = 20, 50
    delta_z = zscore(df['x_delta'], _dz_win)
    price_ret_z = zscore(price_ret, _prz_win)
    vol_z = zscore(df['x_volume'], _prz_win)

    df[_col('effort_vs_result', freq)] = df['x_delta'] / (price_ret.abs() + 0.5)

    df[_col('distribute', freq)] = np.where(
        (delta_z >= 0.5) & (price_ret_z.abs() < 0.3), df['x_delta'],
        np.where(
            (delta_z <= -0.5) & (price_ret_z.abs() > 0.3), df['x_delta'], 0
        )
    )

    df[_col('trend', freq)] = np.where(
        (delta_z >= 1) & (price_ret > 0), df['x_delta'],
        np.where(
            (delta_z <= -1) & (price_ret < 0), df['x_delta'], 0
        )
    )
    df[_col('trend1', freq)] = np.where(
        (delta_z >= 1) & (price_ret > 0), price_ret, 0
    )

    df[_col('absorption', freq)] = np.where(
        (delta_z < 0.5) & (price_ret_z >= 0.5) & (vol_z > 0.2), price_ret,
        np.where(
            (delta_z > -0.5) & (price_ret_z <= -0.5) & (vol_z > 0.2), price_ret, 0
        )
    )

    return df


# ============================================================
# 突破类因子
# ============================================================

def calc_breakout_factors(df: pd.DataFrame, freq: str,
                          close_col: str = 'y_close') -> pd.DataFrame:
    """
    x_break_position, x_trend_break, x_trend_break2,
    x_trend_break_watch, x_vol_ratio
    """
    df = df.copy()

    trend_col = _col('trend', freq)
    if trend_col not in df.columns:
        df = calc_delta_factors(df, freq, close_col)

    breakout_factor = 0
    for win, w in [(5, 0.0), (10, 1.0), (20, 0.0)]:
        upper = df['high'].rolling(win).max().shift(1)
        lower = df['low'].rolling(win).min().shift(1)
        bf = (df[close_col] - (upper + lower) / 2) / (upper - lower).clip(lower=1e-9)
        breakout_factor = breakout_factor + w * bf

    df[_col('break_position', freq)] = breakout_factor

    vol_ratio = df['x_volume'] / df['x_volume'].rolling(48).median()
    vol_pct = df['x_volume'].rolling(100).rank(pct=True)

    df[_col('vol_ratio', freq)] = vol_ratio
    df[_col('trend_break', freq)] = np.where(
        vol_pct > 0.5, breakout_factor * df[trend_col], 0
    )
    df[_col('trend_break2', freq)] = breakout_factor * df[trend_col] * vol_ratio
    df[_col('trend_break_watch', freq)] = df[_col('trend_break2', freq)].shift(3)

    return df


# ============================================================
# VPIN因子 (Volume-Synchronized Probability of Informed Trading)
# ============================================================

# def calc_vpin_factors(df: pd.DataFrame, freq: str,
#                       close_col: str = 'y_close',
#                       volume_col: str = 'x_volume',
#                       n_bucket: int = 20,
#                       bucket_max_volume: float = 0.0,
#                       vol_decay: float = 0.8) -> pd.DataFrame:
#     """
#     基于flowrisk库的VPIN算法，实现为bar级因子。
#
#     核心思路:
#       1. EWMA估计价格波动率
#       2. 用正态CDF对每根bar的成交量做Bulk Classification (买/卖拆分)
#       3. 将成交量灌入固定容量的bucket，bucket满后滚动到下一个
#       4. VPIN = sum(|买量-卖量|) / sum(bucket总量)，在最近n_bucket个bucket上计算
#
#     Args:
#         df:                 bar数据
#         freq:               周期标识
#         close_col:          收盘价列名
#         volume_col:         成交量列名
#         n_bucket:           bucket数量 (滚动窗口长度)
#         bucket_max_volume:  单个bucket最大容量; 0=自动取近期平均成交量*n_bucket
#         vol_decay:          EWMA波动率衰减系数 (0~1, 越大越平滑)
#
#     Returns:
#         新增列: x_vpin_{freq}
#     """
#     df = df.copy()
#
#     if volume_col not in df.columns or close_col not in df.columns:
#         return df
#
#     prices = df[close_col].values.astype(float)
#     volumes = df[volume_col].values.astype(float)
#     n = len(prices)
#
#     if n < 3:
#         return df
#
#     # --- 自动推算bucket_max_volume ---
#     if bucket_max_volume <= 0:
#         avg_vol = np.nanmean(volumes[volumes > 0]) if np.any(volumes > 0) else 1.0
#         bucket_max_volume = avg_vol * n_bucket
#
#     # --- 预计算EWMA波动率和buy_pct (向量化) ---
#     price_diff = np.diff(prices, prepend=prices[0])
#     sq_ret = price_diff ** 2
#
#     # EWMA方差
#     ewma_var = np.empty(n)
#     ewma_var[0] = sq_ret[0]
#     ewma_var[1] = sq_ret[1]
#     for i in range(2, n):
#         ewma_var[i] = (1 - vol_decay) * sq_ret[i] + vol_decay * ewma_var[i - 1]
#
#     vol_est = np.sqrt(np.maximum(ewma_var, 1e-16))
#     z_scores = price_diff / np.maximum(vol_est, 1e-8)
#     # 批量计算norm.cdf
#     buy_pcts = norm.cdf(z_scores)
#     buy_pcts[:2] = 0.5  # 前2根bar初始化为50/50
#
#     # --- Bucket数组 ---
#     bucket_vol = np.zeros(n_bucket)
#     bucket_buy = np.zeros(n_bucket)
#     bucket_sell = np.zeros(n_bucket)
#     cur_idx = 0
#
#     # --- 逐bar计算VPIN (bucket填充必须顺序执行) ---
#     vpin_arr = np.full(n, np.nan)
#
#     for i in range(n):
#         p = prices[i]
#         v = volumes[i]
#
#         if np.isnan(p) or np.isnan(v) or v <= 0:
#             if i > 2:
#                 vpin_arr[i] = vpin_arr[i - 1]
#             continue
#
#         buy_pct = buy_pcts[i]
#
#         # 将成交量灌入bucket
#         remaining = v
#         while remaining > 0:
#             space = bucket_max_volume - bucket_vol[cur_idx]
#             fill = min(remaining, space)
#             buy_fill = buy_pct * fill
#             bucket_vol[cur_idx] += fill
#             bucket_buy[cur_idx] += buy_fill
#             bucket_sell[cur_idx] += fill - buy_fill
#             remaining -= fill
#             if bucket_vol[cur_idx] >= bucket_max_volume:
#                 cur_idx = (cur_idx + 1) % n_bucket
#                 bucket_vol[cur_idx] = 0.0
#                 bucket_buy[cur_idx] = 0.0
#                 bucket_sell[cur_idx] = 0.0
#
#         # 计算VPIN
#         total_vol = bucket_vol.sum()
#         if total_vol > 0:
#             vpin_arr[i] = np.abs(bucket_buy - bucket_sell).sum() / total_vol
#         elif i > 0:
#             vpin_arr[i] = vpin_arr[i - 1]
#
#     df[_col('vpin', freq)] = vpin_arr
#
#     # --- VPIN zscore (window=20) ---
#     vpin_s = df[_col('vpin', freq)]
#     vpin_z = zscore(vpin_s, window=20)
#     df[_col('vpin_zscore', freq)] = vpin_z
#
#     # --- VPIN zscore >= 2 过滤: 小于2的部分赋值为零，大于等于2保留原值 ---
#     df[_col('vpin_zscore_filtered', freq)] = np.where(vpin_z >= 2, vpin_z, 0.0)
#
#     return df


def calc_vpin_factors(df: pd.DataFrame, freq: str,
                      close_col: str = 'y_close',
                      volume_col: str = 'x_volume',
                      n_bucket: int = 20,
                      bucket_size_window: int = 50,
                      vol_decay: float = 0.8) -> pd.DataFrame:
    """
    无未来函数的 VPIN (Volume-Synchronized Probability of Informed Trading)。

    算法步骤:
      1. 计算买卖压力权重 z:
         ΔP = P_i - P_{i-1}，用 EWMA 估计波动率 σ，
         z = CDF(ΔP / σ)，映射到 [0, 1]。
      2. 拆分买卖量:
         V_buy = V × z，V_sell = V × (1 - z)。
      3. 等量桶 (Volume Bucketing):
         bucket_size 由过去 bucket_size_window 根 bar 的滚动均量决定
         （避免使用全局均值导致的未来函数）。
         成交量按顺序灌入固定容量的桶，桶满则滚动到下一个。
      4. 计算 VPIN:
         VPIN = Σ|V_buy_j - V_sell_j| / (n_bucket × bucket_size)，
         在最近 n_bucket 个已填满的桶上计算。

    Args:
        df:                  bar数据
        freq:                周期标识
        close_col:           收盘价列名
        volume_col:          成交量列名
        n_bucket:            滚动窗口桶数量
        bucket_size_window:  用于估算 bucket_size 的滚动窗口长度
        vol_decay:           EWMA 波动率衰减系数 (0~1)

    Returns:
        新增列: x_vpin_{freq}, x_vpin_zscore_{freq}, x_vpin_zscore_filtered_{freq}
    """
    df = df.copy()

    if volume_col not in df.columns or close_col not in df.columns:
        return df

    prices = df[close_col].values.astype(float)
    volumes = df[volume_col].values.astype(float)
    n = len(prices)

    if n < 3:
        return df

    # ── 第1步: 计算买卖压力权重 z ─────────────────────────────────────────
    # ΔP = P_i - P_{i-1}
    price_diff = np.empty(n)
    price_diff[0] = 0.0
    price_diff[1:] = prices[1:] - prices[:-1]

    # EWMA 波动率 σ (因果: 只用当前及过去)
    sq_ret = price_diff ** 2
    ewma_var = np.empty(n)
    ewma_var[0] = sq_ret[0]
    ewma_var[1] = sq_ret[1]
    for i in range(2, n):
        ewma_var[i] = (1 - vol_decay) * sq_ret[i] + vol_decay * ewma_var[i - 1]

    sigma = np.sqrt(np.maximum(ewma_var, 1e-16))
    z_scores = price_diff / np.maximum(sigma, 1e-8)

    # z = CDF(ΔP / σ)，映射到 [0, 1]
    buy_pcts = norm.cdf(z_scores)
    buy_pcts[0] = 0.5  # 首根 bar 无价格变化，中性

    # ── 第2步: 拆分买卖量 (向量化) ────────────────────────────────────────
    vol_buy = volumes * buy_pcts       # V_buy = V × z
    vol_sell = volumes * (1 - buy_pcts)  # V_sell = V × (1 - z)

    # ── 第3步 & 第4步: 等量桶填充 + 逐 bar 计算 VPIN ─────────────────────
    # bucket_size 用滚动均量估算，避免未来函数
    rolling_avg_vol = pd.Series(volumes).rolling(
        bucket_size_window, min_periods=1
    ).mean().values

    # 环形桶数组
    bucket_buy_arr = np.zeros(n_bucket)
    bucket_sell_arr = np.zeros(n_bucket)
    bucket_filled = np.zeros(n_bucket)  # 每个桶已填充量
    cur_idx = 0
    filled_count = 0  # 历史上已填满的桶总数

    vpin_arr = np.full(n, np.nan)

    for i in range(n):
        v = volumes[i]
        vb = vol_buy[i]
        vs = vol_sell[i]

        if np.isnan(prices[i]) or np.isnan(v) or v <= 0:
            if i > 0:
                vpin_arr[i] = vpin_arr[i - 1]
            continue

        # 当前 bar 的 bucket_size: 滚动均量 × n_bucket
        bucket_size = rolling_avg_vol[i] * n_bucket
        if bucket_size <= 0:
            bucket_size = v * n_bucket  # 极端回退

        # 按买卖比例灌入桶
        remaining = v
        buy_ratio = vb / v if v > 0 else 0.5

        while remaining > 1e-9:
            space = bucket_size - bucket_filled[cur_idx]
            fill = min(remaining, space)
            bucket_buy_arr[cur_idx] += buy_ratio * fill
            bucket_sell_arr[cur_idx] += (1 - buy_ratio) * fill
            bucket_filled[cur_idx] += fill
            remaining -= fill

            if bucket_filled[cur_idx] >= bucket_size - 1e-9:
                filled_count += 1
                cur_idx = (cur_idx + 1) % n_bucket
                bucket_buy_arr[cur_idx] = 0.0
                bucket_sell_arr[cur_idx] = 0.0
                bucket_filled[cur_idx] = 0.0

        # 计算 VPIN = Σ|V_buy_j - V_sell_j| / (n_bucket × bucket_size)
        if filled_count >= n_bucket:
            imbalance = np.abs(bucket_buy_arr - bucket_sell_arr).sum()
            vpin_arr[i] = imbalance / (n_bucket * bucket_size)
        elif filled_count > 0:
            # 桶未填满 n_bucket 个时，用已有桶近似
            total_vol = bucket_filled.sum()
            if total_vol > 0:
                imbalance = np.abs(bucket_buy_arr - bucket_sell_arr).sum()
                vpin_arr[i] = imbalance / total_vol

    df[_col('vpin', freq)] = vpin_arr

    # ── VPIN zscore (window=20) ───────────────────────────────────────────
    vpin_s = df[_col('vpin', freq)]
    vpin_z = zscore(vpin_s, window=20)
    df[_col('vpin_zscore', freq)] = vpin_z

    # ── VPIN zscore >= 2 过滤 ─────────────────────────────────────────────
    df[_col('vpin_zscore_filtered', freq)] = np.where(vpin_z >= 2, vpin_z, 0.0)

    return df


# ============================================================
# AD / Qstick / Sharpe / CCI2 / RVI / Klinger / LinearReg_MA
# ============================================================

def _ts_ema(s: pd.Series, period: int) -> pd.Series:
    """EMA 辅助，等价于 ts_EMA(df, period)。"""
    return s.ewm(span=period, adjust=False).mean()


def calc_ad_factor(df: pd.DataFrame, freq: str,
                   close_col: str = 'y_close',
                   volume_col: str = 'x_volume',
                   period: int = 14) -> pd.DataFrame:
    """AD 指标 (Accumulation/Distribution EMA)。"""
    df = df.copy()
    if volume_col not in df.columns:
        return df
    has_hl = all(c in df.columns for c in ['high', 'low'])
    if not has_hl:
        return df

    close, vol = df[close_col], df[volume_col]
    avgvol = vol.rolling(period).mean()
    denom = df['high'] - df['low']
    ad = (2 * close - df['low'] - df['high']) / denom * vol / avgvol
    ad = np.where(denom == 0, 0, ad)
    ad = _ts_ema(pd.Series(ad, index=df.index), period)
    df[_col('ad', freq)] = ad
    return df


def calc_qstick_2_factor(df: pd.DataFrame, freq: str,
                          close_col: str = 'y_close',
                          period: int = 14) -> pd.DataFrame:
    """Qstick_2: (close-open)/close 的滚动均值。"""
    df = df.copy()
    if 'open' not in df.columns:
        return df
    pricemove = (df[close_col] - df['open']) / df[close_col]
    df[_col('qstick', freq)] = pricemove.rolling(period).mean()
    return df


def calc_sharpe_2_factor(df: pd.DataFrame, freq: str,
                          close_col: str = 'y_close',
                          period: int = 20) -> pd.DataFrame:
    """Sharpe_2: 对数收益率的滚动 mean/std。"""
    df = df.copy()
    ret = np.log(df[close_col] / df[close_col].shift(1))
    df[_col('sharpe', freq)] = ret.rolling(period).mean() / ret.rolling(period).std()
    return df



def calc_rvi_2_factor(df: pd.DataFrame, freq: str,
                       close_col: str = 'y_close',
                       period: int = 14) -> pd.DataFrame:
    """RVI_2: 涨跌波动率差异比。"""
    df = df.copy()
    gain = df[close_col].diff()
    gps = gain.clip(lower=0).rolling(period).std()
    gms = (-gain).clip(lower=0).rolling(period).std()
    df[_col('rvi', freq)] = (gps - gms) / (gps + gms)
    return df


def calc_klinger_4_factor(df: pd.DataFrame, freq: str,
                           close_col: str = 'y_close',
                           volume_col: str = 'x_volume',
                           dem_period: int = 13) -> pd.DataFrame:
    """Klinger_4: 基于量价趋势的 KO 信号线。"""
    df = df.copy()
    if volume_col not in df.columns:
        return df
    has_hl = all(c in df.columns for c in ['high', 'low'])
    if not has_hl:
        return df

    short, long_ = 34, 55
    avg_price = (df[close_col] + df['high'] + df['low']) / 3
    sum_vol = df[volume_col].rolling(dem_period).sum()
    sv = np.where(avg_price > avg_price.shift(1),
                  df[volume_col] / sum_vol,
                  -df[volume_col] / sum_vol)
    sv = pd.Series(sv, index=df.index)
    svma_short = sv.ewm(span=short, adjust=False).mean()
    svma_long = sv.ewm(span=long_, adjust=False).mean()
    ko = (svma_long - svma_short).ewm(span=dem_period, adjust=False).mean()
    df[_col('klinger', freq)] = ko
    return df


def calc_linearreg_ma_4_factor(df: pd.DataFrame, freq: str,
                                close_col: str = 'y_close',
                                period: int = 14) -> pd.DataFrame:
    """LinearReg_MA_4: 线性回归预测值与当前价格的对数偏离 EMA。"""
    df = df.copy()
    close = df[close_col]
    x = np.arange(period, dtype=float)
    sumx = x.sum()
    sumx2 = (x * x).sum()
    sumy = close.rolling(period).sum()
    sumxy = close.rolling(period).apply(lambda c: (c * x).sum(), raw=True)
    slope = (period * sumxy - sumx * sumy) / (period * sumx2 - sumx * sumx)
    intercept = (sumy - slope * sumx) / period
    ey = slope * (period - 1) + intercept
    ey = np.log(ey / close)
    ey = _ts_ema(ey, period)
    df[_col('linearreg_ma', freq)] = ey
    return df


# ============================================================
# 一键全算（bar级）
# ============================================================

def calc_all_factors(df: pd.DataFrame, freq: str,
                     close_col: str = 'y_close',
                     has_delta: bool = True) -> pd.DataFrame:
    """
    一次性计算所有bar级因子

    Args:
        df:         bar数据
        freq:       周期标识 (如 '10min', '60min')
        close_col:  收盘价列名
        has_delta:  是否有 x_delta/x_volume (tick合成bar才有)
    """
    df = calc_ma_factors(df, freq, close_col)
    df = calc_momentum_factors(df, freq, close_col)
    df = calc_volatility_factors(df, freq, close_col)
    df = calc_technical_factors(df, freq, close_col)
    df = calc_advanced_factors(df, freq, close_col)

    vol_col = 'x_volume' if 'x_volume' in df.columns else 'volume'
    if vol_col in df.columns:
        df = calc_volume_factors(df, freq, close_col, volume_col=vol_col)

    if any(c in df.columns for c in ('x_vwap_pv', 'x_bvwap_pv', 'x_turnover')):
        df = calc_vwap_factors(df, freq, close_col)

    if has_delta and 'x_delta' in df.columns:
        df = calc_delta_factors(df, freq, close_col)
        if 'high' in df.columns:
            df = calc_breakout_factors(df, freq, close_col)

    # VPIN因子 (需要价格和成交量)
    vol_col = 'x_volume' if 'x_volume' in df.columns else 'volume'
    if vol_col in df.columns:
        df = calc_vpin_factors(df, freq, close_col, volume_col=vol_col)

    # Delta × VPIN 交叉因子
    vpin_col = _col('vpin', freq)
    if 'x_delta' in df.columns and vpin_col in df.columns:
        df[_col('delta_x_vpin', freq)] = df['x_delta'] * df[vpin_col]

    # AD / Qstick / Sharpe / CCI2 / RVI / Klinger / LinearReg_MA
    vol_col = 'x_volume' if 'x_volume' in df.columns else 'volume'
    df = calc_ad_factor(df, freq, close_col, volume_col=vol_col)
    df = calc_qstick_2_factor(df, freq, close_col)
    df = calc_sharpe_2_factor(df, freq, close_col)
    df = calc_rvi_2_factor(df, freq, close_col)
    df = calc_klinger_4_factor(df, freq, close_col, volume_col=vol_col)
    df = calc_linearreg_ma_4_factor(df, freq, close_col)

    return df
