"""
用于准备回测用的
主流程：Tick → 小周期Bar + 大周期Bar → 因子计算 → 合并 → 保存
========================================================================
数据源:  D:/commodity_data/tick/{品种}/{品种}_tick_yyyy-mm-dd.parquet
输出:    D:/commodity_data/tick/{品种}/{freq_label}/{品种}_{freq}.parquet

用户可自定义:
  BAR_FREQ   = '10min'   小周期
  UPPER_FREQ = '60min'   大周期（自动推导，也可手动指定）
"""

import os
import glob
import pandas as pd
import numpy as np
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from factors_backup import calc_all_factors
from bar_bucket_utils import (
    BAR_PERIOD_COL,
    build_trading_blocks,
    build_period_table,
    filter_to_trading_time,
    attach_period_info,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ========== 用户配置 ==========
BASE_DIR = r"D:/commodity_data/tick"
BAR_FREQ = '60min'
START_DATE = '2001-01-01'  # None=处理所有数据; 指定开始日期示例: '2020-01-01'

# 大周期自动推导映射，也可手动覆盖
UPPER_FREQ_MAP = {
    '3min': '15min',
    '5min': '30min',
    '10min': '60min',
    '11min': '60min',
    '12min': '60min',
    '13min': '60min',
    '14min': '60min',
    '15min': '60min',
    '30min': '120min',
    '60min': '240min',
}
UPPER_FREQ = UPPER_FREQ_MAP.get(BAR_FREQ, '60min')

# SYMBOLS = ["A", "C", "M", "Y","JM", "MA", "I", "V", "OI", "P", "RB", "RM",'CF','CY']
SYMBOLS = ["P"]# None=处理所有品种; 指定品种示例: ['RB', 'I', 'AG']
# N_WORKERS = max(2, cpu_count() // 8)  # 控制并行进程数，避免内存溢出
N_WORKERS = 5  # 单进程调试模式


# ============================================================
# Tick → Bar 合成
# ============================================================

def _empty_bar_frame() -> pd.DataFrame:
    """统一空bar结构，避免后续列缺失。"""
    cols = [
        'open', 'high', 'low', 'y_close',
        'x_volume', 'x_turnover', 'x_open_interest',
        'x_delta', 'x_buy_vol', 'x_sell_vol',
        'x_vwap_pv', 'x_bvwap_pv', 'x_bvwap_v',
        BAR_PERIOD_COL,
    ]
    return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name='datetime'))



def tick_to_bar(df: pd.DataFrame, freq: str, trading_day: pd.Timestamp,
                blocks_cache: dict = None, periods_cache: dict = None) -> pd.DataFrame:
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')

    trading_day = pd.Timestamp(trading_day)

    if blocks_cache is not None and trading_day in blocks_cache:
        blocks = blocks_cache[trading_day]
    else:
        blocks = build_trading_blocks(trading_day, source_ts=df.index.to_series())
        if blocks_cache is not None:
            blocks_cache[trading_day] = blocks

    cache_key = (trading_day, freq)
    if periods_cache is not None and cache_key in periods_cache:
        periods = periods_cache[cache_key]
    else:
        periods = build_period_table(blocks, freq)
        if periods_cache is not None:
            periods_cache[cache_key] = periods

    df = filter_to_trading_time(df, blocks)
    if df.empty or periods.empty:
        return _empty_bar_frame()

    # ── 增量计算：首tick用自身累计量，避免开盘集合竞价成交量被丢弃 ──────────
    # 原写法：diff().fillna(0) 会把第一个tick的成交量清零，
    # 导致开盘价所在bar的 VWAP 分子/分母偏低，造成 open > VWAP 的系统性偏差。
    _vol_diff = df['volume'].diff().clip(lower=0)
    _vol_diff.iloc[0] = max(df['volume'].iloc[0], 0)          # 首tick：用自身累计量
    df['vol_diff'] = _vol_diff.astype('float32')

    # total_turnover 必须先用 float64 做 diff，再降精度：
    # float32 仅 7 位有效数字，日内累计成交额 >1e9 时相邻 diff 会被截断为 0。
    _turnover_f64  = df['total_turnover'].astype('float64')
    _turnover_diff = _turnover_f64.diff().clip(lower=0)
    _turnover_diff.iloc[0] = max(_turnover_f64.iloc[0], 0)
    df['turnover_diff'] = _turnover_diff.astype('float32')

    # ── L1：仅用买一/卖一（原有逻辑，保持不变）─────────────────────────────
    prev_b1   = df['b1'].shift(1)
    prev_a1   = df['a1'].shift(1)
    prev_b1_v = df['b1_v'].shift(1)
    prev_a1_v = df['a1_v'].shift(1)

    df['raw_sell'] = 0.0
    mask = df['b1'] < prev_b1
    df.loc[mask, 'raw_sell'] = prev_b1_v[mask]
    mask = (df['b1'] == prev_b1) & (df['b1_v'] < prev_b1_v)
    df.loc[mask, 'raw_sell'] = (prev_b1_v - df['b1_v'])[mask]

    df['raw_buy'] = 0.0
    mask = df['a1'] > prev_a1
    df.loc[mask, 'raw_buy'] = prev_a1_v[mask]
    mask = (df['a1'] == prev_a1) & (df['a1_v'] < prev_a1_v)
    df.loc[mask, 'raw_buy'] = (prev_a1_v - df['a1_v'])[mask]

    total_raw      = df['raw_buy'] + df['raw_sell']
    df['buy_vol']  = np.where(total_raw > 0, df['vol_diff'] * df['raw_buy']  / total_raw, 0.0).astype('float32')
    df['sell_vol'] = np.where(total_raw > 0, df['vol_diff'] * df['raw_sell'] / total_raw, 0.0).astype('float32')
    df['delta_contrib'] = (df['buy_vol'] - df['sell_vol']).astype('float32')

    # ── 开盘tick方向修正 ────────────────────────────────────────────────────
    # shift(1) 在第一行产生 NaN，导致 raw_buy/raw_sell = 0，开盘成交量方向未知。
    # 用开盘价相对前收盘的涨跌方向代理：涨 → 全部归买方，跌 → 全部归卖方，平 → 各半。
    # prev_close 通常不在当日tick文件中；若无法获取则以 0.5/0.5 中性分配，
    # 避免系统性地把开盘大量成交压成 delta=0。
    _first_vol  = df['vol_diff'].iloc[0]
    if _first_vol > 0:
        _open_price = df['last'].iloc[0]
        # 尝试用当日首个b1/a1中点估算前收（仅作方向判断，不追求精确）
        _mid = (df['b1'].iloc[0] + df['a1'].iloc[0]) / 2.0
        if _open_price > _mid:          # 开盘价高于中间价 → 偏买
            df.iloc[0, df.columns.get_loc('buy_vol')]      = _first_vol
            df.iloc[0, df.columns.get_loc('sell_vol')]     = 0.0
            df.iloc[0, df.columns.get_loc('delta_contrib')]= _first_vol
        elif _open_price < _mid:        # 开盘价低于中间价 → 偏卖
            df.iloc[0, df.columns.get_loc('buy_vol')]      = 0.0
            df.iloc[0, df.columns.get_loc('sell_vol')]     = _first_vol
            df.iloc[0, df.columns.get_loc('delta_contrib')]= -_first_vol
        else:                           # 等于中间价 → 各半
            _half = _first_vol / 2.0
            df.iloc[0, df.columns.get_loc('buy_vol')]      = _half
            df.iloc[0, df.columns.get_loc('sell_vol')]     = _half
            df.iloc[0, df.columns.get_loc('delta_contrib')]= 0.0

    # ── L2：中金"买五卖五"逐档累加法（仅在有五档数据时执行）────────────────
    has_l2 = 'b2' in df.columns and 'a2' in df.columns

    if has_l2:
        raw_sell_l2 = np.zeros(len(df), dtype='float32')
        raw_buy_l2  = np.zeros(len(df), dtype='float32')

        for i in range(1, 6):
            pb   = df[f'b{i}'].shift(1).values
            pbv  = df[f'b{i}_v'].shift(1).values
            pa   = df[f'a{i}'].shift(1).values
            pav  = df[f'a{i}_v'].shift(1).values
            last = df['last'].values
            bv   = df[f'b{i}_v'].values
            av   = df[f'a{i}_v'].values

            full_sell = last < pb
            part_sell = (last == pb) & (bv < pbv)
            raw_sell_l2[full_sell] += pbv[full_sell]
            raw_sell_l2[part_sell] += np.clip(pbv - bv, 0, None)[part_sell]

            full_buy = last > pa
            part_buy = (last == pa) & (av < pav)
            raw_buy_l2[full_buy] += pav[full_buy]
            raw_buy_l2[part_buy] += np.clip(pav - av, 0, None)[part_buy]

        total_raw_l2      = raw_buy_l2 + raw_sell_l2
        vol_diff          = df['vol_diff'].values
        safe_ratio_buy  = np.divide(raw_buy_l2,  total_raw_l2, out=np.zeros_like(raw_buy_l2),  where=total_raw_l2 > 0)
        safe_ratio_sell = np.divide(raw_sell_l2, total_raw_l2, out=np.zeros_like(raw_sell_l2), where=total_raw_l2 > 0)
        df['buy_vol_l2']  = (vol_diff * safe_ratio_buy).astype('float32')
        df['sell_vol_l2'] = (vol_diff * safe_ratio_sell).astype('float32')
        df['delta_contrib_l2'] = (df['buy_vol_l2'] - df['sell_vol_l2']).astype('float32')
    # ── L2 结束 ──────────────────────────────────────────────────────────────

    # ── 快速分桶：np.searchsorted（原有逻辑，保持不变）──────────────────────
    bar_starts = periods['bar_start'].values.astype('datetime64[ns]').astype(np.int64)
    bar_ends   = periods['bar_end'].values.astype('datetime64[ns]').astype(np.int64)
    tick_ts    = df.index.values.astype('datetime64[ns]').astype(np.int64)

    bin_idx = np.searchsorted(bar_ends, tick_ts, side='right')
    valid   = (bin_idx < len(bar_starts)) & (bin_idx >= 0)
    valid[valid] &= tick_ts[valid] >= bar_starts[bin_idx[valid]]

    if not valid.any():
        return _empty_bar_frame()

    assigned            = df.iloc[np.flatnonzero(valid)].copy()
    matched_idx         = bin_idx[valid]
    assigned['bar_end'] = periods['bar_end'].values[matched_idx]
    assigned[BAR_PERIOD_COL] = periods[BAR_PERIOD_COL].values[matched_idx]

    # ── VWAP / BVWAP 中间列（tick级预计算）────────────────────────────────
    # VWAP  = Σ(last × vol_diff) / Σ(vol_diff)
    assigned['_vwap_pv'] = (assigned['last'] * assigned['vol_diff']).astype('float64')
    # BVWAP = Σ(b1×b1_v + a1×a1_v) / Σ(b1_v + a1_v)
    _b1_valid = (assigned['b1'] > 0) & (assigned['b1_v'] > 0)
    _a1_valid = (assigned['a1'] > 0) & (assigned['a1_v'] > 0)
    assigned['_bvwap_pv'] = (
        np.where(_b1_valid, assigned['b1'] * assigned['b1_v'], 0.0)
        + np.where(_a1_valid, assigned['a1'] * assigned['a1_v'], 0.0)
    ).astype('float64')
    assigned['_bvwap_v'] = (
        np.where(_b1_valid, assigned['b1_v'], 0.0)
        + np.where(_a1_valid, assigned['a1_v'], 0.0)
    ).astype('float64')

    grouped = assigned.groupby('bar_end', sort=True)
    ohlc    = grouped['last'].agg(open='first', high='max', low='min', y_close='last')

    # VWAP / BVWAP: 输出每根bar的分子分母，由 factors_backup.py 按交易日 cumsum 后算比值
    sum_vwap_pv  = grouped['_vwap_pv'].sum()
    sum_vol      = grouped['vol_diff'].sum()
    sum_bvwap_pv = grouped['_bvwap_pv'].sum()
    sum_bvwap_v  = grouped['_bvwap_v'].sum()

    parts = [
        ohlc,
        sum_vol.rename('x_volume'),
        grouped['turnover_diff'].sum().rename('x_turnover'),
        grouped['open_interest'].last().rename('x_open_interest'),
        grouped['delta_contrib'].sum().rename('x_delta'),
        grouped['buy_vol'].sum().rename('x_buy_vol'),
        grouped['sell_vol'].sum().rename('x_sell_vol'),
        sum_vwap_pv.rename('x_vwap_pv'),
        # x_vwap_v 与 x_volume 完全相同（均为 sum(vol_diff)），不重复存储；
        # factors_backup.py 计算 VWAP 时直接用 x_vwap_pv / x_volume 即可。
        sum_bvwap_pv.rename('x_bvwap_pv'),
        sum_bvwap_v.rename('x_bvwap_v'),
    ]
    if has_l2:
        parts += [
            grouped['delta_contrib_l2'].sum().rename('x_delta_l2'),
            grouped['buy_vol_l2'].sum().rename('x_buy_vol_l2'),
            grouped['sell_vol_l2'].sum().rename('x_sell_vol_l2'),
        ]
    parts.append(grouped[BAR_PERIOD_COL].first())
    bar = pd.concat(parts, axis=1)
    bar.index.name = 'datetime'

    for col in ['dominant_id', 'order_book_id']:
        if col in assigned.columns:
            bar['dominant_id'] = grouped[col].last()
            break

    return bar


def build_upper_bar_from_small(result: pd.DataFrame, freq: str) -> pd.DataFrame:
    """按交易时长逻辑将小周期bar合成为大周期bar。

    聚合规则：
      open           → 第一根小bar的 open
      high/low       → 极值
      y_close        → 最后一根小bar的 y_close
      x_volume       → 求和
      x_turnover     → 求和
      x_open_interest→ 最后一根小bar的值（持仓量取截面快照）
    """
    if result.empty or 'trading_date' not in result.columns:
        return pd.DataFrame(columns=['y_close'])

    upper_frames = []
    trading_dates = pd.to_datetime(result['trading_date']).dt.date
    for trading_day, day_df in result.groupby(trading_dates):
        day_df = day_df.sort_index()
        if day_df.empty:
            continue

        blocks = build_trading_blocks(pd.Timestamp(trading_day), source_ts=day_df.index.to_series())
        periods = build_period_table(blocks, freq)
        assigned = attach_period_info(day_df, periods, closed='right')
        if assigned.empty:
            continue

        grouped = assigned.groupby('bar_end', sort=True)

        upper = grouped['y_close'].last().to_frame('y_close')

        # ── OHLC ──────────────────────────────────────────────────────────
        if 'open' in assigned.columns:
            upper['open'] = grouped['open'].first()
        if 'high' in assigned.columns:
            upper['high'] = grouped['high'].max()
        if 'low' in assigned.columns:
            upper['low'] = grouped['low'].min()

        # ── 量价字段（求和/快照） ──────────────────────────────────────────
        if 'x_volume' in assigned.columns:
            upper['x_volume'] = grouped['x_volume'].sum()
        if 'x_turnover' in assigned.columns:
            upper['x_turnover'] = grouped['x_turnover'].sum()
        if 'x_open_interest' in assigned.columns:
            upper['x_open_interest'] = grouped['x_open_interest'].last()

        upper[BAR_PERIOD_COL] = grouped[BAR_PERIOD_COL].first()
        upper_frames.append(upper)

    if not upper_frames:
        return pd.DataFrame(columns=['y_close'])

    out = pd.concat(upper_frames).sort_index()
    out.index.name = 'datetime'
    return out


# ============================================================
# 主流程
# ============================================================

def _detect_tick_columns(first_file: str) -> list[str]:
    """读取第一个文件，确定可用列。"""
    required = [
        'datetime', 'last', 'volume', 'total_turnover', 'open_interest',
        'b1', 'a1', 'b1_v', 'a1_v'
    ]
    optional = ['trading_date', 'dominant_id', 'order_book_id']
    # L2 五档盘口列
    l2_cols = [f'{side}{i}' for i in range(2, 6) for side in ('b', 'a')]
    l2_cols += [f'{side}{i}_v' for i in range(2, 6) for side in ('b', 'a')]
    first_df = pd.read_parquet(first_file, columns=None)
    cols = required + [c for c in optional if c in first_df.columns]
    cols += [c for c in l2_cols if c in first_df.columns]
    del first_df
    return cols


def _read_one_tick_file(path: str, columns: list[str]) -> pd.DataFrame:
    """读取单个tick parquet文件，立即压缩数据类型。"""
    df = pd.read_parquet(path, columns=columns)
    df['last'] = df['last'].astype('float32')
    df['volume'] = df['volume'].astype('int32')
    # total_turnover 保留 float64：日内累计成交额可达 1e9~1e10，
    # float32 精度不足会导致 diff() 后的增量大量归零，在 tick_to_bar 中再降精度。
    df['open_interest'] = df['open_interest'].astype('int32')
    df['b1'] = df['b1'].astype('float32')
    df['a1'] = df['a1'].astype('float32')
    df['b1_v'] = df['b1_v'].astype('int32')
    df['a1_v'] = df['a1_v'].astype('int32')
    for i in range(2, 6):
        for side in ('b', 'a'):
            col_p = f'{side}{i}'
            col_v = f'{side}{i}_v'
            if col_p in df.columns:
                df[col_p] = df[col_p].astype('float32')
            if col_v in df.columns:
                df[col_v] = df[col_v].astype('int32')
    return df


def _extract_date_from_filename(path: str) -> str:
    """从文件名 {SYMBOL}_tick_yyyy-mm-dd.parquet 提取日期字符串。"""
    import re
    m = re.search(r'(\d{4}-\d{2}-\d{2})\.parquet$', os.path.basename(path))
    return m.group(1) if m else None


def _group_files_by_trading_date(files: list[str]) -> dict[str, list[str]]:
    """
    按 trading_date 分组文件。
    文件名中的日期即为 trading_date（与原始数据一致）。
    同一 trading_date 可能有多个文件（不同合约），合并处理。
    """
    groups = {}
    for f in files:
        date_str = _extract_date_from_filename(f)
        if date_str:
            groups.setdefault(date_str, []).append(f)
    return dict(sorted(groups.items()))


def process_symbol(symbol: str, base_dir: str, bar_freq: str, upper_freq: str,
                   start_date: str = None, use_parallel_days: bool = True):
    """单品种完整流程 — 逐日流式读取tick，避免一次性加载全部数据。

    Args:
        symbol: 品种代码
        base_dir: tick数据根目录
        bar_freq: 小周期频率
        upper_freq: 大周期频率
        start_date: 开始日期，格式'YYYY-MM-DD'，None表示处理所有数据
        use_parallel_days: 是否并行处理日期（当前未使用）
    """
    import gc
    import re

    pattern = os.path.join(base_dir, symbol, f"{symbol}_tick_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"[{symbol}] 无tick文件: {pattern}")
        return None

    try:
        logger.info(f"[{symbol}] 开始处理... ({len(files)} 个tick文件)")
        columns = _detect_tick_columns(files[0])
        date_groups = _group_files_by_trading_date(files)

        # 根据start_date过滤日期
        if start_date:
            start_ts = pd.Timestamp(start_date)
            date_groups = {k: v for k, v in date_groups.items()
                          if pd.Timestamp(k) >= start_ts}
            if not date_groups:
                logger.warning(f"[{symbol}] start_date={start_date}后无数据")
                return None
            logger.info(f"[{symbol}] 从{start_date}开始，共{len(date_groups)}个交易日")

        # ===== 批量读取tick → 合成bar =====
        all_bars = []
        batch_size = 200  # 每批处理200天，减少gc开销
        date_items = list(date_groups.items())
        total_days = len(date_items)

        # 缓存trading blocks和period tables
        blocks_cache = {}
        periods_cache = {}

        for batch_start in range(0, total_days, batch_size):
            batch_end = min(batch_start + batch_size, total_days)
            batch_items = date_items[batch_start:batch_end]

            for date_str, day_files in batch_items:
                trading_day = pd.Timestamp(date_str)
                try:
                    # 读取该交易日的所有tick文件
                    day_dfs = [_read_one_tick_file(f, columns) for f in day_files]
                    day_df = pd.concat(day_dfs, ignore_index=True) if len(day_dfs) > 1 else day_dfs[0]
                    del day_dfs
                except Exception as e:
                    logger.warning(f"[{symbol}] {date_str} 读取失败: {e}")
                    continue


                bar = tick_to_bar(day_df, bar_freq, trading_day, blocks_cache, periods_cache)
                del day_df

                if bar.empty:
                    continue
                bar['trading_date'] = trading_day
                all_bars.append(bar)

            # 每批结束后才gc
            gc.collect()
            if batch_end % 500 == 0 or batch_end == total_days:
                logger.info(f"[{symbol}] tick→bar进度: {batch_end}/{total_days} 天 ({batch_end*100//total_days}%)")

        if not all_bars:
            logger.warning(f"[{symbol}] 无法合成K线")
            return None

        result = pd.concat(all_bars).sort_index()
        del all_bars
        gc.collect()

        # ========== 1. 小周期因子（delta + 均线 + 动量 + 突破） ==========
        result = calc_all_factors(result, freq=bar_freq, has_delta=True)

        # ========== 2. 大周期因子（均线/趋势/动量，无delta） ==========
        upper_bar = build_upper_bar_from_small(result, upper_freq)

        if len(upper_bar) > 60:
            upper_bar = calc_all_factors(upper_bar, freq=upper_freq,
                                         close_col='y_close', has_delta=False)
            upper_factor_cols = [c for c in upper_bar.columns if c.startswith('x_')]

            upper_end = upper_bar[upper_factor_cols].copy()
            del upper_bar
            gc.collect()

            small = result.reset_index().rename(columns={result.index.name or 'index': 'datetime'})
            del result
            upper_merge = upper_end.reset_index().rename(columns={upper_end.index.name or 'index': 'datetime'})
            del upper_end
            gc.collect()
            upper_merge = upper_merge.sort_values('datetime')
            small = small.sort_values('datetime')

            merged = pd.merge_asof(
                small, upper_merge,
                on='datetime', direction='backward'
            )
            del small, upper_merge
            result = merged.set_index('datetime')
            del merged
            gc.collect()
        else:
            logger.warning(f"[{symbol}] 大周期数据不足，跳过大周期因子")

        # ========== 3. 保存 ==========
        # 确保 dominant_id 列存在
        if 'dominant_id' not in result.columns:
            result['dominant_id'] = np.nan

        freq_label = bar_freq.replace('min', 'm')  # 10min -> 10m
        out_dir = rf"D:\commodity_data\{freq_label}"
        os.makedirs(out_dir, exist_ok=True)

        n_rows = len(result)
        n_factors = len([c for c in result.columns if c.startswith('x_')])
        CSV_TAIL_LIMIT = 10000

        # 始终保存完整 parquet
        pq_path = os.path.join(out_dir, f"{symbol}_{bar_freq}.parquet")
        result.to_parquet(pq_path, engine='pyarrow')
        logger.info(f"[{symbol}] parquet已保存: {pq_path}, {n_rows} 条, 因子 {n_factors} 个")

        # CSV: 数据量大时只保留近期 CSV_TAIL_LIMIT 条，节省磁盘和内存
        csv_path = os.path.join(out_dir, f"{symbol}_{bar_freq}.csv")
        if n_rows > CSV_TAIL_LIMIT:
            result.iloc[-CSV_TAIL_LIMIT:].to_csv(csv_path)
            logger.info(f"[{symbol}] csv已保存(近{CSV_TAIL_LIMIT}条): {csv_path}")
        else:
            result.to_csv(csv_path)
            logger.info(f"[{symbol}] csv已保存(全量): {csv_path}, {n_rows} 条")

        del result
        gc.collect()
        return symbol
    except Exception as e:
        logger.error(f"[{symbol}] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        gc.collect()
        return None


def get_symbols(base_dir: str) -> list:
    """从 BASE_DIR 下扫描包含 parquet 文件的子文件夹，作为品种列表"""
    symbols = []
    for name in sorted(os.listdir(base_dir)):
        sub = os.path.join(base_dir, name)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.parquet")):
            symbols.append(name)
    return symbols


def main():
    symbols = SYMBOLS if SYMBOLS else get_symbols(BASE_DIR)
    logger.info(f"小周期={BAR_FREQ}, 大周期={UPPER_FREQ}, 检测到品种: {len(symbols)}个 {symbols}")
    if START_DATE:
        logger.info(f"开始日期: {START_DATE}")
    logger.info(f"使用 {N_WORKERS} 个并行进程处理品种")

    process_func = partial(process_symbol,
                          base_dir=BASE_DIR,
                          bar_freq=BAR_FREQ,
                          upper_freq=UPPER_FREQ,
                          start_date=START_DATE,
                          use_parallel_days=False)

    completed = []
    failed = []

    with Pool(N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(process_func, symbols), 1):
            if result is not None:
                completed.append(result)
            else:
                failed.append(symbols[i-1] if i <= len(symbols) else 'unknown')
            logger.info(f"进度: {i}/{len(symbols)} 完成")

    logger.info(f"全部完成，成功处理 {len(completed)}/{len(symbols)} 个品种")
    if failed:
        logger.warning(f"失败品种: {failed}")


if __name__ == "__main__":
    main()