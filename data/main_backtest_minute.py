import gc
import glob
import logging
import os
import re
from functools import partial
from multiprocessing import Pool

import pandas as pd

from bar_bucket_utils import (
    BAR_PERIOD_COL,
    attach_period_info,
    build_period_table,
    build_trading_blocks,
)
from factors_ import calc_all_factors, calc_tick_factor_bar


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


BASE_DIR = r"D:/commodity_data/tick"
BAR_FREQ = '60min'
START_DATE = '2001-01-01'
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
SYMBOLS = ["P"]
N_WORKERS = 5
CSV_TAIL_LIMIT = 10000


def _detect_tick_columns(first_file: str) -> list[str]:
    required = [
        'datetime', 'last', 'volume', 'total_turnover', 'open_interest',
        'b1', 'a1', 'b1_v', 'a1_v',
    ]
    optional = ['trading_date', 'dominant_id', 'order_book_id']
    depth_cols = [f'{side}{i}' for i in range(2, 6) for side in ('b', 'a')]
    depth_cols += [f'{side}{i}_v' for i in range(2, 6) for side in ('b', 'a')]

    first_df = pd.read_parquet(first_file, columns=None)
    cols = required + [c for c in optional if c in first_df.columns]
    cols += [c for c in depth_cols if c in first_df.columns]
    del first_df
    return cols


def _read_one_tick_file(path: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns)
    df['last'] = df['last'].astype('float32')
    df['volume'] = df['volume'].astype('int32')
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


def _extract_date_from_filename(path: str) -> str | None:
    match = re.search(r'(\d{4}-\d{2}-\d{2})\.parquet$', os.path.basename(path))
    return match.group(1) if match else None


def _group_files_by_trading_date(files: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in files:
        date_str = _extract_date_from_filename(path)
        if date_str:
            groups.setdefault(date_str, []).append(path)
    return dict(sorted(groups.items()))


def _select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    core_cols = [
        'open', 'high', 'low', 'y_close',
        'x_volume', 'x_turnover', 'x_open_interest',
        'trading_date', BAR_PERIOD_COL,
        'dominant_id', 'order_book_id',
    ]

    selected_cols: list[str] = []
    for col in core_cols:
        if col in df.columns and col not in selected_cols:
            selected_cols.append(col)

    for col in df.columns:
        if col.startswith('x_') and col not in selected_cols:
            selected_cols.append(col)

    return df[selected_cols].copy()


def build_upper_bar_from_small(result: pd.DataFrame, freq: str) -> pd.DataFrame:
    if result.empty or 'trading_date' not in result.columns:
        return pd.DataFrame(columns=['y_close'])

    upper_frames: list[pd.DataFrame] = []
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

        if 'open' in assigned.columns:
            upper['open'] = grouped['open'].first()
        if 'high' in assigned.columns:
            upper['high'] = grouped['high'].max()
        if 'low' in assigned.columns:
            upper['low'] = grouped['low'].min()
        if 'x_volume' in assigned.columns:
            upper['x_volume'] = grouped['x_volume'].sum()
        if 'x_turnover' in assigned.columns:
            upper['x_turnover'] = grouped['x_turnover'].sum()
        if 'x_open_interest' in assigned.columns:
            upper['x_open_interest'] = grouped['x_open_interest'].last()
        if 'x_vwap_pv' in assigned.columns:
            upper['x_vwap_pv'] = grouped['x_vwap_pv'].sum()
        if 'x_vwap_v' in assigned.columns:
            upper['x_vwap_v'] = grouped['x_vwap_v'].sum()
        if 'x_bvwap_pv' in assigned.columns:
            upper['x_bvwap_pv'] = grouped['x_bvwap_pv'].sum()
        if 'x_bvwap_v' in assigned.columns:
            upper['x_bvwap_v'] = grouped['x_bvwap_v'].sum()

        upper['trading_date'] = pd.Timestamp(trading_day)
        upper[BAR_PERIOD_COL] = grouped[BAR_PERIOD_COL].first()
        upper.index.name = 'datetime'
        upper_frames.append(upper)

    if not upper_frames:
        return pd.DataFrame(columns=['y_close'])

    return pd.concat(upper_frames).sort_index()


def process_symbol(symbol: str,
                   base_dir: str,
                   bar_freq: str,
                   upper_freq: str,
                   start_date: str | None = None) -> str | None:
    pattern = os.path.join(base_dir, symbol, f"{symbol}_tick_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(f"[{symbol}] no tick files matched: {pattern}")
        return None

    try:
        logger.info(f"[{symbol}] found {len(files)} tick files")
        columns = _detect_tick_columns(files[0])
        date_groups = _group_files_by_trading_date(files)

        if start_date:
            start_ts = pd.Timestamp(start_date)
            date_groups = {
                day: day_files
                for day, day_files in date_groups.items()
                if pd.Timestamp(day) >= start_ts
            }
            if not date_groups:
                logger.warning(f"[{symbol}] no files after start_date={start_date}")
                return None

        all_bars: list[pd.DataFrame] = []
        date_items = list(date_groups.items())
        total_days = len(date_items)
        batch_size = 200

        for batch_start in range(0, total_days, batch_size):
            batch_items = date_items[batch_start: batch_start + batch_size]
            batch_end = batch_start + len(batch_items)

            for date_str, day_files in batch_items:
                trading_day = pd.Timestamp(date_str)
                try:
                    day_frames = [_read_one_tick_file(path, columns) for path in day_files]
                    day_df = pd.concat(day_frames, ignore_index=True) if len(day_frames) > 1 else day_frames[0]
                    del day_frames
                except Exception as exc:
                    logger.warning(f"[{symbol}] failed reading {date_str}: {exc}")
                    continue

                bar = calc_tick_factor_bar(day_df, bar_freq, trading_day=trading_day)
                del day_df

                if bar.empty:
                    continue

                bar['trading_date'] = trading_day
                all_bars.append(bar)

            gc.collect()
            if batch_end % 500 == 0 or batch_end == total_days:
                logger.info(f"[{symbol}] progress {batch_end}/{total_days} trading days")

        if not all_bars:
            logger.warning(f"[{symbol}] no bars were produced")
            return None

        result = pd.concat(all_bars).sort_index()
        del all_bars
        gc.collect()

        result = calc_all_factors(
            result,
            freq=bar_freq,
            close_col='y_close',
            has_delta='x_delta' in result.columns,
        )

        upper_bar = build_upper_bar_from_small(result, upper_freq)
        if not upper_bar.empty:
            upper_bar = calc_all_factors(
                upper_bar,
                freq=upper_freq,
                close_col='y_close',
                has_delta=False,
            )
            upper_factor_cols = [
                col for col in upper_bar.columns
                if col.startswith('x_') and col.endswith(f'_{upper_freq}')
            ]
            if upper_factor_cols:
                small = (
                    result.reset_index()
                    .rename(columns={result.index.name or 'index': 'datetime'})
                    .sort_values('datetime')
                )
                upper_merge = (
                    upper_bar[upper_factor_cols]
                    .reset_index()
                    .rename(columns={upper_bar.index.name or 'index': 'datetime'})
                    .sort_values('datetime')
                )
                merged = pd.merge_asof(
                    small,
                    upper_merge,
                    on='datetime',
                    direction='backward',
                )
                result = merged.set_index('datetime')
        else:
            logger.warning(f"[{symbol}] no upper bars were produced for {upper_freq}")

        result = _select_output_columns(result)

        freq_label = bar_freq.replace('min', 'm')
        out_dir = rf"D:\commodity_data\{freq_label}"
        os.makedirs(out_dir, exist_ok=True)
        factor_count = len([col for col in result.columns if col.startswith('x_')])

        pq_path = os.path.join(out_dir, f"{symbol}_{bar_freq}.parquet")
        result.to_parquet(pq_path, engine='pyarrow')
        logger.info(f"[{symbol}] parquet saved: {pq_path}, rows={len(result)}, factor_cols={factor_count}")

        csv_path = os.path.join(out_dir, f"{symbol}_{bar_freq}.csv")
        if len(result) > CSV_TAIL_LIMIT:
            result.iloc[-CSV_TAIL_LIMIT:].to_csv(csv_path)
            logger.info(f"[{symbol}] csv tail saved: {csv_path}, rows={CSV_TAIL_LIMIT}")
        else:
            result.to_csv(csv_path)
            logger.info(f"[{symbol}] csv saved: {csv_path}, rows={len(result)}, factor_cols={factor_count}")

        del result
        gc.collect()
        return symbol
    except Exception as exc:
        logger.exception(f"[{symbol}] failed: {exc}")
        gc.collect()
        return None


def get_symbols(base_dir: str) -> list[str]:
    symbols: list[str] = []
    for name in sorted(os.listdir(base_dir)):
        subdir = os.path.join(base_dir, name)
        if os.path.isdir(subdir) and glob.glob(os.path.join(subdir, "*.parquet")):
            symbols.append(name)
    return symbols


def main() -> None:
    symbols = SYMBOLS if SYMBOLS else get_symbols(BASE_DIR)
    logger.info(f"bar_freq={BAR_FREQ}, upper_freq={UPPER_FREQ}, symbols={symbols}")
    if START_DATE:
        logger.info(f"start_date={START_DATE}")
    logger.info(f"workers={N_WORKERS}")

    process_func = partial(
        process_symbol,
        base_dir=BASE_DIR,
        bar_freq=BAR_FREQ,
        upper_freq=UPPER_FREQ,
        start_date=START_DATE,
    )

    completed: list[str] = []
    failed: list[str] = []

    with Pool(N_WORKERS) as pool:
        for symbol, result in zip(symbols, pool.imap(process_func, symbols)):
            if result is None:
                failed.append(symbol)
            else:
                completed.append(result)
            logger.info(f"completed {len(completed) + len(failed)}/{len(symbols)} symbols")

    logger.info(f"done: {len(completed)}/{len(symbols)} symbols")
    if failed:
        logger.warning(f"failed symbols: {failed}")


if __name__ == "__main__":
    main()
