"""
main_data — 仅处理1小时数据
流程: Tick → 1H Bar + Tick因子 → Bar因子 → 合并 → 保存
数据源: {BASE_DIR}/{品种}_tick_yyyy-mm-dd.parquet
"""

import os
import gc
import re
import glob
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from tick_factor import TickFactor, calc_delta_factors
from futures_factors import FuturesFactors

script_path = Path(__file__).resolve()
project_root = script_path.parents[1]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ========== 配置 ==========
SYMBOL = "P"
BASE_DIR = os.path.join(r'D:\commodity_data\tick', f'{SYMBOL}')
START_DATE = '2020.01.01'
BATCH_SIZE = 200

# BAR_SLOTS = [
#     ("21:00", "22:00"),
#     ("22:00", "22:55"),
#     ("09:00", "10:00"),
#     ("10:00", "11:15"),
#     ("11:15", "14:15"),
#     ("14:15", "14:55"),
# ]
BAR_SLOTS = [
    ("21:00", "22:59"),
    ("09:00", "11:29"),
    ("11:30", "14:59"),
]

VWAP_START = "21:00"

REQUIRED_COLS = ['datetime', 'last', 'volume', 'total_turnover', 'open_interest',
                 'b1', 'a1', 'b1_v', 'a1_v']
OPTIONAL_COLS = ['trading_date', 'dominant_id', 'order_book_id']
L2_COLS = [f'{s}{i}' for i in range(2, 6) for s in ('b', 'a')] + \
          [f'{s}{i}_v' for i in range(2, 6) for s in ('b', 'a')]


def _detect_columns(first_file: str) -> list[str]:
    schema = pd.read_parquet(first_file, columns=[]).columns.tolist()
    return REQUIRED_COLS + [c for c in OPTIONAL_COLS + L2_COLS if c in schema]


def _read_tick(path: str, columns: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=columns)
    df['last'] = df['last'].astype('float32')
    df['volume'] = df['volume'].astype('int32')
    df['open_interest'] = df['open_interest'].astype('int32')
    for col in ('b1', 'a1'):
        df[col] = df[col].astype('float32')
    for col in ('b1_v', 'a1_v'):
        df[col] = df[col].astype('int32')
    return df


def _extract_date(path: str) -> str:
    m = re.search(r'(\d{4}-\d{2}-\d{2})\.parquet$', os.path.basename(path))
    return m.group(1) if m else None


def _group_by_date(files: list[str]) -> dict[str, list[str]]:
    groups = {}
    for f in files:
        d = _extract_date(f)
        if d:
            groups.setdefault(d, []).append(f)
    return dict(sorted(groups.items()))


def process_data():
    files = sorted(glob.glob(os.path.join(BASE_DIR, "*_tick_*.parquet")))
    if not files:
        logger.warning(f"[{SYMBOL}] 无tick文件")
        return None

    logger.info(f"[{SYMBOL}] {len(files)} 个tick文件")
    columns = _detect_columns(files[0])
    date_groups = _group_by_date(files)

    if START_DATE:
        date_groups = {k: v for k, v in date_groups.items()
                       if pd.Timestamp(k) >= pd.Timestamp(START_DATE)}

    date_items = list(date_groups.items())
    total = len(date_items)
    all_bars = []

    for batch_start in range(0, total, BATCH_SIZE):
        batch = date_items[batch_start:batch_start + BATCH_SIZE]

        for date_str, day_files in batch:
            trading_day = pd.Timestamp(date_str)
            try:
                dfs = [_read_tick(f, columns) for f in day_files]
                tick_df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
                del dfs
            except Exception as e:
                logger.warning(f"[{SYMBOL}] {date_str} 读取失败: {e}")
                continue

            try:
                tf = TickFactor(tick_df, trading_day, bar_slots=BAR_SLOTS, vwap_start=VWAP_START)
                bar = tf.calc_bar_and_factors()
            except Exception as e:
                logger.warning(f"[{SYMBOL}] {date_str} tick因子失败: {e}")
                bar = pd.DataFrame()
            finally:
                del tick_df

            if not bar.empty:
                all_bars.append(bar)

        gc.collect()
        done = min(batch_start + BATCH_SIZE, total)
        if done % 500 == 0 or done == total:
            logger.info(f"[{SYMBOL}] 进度: {done}/{total}")

    if not all_bars:
        logger.warning(f"[{SYMBOL}] 无有效bar")
        return None

    result = pd.concat(all_bars).sort_index()
    del all_bars
    gc.collect()

    # Bar级因子
    tick_cols = [c for c in result.columns if c.startswith('x_')]
    tick_factors = result[tick_cols].copy()

    ff = FuturesFactors(result[['open', 'high', 'low', 'close', 'volume']], tick_factors=tick_factors)
    bar_factors = ff.calculate_all_factors()
    bar_factors.columns = [c if c.startswith('x_') else 'x_' + c for c in bar_factors.columns]

    result = pd.concat([result[['open', 'high', 'low', 'close', 'volume', 'trading_date']],
                         tick_factors, bar_factors], axis=1)
    result.rename(columns={'open': 'y_open', 'close': 'y_close'}, inplace=True)

    # Delta 衍生因子（需要 y_close, x_delta, x_buy_vol, x_sell_vol, volume）
    result = calc_delta_factors(result, freq='', close_col='y_close')

    # 保存
    out_dir = os.path.join(project_root, "data")
    os.makedirs(out_dir, exist_ok=True)
    pq_path = os.path.join(out_dir, f"{SYMBOL}.parquet")
    csv_path = os.path.join(out_dir, f"{SYMBOL}.csv")
    result.to_parquet(pq_path, engine='pyarrow')
    result.to_csv(csv_path)

    n_factors = len([c for c in result.columns if c.startswith('x_')])
    logger.info(f"[{SYMBOL}] 已保存: {pq_path}, {len(result)}条, {n_factors}个因子")

    del result
    gc.collect()
    return SYMBOL


def main():
    result = process_data()
    if result:
        logger.info("完成")
    else:
        logger.warning("处理失败或无有效数据")


if __name__ == "__main__":
    main()
