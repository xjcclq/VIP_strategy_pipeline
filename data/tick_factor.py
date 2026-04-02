"""
TickFactor — Tick级因子计算类
输入: 单日 tick DataFrame + trading_day + bar_slots + vwap_start
输出: bar DataFrame，含 bar_start / bar_end / OHLCV + tick级因子

Bar 分桶规则:
  - 每个 slot = (start_time, end_time)，左开右闭
  - 夜盘 slot (hour >= 20) 对应 prev_day 日期的 tick
  - VWAP 从 vwap_start 时刻开始在当日内累计
  - VPIN 在每个 slot 内用 tick 数据独立计算，桶容量 = slot总量 / n_bucket
"""

import pandas as pd
import numpy as np
from scipy.stats import norm


# ── 模块级工具函数（供 calc_delta_factors 使用）────────────────────────────────

def _col(name: str, freq: str) -> str:
    return f'x_{name}_{freq}' if freq else f'x_{name}'


def zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(1, window // 4)).mean()
    st = s.rolling(window, min_periods=max(1, window // 4)).std().replace(0, np.nan)
    return ((s - m) / st).fillna(0.0)


def calc_delta_factors(df: pd.DataFrame, freq: str,
                       close_col: str = 'y_close') -> pd.DataFrame:
    """
    基于 delta / buy_vol / sell_vol 的衍生因子，需在 tick+bar 合并后调用。
    要求 df 中已有: x_delta, x_buy_vol, x_sell_vol, x_volume(或volume), y_close
    """
    df = df.copy()
    price_ret = df[close_col].pct_change()

    # 基础 volume 列：优先用 x_volume，否则用 volume
    vol_col = 'x_volume' if 'x_volume' in df.columns else 'volume'

    df[_col('cvd', freq)] = df['x_delta'].cumsum()
    df[_col('price_delta', freq)] = price_ret * df['x_delta']
    df[_col('price_delta_volume', freq)] = price_ret * df['x_delta'] * df[vol_col]
    df[_col('price_delta_trend', freq)] = df[_col('price_delta', freq)].cumsum()

    # OFI: (buy_vol - sell_vol) / (buy_vol + sell_vol)
    if 'x_buy_vol' in df.columns and 'x_sell_vol' in df.columns:
        total = df['x_buy_vol'] + df['x_sell_vol']
        df[_col('ofi', freq)] = (df['x_buy_vol'] - df['x_sell_vol']) / total.replace(0, np.nan)

    _dz_win, _prz_win = 20, 50
    delta_z = zscore(df['x_delta'], _dz_win)
    price_ret_z = zscore(price_ret, _prz_win)
    vol_z = zscore(df[vol_col], _prz_win)

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


class TickFactor:

    REQUIRED = ['datetime', 'last', 'volume', 'total_turnover',
                'open_interest', 'b1', 'a1', 'b1_v', 'a1_v']

    def __init__(self, tick_data, trading_day, bar_slots, vwap_start="21:00"):
        self.data = tick_data.copy()
        self.trading_day = pd.Timestamp(trading_day)
        self.bar_slots = bar_slots
        self.vwap_start = vwap_start
        missing = [c for c in self.REQUIRED if c not in self.data.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

    def _slot_ts(self, time_str):
        h, m = map(int, time_str.split(":"))
        base = self.trading_day - pd.Timedelta(days=1) if h >= 20 else self.trading_day
        return base.replace(hour=h, minute=m, second=0, microsecond=0)

    @staticmethod
    def _calc_vpin(prices: np.ndarray, volumes: np.ndarray,
                   n_bucket: int = 5, vol_decay: float = 0.8) -> float:
        """
        slot 内 tick 级 VPIN 计算，返回该 slot 的单个 VPIN 值。

        桶容量 = slot总成交量 / n_bucket，确保刚好产生 n_bucket 个桶。
        返回最终所有桶的平均买卖失衡比例。

        Args:
            prices:    tick 价格数组（last）
            volumes:   tick 成交量增量数组（vol_diff）
            n_bucket:  桶数量，建议 5（将 slot 均分为5段观测）
            vol_decay: EWMA 波动率衰减系数

        Returns:
            vpin: float，0~1 之间，越大说明知情交易越活跃
        """
        n = len(prices)
        if n < 3:
            return np.nan

        total_vol = np.nansum(volumes[volumes > 0])
        if total_vol <= 0:
            return np.nan

        # 桶容量基于 slot 总量，确保统计稳定
        bucket_max = total_vol / n_bucket

        # EWMA 估计波动率，用于 norm.cdf 权重
        price_diff = np.diff(prices, prepend=prices[0])
        ewma_var = np.empty(n)
        ewma_var[0] = ewma_var[1] = price_diff[0] ** 2
        for i in range(2, n):
            ewma_var[i] = (1 - vol_decay) * price_diff[i] ** 2 + vol_decay * ewma_var[i - 1]

        vol_est  = np.sqrt(np.maximum(ewma_var, 1e-16))
        buy_pcts = norm.cdf(price_diff / np.maximum(vol_est, 1e-8))
        buy_pcts[:2] = 0.5   # 前2条 tick 无法估计方向，设为均分

        # 装桶
        bkt_vol  = np.zeros(n_bucket)
        bkt_buy  = np.zeros(n_bucket)
        bkt_sell = np.zeros(n_bucket)
        cur_idx  = 0

        for i in range(n):
            if np.isnan(prices[i]) or np.isnan(volumes[i]) or volumes[i] <= 0:
                continue
            remaining = volumes[i]
            bp = buy_pcts[i]
            while remaining > 0:
                fill = min(remaining, bucket_max - bkt_vol[cur_idx])
                bkt_vol[cur_idx]  += fill
                bkt_buy[cur_idx]  += bp * fill
                bkt_sell[cur_idx] += (1 - bp) * fill
                remaining -= fill
                if bkt_vol[cur_idx] >= bucket_max:
                    cur_idx = (cur_idx + 1) % n_bucket
                    bkt_vol[cur_idx] = bkt_buy[cur_idx] = bkt_sell[cur_idx] = 0.0

        # VPIN = 所有桶的平均买卖失衡 / 桶容量
        total = bkt_vol.sum()
        if total <= 0:
            return np.nan
        return np.abs(bkt_buy - bkt_sell).sum() / total

    def calc_bar_and_factors(self):
        df = self.data.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        # ---------- 增量计算 ----------
        vol_diff = df['volume'].diff().clip(lower=0)
        vol_diff.iloc[0] = max(df['volume'].iloc[0], 0)
        df['vol_diff'] = vol_diff

        to_diff = df['total_turnover'].astype('float64').diff().clip(lower=0)
        to_diff.iloc[0] = max(df['total_turnover'].iloc[0], 0)
        df['turnover_diff'] = to_diff

        # ---------- Delta（L1买卖压力）----------
        prev_b1, prev_a1 = df['b1'].shift(1), df['a1'].shift(1)
        prev_b1_v, prev_a1_v = df['b1_v'].shift(1), df['a1_v'].shift(1)

        raw_sell = pd.Series(0.0, index=df.index)
        m = df['b1'] < prev_b1
        raw_sell[m] = prev_b1_v[m]
        m = (df['b1'] == prev_b1) & (df['b1_v'] < prev_b1_v)
        raw_sell[m] = (prev_b1_v - df['b1_v'])[m]

        raw_buy = pd.Series(0.0, index=df.index)
        m = df['a1'] > prev_a1
        raw_buy[m] = prev_a1_v[m]
        m = (df['a1'] == prev_a1) & (df['a1_v'] < prev_a1_v)
        raw_buy[m] = (prev_a1_v - df['a1_v'])[m]

        total_raw = raw_buy + raw_sell
        df['buy_vol']       = np.where(total_raw > 0, df['vol_diff'] * raw_buy  / total_raw, 0.0)
        df['sell_vol']      = np.where(total_raw > 0, df['vol_diff'] * raw_sell / total_raw, 0.0)
        df['delta_contrib'] = df['buy_vol'] - df['sell_vol']
        df['_vwap_pv']      = df['last'].astype('float64') * df['vol_diff']
        # df['spread']        = (df['a1'] - df['b1']) / df['last']

        # ---------- Slot 分配 ----------
        slot_boundaries = [(self._slot_ts(s), self._slot_ts(e), i)
                           for i, (s, e) in enumerate(self.bar_slots)]
        slot_idx = pd.Series(-1, index=df.index, dtype=int)
        for start_ts, end_ts, i in slot_boundaries:
            mask = (df['datetime'] > start_ts) & (df['datetime'] <= end_ts)
            slot_idx[mask] = i
        df['_slot'] = slot_idx
        valid = df[df['_slot'] >= 0].copy()

        # ---------- 按 slot 聚合，VPIN 在 slot 内用 tick 计算 ----------
        def agg_slot(grp):
            prices  = grp['last'].values.astype(float)
            volumes = grp['vol_diff'].values.astype(float)

            return pd.Series({
                'open':            grp['last'].iloc[0],
                'high':            grp['last'].max(),
                'low':             grp['last'].min(),
                'close':           grp['last'].iloc[-1],
                'volume':          grp['vol_diff'].sum(),
                'x_turnover':      grp['turnover_diff'].sum(),
                'x_open_interest': grp['open_interest'].iloc[-1],
                'x_delta':         grp['delta_contrib'].sum(),
                'x_buy_vol':       grp['buy_vol'].sum(),
                'x_sell_vol':      grp['sell_vol'].sum(),
                # VPIN：slot 内 tick 数据独立计算，桶容量=slot总量/5
                # 'x_vpin':          TickFactor._calc_vpin(prices, volumes, n_bucket=5),
                '_vwap_pv':        grp['_vwap_pv'].sum(),
                '_vwap_v':         grp['vol_diff'].sum(),
            })

        bar = valid.groupby('_slot', sort=True).apply(agg_slot)

        # ---------- 时间列 ----------
        slot_start = {i: self._slot_ts(s) for i, (s, _) in enumerate(self.bar_slots)}
        slot_end   = {i: self._slot_ts(e) for i, (_, e) in enumerate(self.bar_slots)}
        bar['bar_start'] = bar.index.map(slot_start)
        bar['bar_end']   = bar.index.map(slot_end)
        bar.index = bar['bar_end']
        bar.index.name = 'datetime'

        # ---------- VWAP / OFI ----------
        bar['x_vwap']     = bar['_vwap_pv'].cumsum() / bar['_vwap_v'].cumsum().replace(0, np.nan)
        bar['x_vwap_dev'] = (bar['close'] - bar['x_vwap']) / bar['x_vwap']
        bar['x_ofi']      = (bar['x_delta'] / bar['volume'].replace(0, np.nan)).fillna(0.0)
        bar = bar.drop(columns=['_vwap_pv', '_vwap_v'], errors='ignore')
        bar = bar[bar['volume'] > 0].copy()
        bar['trading_date'] = self.trading_day

        # ---------- VPIN zscore（跨 bar 滚动标准化，用于异常检测）----------
        # vpin_s = bar['x_vpin']
        # vpin_z = (vpin_s - vpin_s.rolling(20).mean()) / vpin_s.rolling(20).std().clip(lower=1e-9)
        # bar['x_vpin_zscore']          = vpin_z
        # bar['x_vpin_zscore_filtered'] = np.where(vpin_z >= 2, vpin_z, 0.0)

        # ---------- 列顺序 ----------
        front = ['bar_start', 'bar_end', 'open', 'high', 'low', 'close', 'volume', 'trading_date']
        x_cols = [c for c in bar.columns if c.startswith('x_')]
        return bar[front + x_cols]