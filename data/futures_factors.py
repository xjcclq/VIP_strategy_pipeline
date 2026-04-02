import pandas as pd
import numpy as np
import talib
from scipy.stats import norm


class FuturesFactors:
    """期货量价因子计算类"""

    def __init__(self, data: pd.DataFrame,
                 tick_factors: pd.DataFrame = None,
                 add_session_features: bool = True):
        self.data = data.copy()
        self.tick_factors = tick_factors
        self.add_session_features = add_session_features
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype('float64')
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)
        if self.data.empty:
            raise ValueError("输入数据为空")
        missing_cols = [col for col in numeric_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        print("FuturesFactors初始化完成")

    # ==================== 价格因子 ====================

    def returns(self, periods: int = 1) -> pd.Series:
        return self.data['close'].pct_change(periods)

    def log_returns(self, periods: int = 1) -> pd.Series:
        return np.log(self.data['close'] / self.data['close'].shift(periods))

    def momentum(self, window: int = 20) -> pd.Series:
        return self.data['close'] / self.data['close'].shift(window) - 1

    def rsi(self, window: int = 14) -> pd.Series:
        result = talib.RSI(self.data['close'].values, timeperiod=window)
        return pd.Series(result, index=self.data.index)

    def williams_r(self, window: int = 14) -> pd.Series:
        result = talib.WILLR(self.data['high'].values, self.data['low'].values,
                             self.data['close'].values, timeperiod=window)
        return pd.Series(result, index=self.data.index)

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        macd, sig, hist = talib.MACD(self.data['close'].values,
                                     fastperiod=fast, slowperiod=slow, signalperiod=signal)
        return pd.DataFrame({'macd': macd, 'signal': sig, 'histogram': hist},
                            index=self.data.index)

    def bollinger_position(self, window: int = 20, std: float = 2) -> pd.Series:
        middle  = self.data['close'].rolling(window).mean()
        std_dev = self.data['close'].rolling(window).std()
        upper   = middle + std * std_dev
        lower   = middle - std * std_dev
        return (self.data['close'] - lower) / (upper - lower)

    def boll_upper(self, window: int = 20, std: float = 2) -> pd.Series:
        ma = self.data['close'].rolling(window).mean()
        return ma + std * self.data['close'].rolling(window).std()

    def boll_lower(self, window: int = 20, std: float = 2) -> pd.Series:
        ma = self.data['close'].rolling(window).mean()
        return ma - std * self.data['close'].rolling(window).std()

    def price_ma250_dev(self) -> pd.Series:
        ma250 = self.data['close'].rolling(250, min_periods=1).mean()
        return (self.data['close'] - ma250) / ma250 * 100

    def intraday_return(self) -> pd.Series:
        return (self.data['close'] - self.data['open']) / self.data['open']

    def overnight_return(self) -> pd.Series:
        return (self.data['open'] - self.data['close'].shift(1)) / self.data['close'].shift(1)

    def price_acceleration(self, window: int = 5) -> pd.Series:
        """收益率的变化率，捕捉动量加速/减速"""
        return self.returns().diff(window)

    # ==================== 成交量因子 ====================

    def volume_change_rate(self, periods: int = 1) -> pd.Series:
        return self.data['volume'].pct_change(periods)

    def volume_momentum(self, window: int = 20) -> pd.Series:
        return self.data['volume'] / self.data['volume'].shift(window) - 1

    def relative_volume(self, window: int = 20) -> pd.Series:
        return self.data['volume'] / self.data['volume'].rolling(window).mean()

    def volume_ma_ratio(self, short: int = 5, long: int = 20) -> pd.Series:
        return self.data['volume'].rolling(short).mean() / self.data['volume'].rolling(long).mean()

    def volume_acceleration(self, window: int = 5) -> pd.Series:
        return self.volume_change_rate().diff(window)

    def obv(self) -> pd.Series:
        result = talib.OBV(self.data['close'].values, self.data['volume'].values)
        return pd.Series(result, index=self.data.index)

    # ==================== 量价结合因子 ====================

    def vwap(self) -> pd.Series:
        return (self.data['high'] + self.data['low'] + self.data['close']) / 3

    def price_vwap_deviation(self, window: int = 20) -> pd.Series:
        vwap    = self.vwap()
        vwap_ma = vwap.rolling(window).mean()
        return (self.data['close'] - vwap_ma) / vwap_ma

    def money_flow_index(self, window: int = 14) -> pd.Series:
        result = talib.MFI(self.data['high'].values, self.data['low'].values,
                           self.data['close'].values, self.data['volume'].values,
                           timeperiod=window)
        return pd.Series(result, index=self.data.index)

    def price_volume_trend(self) -> pd.Series:
        return (self.data['close'].pct_change() * self.data['volume']).cumsum()

    def ease_of_movement(self, window: int = 14, scale: int = 10000) -> pd.Series:
        high, low, volume = self.data['high'], self.data['low'], self.data['volume']
        distance  = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
        box_ratio = volume / scale / (high - low)
        return (distance / box_ratio).rolling(window).mean()

    def volume_weighted_return(self, window: int = 20) -> pd.Series:
        log_ret   = self.log_returns()
        total_vol = self.data['volume'].rolling(window).sum()
        return (log_ret * self.data['volume']).rolling(window).sum() / total_vol

    def money_flow_ratio(self, window: int = 14) -> pd.Series:
        tp  = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        mf  = tp * self.data['volume']
        pos = mf.where(tp > tp.shift(1), 0).rolling(window).sum()
        neg = mf.where(tp < tp.shift(1), 0).rolling(window).sum()
        return pos / (pos + neg.abs())

    def volume_price_correlation(self, window: int = 20) -> pd.Series:
        return self.log_returns().rolling(window).corr(self.volume_change_rate())

    def volume_profile_indicator_fast(self, window: int = 20, bins: int = 10) -> pd.Series:
        close_arr = self.data['close'].values.astype(float)
        vol_arr   = self.data['volume'].values.astype(float)
        n         = len(close_arr)
        results   = np.full(n, np.nan)
        for i in range(window - 1, n):
            prices  = close_arr[i - window + 1: i + 1]
            volumes = vol_arr[i - window + 1: i + 1]
            pmin, pmax = prices.min(), prices.max()
            if pmax == pmin or np.isnan(pmin):
                continue
            bin_idx    = np.clip(
                np.digitize(prices, np.linspace(pmin, pmax, bins + 1)) - 1, 0, bins - 1
            )
            results[i] = np.argmax(np.bincount(bin_idx, weights=volumes, minlength=bins)) / bins
        return pd.Series(results, index=self.data.index)

    # ==================== 波动率因子 ====================

    def realized_volatility(self, window: int = 20) -> pd.Series:
        return self.log_returns().rolling(window).std() * np.sqrt(252)

    def intraday_volatility(self) -> pd.Series:
        return np.log(self.data['high'] / self.data['low'])

    def overnight_volatility(self) -> pd.Series:
        return np.log(self.data['open'] / self.data['close'].shift(1))

    def atr(self, window: int = 14) -> pd.Series:
        result = talib.ATR(self.data['high'].values, self.data['low'].values,
                           self.data['close'].values, timeperiod=window)
        return pd.Series(result, index=self.data.index)

    def parkinson_volatility(self, window: int = 20) -> pd.Series:
        log_hl2 = np.log(self.data['high'] / self.data['low']) ** 2
        return np.sqrt(log_hl2.rolling(window).mean() / (4 * np.log(2)))

    def volatility_ratio(self, short: int = 10, long: int = 30) -> pd.Series:
        log_ret = self.log_returns()
        return log_ret.rolling(short).std() / log_ret.rolling(long).std()

    def skewness(self, window: int = 20) -> pd.Series:
        return self.log_returns().rolling(window).skew()

    def kurtosis(self, window: int = 20) -> pd.Series:
        return self.log_returns().rolling(window).kurt()

    # ==================== 流动性因子 ====================

    def bid_ask_spread_proxy(self, window: int = 20) -> pd.Series:
        spread = (self.data['high'] - self.data['low']) / self.data['close']
        return spread.rolling(window).mean()

    def price_impact(self, window: int = 5) -> pd.Series:
        return self.returns().abs().rolling(window).corr(self.volume_change_rate())

    def market_depth_imbalance(self, window: int = 20) -> pd.Series:
        close, high, low, vol = (self.data['close'], self.data['high'],
                                 self.data['low'], self.data['volume'])
        high_vol = (close == high).astype(int) * vol
        low_vol  = (close == low).astype(int) * vol
        return (high_vol.rolling(window).sum() - low_vol.rolling(window).sum()) / \
               vol.rolling(window).sum()

    def amihud_illiquidity(self, window: int = 20) -> pd.Series:
        ret_abs  = self.log_returns().abs()
        turnover = self.data['close'] * self.data['volume']
        illiq    = ret_abs / turnover.replace(0, np.nan)
        mu       = illiq.rolling(window).mean()
        std      = illiq.rolling(window).std().replace(0, np.nan)
        return (illiq - mu) / std

    def amihud_volatility(self, window: int = 20) -> pd.Series:
        ret_abs  = self.log_returns().abs()
        turnover = self.data['close'] * self.data['volume']
        illiq    = ret_abs / turnover.replace(0, np.nan)
        vol      = illiq.rolling(window).std()
        mu       = vol.rolling(window).mean()
        std      = vol.rolling(window).std().replace(0, np.nan)
        return (vol - mu) / std

    def liquidity_indicator(self, window: int = 20) -> pd.Series:
        turnover   = self.data['volume'] * self.data['close']
        volatility = self.log_returns().abs()
        return (turnover / (volatility + 1e-8)).rolling(window).mean()

    # ==================== 技术指标 ====================

    def stochastic_oscillator(self, k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        slowk, slowd = talib.STOCH(self.data['high'].values, self.data['low'].values,
                                   self.data['close'].values,
                                   fastk_period=k_window, slowk_period=3, slowd_period=d_window)
        return pd.DataFrame({'stoch_k': slowk, 'stoch_d': slowd}, index=self.data.index)

    def cci(self, window: int = 20) -> pd.Series:
        result = talib.CCI(self.data['high'].values, self.data['low'].values,
                           self.data['close'].values, timeperiod=window)
        return pd.Series(result, index=self.data.index)

    def adx(self, window: int = 14) -> pd.Series:
        result = talib.ADX(self.data['high'].values, self.data['low'].values,
                           self.data['close'].values, timeperiod=window)
        return pd.Series(result, index=self.data.index)

    def aroon(self, window: int = 14) -> pd.DataFrame:
        aroon_up, aroon_down = talib.AROON(self.data['high'].values,
                                           self.data['low'].values, timeperiod=window)
        return pd.DataFrame({
            'aroon_up':         aroon_up,
            'aroon_down':       aroon_down,
            'aroon_oscillator': aroon_up - aroon_down
        }, index=self.data.index)

    def trix(self, window: int = 14) -> pd.Series:
        return pd.Series(talib.TRIX(self.data['close'].values, timeperiod=window),
                         index=self.data.index)

    def chaikin_oscillator(self, fast: int = 3, slow: int = 10) -> pd.Series:
        result = talib.ADOSC(self.data['high'].values, self.data['low'].values,
                             self.data['close'].values, self.data['volume'].values,
                             fastperiod=fast, slowperiod=slow)
        return pd.Series(result, index=self.data.index)

    def accumulation_distribution(self) -> pd.Series:
        result = talib.AD(self.data['high'].values, self.data['low'].values,
                          self.data['close'].values, self.data['volume'].values)
        return pd.Series(result, index=self.data.index)

    def parabolic_sar(self, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        result = talib.SAR(self.data['high'].values, self.data['low'].values,
                           acceleration=acceleration, maximum=maximum)
        return pd.Series(result, index=self.data.index)

    # ==================== 通道/位置因子 ====================

    def keltner_channel_position(self, window: int = 20, multiplier: float = 2) -> pd.Series:
        ema = self.data['close'].ewm(span=window).mean()
        atr = self.atr(window)
        return (self.data['close'] - (ema - multiplier * atr)) / (2 * multiplier * atr)

    def donchian_channel_position(self, window: int = 20) -> pd.Series:
        high_max = self.data['high'].rolling(window).max()
        low_min  = self.data['low'].rolling(window).min()
        return (self.data['close'] - low_min) / (high_max - low_min)

    def price_channel_breakout(self, window: int = 20) -> pd.Series:
        high_max = self.data['high'].rolling(window).max().shift(1)
        low_min  = self.data['low'].rolling(window).min().shift(1)
        return (self.data['close'] > high_max).astype(int) + \
               (self.data['close'] < low_min).astype(int) * -1

    def close_position_ratio(self) -> pd.Series:
        return (self.data['close'] - self.data['low']) / (self.data['high'] - self.data['low'])

    def high_low_ratio(self, window: int = 20) -> pd.Series:
        return (self.data['high'] / self.data['low']).rolling(window).mean()

    # ==================== 统计/高级因子 ====================

    def hurst_exponent(self, window: int = 100) -> pd.Series:
        log_price = np.log(self.data['close']).values.astype(float)
        n         = len(log_price)
        lags      = np.arange(2, 20)
        log_lags  = np.log(lags)
        lags_mean = log_lags.mean()
        lags_var  = ((log_lags - lags_mean) ** 2).sum()

        log_tau_matrix = np.full((n, len(lags)), np.nan)
        for j, lag in enumerate(lags):
            lag  = int(lag)
            diff = log_price[lag:] - log_price[:-lag]
            s    = pd.Series(diff)
            win  = window - lag
            if win < 2:
                continue
            std_vals = s.rolling(win, min_periods=win).std(ddof=0).values
            std_vals = np.sqrt(np.maximum(std_vals, 0))
            valid    = std_vals > 0
            log_vals = np.where(valid, np.log(std_vals), np.nan)
            pos      = np.arange(len(log_vals)) + lag
            mask     = (pos >= window - 1) & (pos < n) & valid
            log_tau_matrix[pos[mask], j] = log_vals[mask]

        hurst_arr = np.full(n, np.nan)
        for i in range(window - 1, n):
            lt = log_tau_matrix[i]
            if not np.any(np.isnan(lt)):
                slope        = ((lt - lt.mean()) * (log_lags - lags_mean)).sum() / lags_var
                hurst_arr[i] = slope * 2.0

        return pd.Series(hurst_arr, index=self.data.index)

    def fractal_dimension(self, window: int = 20) -> pd.Series:
        close_arr = self.data['close'].values.astype(float)
        n         = len(close_arr)
        log_denom = np.log(2 * (window - 1))
        fd_arr    = np.full(n, np.nan)
        for i in range(window - 1, n):
            prices = close_arr[i - window + 1: i + 1]
            if np.isnan(prices).any():
                continue
            L = np.sum(np.abs(np.diff(prices)))
            if L > 0:
                fd_arr[i] = 1 + np.log(L) / log_denom
        return pd.Series(fd_arr, index=self.data.index)

    def trend_strength(self, window: int = 20) -> pd.Series:
        price_change = self.data['close'] - self.data['close'].shift(window)
        price_vol    = self.data['close'].rolling(window).std()
        return price_change.abs() / (price_vol + 1e-8)

    def ad_factor(self, period: int = 14) -> pd.Series:
        close, high, low, vol = (self.data['close'], self.data['high'],
                                 self.data['low'], self.data['volume'])
        denom  = (high - low).replace(0, np.nan)
        ad_raw = (2 * close - low - high) / denom * vol / vol.rolling(period).mean()
        return ad_raw.ewm(span=period, adjust=False).mean()

    def qstick(self, period: int = 14) -> pd.Series:
        return ((self.data['close'] - self.data['open']) / self.data['close']).rolling(period).mean()

    def sharpe_ratio(self, period: int = 20) -> pd.Series:
        ret = np.log(self.data['close'] / self.data['close'].shift(1))
        return ret.rolling(period).mean() / ret.rolling(period).std()

    def rvi(self, period: int = 14) -> pd.Series:
        gain = self.data['close'].diff()
        gps  = gain.clip(lower=0).rolling(period).std()
        gms  = (-gain).clip(lower=0).rolling(period).std()
        return (gps - gms) / (gps + gms)

    def klinger(self, dem_period: int = 13) -> pd.Series:
        close, high, low, vol = (self.data['close'], self.data['high'],
                                 self.data['low'], self.data['volume'])
        avg_price = (close + high + low) / 3
        sv = pd.Series(
            np.where(avg_price > avg_price.shift(1),
                     vol / vol.rolling(dem_period).sum(),
                     -vol / vol.rolling(dem_period).sum()),
            index=self.data.index
        )
        return (sv.ewm(span=55, adjust=False).mean() -
                sv.ewm(span=34, adjust=False).mean()).ewm(span=dem_period, adjust=False).mean()

    def linearreg_ma(self, period: int = 14) -> pd.Series:
        close = self.data['close']
        x     = np.arange(period, dtype=float)
        sumx  = x.sum(); sumx2 = (x * x).sum()
        sumy  = close.rolling(period).sum()
        sumxy = close.rolling(period).apply(lambda c: (c * x).sum(), raw=True)
        slope = (period * sumxy - sumx * sumy) / (period * sumx2 - sumx * sumx)
        intercept = (sumy - slope * sumx) / period
        ey = np.log((slope * (period - 1) + intercept) / close)
        return ey.ewm(span=period, adjust=False).mean()

    # ==================== Kyle Lambda ====================

    def kyle_lambda(self, window: int = 20) -> pd.Series:
        """Kyle Lambda 同期：λ = Cov(R_t, OFI_t) / Var(OFI_t)"""
        tf = self.tick_factors
        if tf is None or ('x_ofi' not in tf.columns and 'x_delta' not in tf.columns):
            return pd.Series(np.nan, index=self.data.index)
        ret = self.log_returns()
        ofi = tf['x_ofi'].reindex(self.data.index) if 'x_ofi' in tf.columns else \
              tf['x_delta'].reindex(self.data.index) / self.data['volume'].replace(0, np.nan)
        cov = ret.rolling(window).cov(ofi)
        var = ofi.rolling(window).var().replace(0, np.nan)
        return cov / var

    def kyle_lambda_lag(self, window: int = 20) -> pd.Series:
        """Kyle Lambda 滞后：λ = Cov(R_t, OFI_{t-1}) / Var(OFI_{t-1})"""
        tf = self.tick_factors
        if tf is None or ('x_ofi' not in tf.columns and 'x_delta' not in tf.columns):
            return pd.Series(np.nan, index=self.data.index)
        ret     = self.log_returns()
        ofi     = tf['x_ofi'].reindex(self.data.index) if 'x_ofi' in tf.columns else \
                  tf['x_delta'].reindex(self.data.index) / self.data['volume'].replace(0, np.nan)
        ofi_lag = ofi.shift(1)
        cov     = ret.rolling(window).cov(ofi_lag)
        var     = ofi_lag.rolling(window).var().replace(0, np.nan)
        return cov / var

    # ==================== 会话特征 ====================

    def session_features(self) -> pd.DataFrame:
        """
        会话虚拟变量：
          x_is_night    : 0=白盘 / 1=夜盘
          x_session_sin : sin(π × 盘内归一化进度)，捕捉开/收盘对称效应

        棕榈油交易时段：
          白盘  09:00-15:00（540-900 min，含午休线性插值）
          夜盘  21:00-23:00（1260-1380 min）
        日线数据（无时间分量）两列均填 0。
        """
        idx      = pd.to_datetime(self.data.index)
        has_time = (idx.hour != 0).any() or (idx.minute != 0).any()

        if not has_time:
            print("[会话特征] 日线数据，x_is_night/x_session_sin 均填 0")
            return pd.DataFrame({
                'is_night':    0.0,
                'session_sin': 0.0,
            }, index=self.data.index)

        hm = idx.hour * 60 + idx.minute   # 距午夜分钟数

        # ── 夜盘哑变量：21:00(1260) ≤ hm < 23:00(1380) ──────────────────
        is_night = ((hm >= 1260) & (hm < 1380)).astype(float)

        # ── 盘内归一化进度 [0, 1] ────────────────────────────────────────
        # 白盘 09:00(540)–15:00(900)：progress = (hm - 540) / 360
        # 夜盘 21:00(1260)–23:00(1380)：progress = (hm - 1260) / 120
        # 其余时段默认 0.5
        progress = pd.Series(0.5, index=idx, dtype=float)

        day_mask   = (hm >= 540)  & (hm < 900)
        night_mask = (hm >= 1260) & (hm < 1380)

        progress[day_mask]   = (hm[day_mask]   - 540)  / 360.0
        progress[night_mask] = (hm[night_mask]  - 1260) / 120.0
        progress             = progress.clip(0.0, 1.0)

        # sin(π·p)：盘初≈0 → 盘中≈1 → 盘尾≈0，天然捕捉开/收盘对称效应
        session_sin = np.sin(np.pi * progress)

        night_cnt = int(is_night.sum())
        day_cnt   = len(is_night) - night_cnt
        print(f"[会话特征] 白盘={day_cnt} bars  夜盘={night_cnt} bars  "
              f"sin均值={session_sin.mean():.3f}")

        return pd.DataFrame({
            'is_night':    is_night,
            'session_sin': session_sin,
        }, index=self.data.index)

    # ==================== 综合计算 ====================

    def calculate_all_factors(self) -> pd.DataFrame:
        factors = pd.DataFrame(index=self.data.index)

        # ── 价格 / 动量 ──────────────────────────────────────────────────────
        factors['return_1d']               = self.returns(1)
        factors['return_5d']               = self.returns(5)
        factors['log_return_1d']           = self.log_returns(1)
        factors['momentum_20d']            = self.momentum(20)
        factors['rsi_14d']                 = self.rsi(14)
        factors['williams_r']              = self.williams_r(14)
        factors['price_acceleration']      = self.price_acceleration()
        factors['intraday_return']         = self.intraday_return()
        factors['overnight_return']        = self.overnight_return()

        macd_data = self.macd()
        factors['macd']                    = macd_data['macd']
        factors['macd_signal']             = macd_data['signal']
        factors['macd_histogram']          = macd_data['histogram']

        factors['bollinger_position']      = self.bollinger_position()
        factors['boll_upper']              = self.boll_upper()
        factors['boll_lower']              = self.boll_lower()
        factors['price_ma250_dev']         = self.price_ma250_dev()

        # ── 成交量 ───────────────────────────────────────────────────────────
        factors['volume_change_rate']      = self.volume_change_rate()
        factors['volume_momentum']         = self.volume_momentum()
        factors['relative_volume']         = self.relative_volume()
        factors['volume_ma_ratio']         = self.volume_ma_ratio()
        factors['volume_acceleration']     = self.volume_acceleration()
        factors['obv']                     = self.obv()

        # ── 量价结合 ─────────────────────────────────────────────────────────
        factors['price_vwap_deviation']    = self.price_vwap_deviation()
        factors['money_flow_index']        = self.money_flow_index()
        factors['price_volume_trend']      = self.price_volume_trend()
        factors['ease_of_movement']        = self.ease_of_movement()
        factors['volume_weighted_return']  = self.volume_weighted_return()
        factors['money_flow_ratio']        = self.money_flow_ratio()
        factors['volume_price_correlation'] = self.volume_price_correlation()
        factors['volume_profile_indicator'] = self.volume_profile_indicator_fast()

        # ── 波动率 ───────────────────────────────────────────────────────────
        factors['parkinson_volatility']    = self.parkinson_volatility()
        factors['realized_volatility']     = self.realized_volatility()
        factors['intraday_volatility']     = self.intraday_volatility()
        factors['overnight_volatility']    = self.overnight_volatility()
        factors['atr']                     = self.atr()
        factors['volatility_ratio']        = self.volatility_ratio()
        factors['skewness']                = self.skewness()
        factors['kurtosis']                = self.kurtosis()

        # ── 流动性 ───────────────────────────────────────────────────────────
        factors['amihud_illiquidity']      = self.amihud_illiquidity()
        factors['amihud_volatility']       = self.amihud_volatility()
        factors['bid_ask_spread_proxy']    = self.bid_ask_spread_proxy()
        factors['price_impact']            = self.price_impact()
        factors['market_depth_imbalance']  = self.market_depth_imbalance()
        factors['liquidity_indicator']     = self.liquidity_indicator()

        # ── 技术指标 ─────────────────────────────────────────────────────────
        stoch_data = self.stochastic_oscillator()
        factors['stoch_k']                 = stoch_data['stoch_k']
        factors['stoch_d']                 = stoch_data['stoch_d']
        factors['cci']                     = self.cci()
        factors['adx']                     = self.adx()

        aroon_data = self.aroon()
        factors['aroon_up']                = aroon_data['aroon_up']
        factors['aroon_down']              = aroon_data['aroon_down']
        factors['aroon_oscillator']        = aroon_data['aroon_oscillator']

        factors['trix']                    = self.trix()
        factors['chaikin_oscillator']      = self.chaikin_oscillator()
        factors['accumulation_distribution'] = self.accumulation_distribution()
        factors['parabolic_sar']           = self.parabolic_sar()

        # ── 通道/位置 ────────────────────────────────────────────────────────
        factors['keltner_position']        = self.keltner_channel_position()
        factors['donchian_position']       = self.donchian_channel_position()
        factors['price_channel_breakout']  = self.price_channel_breakout()
        factors['close_position_ratio']    = self.close_position_ratio()
        factors['high_low_ratio']          = self.high_low_ratio()

        # ── 统计/高级因子 ────────────────────────────────────────────────────
        factors['hurst_exponent']          = self.hurst_exponent()
        factors['fractal_dimension']       = self.fractal_dimension()
        factors['trend_strength']          = self.trend_strength()
        factors['ad']                      = self.ad_factor()
        factors['qstick']                  = self.qstick()
        factors['sharpe']                  = self.sharpe_ratio()
        factors['rvi']                     = self.rvi()
        factors['klinger']                 = self.klinger()
        factors['linearreg_ma']            = self.linearreg_ma()

        # ── Kyle Lambda（需要 tick_factors）─────────────────────────────────
        factors['kyle_lambda']             = self.kyle_lambda()
        factors['kyle_lambda_lag']         = self.kyle_lambda_lag()

        # ── 会话特征虚拟变量 ─────────────────────────────────────────────────
        if self.add_session_features:
            sess = self.session_features()
            factors['is_night']    = sess['is_night']
            factors['session_sin'] = sess['session_sin']

        return factors