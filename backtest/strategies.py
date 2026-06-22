"""交易策略模块 - 包含经典和实战策略"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name: str = "BaseStrategy"):
        self.name = name

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Returns
        -------
        pd.Series
            信号序列: 1=买入, -1=卖出, 0=持有
        """
        pass


class MovingAverageCross(BaseStrategy):
    """
    双均线交叉策略

    短期均线上穿长期均线时买入 (金叉)
    短期均线下穿长期均线时卖出 (死叉)
    """

    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(name=f"MA_Cross({short_window},{long_window})")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        short_ma = close.rolling(window=self.short_window).mean()
        long_ma = close.rolling(window=self.long_window).mean()

        signals = pd.Series(0, index=data.index)
        cross_over = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        signals[cross_over] = 1
        cross_under = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        signals[cross_under] = -1
        return signals


class MomentumStrategy(BaseStrategy):
    """
    动量策略

    N 日收益率超过阈值时买入，低于负阈值时卖出
    """

    def __init__(self, lookback: int = 20, threshold: float = 0.05):
        super().__init__(name=f"Momentum({lookback},{threshold:.0%})")
        self.lookback = lookback
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        returns = close.pct_change(periods=self.lookback)
        signals = pd.Series(0, index=data.index)
        signals[returns > self.threshold] = 1
        signals[returns < -self.threshold] = -1
        return signals


class BuyAndHold(BaseStrategy):
    """买入持有策略（基准）"""

    def __init__(self):
        super().__init__(name="Buy & Hold")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        signals.iloc[0] = 1
        return signals


class BollingerBands(BaseStrategy):
    """
    布林带均值回归策略

    价格触及下轨时买入，触及上轨时卖出
    适用于震荡行情
    """

    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(name=f"Bollinger({window},{num_std})")
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        ma = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()

        upper_band = ma + self.num_std * std
        lower_band = ma - self.num_std * std

        signals = pd.Series(0, index=data.index)

        # 价格从下轨下方回升触碰下轨 -> 买入
        signals[(close <= lower_band) & (close.shift(1) > lower_band.shift(1))] = 1
        # 价格从上轨上方回落触碰上轨 -> 卖出
        signals[(close >= upper_band) & (close.shift(1) < upper_band.shift(1))] = -1

        return signals


class RSIStrategy(BaseStrategy):
    """
    RSI 超买超卖策略

    RSI 低于超卖线时买入，高于超买线时卖出
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(name=f"RSI({period},{oversold},{overbought})")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _calc_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        rsi = self._calc_rsi(close)

        signals = pd.Series(0, index=data.index)

        # RSI 从超卖区回升 -> 买入
        signals[(rsi > self.oversold) & (rsi.shift(1) <= self.oversold)] = 1
        # RSI 从超买区回落 -> 卖出
        signals[(rsi < self.overbought) & (rsi.shift(1) >= self.overbought)] = -1

        return signals


class MACDStrategy(BaseStrategy):
    """
    MACD 趋势跟踪策略

    DIF 上穿 DEA 时买入（金叉），下穿时卖出（死叉）
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(name=f"MACD({fast},{slow},{signal})")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]

        # 计算 EMA
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()

        # DIF = 快线 - 慢线
        dif = ema_fast - ema_slow
        # DEA = DIF 的信号线
        dea = dif.ewm(span=self.signal, adjust=False).mean()

        signals = pd.Series(0, index=data.index)

        # DIF 上穿 DEA -> 买入
        cross_over = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        signals[cross_over] = 1

        # DIF 下穿 DEA -> 卖出
        cross_under = (dif < dea) & (dif.shift(1) >= dea.shift(1))
        signals[cross_under] = -1

        return signals


class VolumeBreakout(BaseStrategy):
    """
    成交量突破策略

    价格突破近期高点 + 成交量放大时买入
    价格跌破近期低点时卖出
    """

    def __init__(self, lookback: int = 20, volume_ratio: float = 1.5):
        super().__init__(name=f"VolBreakout({lookback},{volume_ratio})")
        self.lookback = lookback
        self.volume_ratio = volume_ratio

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        volume = data["volume"]

        recent_high = close.rolling(window=self.lookback).max()
        recent_low = close.rolling(window=self.lookback).min()
        avg_volume = volume.rolling(window=self.lookback).mean()

        signals = pd.Series(0, index=data.index)

        # 价格创近期新高 + 成交量放大 -> 买入
        buy_signal = (close == recent_high) & (volume > avg_volume * self.volume_ratio)
        signals[buy_signal] = 1

        # 价格创近期新低 -> 卖出
        sell_signal = close == recent_low
        signals[sell_signal] = -1

        return signals


