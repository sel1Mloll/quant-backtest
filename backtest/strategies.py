"""交易策略模块"""

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

    当短期均线上穿长期均线时买入 (金叉)
    当短期均线下穿长期均线时卖出 (死叉)
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

        # 金叉
        cross_over = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        signals[cross_over] = 1

        # 死叉
        cross_under = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        signals[cross_under] = -1

        return signals


class MomentumStrategy(BaseStrategy):
    """
    动量策略

    当 N 日收益率超过阈值时买入
    当 N 日收益率低于负阈值时卖出
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
