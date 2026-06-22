"""风险管理模块 - 仓位管理与止损"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BasePositionSizer(ABC):
    """仓位管理器基类"""

    @abstractmethod
    def get_position_size(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """
        返回每日仓位比例 [0, 1]

        Parameters
        ----------
        data : pd.DataFrame
            行情数据
        signals : pd.Series
            交易信号 (1=买入, -1=卖出, 0=持有)

        Returns
        -------
        pd.Series
            每日仓位比例
        """
        pass


class FixedRatioSizer(BasePositionSizer):
    """
    固定比例仓位管理

    每次开仓使用固定比例的资金，剩余资金留作现金缓冲。
    """

    def __init__(self, fraction: float = 0.5):
        """
        Parameters
        ----------
        fraction : float
            每次开仓使用的资金比例 (0, 1]
        """
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction

    def get_position_size(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        return pd.Series(self.fraction, index=signals.index)


class KellySizer(BasePositionSizer):
    """
    凯利公式仓位管理

    f* = (p * b - q) / b
    其中 p = 胜率, q = 1-p, b = 盈亏比（平均盈利 / 平均亏损）

    默认使用半凯利（half-Kelly），降低波动和破产风险。
    """

    def __init__(self, win_rate: float = None, avg_win: float = None,
                 avg_loss: float = None, half_kelly: bool = True):
        """
        Parameters
        ----------
        win_rate : float, optional
            胜率 [0, 1]，None 时保守默认 0.25
        avg_win : float, optional
            平均盈利率，None 时保守默认 0.05
        avg_loss : float, optional
            平均亏损率，None 时保守默认 0.03
        half_kelly : bool, default True
            是否使用半凯利
        """
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.half_kelly = half_kelly

    def get_position_size(self, data: pd.DataFrame, signals: pd.Series) -> pd.Series:
        if self.win_rate is None or self.avg_win is None or self.avg_loss is None:
            # 参数不足时保守地固定比例 25%
            return pd.Series(0.25, index=signals.index)

        b = self.avg_win / abs(self.avg_loss) if self.avg_loss != 0 else 1.0
        p = self.win_rate
        q = 1.0 - p

        kelly = (p * b - q) / b if b > 0 else 0.0
        kelly = max(0.0, min(kelly, 1.0))

        if self.half_kelly:
            kelly *= 0.5

        return pd.Series(kelly, index=signals.index)
