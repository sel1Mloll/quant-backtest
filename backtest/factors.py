"""因子计算模块 - 技术因子与多因子打分系统"""

import pandas as pd
import numpy as np
from .strategies import BaseStrategy


def calc_momentum_factors(data: pd.DataFrame) -> pd.DataFrame:
    """计算动量类因子"""
    close = data["close"]
    factors = pd.DataFrame(index=data.index)

    factors["mom_1d"] = close.pct_change(1)
    factors["mom_5d"] = close.pct_change(5)
    factors["mom_10d"] = close.pct_change(10)
    factors["mom_20d"] = close.pct_change(20)

    # 动量稳定性：过去20天正收益天数占比
    factors["mom_stability"] = (close.diff() > 0).rolling(20).sum() / 20

    return factors


def calc_volatility_factors(data: pd.DataFrame) -> pd.DataFrame:
    """计算波动率类因子"""
    close = data["close"]
    high = data["high"]
    low = data["low"]
    factors = pd.DataFrame(index=data.index)

    # 历史波动率
    daily_returns = close.pct_change()
    factors["vol_20d"] = daily_returns.rolling(20).std()

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    factors["atr"] = tr.rolling(14).mean()
    factors["atr_ratio"] = factors["atr"] / close

    # 最大回撤
    cumulative_max = close.cummax()
    factors["drawdown"] = (close - cumulative_max) / cumulative_max
    factors["max_drawdown_20d"] = factors["drawdown"].rolling(20).min()

    return factors


def calc_volume_factors(data: pd.DataFrame) -> pd.DataFrame:
    """计算成交量类因子"""
    volume = data["volume"]
    close = data["close"]
    factors = pd.DataFrame(index=data.index)

    factors["vol_ratio_5d"] = volume / volume.rolling(5).mean()
    factors["vol_ratio_20d"] = volume / volume.rolling(20).mean()

    amount = data.get("amount", close * volume)
    factors["amount_ratio"] = amount / amount.rolling(20).mean()

    # 量价相关性
    factors["vol_price_corr"] = close.pct_change().rolling(10).corr(volume)

    return factors


def calc_price_factors(data: pd.DataFrame) -> pd.DataFrame:
    """计算价格形态类因子"""
    close = data["close"]
    high = data["high"]
    low = data["low"]
    factors = pd.DataFrame(index=data.index)

    # 价格在近期高低点中的位置
    rolling_high = high.rolling(20).max()
    rolling_low = low.rolling(20).min()
    factors["price_position"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)

    # 距离均线的偏离度
    ma_20 = close.rolling(20).mean()
    factors["ma_distance"] = (close - ma_20) / ma_20

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    factors["rsi_14"] = 100 - (100 / (1 + rs))

    return factors


class FactorEngine:
    """
    因子引擎 - 计算、标准化、组合因子

    Parameters
    ----------
    data : pd.DataFrame
        行情数据
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.factors = pd.DataFrame(index=data.index)
        self._calc_all_factors()

    def _calc_all_factors(self):
        """计算所有因子"""
        for calc_fn in [calc_momentum_factors, calc_volatility_factors,
                         calc_volume_factors, calc_price_factors]:
            df = calc_fn(self.data)
            for col in df.columns:
                self.factors[col] = df[col]

    def get_factor(self, name: str) -> pd.Series:
        return self.factors[name]

    def get_all_factors(self) -> pd.DataFrame:
        return self.factors

    def rank_factor(self, factor: pd.Series, ascending: bool = True) -> pd.Series:
        """对因子进行排名归一化 [0,1]"""
        return factor.rank(pct=True, ascending=ascending)

    def composite_score(self, factor_weights: dict) -> pd.Series:
        """
        多因子加权打分

        Parameters
        ----------
        factor_weights : dict
            {因子名: 权重}，正数=越大越好，负数=越小越好
            如: {"mom_20d": 0.4, "vol_20d": -0.2}
        """
        score = pd.Series(0.0, index=self.factors.index)
        for factor_name, weight in factor_weights.items():
            if factor_name not in self.factors.columns:
                continue
            factor_values = self.factors[factor_name].fillna(0)
            ascending = weight < 0
            ranked = self.rank_factor(factor_values, ascending=ascending)
            score += ranked * abs(weight)
        return score

    def generate_signals_from_score(
        self, score: pd.Series, top_pct: float = 0.2, bottom_pct: float = 0.2
    ) -> pd.Series:
        """根据综合得分生成信号"""
        signals = pd.Series(0, index=score.index)
        signals[score >= score.quantile(1 - top_pct)] = 1
        signals[score <= score.quantile(bottom_pct)] = -1
        return signals


class FactorStrategy(BaseStrategy):
    """
    多因子策略 - 基于综合因子打分生成交易信号
    """

    def __init__(
        self,
        weights: dict = None,
        top_pct: float = 0.2,
        bottom_pct: float = 0.2,
    ):
        super().__init__(name="FactorStrategy")
        if weights is None:
            weights = {
                "mom_20d": 0.3,
                "vol_20d": -0.2,
                "vol_ratio_20d": 0.2,
                "price_position": 0.15,
                "ma_distance": 0.15,
            }
        self.weights = weights
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        engine = FactorEngine(data)
        score = engine.composite_score(self.weights)
        signals = engine.generate_signals_from_score(score, self.top_pct, self.bottom_pct)
        return signals
