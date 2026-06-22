"""交易策略单元测试"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from backtest.strategies import (
    MovingAverageCross, MomentumStrategy, BuyAndHold,
    BollingerBands, RSIStrategy, MACDStrategy, VolumeBreakout,
)


@pytest.fixture
def uptrend_data():
    """持续上涨 + 末尾下跌的行情，用于测试信号"""
    n = 80
    dates = pd.date_range("2024-01-01", periods=n)
    close = np.zeros(n)
    close[:40] = 100 + np.arange(40) * 0.5        # 上涨
    close[40:60] = 120 + np.arange(20) * 0.3       # 继续涨
    close[60:] = 126 - np.arange(20) * 1.0          # 下跌
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(100000, 500000, n),
    })


class TestMovingAverageCross:
    def test_basic_signal_types(self, uptrend_data):
        """信号应只包含 -1, 0, 1"""
        strategy = MovingAverageCross(10, 30)
        signals = strategy.generate_signals(uptrend_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_short_above_long_should_not_be_short(self, uptrend_data):
        """短均线明显高于长均线时不应有卖出信号"""
        strategy = MovingAverageCross(3, 20)
        signals = strategy.generate_signals(uptrend_data)
        # 连续上涨期间不会有死叉
        early = signals.iloc[30:50]
        assert (early == -1).sum() == 0

    def test_name_format(self):
        strategy = MovingAverageCross(5, 20)
        assert "MA_Cross" in strategy.name


class TestBuyAndHold:
    def test_only_first_day_is_buy(self, uptrend_data):
        strategy = BuyAndHold()
        signals = strategy.generate_signals(uptrend_data)
        assert signals.iloc[0] == 1
        assert (signals.iloc[1:] == 0).all()

    def test_name(self):
        assert "Buy" in BuyAndHold().name


class TestMomentum:
    def test_signal_values(self, uptrend_data):
        strategy = MomentumStrategy(lookback=10, threshold=0.02)
        signals = strategy.generate_signals(uptrend_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_strong_upward_has_buy_signals(self, uptrend_data):
        """连续上涨应产生买入信号"""
        strategy = MomentumStrategy(lookback=5, threshold=0.01)
        signals = strategy.generate_signals(uptrend_data)
        assert (signals == 1).sum() > 0


class TestBollingerBands:
    def test_signal_values(self, uptrend_data):
        strategy = BollingerBands(20, 2.0)
        signals = strategy.generate_signals(uptrend_data)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestRSIStrategy:
    def test_signal_values(self, uptrend_data):
        strategy = RSIStrategy(14, 30, 70)
        signals = strategy.generate_signals(uptrend_data)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestMACDStrategy:
    def test_signal_values(self, uptrend_data):
        strategy = MACDStrategy(12, 26, 9)
        signals = strategy.generate_signals(uptrend_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_dif_above_dea_triggers_buy(self, uptrend_data):
        """快线上穿慢线时应有买入信号"""
        close = uptrend_data["close"]
        ema_fast = close.ewm(span=5, adjust=False).mean()
        ema_slow = close.ewm(span=20, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=5, adjust=False).mean()

        strategy = MACDStrategy(5, 20, 5)
        signals = strategy.generate_signals(uptrend_data)
        cross_over = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        assert (signals[cross_over] == 1).all()


class TestVolumeBreakout:
    def test_signal_values(self, uptrend_data):
        strategy = VolumeBreakout(10, 1.5)
        signals = strategy.generate_signals(uptrend_data)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestBaseStrategyAbstraction:
    def test_cannot_instantiate_base(self):
        from backtest.strategies import BaseStrategy
        import pytest
        with pytest.raises(TypeError):
            BaseStrategy("test")
