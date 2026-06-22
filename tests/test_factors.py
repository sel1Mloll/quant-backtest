"""因子计算单元测试"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from backtest.factors import FactorEngine, FactorStrategy


@pytest.fixture
def market_data():
    """60 天的行情数据"""
    np.random.seed(99)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n)
    prices = 100 * np.exp(np.random.randn(n).cumsum() * 0.01)
    close = np.maximum(prices, 1.0)
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(100000, 500000, n),
    })


class TestFactorEngine:
    def test_calculates_all_factors(self, market_data):
        engine = FactorEngine(market_data)
        factors = engine.get_all_factors()
        assert isinstance(factors, pd.DataFrame)
        assert len(factors.columns) > 0
        expected = ["mom_1d", "mom_5d", "vol_20d", "atr", "atr_ratio",
                     "vol_ratio_5d", "vol_ratio_20d",
                     "price_position", "ma_distance", "rsi_14"]
        for col in expected:
            assert col in factors.columns, f"缺少因子列: {col}"

    def test_factor_values_not_all_nan(self, market_data):
        engine = FactorEngine(market_data)
        factors = engine.get_all_factors()
        for col in factors.columns:
            assert factors[col].notna().sum() > 0, f"因子列全是 NaN: {col}"

    def test_composite_score_range(self, market_data):
        engine = FactorEngine(market_data)
        score = engine.composite_score({"mom_20d": 0.5, "vol_20d": -0.5})
        assert score.min() >= 0
        assert score.max() <= 1

    def test_signal_generation_from_score(self, market_data):
        engine = FactorEngine(market_data)
        score = engine.composite_score({"mom_20d": 1.0})
        signals = engine.generate_signals_from_score(score, top_pct=0.3, bottom_pct=0.3)
        assert set(signals.unique()).issubset({-1, 0, 1})
        assert (signals == 1).sum() > 0 or (signals == -1).sum() > 0

    def test_get_factor_returns_series(self, market_data):
        engine = FactorEngine(market_data)
        factor = engine.get_factor("mom_5d")
        assert isinstance(factor, pd.Series)
        assert len(factor) == len(market_data)

    def test_rank_factor_normalization(self, market_data):
        engine = FactorEngine(market_data)
        factor = engine.get_factor("mom_5d")
        ranked = engine.rank_factor(factor.fillna(0))
        assert ranked.min() >= 0 and ranked.max() <= 1


class TestFactorStrategy:
    def test_signal_values(self, market_data):
        strategy = FactorStrategy()
        signals = strategy.generate_signals(market_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_no_lookahead_bias(self, market_data):
        """
        验证没有前视偏差：信号 day[t] 只使用 day[t-1] 及之前的数据。
        """
        strategy = FactorStrategy()
        signals = strategy.generate_signals(market_data)
        engine = FactorEngine(market_data)
        score = engine.composite_score(strategy.weights)
        expected_signals = engine.generate_signals_from_score(
            score.shift(1), strategy.top_pct, strategy.bottom_pct,
        )
        pd.testing.assert_series_equal(signals, expected_signals)

    def test_custom_weights(self, market_data):
        weights = {"mom_5d": 0.6, "vol_ratio_5d": 0.4}
        strategy = FactorStrategy(weights=weights)
        signals = strategy.generate_signals(market_data)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_name(self):
        assert "Factor" in FactorStrategy().name

    def test_different_thresholds(self, market_data):
        """不同分位数阈值应产生不同数量的信号"""
        s1 = FactorStrategy(top_pct=0.1, bottom_pct=0.1).generate_signals(market_data)
        s2 = FactorStrategy(top_pct=0.4, bottom_pct=0.4).generate_signals(market_data)
        assert (s1 != 0).sum() <= (s2 != 0).sum()

    def test_risk_manager_import_path(self):
        """验证 risk_manager 可导入"""
        from backtest.risk_manager import FixedRatioSizer, KellySizer, BasePositionSizer
        assert BasePositionSizer is not None
        assert FixedRatioSizer is not None
        assert KellySizer is not None
