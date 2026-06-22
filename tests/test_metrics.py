"""绩效指标单元测试"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from backtest.metrics import calculate_metrics


class TestCalculateMetrics:
    @pytest.fixture
    def steady_growth(self):
        """稳定上涨的净值曲线"""
        dates = pd.date_range("2024-01-01", periods=252, freq="D")
        values = 100000 * (1.001 ** np.arange(252))
        return pd.Series(values, index=dates)

    @pytest.fixture
    def flat_curve(self):
        """完全不动的净值"""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        return pd.Series([100000] * 50, index=dates)

    def test_returns_expected_keys(self, steady_growth):
        metrics = calculate_metrics(steady_growth)
        expected = ["总收益率", "年化收益率", "年化波动率",
                     "夏普比率", "索提诺比率", "最大回撤",
                     "Calmar比率", "日胜率", "交易天数"]
        for k in expected:
            assert k in metrics, f"缺少指标: {k}"

    def test_steady_growth_positive_returns(self, steady_growth):
        metrics = calculate_metrics(steady_growth)
        assert float(metrics["总收益率"].rstrip("%")) > 0
        assert float(metrics["年化收益率"].rstrip("%")) > 0
        assert float(metrics["夏普比率"]) > 0

    def test_flat_curve_zero_sharpe(self, flat_curve):
        metrics = calculate_metrics(flat_curve)
        assert float(metrics["总收益率"].rstrip("%")) == 0
        assert float(metrics["夏普比率"]) == 0.0

    def test_max_drawdown_is_zero_or_negative(self, steady_growth):
        metrics = calculate_metrics(steady_growth)
        dd = float(metrics["最大回撤"].rstrip("%"))
        assert dd <= 0

    def test_win_rate_between_0_and_1(self, steady_growth):
        metrics = calculate_metrics(steady_growth)
        wr = float(metrics["日胜率"].rstrip("%")) / 100
        assert 0 <= wr <= 1

    def test_trading_days_positive(self, steady_growth):
        metrics = calculate_metrics(steady_growth)
        assert metrics["交易天数"] > 0

    def test_empty_series_returns_empty_dict(self):
        empty = pd.Series([], dtype=float)
        assert calculate_metrics(empty) == {}

    def test_single_value_returns_empty_dict(self):
        single = pd.Series([100000])
        assert calculate_metrics(single) == {}

    def test_calmar_ratio_positive_for_growth(self, steady_growth):
        """有涨有跌的行情应产生合理的 Calmar 比率"""
        dates = pd.date_range("2024-01-01", periods=252, freq="D")
        values = 100000 * (1 + 0.001 * np.sin(np.arange(252) * 0.1) + 0.0008 * np.arange(252) / 100)
        curve = pd.Series(values, index=dates)
        metrics = calculate_metrics(curve)
        calmar = float(metrics["Calmar比率"])
        assert calmar >= 0, f"上涨行情的 Calmar 应为非负, got {calmar}"

    def test_negative_returns_negative_metrics(self):
        """持续下跌的净值应得到负收益和负夏普"""
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        values = 100000 * (0.998 ** np.arange(100))
        curve = pd.Series(values, index=dates)
        metrics = calculate_metrics(curve)
        assert float(metrics["总收益率"].rstrip("%")) < 0
        assert float(metrics["夏普比率"]) < 0
        assert float(metrics["最大回撤"].rstrip("%")) < 0
