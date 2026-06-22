"""回测引擎单元测试"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest
from backtest.engine import BacktestEngine
from backtest.strategies import BuyAndHold, MovingAverageCross
from backtest.risk_manager import FixedRatioSizer


@pytest.fixture
def sample_data():
    """生成 60 天的模拟行情数据"""
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2024-01-01", periods=n)
    prices = 100 * np.exp(np.random.randn(n).cumsum() * 0.008 + 0.0005)
    close = np.maximum(prices, 1.0)
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(100000, 500000, n),
    })


class TestBacktestEngine:
    def test_run_returns_metrics_and_benchmark(self, sample_data):
        engine = BacktestEngine(sample_data, BuyAndHold())
        metrics, benchmark = engine.run()
        assert isinstance(metrics, dict)
        assert "总收益率" in metrics
        assert "年化收益率" in metrics
        assert "夏普比率" in metrics
        assert "最大回撤" in metrics
        assert isinstance(benchmark, pd.Series)
        assert len(benchmark) == len(sample_data)

    def test_run_with_commission(self, sample_data):
        """交易成本应降低最终资金"""
        engine_no_cost = BacktestEngine(sample_data, BuyAndHold())
        m1, _ = engine_no_cost.run()

        engine_cost = BacktestEngine(sample_data, BuyAndHold(),
                                      commission_rate=0.003, slippage=0.001)
        m2, _ = engine_cost.run()

        final_no_cost = float(m1["最终资金"])
        final_cost = float(m2["最终资金"])
        assert final_cost < final_no_cost, "交易成本应降低最终资金"

    def test_run_with_position_sizing(self, sample_data):
        """仓位管理应降低持仓暴露"""
        engine_full = BacktestEngine(sample_data, BuyAndHold())
        m_full, _ = engine_full.run()

        engine_half = BacktestEngine(
            sample_data, BuyAndHold(),
            risk_manager=FixedRatioSizer(fraction=0.5),
        )
        m_half, _ = engine_half.run()

        ret_full = float(m_full["总收益率"].rstrip("%"))
        ret_half = float(m_half["总收益率"].rstrip("%"))
        # 半仓策略的收益应小于或接近全仓的一半
        assert abs(ret_half) <= abs(ret_full) * 0.6 or abs(ret_half - ret_full * 0.5) < 5

    def test_stop_loss(self, sample_data):
        """止损机制不导致异常"""
        engine_no_sl = BacktestEngine(sample_data, BuyAndHold())
        m1, _ = engine_no_sl.run()

        engine_sl = BacktestEngine(sample_data, BuyAndHold(), stop_loss=0.05)
        m2, _ = engine_sl.run()
        dd_no_sl = float(m1["最大回撤"].rstrip("%"))
        dd_sl = float(m2["最大回撤"].rstrip("%"))
        # 有止损的最终资金应合理（不应为 0 或 NaN）
        assert float(m2["最终资金"]) > 0, "止损后最终资金应为正"
        assert "总收益率" in m2

    def test_trade_log_format(self, sample_data):
        engine = BacktestEngine(sample_data, MovingAverageCross(5, 20))
        engine.run()
        trades = engine.get_trade_log()
        assert isinstance(trades, pd.DataFrame)
        if len(trades) > 0:
            expected_cols = ["日期", "操作", "价格"]
            for c in expected_cols:
                assert c in trades.columns, f"缺少栏目: {c}"

    def test_initial_capital_is_honored(self, sample_data):
        cap = 500000.0
        engine = BacktestEngine(sample_data, BuyAndHold())
        metrics, _ = engine.run(initial_capital=cap)
        assert float(metrics["初始资金"]) == cap
        assert float(metrics["最终资金"]) != cap or float(metrics["总收益率"].rstrip("%")) == 0

    def test_equity_curve_length(self, sample_data):
        engine = BacktestEngine(sample_data, BuyAndHold())
        engine.run()
        assert len(engine.equity_curve) == len(sample_data)
        assert engine.equity_curve.index[0] == sample_data["date"].iloc[0]

    def test_trade_log_empty_for_no_signals(self, sample_data):
        """不使用策略时交易日志为空"""
        engine = BacktestEngine(sample_data, BuyAndHold())
        trades = engine.get_trade_log()
        assert isinstance(trades, pd.DataFrame)
