"""回测引擎模块"""

import pandas as pd
import numpy as np
from .strategies import BaseStrategy
from .metrics import calculate_metrics, plot_results


class BacktestEngine:
    """量化回测引擎，接收股票数据和策略，运行回测并输出绩效指标。"""

    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy):
        self.data = data.sort_values("date").reset_index(drop=True)
        self.strategy = strategy
        self.signals = None
        self.positions = None
        self.equity_curve = None

    def run(self, initial_capital: float = 100000.0):
        """运行回测，返回绩效指标和基准净值"""
        # 1. 生成信号
        self.signals = self.strategy.generate_signals(self.data)

        # 2. 计算持仓状态
        self.positions = self.signals.replace(-1, 0).cumsum()
        self.positions = (self.positions > 0).astype(int)

        # 3. 计算每日收益率
        daily_returns = self.data["close"].pct_change().fillna(0)
        strategy_returns = daily_returns * self.positions.shift(1).fillna(0)

        # 4. 计算净值曲线
        self.equity_curve = (1 + strategy_returns).cumprod() * initial_capital

        # 5. 基准净值
        benchmark_curve = (1 + daily_returns).cumprod() * initial_capital

        # 6. 计算绩效指标
        metrics = calculate_metrics(self.equity_curve)
        metrics["初始资金"] = f"{initial_capital:.2f}"
        metrics["最终资金"] = f"{self.equity_curve.iloc[-1]:.2f}"
        return metrics, benchmark_curve

    def plot_result(self, benchmark_curve=None, save_path=None):
        """绘制回测图表"""
        plot_results(
            equity_curve=self.equity_curve,
            benchmark=benchmark_curve,
            strategy_name=self.strategy.name,
            save_path=save_path,
        )

    def get_trade_log(self):
        """生成交易记录"""
        if self.signals is None:
            return pd.DataFrame()
        trades = []
        for i in range(len(self.signals)):
            if self.signals.iloc[i] != 0:
                label = "买入" if self.signals.iloc[i] == 1 else "卖出"
                trades.append({"日期": self.data.iloc[i]["date"], "操作": label, "价格": self.data.iloc[i]["close"]})
        return pd.DataFrame(trades)
