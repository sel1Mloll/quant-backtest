"""回测引擎模块"""

import pandas as pd
import numpy as np
from .strategies import BaseStrategy
from .metrics import calculate_metrics, plot_results
from .risk_manager import BasePositionSizer


class BacktestEngine:
    """
    量化回测引擎，接收股票数据和策略，运行回测并输出绩效指标。

    Parameters
    ----------
    data : pd.DataFrame
        行情数据
    strategy : BaseStrategy
        交易策略实例
    commission_rate : float, default 0.0
        佣金费率，如 0.0003 表示万三
    slippage : float, default 0.0
        滑点比例，如 0.001 表示千一
    stop_loss : float, default 0.0
        止损比例，如 0.05 表示亏损 5% 时强制平仓；0 表示不使用
    risk_manager : BasePositionSizer, optional
        仓位管理器，控制每次开仓的资金比例
    """

    def __init__(self, data: pd.DataFrame, strategy: BaseStrategy,
                 commission_rate: float = 0.0, slippage: float = 0.0,
                 stop_loss: float = 0.0, risk_manager: BasePositionSizer = None):
        self.data = data.sort_values("date").reset_index(drop=True)
        self.strategy = strategy
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.risk_manager = risk_manager
        self.signals = None
        self.positions = None
        self.equity_curve = None

    def _build_positions(self):
        """根据信号构建基础仓位序列 (0 或 1)"""
        self.positions = self.signals.replace(-1, 0).cumsum()
        self.positions = (self.positions > 0).astype(int)

    def _apply_stop_loss(self):
        """
        止损：当持仓期间价格从入场点回撤超过 stop_loss 时强制平仓。
        通过遍历实现，因为止损逻辑是顺序依赖的。
        """
        if self.stop_loss <= 0 or self.positions is None:
            return

        close = self.data["close"].values
        pos = self.positions.values.copy()
        signals_arr = self.signals.values.copy()

        in_position = False
        entry_price = 0.0

        for i in range(len(pos)):
            if pos[i] == 1 and not in_position:
                in_position = True
                entry_price = close[i]
            elif pos[i] == 0 and in_position:
                in_position = False
            elif in_position:
                if close[i] < entry_price * (1.0 - self.stop_loss):
                    pos[i] = 0
                    signals_arr[i] = -1
                    in_position = False

        self.positions = pd.Series(pos, index=self.positions.index)
        self.signals = pd.Series(signals_arr, index=self.signals.index)

    def _calculate_returns(self, initial_capital: float):
        """计算策略日收益率，含交易成本和仓位管理"""
        daily_returns = self.data["close"].pct_change().fillna(0)

        in_market = self.positions.shift(1).fillna(0).astype(float)

        position_mult = 1.0
        if self.risk_manager is not None:
            size_series = self.risk_manager.get_position_size(self.data, self.signals)
            if isinstance(size_series, pd.Series):
                position_mult = size_series.shift(1).fillna(0)
            else:
                position_mult = pd.Series(size_series, index=daily_returns.index).shift(1).fillna(0)

        effective_position = in_market * position_mult
        strategy_returns = daily_returns * effective_position

        position_changed = (self.positions != self.positions.shift(1).fillna(0)).astype(int)
        trade_cost = position_changed * (self.commission_rate + self.slippage)
        strategy_returns = strategy_returns - trade_cost

        return strategy_returns, daily_returns

    def run(self, initial_capital: float = 100000.0):
        """运行回测，返回绩效指标和基准净值"""
        self.signals = self.strategy.generate_signals(self.data)
        self._build_positions()
        self._apply_stop_loss()

        strategy_returns, daily_returns = self._calculate_returns(initial_capital)

        self.equity_curve = (1 + strategy_returns).cumprod() * initial_capital
        self.equity_curve.index = self.data["date"]

        benchmark_curve = (1 + daily_returns).cumprod() * initial_capital
        benchmark_curve.index = self.data["date"]

        metrics = calculate_metrics(self.equity_curve)
        metrics["初始资金"] = f"{initial_capital:.2f}"
        metrics["最终资金"] = f"{self.equity_curve.iloc[-1]:.2f}"

        total_cost_pct = strategy_returns.sum() - (daily_returns * self.positions.shift(1).fillna(0)).sum()
        if abs(total_cost_pct) > 1e-8:
            metrics["交易成本占比"] = f"{abs(total_cost_pct):.4%}"

        return metrics, benchmark_curve

    def get_trade_log(self):
        """生成交易记录"""
        if self.signals is None:
            return pd.DataFrame()
        trades = []
        for i in range(len(self.signals)):
            if self.signals.iloc[i] != 0:
                action = "买入" if self.signals.iloc[i] == 1 else "卖出"
                trade = {"日期": self.data.iloc[i]["date"], "操作": action, "价格": self.data.iloc[i]["close"]}
                trades.append(trade)
        return pd.DataFrame(trades)

    def plot_result(self, benchmark_curve=None, save_path=None, strategy_params=None):
        """绘制回测结果图表"""
        plot_results(
            equity_curve=self.equity_curve,
            benchmark=benchmark_curve,
            strategy_name=self.strategy.name,
            save_path=save_path,
            trades=self.get_trade_log(),
            strategy_params=strategy_params,
        )
