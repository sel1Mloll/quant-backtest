"""绩效评估指标模块"""

import pandas as pd
import numpy as np


def calculate_metrics(equity_curve: pd.Series) -> dict:
    """计算策略绩效指标"""
    daily_returns = equity_curve.pct_change().dropna()
    if len(daily_returns) == 0:
        return {}

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    n_years = len(daily_returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_volatility = daily_returns.std() * np.sqrt(252)

    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()

    winning_trades = daily_returns[daily_returns > 0]
    win_rate = len(winning_trades) / len(daily_returns) if len(daily_returns) > 0 else 0

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "总收益率": f"{total_return:.2%}",
        "年化收益率": f"{annual_return:.2%}",
        "年化波动率": f"{annual_volatility:.2%}",
        "夏普比率": f"{sharpe_ratio:.2f}",
        "最大回撤": f"{max_drawdown:.2%}",
        "胜率": f"{win_rate:.2%}",
        "Calmar比率": f"{calmar_ratio:.2f}",
        "交易天数": len(daily_returns),
    }


def plot_results(equity_curve, benchmark=None, strategy_name="策略", save_path=None):
    """绘制回测结果图表"""
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"回测结果 - {strategy_name}", fontsize=14)

    # 净值曲线
    ax1 = axes[0, 0]
    ax1.plot(equity_curve.index, equity_curve, label=strategy_name, linewidth=1.5)
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark, label="基准", linewidth=1.5, alpha=0.7)
    ax1.set_title("净值曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 每日收益率
    ax2 = axes[0, 1]
    daily_returns = equity_curve.pct_change().dropna()
    ax2.bar(daily_returns.index, daily_returns, width=1, alpha=0.6)
    ax2.axhline(y=0, color="r", linestyle="-", linewidth=0.5)
    ax2.set_title("每日收益率")
    ax2.grid(True, alpha=0.3)

    # 回撤曲线
    ax3 = axes[1, 0]
    cumulative_max = equity_curve.cummax()
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    ax3.fill_between(drawdown.index, 0, drawdown, color="red", alpha=0.3)
    ax3.plot(drawdown.index, drawdown, color="red", linewidth=1)
    ax3.set_title("回撤曲线")
    ax3.grid(True, alpha=0.3)

    # 收益率分布
    ax4 = axes[1, 1]
    ax4.hist(daily_returns, bins=50, alpha=0.6, color="steelblue")
    ax4.axvline(x=0, color="r", linestyle="-", linewidth=0.5)
    ax4.set_title("收益率分布")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存到: {save_path}")
    plt.show()
