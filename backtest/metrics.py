"""绩效评估指标模块 - 专业版"""

import pandas as pd
import numpy as np
import os


def calculate_metrics(equity_curve: pd.Series) -> dict:
    """计算策略绩效指标（含索提诺比率等）"""
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

    # 索提诺比率（只考虑下行风险）
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    winning_days = daily_returns[daily_returns > 0]
    win_rate = len(winning_days) / len(daily_returns) if len(daily_returns) > 0 else 0
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "总收益率":     f"{total_return:.2%}",
        "年化收益率":   f"{annual_return:.2%}",
        "年化波动率":   f"{annual_volatility:.2%}",
        "夏普比率":     f"{sharpe_ratio:.2f}",
        "索提诺比率":   f"{sortino_ratio:.2f}",
        "最大回撤":     f"{max_drawdown:.2%}",
        "Calmar比率":   f"{calmar_ratio:.2f}",
        "日胜率":       f"{win_rate:.2%}",
        "交易天数":     len(daily_returns),
    }


def plot_results(equity_curve, benchmark=None, strategy_name="策略", save_path=None,
                 trades=None, strategy_params=None):
    """绘制专业回测结果图表"""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.dates import AutoDateLocator, DateFormatter
    from scipy import stats

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    daily_ret = equity_curve.pct_change().dropna()
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    rolling_sr = daily_ret.rolling(60).mean() / daily_ret.rolling(60).std() * np.sqrt(252)

    # 月度收益热力图数据
    monthly_ret = daily_ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    years = sorted(set(monthly_ret.index.year))
    heatmap = pd.DataFrame(np.nan, index=years, columns=range(1, 13))
    for idx, val in monthly_ret.items():
        heatmap.loc[idx.year, idx.month] = val

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

    # ---- 第1行: 净值曲线 ----
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity_curve.index, equity_curve, label=strategy_name, linewidth=2, color="#1976D2")
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark, label="基准", linewidth=1.5, color="#9E9E9E", alpha=0.7)
        aligned = pd.concat([equity_curve, benchmark], axis=1).dropna()
        better = aligned.iloc[:, 0] >= aligned.iloc[:, 1]
        ax1.fill_between(aligned.index, aligned.iloc[:, 0], aligned.iloc[:, 1],
                         where=better.values, color="green", alpha=0.08)
        ax1.fill_between(aligned.index, aligned.iloc[:, 0], aligned.iloc[:, 1],
                         where=~better.values, color="red", alpha=0.08)
    if trades is not None and len(trades) > 0:
        tc = trades.columns[0]
        op = trades.columns[1]
        for _, r in trades.iterrows():
            d = pd.Timestamp(r[tc])
            if d in equity_curve.index:
                p = equity_curve[d]
                m = "^" if "买入" in str(r[op]) else "v"
                c = "#E53935" if "买入" in str(r[op]) else "#43A047"
                ax1.scatter(d, p, color=c, marker=m, s=60, zorder=5)
    ax1.axhline(y=equity_curve.iloc[0], color="gray", ls="--", lw=0.5, alpha=0.5)
    ax1.set_title("净值曲线", fontsize=13, fontweight="bold")

    # ---- 修复1: X轴刻度重叠 ----
    locator = AutoDateLocator(minticks=5, maxticks=10)
    formatter = DateFormatter("%Y-%m")
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 文字说明
    ax1.text(0.02, 0.02, "▸ 蓝色曲线为策略净值，灰色为持有不动基准\n▸ 红/绿填充表示策略跑输/跑赢基准的区间", 
             transform=ax1.transAxes, fontsize=8, color="gray", va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ---- 第2行左: 回撤 ----
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(drawdown.index, 0, drawdown, color="#E53935", alpha=0.3)
    ax2.plot(drawdown.index, drawdown, color="#E53935", lw=1.5)
    if not drawdown.isna().all():
        mi = drawdown.idxmin()
        mv = drawdown.min()
        ax2.annotate(f"最大回撤: {mv:.2%}", xy=(mi, mv), xytext=(mi, mv * 0.5),
                     arrowprops=dict(arrowstyle="->", color="darkred"),
                     fontsize=9, color="darkred", fontweight="bold")
    ax2.axhline(y=0, color="gray", lw=0.5)
    ax2.set_title("回撤曲线", fontsize=12, fontweight="bold")
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, -0.25, "回撤 = 当前净值距离历史最高点的跌幅", 
             transform=ax2.transAxes, fontsize=7, color="gray", ha="center")

    # ---- 第2行中左: 每日收益率（修复2：加滚动均线）----
    ax3 = fig.add_subplot(gs[1, 1])
    bar_colors = ["#4CAF50" if v >= 0 else "#EF5350" for v in daily_ret]
    ax3.bar(daily_ret.index, daily_ret, color=bar_colors, width=1, alpha=0.35)
    # 加20日滚动均线让趋势更清晰
    rolling_mean = daily_ret.rolling(20).mean()
    ax3.plot(rolling_mean.index, rolling_mean, color="#FF6F00", linewidth=1.5, label="20日均线")
    ax3.axhline(y=0, color="gray", lw=0.5)
    ax3.set_title("每日收益率", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.xaxis.set_major_locator(locator)
    ax3.xaxis.set_major_formatter(formatter)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, -0.25, "绿色=盈利日, 红色=亏损日, 橙色线=20日滚动均值", 
             transform=ax3.transAxes, fontsize=7, color="gray", ha="center")

    # ---- 第2行中右: 滚动夏普 ----
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(rolling_sr.index, rolling_sr, color="#7B1FA2", lw=1.5)
    ax4.axhline(y=0, color="gray", ls="--", lw=0.5)
    ax4.axhline(y=1, color="green", ls="--", lw=0.5, alpha=0.5, label="优秀=1")
    ax4.axhline(y=-1, color="red", ls="--", lw=0.5, alpha=0.5, label="亏损=-1")
    ax4.set_title("滚动夏普 (60日)", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.xaxis.set_major_locator(locator)
    ax4.xaxis.set_major_formatter(formatter)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.text(0.5, -0.25, "夏普>1优秀, <0说明风险调整后收益为负", 
             transform=ax4.transAxes, fontsize=7, color="gray", ha="center")

    # ---- 第2行右: 收益率分布（修复3：加正态曲线+统计量）----
    ax5_r = fig.add_subplot(gs[1, 3])
    n, bins, patches = ax5_r.hist(daily_ret, bins=60, alpha=0.6, color="#5C6BC0", 
                                   edgecolor="white", linewidth=0.5, density=True)
    # 拟合正态分布曲线
    mu, std = daily_ret.mean(), daily_ret.std()
    x = np.linspace(daily_ret.min(), daily_ret.max(), 200)
    ax5_r.plot(x, stats.norm.pdf(x, mu, std), "r--", lw=2, alpha=0.7, label="正态拟合")
    ax5_r.axvline(x=0, color="#E53935", ls="--", lw=1)
    ax5_r.axvline(x=mu, color="green", ls="--", lw=1, alpha=0.7, label=f"均值={mu:.3f}")
    ax5_r.set_title("收益率分布", fontsize=12, fontweight="bold")
    ax5_r.legend(fontsize=8)
    ax5_r.grid(True, alpha=0.3)
    # 标注关键统计量
    stats_text = f"均值: {mu:.4f}\n标准差: {std:.4f}\n偏度: {daily_ret.skew():.2f}\n峰度: {daily_ret.kurtosis():.2f}"
    ax5_r.text(0.95, 0.95, stats_text, transform=ax5_r.transAxes, fontsize=7,
               verticalalignment="top", horizontalalignment="right",
               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax5_r.text(0.5, -0.25, "红色虚线=正态分布, 绿色虚线=均值, 红色竖线=零收益线", 
               transform=ax5_r.transAxes, fontsize=7, color="gray", ha="center")

    # ---- 第3行左: 月度热力图 ----
    ax5 = fig.add_subplot(gs[2, :2])
    im = ax5.imshow(heatmap.values, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)
    ax5.set_xticks(range(12))
    ax5.set_xticklabels([f"{i+1}月" for i in range(12)], fontsize=8)
    ax5.set_yticks(range(len(years)))
    ax5.set_yticklabels(years, fontsize=9)
    ax5.set_title("月度收益率热力图", fontsize=12, fontweight="bold")
    for i in range(len(years)):
        for j in range(12):
            v = heatmap.values[i, j]
            if not np.isnan(v):
                c = "white" if abs(v) > 0.05 else "black"
                ax5.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=8, color=c)
    plt.colorbar(im, ax=ax5, shrink=0.8)
    ax5.text(0.5, -0.15, "绿色=盈利月, 红色=亏损月, 数值为月度收益率", 
             transform=ax5.transAxes, fontsize=7, color="gray", ha="center")

    # ---- 第3行中右: 月度收益柱状图（新增）----
    ax5_b = fig.add_subplot(gs[2, 2])
    monthly_bar = monthly_ret.dropna()
    bar_colors_month = ["#4CAF50" if v >= 0 else "#EF5350" for v in monthly_bar.values]
    ax5_b.bar(range(len(monthly_bar)), monthly_bar.values, color=bar_colors_month, alpha=0.7)
    ax5_b.axhline(y=0, color="gray", lw=0.5)
    ax5_b.set_title("月度收益柱状图", fontsize=12, fontweight="bold")
    ax5_b.set_xticks(range(0, len(monthly_bar), max(1, len(monthly_bar)//6)))
    ax5_b.set_xticklabels([monthly_bar.index[i].strftime("%Y-%m") for i in range(0, len(monthly_bar), max(1, len(monthly_bar)//6))], 
                          fontsize=7, rotation=30, ha="right")
    ax5_b.grid(True, alpha=0.3)
    ax5_b.text(0.5, -0.15, "每月收益汇总，直观对比各月表现", 
               transform=ax5_b.transAxes, fontsize=7, color="gray", ha="center")

    # ---- 第3行右: 核心指标面板 ----
    ax6 = fig.add_subplot(gs[2, 3])
    ax6.axis("off")
    m = calculate_metrics(equity_curve)

    br = ""
    if benchmark is not None:
        br = f"{(benchmark.iloc[-1] / benchmark.iloc[0]) - 1:.2%}"

    left = ["总收益率", "年化收益率", "年化波动率", "夏普比率", "索提诺比率"]
    right = ["最大回撤", "日胜率", "Calmar比率", "交易天数", ""]

    ax6.text(0.12, 0.96, "核心绩效指标", fontsize=14, fontweight="bold",
             color="#1A237E", transform=ax6.transAxes)
    ax6.text(0.12, 0.88, f"策略: {strategy_name}  基准收益: {br}", fontsize=10,
             color="gray", transform=ax6.transAxes)
    if strategy_params:
        param_str = " | ".join([f"{k}={v}" for k, v in strategy_params.items()])
        ax6.text(0.12, 0.80, f"参数: {param_str}", fontsize=9,
                 color="#666", style="italic", transform=ax6.transAxes)

    for i, k in enumerate(left):
        y = 0.65 - i * 0.12
        ax6.text(0.15, y, k, fontsize=10, color="#333", transform=ax6.transAxes)
        ax6.text(0.50, y, str(m.get(k, "")), fontsize=11, fontweight="bold",
                 color="#1565C0", transform=ax6.transAxes)
    for i, k in enumerate(right):
        y = 0.65 - i * 0.12
        ax6.text(0.62, y, k, fontsize=10, color="#333", transform=ax6.transAxes)
        c = "#E53935" if "回撤" in k else "#1565C0"
        ax6.text(0.88, y, str(m.get(k, "")), fontsize=11, fontweight="bold",
                 color=c, transform=ax6.transAxes)

    ax6.plot([0.08, 0.96], [0.69, 0.69], color="lightgray", lw=1,
             transform=ax6.transAxes, clip_on=False)

    # ---- 第4行: 总评（新增）----
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis("off")
    m = calculate_metrics(equity_curve)
    s = float(m.get("夏普比率", "0").replace("−", "-"))
    dd = float(m.get("最大回撤", "0%").replace("%", "").replace("−", "-"))
    ret = float(m.get("总收益率", "0%").replace("%", "").replace("−", "-"))

    if s >= 1:
        sr_comment = f"夏普比率{s:.2f} > 1，风险调整后收益优秀"
    elif s >= 0:
        sr_comment = f"夏普比率{s:.2f}，表现一般，超额收益有限"
    else:
        sr_comment = f"夏普比率{s:.2f} < 0，策略承担的风险未获得足够补偿"

    if dd < -15:
        dd_comment = f"最大回撤{dd:.0f}%，回撤较大，需注意仓位管理"
    elif dd < -5:
        dd_comment = f"最大回撤{dd:.0f}%，回撤可控"
    else:
        dd_comment = f"最大回撤{dd:.0f}%，回撤较小，风控良好"

    if ret > 10:
        ret_comment = f"总收益率{ret:.0f}%，收益较好"
    elif ret > 0:
        ret_comment = f"总收益率{ret:.0f}%，小幅盈利"
    else:
        ret_comment = f"总收益率{ret:.0f}%，策略亏损需优化"

    summary = f"""[回测总评]
{ret_comment} | {sr_comment} | {dd_comment}
说明：本报告不含交易成本（佣金、滑点），回测结果为理论值，实盘需酌情调整。"""
    ax7.text(0.5, 0.5, summary, transform=ax7.transAxes, fontsize=10, color="#333",
             ha="center", va="center", linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#DDD"))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"图表已保存到: {save_path}")
    plt.show()
