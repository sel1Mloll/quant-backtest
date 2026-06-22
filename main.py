"""A股量化回测系统 - 主入口"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.data_fetcher import DataFetcher
from backtest.strategies import (
    MovingAverageCross, MomentumStrategy, BuyAndHold,
    BollingerBands, RSIStrategy, MACDStrategy, VolumeBreakout,
)
from backtest.engine import BacktestEngine
from backtest.factors import FactorStrategy
from backtest.risk_manager import FixedRatioSizer, KellySizer


def main():
    """运行一个示例回测"""

    # 1. 选择数据来源
    print("选择数据来源:")
    print("1. 获取腾讯接口真实数据（推荐）")
    print("2. 使用内置样本数据（离线演示）")
    choice = input("请输入 (1/2): ") or "1"

    fetcher = DataFetcher()

    if choice == "1":
        symbol = input("请输入股票代码（回车默认 600519 贵州茅台）: ") or "600519"
        try:
            print(f"正在获取 {symbol} 数据...")
            df = fetcher.get_stock_daily_tx(symbol)
        except Exception as e:
            print(f"获取失败: {e}")
            print("切换到样本数据...")
            df = fetcher.load_sample_data()
    else:
        df = fetcher.load_sample_data()

    print(f"共 {len(df)} 条日线数据")
    print(f"日期范围: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    print()

    # 2. 选择策略
    print("选择策略:")
    menu = [
        ("1", "双均线交叉", "MA_Cross"),
        ("2", "布林带均值回归", "Bollinger Bands"),
        ("3", "RSI 超买超卖", "RSI"),
        ("4", "MACD 趋势跟踪", "MACD"),
        ("5", "成交量突破", "Volume Breakout"),
        ("6", "动量策略", "Momentum"),
        ("7", "多因子策略", "Factor Strategy"),
        ("8", "买入持有（基准）", "Buy & Hold"),
    ]
    for key, name, desc in menu:
        print(f"  {key}. {name} ({desc})")
    choice = input("请输入 (1-8): ") or "1"

    if choice == "1":
        s = int(input("短期均线窗口（默认 5）: ") or "5")
        l = int(input("长期均线窗口（默认 20）: ") or "20")
        strategy = MovingAverageCross(short_window=s, long_window=l)
    elif choice == "2":
        w = int(input("布林带窗口（默认 20）: ") or "20")
        std = float(input("标准差倍数（默认 2.0）: ") or "2.0")
        strategy = BollingerBands(window=w, num_std=std)
    elif choice == "3":
        p = int(input("RSI 周期（默认 14）: ") or "14")
        osold = float(input("超卖线（默认 30）: ") or "30")
        obought = float(input("超买线（默认 70）: ") or "70")
        strategy = RSIStrategy(period=p, oversold=osold, overbought=obought)
    elif choice == "4":
        f = int(input("快线周期（默认 12）: ") or "12")
        s_val = int(input("慢线周期（默认 26）: ") or "26")
        sig = int(input("信号周期（默认 9）: ") or "9")
        strategy = MACDStrategy(fast=f, slow=s_val, signal=sig)
    elif choice == "5":
        lb = int(input("回溯天数（默认 20）: ") or "20")
        vr = float(input("成交量倍数（默认 1.5）: ") or "1.5")
        strategy = VolumeBreakout(lookback=lb, volume_ratio=vr)
    elif choice == "6":
        lb = int(input("回溯天数（默认 20）: ") or "20")
        th = float(input("阈值（默认 0.05）: ") or "0.05")
        strategy = MomentumStrategy(lookback=lb, threshold=th)
    elif choice == "7":
        strategy = FactorStrategy()
    else:
        strategy = BuyAndHold()

    # 兼容 sample data 没有 volume 的情况
    if "volume" not in df.columns and "amount" in df.columns:
        df["volume"] = df["amount"]

    # 3. 风控设置
    print("\n风控设置（回车使用默认值，即不启用）:")
    commission = input("  佣金费率（如 0.0003=万三，默认 0）: ") or "0"
    slippage = input("  滑点比例（如 0.001=千一，默认 0）: ") or "0"
    stop_loss = input("  止损比例（如 0.05=5%，默认 0 不启用）: ") or "0"

    print("  仓位管理:")
    print("    0. 不启用（默认，全仓）")
    print("    1. 固定比例（如 50%）")
    print("    2. 凯利公式")
    sizing_choice = input("  请选择 (0/1/2): ") or "0"

    risk_manager = None
    if sizing_choice == "1":
        frac = float(input("    仓位比例（如 0.5=50%，默认 0.5）: ") or "0.5")
        risk_manager = FixedRatioSizer(fraction=frac)
    elif sizing_choice == "2":
        wr = float(input("    胜率（如 0.55=55%，默认 0.5）: ") or "0.5")
        aw = float(input("    平均盈利率（如 0.03=3%，默认 0.03）: ") or "0.03")
        al = float(input("    平均亏损率（如 0.02=2%，默认 0.02）: ") or "0.02")
        hk = input("    使用半凯利（y/n，默认 y）: ") or "y"
        risk_manager = KellySizer(win_rate=wr, avg_win=aw, avg_loss=al, half_kelly=(hk.lower() == "y"))

    # 4. 运行回测
    print(f"\n运行回测 - 策略: {strategy.name}")
    engine = BacktestEngine(
        df, strategy,
        commission_rate=float(commission),
        slippage=float(slippage),
        stop_loss=float(stop_loss),
        risk_manager=risk_manager,
    )
    metrics, benchmark = engine.run()

    # 5. 输出结果
    print("\n" + "=" * 40)
    print("回测结果")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 6. 交易记录
    trades = engine.get_trade_log()
    if len(trades) > 0:
        trade_date_col = trades.columns[0]
        trade_op_col = trades.columns[1]
        trade_price_col = trades.columns[2]
        print(f"\n交易记录 (共 {len(trades)} 笔):")
        for _, row in trades.head(10).iterrows():
            print(f"  {row[trade_date_col].date()}  {row[trade_op_col]}  @ {row[trade_price_col]:.2f}")
        if len(trades) > 10:
            print(f"  ... 共 {len(trades)} 笔")
        trades.to_csv(os.path.join("outputs", f"{strategy.name}_trades.csv"), index=False)
    else:
        print("\n无交易记录")

    # 7. 保存图表
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{strategy.name}_result.png")
    print(f"\n正在生成图表...")
    engine.plot_result(benchmark_curve=benchmark, save_path=save_path)
    print(f"图表已保存到: {save_path}")


if __name__ == "__main__":
    main()
