"""A股量化回测系统 - 主入口"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest.data_fetcher import DataFetcher
from backtest.strategies import MovingAverageCross, MomentumStrategy, BuyAndHold
from backtest.engine import BacktestEngine


def main():
    """运行一个示例回测"""

    # 1. 获取数据
    print("正在获取数据...")
    fetcher = DataFetcher()
    symbol = input("请输入股票代码（如 000001 平安银行，直接回车默认 000001）: ") or "000001"
    df = fetcher.get_stock_daily(symbol)

    print(f"已获取 {symbol} 共 {len(df)} 条日线数据")
    print(f"日期范围: {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")

    # 2. 选择策略
    print("\n请选择策略:")
    print("1. 双均线交叉 (默认)")
    print("2. 动量策略")
    choice = input("请输入 (1/2): ") or "1"

    if choice == "2":
        lookback = int(input("请输入回看天数（默认 20）: ") or "20")
        threshold = float(input("请输入阈值（默认 0.05）: ") or "0.05")
        strategy = MomentumStrategy(lookback=lookback, threshold=threshold)
    else:
        short_w = int(input("请输入短期均线窗口（默认 5）: ") or "5")
        long_w = int(input("请输入长期均线窗口（默认 20）: ") or "20")
        strategy = MovingAverageCross(short_window=short_w, long_window=long_w)

    # 3. 运行回测
    print(f"\n运行回测 - 策略: {strategy.name}")
    engine = BacktestEngine(df, strategy)
    metrics, benchmark = engine.run()

    # 4. 输出结果
    print("\n" + "=" * 40)
    print("回测结果")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 5. 交易记录
    trades = engine.get_trade_log()
    if len(trades) > 0:
        print(f"\n交易记录（共 {len(trades)} 笔）:")
        for _, row in trades.head(10).iterrows():
            print(f"  {row['日期'].date()}  {row['操作']}  @ {row['价格']:.2f}")
        if len(trades) > 10:
            print(f"  ... 共 {len(trades)} 笔")
    else:
        print("\n无交易记录")

    # 6. 保存图表
    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{strategy.name}_result.png")
    print(f"\n图表保存中...")
    engine.plot_result(benchmark_curve=benchmark, save_path=save_path)


if __name__ == "__main__":
    main()
