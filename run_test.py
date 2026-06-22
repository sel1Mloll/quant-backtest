import sys, os
sys.path.insert(0, "E:\\quant-backtest")
os.environ["PYTHONIOENCODING"] = "utf-8"

from backtest.data_fetcher import DataFetcher
from backtest.strategies import MovingAverageCross, MomentumStrategy
from backtest.engine import BacktestEngine

fetcher = DataFetcher()
df = fetcher.load_sample_data()

date_col = "date"
close_col = "close"

print(f"数据: {len(df)} 条记录")
print(f"日期范围: {df[date_col].iloc[0].date()} ~ {df[date_col].iloc[-1].date()}")
print()

# 策略1: 双均线交叉
strategy1 = MovingAverageCross(short_window=5, long_window=20)
engine1 = BacktestEngine(df, strategy1)
metrics1, benchmark1 = engine1.run()

print("=" * 50)
print("策略1: 双均线交叉(5,20)")
print("=" * 50)
for k, v in metrics1.items():
    print(f"{k}: {v}")

# 交易记录
trades1 = engine1.get_trade_log()
if len(trades1) > 0:
    print(f"\n交易记录: {len(trades1)} 笔")
    trade_date_col = trades1.columns[0]
    trade_op_col = trades1.columns[1]
    trade_price_col = trades1.columns[2]
    for _, row in trades1.head(8).iterrows():
        print(f"  {row[trade_date_col].date()}  {row[trade_op_col]}  @ {row[trade_price_col]:.2f}")

# 保存图表
save_path1 = "E:\\quant-backtest\\outputs\\MA_Cross_result.png"
os.makedirs("E:\\quant-backtest\\outputs", exist_ok=True)
engine1.plot_result(benchmark_curve=benchmark1, save_path=save_path1)
print(f"\n图表已保存到: {save_path1}")
