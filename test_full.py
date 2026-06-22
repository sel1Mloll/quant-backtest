import sys, os
sys.path.insert(0, "E:\\quant-backtest")
os.environ["PYTHONIOENCODING"] = "utf-8"

from backtest.data_fetcher import DataFetcher
from backtest.engine import BacktestEngine
from backtest.strategies import (
    MovingAverageCross, MomentumStrategy, BuyAndHold,
    BollingerBands, RSIStrategy, MACDStrategy, VolumeBreakout
)
from backtest.factors import FactorEngine, FactorStrategy

fetcher = DataFetcher()

df = fetcher.get_stock_daily_tx("600519")

# 检查列名，给 volume 兼容
if "volume" not in df.columns and "amount" in df.columns:
    # 用 amount 近似替代 volume
    df["volume"] = df["amount"]

strategies = [
    ("MA_Cross(5,20)", MovingAverageCross(5, 20)),
    ("Bollinger(20,2)", BollingerBands(20, 2.0)),
    ("RSI(14,30,70)", RSIStrategy(14, 30, 70)),
    ("MACD(12,26,9)", MACDStrategy(12, 26, 9)),
    ("VolumeBreakout(20,1.5)", VolumeBreakout(20, 1.5)),
    ("Buy&Hold", BuyAndHold()),
]

print(f"=== 策略对比: 贵州茅台 (600519) ===")
print(f"数据: {len(df)} 条, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
print()

results = []
for name, strategy in strategies:
    engine = BacktestEngine(df, strategy)
    metrics, _ = engine.run()
    results.append({
        "策略": name,
        "收益率": metrics.get("总收益率", "err"),
        "年化": metrics.get("年化收益率", "err"),
        "夏普": metrics.get("夏普比率", "err"),
        "最大回撤": metrics.get("最大回撤", "err"),
        "胜率": metrics.get("胜率", "err"),
    })
    print(f"{name:25s} | 收益: {metrics.get('总收益率','err'):>8s} | 夏普: {metrics.get('夏普比率','err'):>6s} | 回撤: {metrics.get('最大回撤','err'):>8s} | 胜率: {metrics.get('胜率','err'):>6s}")

print()
print("=== Factor Engine Test ===")
fe = FactorEngine(df)
all_factors = fe.get_all_factors()
print(f"Factors calculated: {len(all_factors.columns)}")
for col in all_factors.columns:
    print(f"  - {col}")

print()
print("=== Factor Strategy Test ===")
fs = FactorStrategy()
engine2 = BacktestEngine(df, fs)
metrics2, _ = engine2.run()
print(f"FactorStrategy | Return: {metrics2.get('总收益率','err')} | Sharpe: {metrics2.get('夏普比率','err')}")
print()
print("DONE!")
