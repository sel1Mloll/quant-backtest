import sys, os
sys.path.insert(0, "E:\\quant-backtest")
from backtest.data_fetcher import DataFetcher

fetcher = DataFetcher()
df = fetcher.get_stock_daily_tx("601318")
dt_col = df.columns[0]
cl_col = df.columns[3]
print(f"中国平安: 共 {len(df)} 条数据")
print(f"日期范围: {df[dt_col].iloc[0].date()} ~ {df[dt_col].iloc[-1].date()}")
print(f"最新收盘价: {df[cl_col].iloc[-1]:.2f}")
print()
df2 = fetcher.get_stock_daily_tx("000001")
print(f"平安银行: 共 {len(df2)} 条数据")
print(f"日期范围: {df2[dt_col].iloc[0].date()} ~ {df2[dt_col].iloc[-1].date()}")
print(f"最新收盘价: {df2[cl_col].iloc[-1]:.2f}")
