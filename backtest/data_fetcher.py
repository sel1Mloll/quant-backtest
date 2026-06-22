"""股票数据获取模块 - 使用 AkShare 获取 A 股数据"""

import pandas as pd
from datetime import datetime, timedelta


class DataFetcher:
    """获取 A 股历史行情数据"""

    def __init__(self):
        try:
            import akshare as ak
            self.ak = ak
        except ImportError:
            raise ImportError("请先安装 akshare: pip install akshare")

    def get_stock_daily(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """
        获取个股日线数据

        Parameters
        ----------
        symbol : str
            股票代码，如 "000001"（平安银行）
        start_date : str
            开始日期，格式 "YYYYMMDD"，默认一年前
        end_date : str
            结束日期，格式 "YYYYMMDD"，默认今天

        Returns
        -------
        pd.DataFrame
            包含 OHLCV 数据的 DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start = datetime.now() - timedelta(days=365)
            start_date = start.strftime("%Y%m%d")

        try:
            df = self.ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

            # 标准化列名
            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            raise RuntimeError(f"获取 {symbol} 数据失败: {e}")

    def get_stock_daily_tx(
        self, symbol: str, start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """
        通过腾讯接口获取个股日线数据（同花顺/腾讯数据源）

        Parameters
        ----------
        symbol : str
            股票代码，如 "000001"（平安银行）
        start_date : str
            开始日期，格式 "YYYYMMDD"，默认一年前
        end_date : str
            结束日期，格式 "YYYYMMDD"，默认今天

        Returns
        -------
        pd.DataFrame
            包含 OHLCV 数据的 DataFrame
        """
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("请先安装 akshare: pip install akshare")

        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start = datetime.now() - timedelta(days=365)
            start_date = start.strftime("%Y%m%d")

        # 判断市场标识: 6开头是沪市(sh)，其他是深市(sz)
        if symbol.startswith("6") or symbol.startswith("9"):
            tx_symbol = f"sh{symbol}"
        else:
            tx_symbol = f"sz{symbol}"

        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=tx_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                }
            )
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            raise RuntimeError(f"获取 {symbol} 数据失败: {e}")

    def get_index_daily(
        self, symbol: str = "000001", start_date: str = None, end_date: str = None
    ) -> pd.DataFrame:
        """获取大盘指数日线数据，默认上证指数"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start = datetime.now() - timedelta(days=365)
            start_date = start.strftime("%Y%m%d")

        try:
            df = self.ak.stock_zh_index_daily(symbol=f"sh{symbol}")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            mask = (df["date"] >= pd.Timestamp(start_date)) & (
                df["date"] <= pd.Timestamp(end_date)
            )
            return df[mask].reset_index(drop=True)
        except Exception as e:
            raise RuntimeError(f"获取指数 {symbol} 数据失败: {e}")

    def get_stock_list(self) -> pd.DataFrame:
        """获取全部 A 股列表"""
        try:
            df = self.ak.stock_zh_a_spot_em()
            return df[["代码", "名称"]]
        except Exception as e:
            raise RuntimeError(f"获取股票列表失败: {e}")
    def load_sample_data(self) -> pd.DataFrame:
        """加载内置的样本数据进行演示（无需网络）"""
        import os
        data_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(data_dir, "sample_data.csv")
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        print(f"[使用样本数据] {len(df)} 条, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        return df
