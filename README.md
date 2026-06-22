# A股量化回测系统

一个基于 Python 的 A 股量化回测系统，支持多种策略的回测与绩效评估。提供命令行和 Web 两种使用方式。

## 功能

- 通过 AkShare / 腾讯接口获取 A 股历史行情数据
- 内置 7 种交易策略：双均线交叉、布林带、RSI、MACD、成交量突破、动量、多因子
- 支持仓位管理：固定比例、凯利公式
- 支持佣金、滑点、止损等风控设置
- 计算夏普比率、索提诺比率、最大回撤、Calmar 比率等专业指标
- 交互式 Web 仪表盘：净值曲线、回撤、滚动夏普、月度热力图、收益分布

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 命令行模式
python main.py

# 3. Web 模式（推荐）
pip install flask
python webapp/app.py
# 浏览器自动打开 http://localhost:5000
```

Windows 用户也可以双击 `webapp/run.bat` 一键启动 Web 界面。

## 项目结构

```
quant-backtest/
├── backtest/
│   ├── engine.py          # 回测引擎
│   ├── strategies.py      # 交易策略
│   ├── factors.py         # 多因子打分
│   ├── risk_manager.py    # 仓位管理
│   ├── data_fetcher.py    # 数据获取
│   ├── metrics.py         # 绩效评估与绘图
│   └── sample_data.csv    # 内置样本数据
├── webapp/
│   ├── app.py             # Flask 后端 API
│   ├── templates/
│   │   └── index.html     # 前端仪表盘
│   └── run.bat            # Windows 一键启动
├── tests/                 # 单元测试
├── outputs/               # 输出图表与交易记录
├── main.py                # 命令行入口
└── requirements.txt       # 依赖清单
```

## Web 界面

左侧配置面板选择数据源、策略、参数和风控设置，点击「运行回测」即可看到：

- **KPI 卡片**：总收益率、年化收益率、夏普比率、最大回撤等
- **净值曲线**：策略净值 vs 基准，支持缩放
- **回撤曲线**：实时最大回撤可视化
- **滚动夏普**：60 日滚动夏普比率趋势
- **月度热力图**：按年月展示收益率
- **收益分布**：直方图 + 正态拟合
- **交易记录**：完整买卖明细表

## 策略说明

| 策略 | 说明 |
|------|------|
| 双均线交叉 | 短期均线上穿长期均线买入，下穿卖出 |
| 布林带 | 价格触及下轨买入，触及上轨卖出 |
| RSI | RSI 低于超卖线买入，高于超买线卖出 |
| MACD | DIF 上穿 DEA 买入，下穿卖出 |
| 成交量突破 | 价格创新高 + 放量时买入 |
| 动量 | N 日收益率超过阈值时买入 |
| 多因子 | 基于动量、波动率、量价等因子综合打分 |
| 买入持有 | 基准策略，第一天买入后持有不动 |

## 回测指标

| 指标 | 说明 |
|------|------|
| 总收益率 | 回测期间总收益 |
| 年化收益率 | 按 252 个交易日折算 |
| 年化波动率 | 收益率标准差 x sqrt(252) |
| 夏普比率 | (年化收益 - 无风险利率) / 年化波动率 |
| 索提诺比率 | 仅考虑下行风险的风险调整收益 |
| 最大回撤 | 净值从高点到低点的最大跌幅 |
| Calmar比率 | 年化收益率 / 最大回撤 |
| 日胜率 | 正收益天数占比 |

## 扩展

在 `backtest/strategies.py` 中添加自定义策略，继承 `BaseStrategy` 类并实现 `generate_signals` 方法即可：

```python
from backtest.strategies import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="MyStrategy")

    def generate_signals(self, data):
        # data 包含 date, open, high, low, close, volume 列
        # 返回 pd.Series: 1=买入, -1=卖出, 0=持有
        signals = pd.Series(0, index=data.index)
        # ... 你的策略逻辑
        return signals
```
