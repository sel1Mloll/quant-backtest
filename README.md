# A股量化回测系统

一个基于 Python 的 A 股量化回测系统，支持多种策略的回测与绩效评估。

## 功能

- 通过 AkShare 获取 A 股历史行情数据
- 支持双均线交叉、动量等交易策略
- 计算夏普比率、最大回撤、年化收益率等绩效指标
- 可视化净值曲线、回撤曲线、收益率分布

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行回测
python main.py
```

## 项目结构

```
quant-backtest/
├── backtest/
│   ├── engine.py        # 回测引擎
│   ├── strategies.py    # 交易策略
│   ├── data_fetcher.py  # 数据获取
│   └── metrics.py       # 绩效评估
├── main.py               # 主入口
├── requirements.txt      # 依赖清单
└── outputs/              # 输出图表
```

## 回测指标

- 总收益率 / 年化收益率
- 年化波动率
- 夏普比率
- 最大回撤
- 胜率
- Calmar比率

## 扩展

在 `backtest/strategies.py` 中添加自定义策略，继承 `BaseStrategy` 类并实现 `generate_signals` 方法即可。
