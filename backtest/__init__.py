"""A股量化回测系统"""

from .engine import BacktestEngine
from .strategies import MovingAverageCross, MomentumStrategy
from .metrics import calculate_metrics
from .data_fetcher import DataFetcher

__version__ = "0.1.0"
"""A股量化回测系统"""

from .engine import BacktestEngine
from .strategies import (
    MovingAverageCross,
    MomentumStrategy,
    BuyAndHold,
    BollingerBands,
    RSIStrategy,
    MACDStrategy,
    VolumeBreakout,
)
from .metrics import calculate_metrics
from .data_fetcher import DataFetcher
from .factors import FactorEngine, FactorStrategy

__version__ = "0.2.0"
