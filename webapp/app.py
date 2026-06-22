"""A股量化回测系统 - Web应用后端"""

import sys
import os
import io
from contextlib import redirect_stdout

# 将项目根目录加入 Python 路径，以便导入 backtest 包
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from backtest.data_fetcher import DataFetcher
from backtest.strategies import (
    MovingAverageCross, MomentumStrategy, BuyAndHold,
    BollingerBands, RSIStrategy, MACDStrategy, VolumeBreakout,
)
from backtest.engine import BacktestEngine
from backtest.factors import FactorStrategy
from backtest.risk_manager import FixedRatioSizer, KellySizer

app = Flask(__name__)


# ── 路由 ──────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/backtest', methods=['POST'])
def api_backtest():
    """运行回测并返回 JSON 结果"""
    try:
        config = request.json
        df = _fetch_data(config)
        strategy = _create_strategy(config.get('strategy', 'ma_cross'), config.get('params', {}))
        risk_cfg = config.get('risk', {})
        risk_manager = _create_risk_manager(risk_cfg)

        engine = BacktestEngine(
            df, strategy,
            commission_rate=risk_cfg.get('commission', 0),
            slippage=risk_cfg.get('slippage', 0),
            stop_loss=risk_cfg.get('stop_loss', 0),
            risk_manager=risk_manager,
        )
        initial_capital = config.get('initial_capital', 100000)
        metrics, benchmark = engine.run(initial_capital=initial_capital)

        equity = engine.equity_curve
        trades_df = engine.get_trade_log()

        daily_ret = equity.pct_change().dropna()
        cum_max = equity.cummax()
        drawdown = (equity - cum_max) / cum_max

        rolling_sr = daily_ret.rolling(60).mean() / daily_ret.rolling(60).std() * np.sqrt(252)

        monthly_ret = daily_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        years = sorted(set(monthly_ret.index.year))
        monthly_data = []
        for year in years:
            for month in range(1, 13):
                mask = (monthly_ret.index.year == year) & (monthly_ret.index.month == month)
                vals = monthly_ret[mask]
                if len(vals) > 0 and not np.isnan(vals.iloc[0]):
                    monthly_data.append([month - 1, years.index(year), round(float(vals.iloc[0]), 4)])

        hist_counts, hist_bins = np.histogram(daily_ret, bins=50, density=True)

        trades = []
        if len(trades_df) > 0:
            for _, row in trades_df.iterrows():
                d = row.iloc[0]
                trades.append({
                    'date': d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d),
                    'action': str(row.iloc[1]),
                    'price': round(float(row.iloc[2]), 2),
                })

        dates = [d.strftime('%Y-%m-%d') for d in equity.index]
        result = {
            'success': True,
            'strategy_name': strategy.name,
            'metrics': metrics,
            'equity_curve': {
                'dates': dates,
                'values': [round(float(v), 2) for v in equity.values],
                'benchmark': [round(float(v), 2) for v in benchmark.values],
            },
            'drawdown': {
                'dates': dates,
                'values': [round(float(v), 4) for v in drawdown.values],
            },
            'daily_returns': {
                'dates': dates[1:],
                'values': [round(float(v), 4) for v in daily_ret.values],
            },
            'rolling_sharpe': {
                'dates': [d.strftime('%Y-%m-%d') for d in rolling_sr.dropna().index],
                'values': [round(float(v), 2) for v in rolling_sr.dropna().values],
            },
            'monthly_returns': {
                'years': years,
                'data': monthly_data,
            },
            'distribution': {
                'bins': [round(float(b), 4) for b in hist_bins[:-1]],
                'counts': [round(float(c), 4) for c in hist_counts],
                'mean': round(float(daily_ret.mean()), 4),
                'std': round(float(daily_ret.std()), 4),
                'skew': round(float(daily_ret.skew()), 2),
                'kurtosis': round(float(daily_ret.kurtosis()), 2),
            },
            'trades': trades,
            'data_info': {
                'total_rows': len(df),
                'start_date': str(df['date'].iloc[0].date()),
                'end_date': str(df['date'].iloc[-1].date()),
            },
        }
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 400


def _fetch_data(config):
    source = config.get('data_source', 'sample')
    buf = io.StringIO()
    with redirect_stdout(buf):
        if source == 'real':
            symbol = config.get('symbol', '600519')
            try:
                fetcher = DataFetcher()
                df = fetcher.get_stock_daily_tx(symbol)
            except Exception:
                df = DataFetcher().load_sample_data()
        else:
            df = DataFetcher().load_sample_data()
    if 'volume' not in df.columns and 'amount' in df.columns:
        df['volume'] = df['amount']
    return df


def _create_strategy(key, params):
    mapping = {
        'ma_cross': lambda: MovingAverageCross(
            short_window=params.get('short_window', 5),
            long_window=params.get('long_window', 20)),
        'bollinger': lambda: BollingerBands(
            window=params.get('window', 20),
            num_std=params.get('num_std', 2.0)),
        'rsi': lambda: RSIStrategy(
            period=params.get('period', 14),
            oversold=params.get('oversold', 30),
            overbought=params.get('overbought', 70)),
        'macd': lambda: MACDStrategy(
            fast=params.get('fast', 12),
            slow=params.get('slow', 26),
            signal=params.get('signal', 9)),
        'volume_breakout': lambda: VolumeBreakout(
            lookback=params.get('lookback', 20),
            volume_ratio=params.get('volume_ratio', 1.5)),
        'momentum': lambda: MomentumStrategy(
            lookback=params.get('lookback', 20),
            threshold=params.get('threshold', 0.05)),
        'factor': lambda: FactorStrategy(
            top_pct=params.get('top_pct', 0.2),
            bottom_pct=params.get('bottom_pct', 0.2)),
        'buy_hold': lambda: BuyAndHold(),
    }
    return mapping.get(key, mapping['buy_hold'])()


def _create_risk_manager(cfg):
    sizing = cfg.get('sizing', 'none')
    if sizing == 'fixed':
        return FixedRatioSizer(fraction=cfg.get('fraction', 0.5))
    elif sizing == 'kelly':
        return KellySizer(
            win_rate=cfg.get('win_rate', 0.5),
            avg_win=cfg.get('avg_win', 0.03),
            avg_loss=cfg.get('avg_loss', 0.02),
            half_kelly=cfg.get('half_kelly', True))
    return None


if __name__ == '__main__':
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open('http://localhost:5000')

    print('=' * 50)
    print('  A股量化回测系统')
    print('  访问地址: http://localhost:5000')
    print('=' * 50)
    threading.Timer(1.5, open_browser).start()
    app.run(debug=False, port=5000)
