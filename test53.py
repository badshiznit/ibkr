import argparse
import datetime
import os
import pytz
import pandas as pd
import pandas_market_calendars as mcal
from ib_insync import *
import numpy as np
import time
import ta  # Import the technical analysis library

# Configuration (Paramètres globaux)
LOCAL_IP = '127.0.0.1'
PORT = 7496
CLIENT_ID = 1
SYMBOL = 'ES'
EXPIRY = '202409'
EXCHANGE = 'CME'
TIMEZONE = 'Europe/Zurich'
DAYS_TO_FETCH = 100
TICK_SIZE = 0.25
TRADE_AMOUNT = 100
POINT_VALUE = 50  # Valeur en USD par point de mouvement de l'indice

# Paramètres de Stop-Loss et Take-Profit
SL_POINT = 4  # Stop-Loss en points de base
TP_RATIO_1 = 1 * SL_POINT  # Take-Profit ratio pour risk_reward_ratio = 1
TP_RATIO_2 = 2 * SL_POINT  # Take-Profit ratio pour risk_reward_ratio = 2
TP_RATIO_3 = 3 * SL_POINT  # Take-Profit ratio pour risk_reward_ratio = 3

# Frais de passage d'ordre (par trade, donc par entrée ou sortie)
ORDER_FEE = 2.0  # Frais d'exécution par passage d'ordre chez IBKR en USD

def connect_ib():
    ib = IB()
    try:
        ib.connect(LOCAL_IP, PORT, clientId=CLIENT_ID)
    except Exception as e:
        print(f"Connection error: {e}")
        return None
    return ib

def fetch_data(ib, symbol, expiry, exchange, trading_days):
    contract = Future(symbol=symbol, lastTradeDateOrContractMonth=expiry, exchange=exchange)
    bars = []
    for day in trading_days:
        end_date = day.strftime('%Y%m%d %H:%M:%S')
        day_bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_date,
            durationStr='1 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )
        bars.extend(day_bars)
    return bars

def convert_to_local(df, timezone):
    local_tz = pytz.timezone(timezone)
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert(local_tz).dt.tz_localize(None)
    df.set_index('date', inplace=True)
    return df

def calculate_indicators(df):
    # RSI on 1-minute data
    df['rsi_1min'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # Resample for other timeframes and calculate RSI
    timeframes = {'rsi_5min': '5T', 'rsi_15min': '15T', 'rsi_1h': '1H'}
    for rsi_label, timeframe in timeframes.items():
        df_resampled = df['close'].resample(timeframe).last().dropna()
        rsi_resampled = ta.momentum.RSIIndicator(df_resampled, window=14).rsi()
        rsi_resampled = rsi_resampled.reindex(df.index, method='ffill')
        df[rsi_label] = rsi_resampled

    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=1000).ema_indicator()

    # Calcul des Bandes de Bollinger
    bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bb_indicator.bollinger_hband()
    df['bb_lower'] = bb_indicator.bollinger_lband()

    # Calcul du SMA pour la tendance
    df['sma'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()  # Par exemple, SMA de 50 périodes

    return df

def calculate_open_position_percentage(prev_daily_high, prev_daily_low, mrkt_open):
    range_prev_day = prev_daily_high - prev_daily_low
    open_position_pct = ((mrkt_open - prev_daily_low) / range_prev_day) * 100
    return open_position_pct

def calculate_trading_data(df):
    trading_data = []
    grouped = df.groupby(df.index.date)
    for date, group in grouped:
        mrkt_open = group.at_time('15:30')['open'].iloc[0] if not group.at_time('15:30').empty else None
        mrkt_close = group.at_time('22:00')['close'].iloc[-1] if not group.at_time('22:00').empty else None

        day_data = {
            'date': date,
            'prem_high': group.between_time('10:00', '15:29')['high'].max(),
            'prem_low': group.between_time('10:00', '15:29')['low'].min(),
            'daily_high': group['high'].max(),
            'daily_low': group['low'].min(),
            'opening_price': group.iloc[0]['open'],  # Price at midnight
            'closing_price': group.iloc[-1]['close'],
            'mrkt_open': mrkt_open,
            'mrkt_close': mrkt_close,
            'mrkt_high': group.between_time('15:30', '22:00')['high'].max(),
            'mrkt_low': group.between_time('15:30', '22:00')['low'].min()
        }
        trading_data.append(day_data)
    return pd.DataFrame(trading_data)

def add_previous_day_levels(trading_data):
    trading_data['prev_m_high'] = trading_data['mrkt_high'].shift(1)
    trading_data['prev_m_low'] = trading_data['mrkt_low'].shift(1)
    trading_data['prev_mrkt_close'] = trading_data['mrkt_close'].shift(1)
    trading_data['range_prev_day'] = trading_data['prev_m_high'] - trading_data['prev_m_low']
    trading_data['open_position_pct'] = calculate_open_position_percentage(
        trading_data['prev_m_high'], trading_data['prev_m_low'], trading_data['mrkt_open']
    )
    trading_data['open_vs_prv_close'] = trading_data['mrkt_open'] - trading_data['prev_mrkt_close']
    trading_data['open_vs_mdn8t'] = trading_data['mrkt_open'] - trading_data['opening_price']
    return trading_data

def identify_crossovers(df, trading_data, params):
    crossovers = []
    for idx, day_data in trading_data.iterrows():
        date = day_data['date']
        prev_m_high = day_data['prev_m_high']
        prev_m_low = day_data['prev_m_low']
        prev_mrkt_close = day_data['prev_mrkt_close']
        prem_high = day_data['prem_high']
        prem_low = day_data['prem_low']
        open_position_pct = day_data['open_position_pct']
        open_vs_prv_close = day_data['open_vs_prv_close']
        open_vs_mdn8t = day_data['open_vs_mdn8t']
        day_df = df[df.index.date == date].between_time('15:30', '22:00')

        for index, row in day_df.iterrows():
            time = row.name
            close = row['close']
            open_ = row['open']
            high = row['high']
            low = row['low']
            rsi_1min = row['rsi_1min']  # RSI on 1-minute data
            rsi_5min = row['rsi_5min']
            rsi_15min = row['rsi_15min']
            rsi_1h = row['rsi_1h']
            atr = row['atr']
            ema1000 = row['ema']

            # Bandes de Bollinger
            bb_upper = row['bb_upper']
            bb_lower = row['bb_lower']

            # SMA
            sma = row['sma']

            color = 'green' if close > open_ else 'red'
            size = close - open_

            if params['enable_prev_high_crossover'] and prev_m_high and low <= prev_m_high <= high:
                crossovers.append({
                    'time': time,
                    'level': 'prev_m_high',
                    'value': prev_m_high,
                    'size': size,
                    'color': color,
                    'opn_pos_pct': open_position_pct,
                    'opn_vs_prv_cls': open_vs_prv_close,
                    'opn_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio'],
                    'rsi_1min': rsi_1min,
                    'rsi_5min': rsi_5min,
                    'rsi_15min': rsi_15min,
                    'rsi_1h': rsi_1h,
                    'atr': atr,
                    'ema1000': ema1000,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'sma': sma
                })
            if params['enable_prev_close_crossover'] and prev_mrkt_close and low <= prev_mrkt_close <= high:
                crossovers.append({
                    'time': time,
                    'level': 'prev_mrkt_close',
                    'value': prev_mrkt_close,
                    'size': size,
                    'color': color,
                    'opn_pos_pct': open_position_pct,
                    'opn_vs_prv_cls': open_vs_prv_close,
                    'opn_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio'],
                    'rsi_1min': rsi_1min,
                    'rsi_5min': rsi_5min,
                    'rsi_15min': rsi_15min,
                    'rsi_1h': rsi_1h,
                    'atr': atr,
                    'ema1000': ema1000,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'sma': sma
                })
            if params['enable_prev_low_crossover'] and prev_m_low and low <= prev_m_low <= high:
                crossovers.append({
                    'time': time,
                    'level': 'prev_m_low',
                    'value': prev_m_low,
                    'size': size,
                    'color': color,
                    'opn_pos_pct': open_position_pct,
                    'opn_vs_prv_cls': open_vs_prv_close,
                    'opn_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio'],
                    'rsi_1min': rsi_1min,
                    'rsi_5min': rsi_5min,
                    'rsi_15min': rsi_15min,
                    'rsi_1h': rsi_1h,
                    'atr': atr,
                    'ema1000': ema1000,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'sma': sma
                })
            if params['enable_premarket_high_crossover'] and prem_high and low <= prem_high <= high:
                crossovers.append({
                    'time': time,
                    'level': 'prem_high',
                    'value': prem_high,
                    'size': size,
                    'color': color,
                    'opn_pos_pct': open_position_pct,
                    'opn_vs_prv_cls': open_vs_prv_close,
                    'opn_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio'],
                    'rsi_1min': rsi_1min,
                    'rsi_5min': rsi_5min,
                    'rsi_15min': rsi_15min,
                    'rsi_1h': rsi_1h,
                    'atr': atr,
                    'ema1000': ema1000,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'sma': sma
                })
            if params['enable_premarket_low_crossover'] and prem_low and low <= prem_low <= high:
                crossovers.append({
                    'time': time,
                    'level': 'prem_low',
                    'value': prem_low,
                    'size': size,
                    'color': color,
                    'opn_pos_pct': open_position_pct,
                    'opn_vs_prv_cls': open_vs_prv_close,
                    'opn_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio'],
                    'rsi_1min': rsi_1min,
                    'rsi_5min': rsi_5min,
                    'rsi_15min': rsi_15min,
                    'rsi_1h': rsi_1h,
                    'atr': atr,
                    'ema1000': ema1000,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'sma': sma
                })
    return crossovers

def simulate_trades(df, crossovers, risk_reward_ratio):
    trades = []
    in_trade = False
    last_close_time = None

    for crossover in crossovers:
        entry_time = crossover['time']

        if last_close_time and entry_time <= last_close_time:
            continue

        if in_trade:
            continue

        crossover_value = crossover['value']
        entry_price = crossover_value

        if df.loc[entry_time]['open'] > crossover_value:  # Le prix vient d'en haut
            direction = 'long'
            stop_loss, take_profit = calculate_stop_loss_take_profit(entry_price, risk_reward_ratio, direction)

        else:  # Le prix vient d'en dessous
            direction = 'short'
            stop_loss, take_profit = calculate_stop_loss_take_profit(entry_price, risk_reward_ratio, direction)

        in_trade = True

        for index, row in df.loc[df.index > entry_time].iterrows():
            if direction == 'long':
                if row['low'] <= stop_loss:  # Stop-Loss atteint
                    trades.append(record_trade(crossover, row, direction, entry_price, stop_loss, take_profit, 'loss', risk_reward_ratio))
                    in_trade = False
                    last_close_time = row.name
                    break
                elif row['high'] >= take_profit:  # Take-Profit atteint
                    trades.append(record_trade(crossover, row, direction, entry_price, stop_loss, take_profit, 'win', risk_reward_ratio))
                    in_trade = False
                    last_close_time = row.name
                    break
            else:  # direction == 'short'
                if row['high'] >= stop_loss:  # Stop-Loss atteint
                    trades.append(record_trade(crossover, row, direction, entry_price, stop_loss, take_profit, 'loss', risk_reward_ratio))
                    in_trade = False
                    last_close_time = row.name
                    break
                elif row['low'] <= take_profit:  # Take-Profit atteint
                    trades.append(record_trade(crossover, row, direction, entry_price, stop_loss, take_profit, 'win', risk_reward_ratio))
                    in_trade = False
                    last_close_time = row.name
                    break

    return trades

def calculate_stop_loss_take_profit(entry_price, risk_reward_ratio, direction):
    if direction == 'long':
        if risk_reward_ratio == 0.5:
            stop_loss = entry_price - 2 * SL_POINT
            take_profit = entry_price + SL_POINT
        elif risk_reward_ratio == 0.33:
            stop_loss = entry_price - 3 * SL_POINT
            take_profit = entry_price + SL_POINT
        else:
            stop_loss = entry_price - SL_POINT
            if risk_reward_ratio == 1:
                take_profit = entry_price + TP_RATIO_1
            elif risk_reward_ratio == 2:
                take_profit = entry_price + TP_RATIO_2
            elif risk_reward_ratio == 3:
                take_profit = entry_price + TP_RATIO_3
    else:  # direction == 'short'
        if risk_reward_ratio == 0.5:
            stop_loss = entry_price + 2 * SL_POINT
            take_profit = entry_price - SL_POINT
        elif risk_reward_ratio == 0.33:
            stop_loss = entry_price + 3 * SL_POINT
            take_profit = entry_price - SL_POINT
        else:
            stop_loss = entry_price + SL_POINT
            if risk_reward_ratio == 1:
                take_profit = entry_price - TP_RATIO_1
            elif risk_reward_ratio == 2:
                take_profit = entry_price - TP_RATIO_2
            elif risk_reward_ratio == 3:
                take_profit = entry_price - TP_RATIO_3
    return stop_loss, take_profit

def record_trade(crossover, row, direction, entry_price, stop_loss, take_profit, outcome, risk_reward_ratio):
    net_profit_usd = calculate_net_profit(outcome, stop_loss, take_profit, entry_price, direction, risk_reward_ratio)
    return {
        'entry_time': crossover['time'],
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'direction': direction,
        'outcome': outcome,
        'net_profit_usd': net_profit_usd,
        'fees': 2 * ORDER_FEE,
        'close_time': row.name,
        'level': crossover['level'],
        'value': crossover['value'],
        'opn_pos_pct': crossover['opn_pos_pct'],
        'opn_vs_prv_cls': crossover['opn_vs_prv_cls'],
        'opn_vs_mdn8t': crossover['opn_vs_mdn8t'],
        'rsi_1min': crossover['rsi_1min'],
        'rsi_5min': crossover['rsi_5min'],
        'rsi_15min': crossover['rsi_15min'],
        'rsi_1h': crossover['rsi_1h'],
        'atr': crossover['atr'],
        'ema1000': crossover['ema1000'],
        'bb_upper': crossover['bb_upper'],
        'bb_lower': crossover['bb_lower'],
        'sma': crossover['sma'],
        'risk': risk_reward_ratio
    }

def calculate_net_profit(outcome, stop_loss, take_profit, entry_price, direction, risk_reward_ratio):
    if direction == 'long':
        if risk_reward_ratio == 0.5:
            sl_value = 2 * SL_POINT
        elif risk_reward_ratio == 0.33:
            sl_value = 3 * SL_POINT
        else:
            sl_value = SL_POINT
    else:  # direction == 'short'
        if risk_reward_ratio == 0.5:
            sl_value = 2 * SL_POINT
        elif risk_reward_ratio == 0.33:
            sl_value = 3 * SL_POINT
        else:
            sl_value = SL_POINT

    if outcome == 'win':
        if direction == 'long':
            return (take_profit - entry_price) * POINT_VALUE - 2 * ORDER_FEE
        else:
            return (entry_price - take_profit) * POINT_VALUE - 2 * ORDER_FEE
    else:
        return (-sl_value * POINT_VALUE) - 2 * ORDER_FEE

def calculate_summary(trades, params):
    total_gains_usd = sum(trade['net_profit_usd'] for trade in trades if trade['outcome'] == 'win')
    total_losses_usd = sum(trade['net_profit_usd'] for trade in trades if trade['outcome'] == 'loss')
    total_fees_usd = sum(trade['fees'] for trade in trades)

    total_trades = len(trades)
    wins = len([t for t in trades if t['outcome'] == 'win'])
    losses = len([t for t in trades if t['outcome'] == 'loss'])

    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    loss_rate = (losses / total_trades) * 100 if total_trades > 0 else 0
    profit_factor = (total_gains_usd / -total_losses_usd) if total_losses_usd < 0 else float('inf')

    summary = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'profit_factor': profit_factor,
        'total_fees_usd': total_fees_usd,
        'net_profit_usd': total_gains_usd + total_losses_usd,
        'active_parameter': params['active_parameter'],
        'risk_reward_ratio': params['risk_reward_ratio']
    }

    return summary

def get_last_trading_days(days_to_fetch):
    cme_equity = mcal.get_calendar('CME_Equity')
    end_date = datetime.datetime.now()
    start_date = end_date - pd.Timedelta(days=200)
    schedule = cme_equity.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index[-(days_to_fetch + 1):].tolist()
    return trading_days

def write_trades_to_csv(trades, filename='trades.csv'):
    columns = [
        "EntryTime", "Price", "SL", "TP", "Direction", "Outcome",
        "CloseTime", "Level", "Value", "OpnPosPct",
        "OpnVsPrvCls", "OpnVsMdn8t", "RSI_1min", "RSI_5min", "RSI_15min", "RSI_1h",
        "ATR", "EMA1000", "BB_Upper", "BB_Lower", "SMA", "Risk", "NetProfitUSD", "FeesUSD"
    ]
    trades_data = [{
        'EntryTime': trade['entry_time'].strftime('%y-%m-%d %H:%M:%S'),
        'Price': trade['entry_price'],
        'SL': trade['stop_loss'],
        'TP': trade['take_profit'],
        'Direction': trade['direction'],
        'Outcome': trade['outcome'],
        'CloseTime': trade['close_time'].strftime('%y-%m-%d %H:%M:%S'),
        'Level': trade['level'],
        'Value': trade['value'],
        'OpnPosPct': trade['opn_pos_pct'],
        'OpnVsPrvCls': trade['opn_vs_prv_cls'],
        'OpnVsMdn8t': trade['opn_vs_mdn8t'],
        'RSI_1min': trade['rsi_1min'],
        'RSI_5min': trade['rsi_5min'],
        'RSI_15min': trade['rsi_15min'],
        'RSI_1h': trade['rsi_1h'],
        'ATR': trade['atr'],
        'EMA1000': trade['ema1000'],
        'BB_Upper': trade['bb_upper'],
        'BB_Lower': trade['bb_lower'],
        'SMA': trade['sma'],
        'Risk': trade['risk'],
        'NetProfitUSD': trade['net_profit_usd'],
        'FeesUSD': trade['fees']
    } for trade in trades]

    df_trades = pd.DataFrame(trades_data, columns=columns)

    if not os.path.isfile(filename):
        df_trades.to_csv(filename, index=False, columns=columns)
    else:
        df_trades.to_csv(filename, mode='a', header=False, index=False, columns=columns)

def main():
    risk_reward_ratios = [0.5, 0.33, 1, 2, 3]

    crossover_params = [
        {"enable_prev_high_crossover": True, "enable_prev_low_crossover": False, "enable_prev_close_crossover": False, "enable_premarket_high_crossover": False, "enable_premarket_low_crossover": False},
        {"enable_prev_high_crossover": False, "enable_prev_low_crossover": True, "enable_prev_close_crossover": False, "enable_premarket_high_crossover": False, "enable_premarket_low_crossover": False},
        {"enable_prev_high_crossover": False, "enable_prev_low_crossover": False, "enable_prev_close_crossover": True, "enable_premarket_high_crossover": False, "enable_premarket_low_crossover": False},
        {"enable_prev_high_crossover": False, "enable_prev_low_crossover": False, "enable_prev_close_crossover": False, "enable_premarket_high_crossover": True, "enable_premarket_low_crossover": False},
        {"enable_prev_high_crossover": False, "enable_prev_low_crossover": False, "enable_prev_close_crossover": False, "enable_premarket_high_crossover": False, "enable_premarket_low_crossover": True},
    ]

    ib = connect_ib()
    if ib is None:
        return

    trading_days = get_last_trading_days(DAYS_TO_FETCH)
    bars = fetch_data(ib, SYMBOL, EXPIRY, EXCHANGE, trading_days)
    if not bars:
        print("No data retrieved.")
        return

    df = util.df(bars)
    df = convert_to_local(df, TIMEZONE)

    df = calculate_indicators(df)

    trading_data = calculate_trading_data(df)
    trading_data = add_previous_day_levels(trading_data)

    # Use only the last DAYS_TO_FETCH days for simulation
    simulation_data = trading_data.iloc[1:]

    total_start_time = time.time()

    profit_factors = []

    for ratio in risk_reward_ratios:
        for params in crossover_params:
            params['risk_reward_ratio'] = ratio

            # Définir l'active_parameter basé sur les croisements activés
            for key, value in params.items():
                if value and key.startswith('enable'):
                    params['active_parameter'] = key
                    break

            crossovers = identify_crossovers(df, simulation_data, params)
            trades = simulate_trades(df, crossovers, ratio)
            summary = calculate_summary(trades, params)

            profit_factors.append(summary['profit_factor'])

            print(f"\nSummary: {params['active_parameter']} {ratio}:1")
            print(f"Total Trades: {summary['total_trades']}")
            print(f"Wins: {summary['wins']} ({summary['win_rate']:.2f}%)")
            print(f"Losses: {summary['losses']} ({summary['loss_rate']:.2f}%)")
            print(f"Profit Factor: {summary['profit_factor']:.2f}")
            print(f"Net Profit (USD): {summary['net_profit_usd']:.2f}")
            print(f"Total Fees (USD): {summary['total_fees_usd']:.2f}")

            if trades:
                print("\nTrades:")
                header = "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<20} {:<15} {:<10} {:<15} {:<15} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "EntryTime", "Price", "SL", "TP", "Direction", "Outcome", "CloseTime", "Level", "Value", "OpnPosPct",
                    "OpnVsPrvCls", "OpnVsMdn8t", "RSI_1min", "RSI_5min", "RSI_15min", "RSI_1h", "ATR", "EMA1000", "BB_Upper", "BB_Lower", "SMA", "Risk"
                )
                print(header)
                print("-" * len(header))
                for trade in trades:
                    print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10} {:<10} {:<20} {:<15} {:<10.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
                        trade['entry_time'].strftime('%y-%m-%d %H:%M:%S'), trade['entry_price'], trade['stop_loss'], trade['take_profit'],
                        trade['direction'], trade['outcome'], trade['close_time'].strftime('%y-%m-%d %H:%M:%S'), trade['level'], trade['value'],
                        trade['opn_pos_pct'], trade['opn_vs_prv_cls'], trade['opn_vs_mdn8t'], trade['rsi_1min'], trade['rsi_5min'], trade['rsi_15min'],
                        trade['rsi_1h'], trade['atr'], trade['ema1000'], trade['bb_upper'], trade['bb_lower'], trade['sma'], trade['risk']
                    ))

                write_trades_to_csv(trades)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total execution time: {total_duration / 60:.2f} minutes")

    if profit_factors:
        min_profit_factor = min(profit_factors)
        max_profit_factor = max(profit_factors)
        avg_profit_factor = sum(profit_factors) / len(profit_factors)
        print(f"\nFinal Summary:")
        print(f"Minimum Profit Factor: {min_profit_factor:.2f}")
        print(f"Maximum Profit Factor: {max_profit_factor:.2f}")
        print(f"Average Profit Factor: {avg_profit_factor:.2f}")

    ib.disconnect()

if __name__ == "__main__":
    main()
