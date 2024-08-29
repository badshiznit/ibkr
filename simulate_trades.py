import argparse
import datetime
import os
import pytz
import pandas as pd
import pandas_market_calendars as mcal
from ib_insync import *
from numba import njit
import numpy as np
import time

# Configuration (Paramètres globaux)
LOCAL_IP = '127.0.0.1'
PORT = 7496
CLIENT_ID = 1
SYMBOL = 'ES'
EXPIRY = '202409'
EXCHANGE = 'CME'
TIMEZONE = 'Europe/Zurich'
DAYS_TO_FETCH = 5
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

@njit
def calculate_rsi_numba(close_prices, period=14):
    delta = np.diff(close_prices)
    gain = np.where(delta > 0, delta, 0)
    loss = -np.where(delta < 0, delta, 0)
    avg_gain = np.zeros_like(close_prices)
    avg_loss = np.zeros_like(close_prices)
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    for i in range(period + 1, len(close_prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@njit
def calculate_atr_numba(high_prices, low_prices, close_prices, period=14):
    tr = np.maximum(high_prices - low_prices, np.maximum(np.abs(high_prices - np.roll(close_prices, 1)), np.abs(low_prices - np.roll(close_prices, 1))))
    tr[0] = 0
    atr = np.zeros_like(close_prices)
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(close_prices)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr

def calculate_ema(df, period=1000):
    df['ema'] = df['close'].ewm(span=period, adjust=False).mean()
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
            color = 'green' if close > open_ else 'red'
            size = close - open_

            if params['enable_prev_high_crossover'] and prev_m_high and low <= prev_m_high <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prev_m_high',
                    'crossover_value': prev_m_high,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio']
                })
            if params['enable_prev_close_crossover'] and prev_mrkt_close and low <= prev_mrkt_close <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prev_mrkt_close',
                    'crossover_value': prev_mrkt_close,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio']
                })
            if params['enable_prev_low_crossover'] and prev_m_low and low <= prev_m_low <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prev_m_low',
                    'crossover_value': prev_m_low,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio']
                })
            if params['enable_premarket_high_crossover'] and prem_high and low <= prem_high <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prem_high',
                    'crossover_value': prem_high,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio']
                })
            if params['enable_premarket_low_crossover'] and prem_low and low <= prem_low <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prem_low',
                    'crossover_value': prem_low,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t,
                    'risk': params['risk_reward_ratio']
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

        crossover_value = crossover['crossover_value']
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
    net_profit_usd = calculate_net_profit(outcome, stop_loss, take_profit, entry_price, direction)
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
        'crossover': crossover['crossover'],
        'crossover_value': crossover['crossover_value'],
        'open_position_pct': crossover['open_position_pct'],
        'open_vs_prv_close': crossover['open_vs_prv_close'],
        'open_vs_mdn8t': crossover['open_vs_mdn8t'],
        'rsi': row['rsi'],
        'atr': row['atr'],
        'ema': row['ema'] - entry_price,
        'risk': risk_reward_ratio
    }

def calculate_net_profit(outcome, stop_loss, take_profit, entry_price, direction):
    if outcome == 'win':
        if direction == 'long':
            return (take_profit - entry_price) * POINT_VALUE - 2 * ORDER_FEE
        else:
            return (entry_price - take_profit) * POINT_VALUE - 2 * ORDER_FEE
    else:
        return (-SL_POINT * POINT_VALUE) - 2 * ORDER_FEE

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
        "CloseTime", "CrossLevel", "CrossValue", "OpenPosPct", 
        "OpenVsPrvClose", "OpenVsMdn8t", "RSI", "ATR", "EMA", "Risk", "NetProfitUSD", "FeesUSD"
    ]
    trades_data = [{
        'EntryTime': trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'Price': trade['entry_price'],
        'SL': trade['stop_loss'],
        'TP': trade['take_profit'],
        'Direction': trade['direction'],
        'Outcome': trade['outcome'],
        'CloseTime': trade['close_time'].strftime('%Y-%m-%d %H:%M:%S'),
        'CrossLevel': trade['crossover'],
        'CrossValue': trade['crossover_value'],
        'OpenPosPct': trade['open_position_pct'],
        'OpenVsPrvClose': trade['open_vs_prv_close'],
        'OpenVsMdn8t': trade['open_vs_mdn8t'],
        'RSI': trade['rsi'],
        'ATR': trade['atr'],
        'EMA': trade['ema'],
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

    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values

    df['rsi'] = calculate_rsi_numba(close_prices)
    df['atr'] = calculate_atr_numba(high_prices, low_prices, close_prices)
    df = calculate_ema(df)

    trading_data = calculate_trading_data(df)
    trading_data = add_previous_day_levels(trading_data)

    # Use only the last DAYS_TO_FETCH days for simulation
    simulation_data = trading_data.iloc[1:]

    total_start_time = time.time()

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

            print(f"\nSummary: {params['active_parameter']} {ratio}:1")
            print(f"Total Trades: {summary['total_trades']}")
            print(f"Wins: {summary['wins']} ({summary['win_rate']:.2f}%)")
            print(f"Losses: {summary['losses']} ({summary['loss_rate']:.2f}%)")
            print(f"Profit Factor: {summary['profit_factor']:.2f}")
            print(f"Net Profit (USD): {summary['net_profit_usd']:.2f}")
            print(f"Total Fees (USD): {summary['total_fees_usd']:.2f}")

            if trades:
                print("\nTrades:")
                header = "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<20} {:<15} {:<10} {:<15} {:<15} {:<15} {:<10} {:<10} {:<10} {:<10}".format(
                    "EntryTime", "Price", "SL", "TP", "Direction", "Outcome", "CloseTime", "CrossLevel", "CrossValue", "OpenPosPct", "OpenVsPrvClose", "OpenVsMdn8t", "RSI", "ATR", "EMA", "Risk"
                )
                print(header)
                print("-" * len(header))
                for trade in trades:
                    print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10} {:<10} {:<20} {:<15} {:<10.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
                        trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'), trade['entry_price'], trade['stop_loss'], trade['take_profit'],
                        trade['direction'], trade['outcome'], trade['close_time'].strftime('%Y-%m-%d %H:%M:%S'), trade['crossover'], trade['crossover_value'],
                        trade['open_position_pct'], trade['open_vs_prv_close'], trade['open_vs_mdn8t'], trade['rsi'], trade['atr'], trade['ema'], trade['risk']
                    ))

                write_trades_to_csv(trades)

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total execution time: {total_duration / 60:.2f} minutes")

    ib.disconnect()

if __name__ == "__main__":
    main()
