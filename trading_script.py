import argparse
import datetime
from ib_insync import *
import pandas as pd
import pytz
import pandas_market_calendars as mcal

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
STOP_LOSS_TICKS = 75
TRADE_AMOUNT = 100

def parse_args():
    parser = argparse.ArgumentParser(description="Trading Script")
    parser.add_argument("--risk_reward_ratio", type=int, default=2, help="Risk/Reward Ratio (e.g., 1:2 -> 2)")
    parser.add_argument("--enable_prev_high_crossover", action='store_true', help="Enable Previous High Crossover")
    parser.add_argument("--enable_prev_low_crossover", action='store_true', help="Enable Previous Low Crossover")
    parser.add_argument("--enable_prev_close_crossover", action='store_true', help="Enable Previous Close Crossover")
    parser.add_argument("--enable_premarket_high_crossover", action='store_true', help="Enable Premarket High Crossover")
    parser.add_argument("--enable_premarket_low_crossover", action='store_true', help="Enable Premarket Low Crossover")
    return parser.parse_args()

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

def calculate_rsi(df, period=14):
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['rsi'] = rsi
    return df

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    df['atr'] = atr
    return df

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
            'prem_high': group.between_time('10:00', '15:30')['high'].max(),
            'prem_low': group.between_time('10:00', '15:30')['low'].min(),
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

def identify_crossovers(df, trading_data, args):
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

            if args.enable_prev_high_crossover and prev_m_high and low <= prev_m_high <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prev_m_high',
                    'crossover_value': prev_m_high,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t
                })
            if args.enable_prev_close_crossover and prev_mrkt_close and low <= prev_mrkt_close <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prev_mrkt_close',
                    'crossover_value': prev_mrkt_close,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t
                })
            if args.enable_prev_low_crossover and prev_m_low and low <= prev_m_low <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prev_m_low',
                    'crossover_value': prev_m_low,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t
                })
            if args.enable_premarket_high_crossover and prem_high and low <= prem_high <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prem_high',
                    'crossover_value': prem_high,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t
                })
            if args.enable_premarket_low_crossover and prem_low and low <= prem_low <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prem_low',
                    'crossover_value': prem_low,
                    'size': size,
                    'color': color,
                    'open_position_pct': open_position_pct,
                    'open_vs_prv_close': open_vs_prv_close,
                    'open_vs_mdn8t': open_vs_mdn8t
                })
    return crossovers

def simulate_trades(df, crossovers, risk_reward_ratio):
    trades = []
    in_trade = False
    last_close_time = None

    for crossover in crossovers:
        if in_trade:
            continue

        entry_time = crossover['time']
        if last_close_time and entry_time <= last_close_time:
            continue

        # Obtenir le prix de clôture de la bougie précédente
        prev_candle_close = df.loc[entry_time - pd.Timedelta(minutes=1)]['close']

        entry_price = df.loc[entry_time]['close']
        level = crossover['crossover_value']
        open_position_pct = crossover['open_position_pct']
        open_vs_prv_close = crossover['open_vs_prv_close']
        open_vs_mdn8t = crossover['open_vs_mdn8t']
        rsi_value = df.loc[entry_time]['rsi']
        atr_value = df.loc[entry_time]['atr']
        ema_value = df.loc[entry_time]['ema']

        # Décider de la direction (long/short) en fonction de la position de la bougie précédente par rapport au niveau de croisement
        if prev_candle_close < level:
            direction = 'short'
            stop_loss = entry_price + (STOP_LOSS_TICKS * TICK_SIZE)
            take_profit = entry_price - (STOP_LOSS_TICKS * risk_reward_ratio * TICK_SIZE)
        else:
            direction = 'long'
            stop_loss = entry_price - (STOP_LOSS_TICKS * TICK_SIZE)
            take_profit = entry_price + (STOP_LOSS_TICKS * risk_reward_ratio * TICK_SIZE)

        in_trade = True
        for index, row in df.loc[df.index >= entry_time].iterrows():
            if direction == 'long':
                if row['low'] <= stop_loss:
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'direction': direction,
                        'outcome': 'loss',
                        'close_time': row.name,
                        'crossover': crossover['crossover'],
                        'crossover_value': crossover['crossover_value'],
                        'open_position_pct': open_position_pct,
                        'open_vs_prv_close': open_vs_prv_close,
                        'open_vs_mdn8t': open_vs_mdn8t,
                        'rsi': rsi_value,
                        'atr': atr_value,
                        'ema': ema_value
                    })
                    in_trade = False
                    last_close_time = row.name
                    break
                elif row['high'] >= take_profit:
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'direction': direction,
                        'outcome': 'win',
                        'close_time': row.name,
                        'crossover': crossover['crossover'],
                        'crossover_value': crossover['crossover_value'],
                        'open_position_pct': open_position_pct,
                        'open_vs_prv_close': open_vs_prv_close,
                        'open_vs_mdn8t': open_vs_mdn8t,
                        'rsi': rsi_value,
                        'atr': atr_value,
                        'ema': ema_value
                    })
                    in_trade = False
                    last_close_time = row.name
                    break
            else:
                if row['high'] >= stop_loss:
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'direction': direction,
                        'outcome': 'loss',
                        'close_time': row.name,
                        'crossover': crossover['crossover'],
                        'crossover_value': crossover['crossover_value'],
                        'open_position_pct': open_position_pct,
                        'open_vs_prv_close': open_vs_prv_close,
                        'open_vs_mdn8t': open_vs_mdn8t,
                        'rsi': rsi_value,
                        'atr': atr_value,
                        'ema': ema_value
                    })
                    in_trade = False
                    last_close_time = row.name
                    break
                elif row['low'] <= take_profit:
                    trades.append({
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'direction': direction,
                        'outcome': 'win',
                        'close_time': row.name,
                        'crossover': crossover['crossover'],
                        'crossover_value': crossover['crossover_value'],
                        'open_position_pct': open_position_pct,
                        'open_vs_prv_close': open_vs_prv_close,
                        'open_vs_mdn8t': open_vs_mdn8t,
                        'rsi': rsi_value,
                        'atr': atr_value,
                        'ema': ema_value
                    })
                    in_trade = False
                    last_close_time = row.name
                    break
    return trades

def calculate_summary(trades, risk_reward_ratio, args):
    total_pnl = 0
    total_invested = 0
    total_return = 0
    total_gains = 0
    total_losses = 0
    win_amounts = []
    loss_amounts = []
    trading_days = set()

    for trade in trades:
        pnl = TRADE_AMOUNT * risk_reward_ratio if trade['outcome'] == 'win' else -TRADE_AMOUNT
        total_pnl += pnl
        total_return += pnl
        trading_days.add(trade['entry_time'].date())
        if trade['outcome'] == 'win':
            total_gains += TRADE_AMOUNT * risk_reward_ratio
            win_amounts.append(TRADE_AMOUNT * risk_reward_ratio)
        else:
            total_losses += TRADE_AMOUNT
            loss_amounts.append(TRADE_AMOUNT)

    total_trades = len(trades)
    wins = len([t for t in trades if t['outcome'] == 'win'])
    losses = len([t for t in trades if t['outcome'] == 'loss'])

    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    loss_rate = (losses / total_trades) * 100 if total_trades > 0 else 0
    avg_win_per_trade = (sum(win_amounts) / len(win_amounts)) if win_amounts else 0
    avg_loss_per_trade = (sum(loss_amounts) / len(loss_amounts)) if loss_amounts else 0
    profit_factor = (total_gains / total_losses) if total_losses > 0 else float('inf')

    total_invested = total_trades * TRADE_AMOUNT
    number_of_trading_days = len(trading_days)

    summary = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'total_return': total_return,
        'avg_win_per_trade': avg_win_per_trade,
        'avg_loss_per_trade': avg_loss_per_trade,
        'profit_factor': profit_factor,
        'total_invested': total_invested,
        'number_of_trading_days': number_of_trading_days,
        'active_parameter': None,
        'risk_reward_ratio': risk_reward_ratio
    }

    if args.enable_prev_high_crossover:
        summary['active_parameter'] = 'enable_prev_high_crossover'
    elif args.enable_prev_low_crossover:
        summary['active_parameter'] = 'enable_prev_low_crossover'
    elif args.enable_prev_close_crossover:
        summary['active_parameter'] = 'enable_prev_close_crossover'
    elif args.enable_premarket_high_crossover:
        summary['active_parameter'] = 'enable_premarket_high_crossover'
    elif args.enable_premarket_low_crossover:
        summary['active_parameter'] = 'enable_premarket_low_crossover'

    return summary

def get_last_trading_days(days_to_fetch):
    cme_equity = mcal.get_calendar('CME_Equity')
    end_date = datetime.datetime.now()
    start_date = end_date - pd.Timedelta(days=200)
    schedule = cme_equity.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index[-(days_to_fetch + 1):].tolist()  # Fetch DAYS_TO_FETCH + 1 days
    return trading_days

def main():
    args = parse_args()

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
    df = calculate_rsi(df)
    df = calculate_atr(df)
    df = calculate_ema(df)

    trading_data = calculate_trading_data(df)
    trading_data = add_previous_day_levels(trading_data)

    # Use only the last DAYS_TO_FETCH days for simulation
    simulation_data = trading_data.iloc[1:]

    crossovers = identify_crossovers(df, simulation_data, args)
    trades = simulate_trades(df, crossovers, args.risk_reward_ratio)
    summary = calculate_summary(trades, args.risk_reward_ratio, args)

    active_param = summary['active_parameter']
    risk_reward = summary['risk_reward_ratio']
    print(f"\nSummary: {active_param} {risk_reward}:1")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Wins: {summary['wins']} ({summary['win_rate']:.2f}%)")
    print(f"Losses: {summary['losses']} ({summary['loss_rate']:.2f}%)")
    print(f"Total Return: ${summary['total_return']:.2f}")
    print(f"Avg Win per Trade: ${summary['avg_win_per_trade']:.2f}")
    print(f"Avg Loss per Trade: ${summary['avg_loss_per_trade']:.2f}")
    print(f"Profit Factor: {summary['profit_factor']:.2f}")
    print(f"Total Invested: ${summary['total_invested']:.2f}")
    print(f"Number of Trading Days: {summary['number_of_trading_days']}")

    if trades:
        print("\nTrades:")
        header = "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<20} {:<15} {:<10} {:<15} {:<15} {:<15} {:<10} {:<10} {:<10}".format(
            "EntryTime", "Price", "SL", "TP", "Direction", "Outcome", "CloseTime", "CrossLevel", "CrossValue", "OpenPosPct", "OpenVsPrvClose", "OpenVsMdn8t", "RSI", "ATR", "EMA"
        )
        print(header)
        print("-" * len(header))
        for trade in trades:
            print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10} {:<10} {:<20} {:<15} {:<10.2f} {:<15.2f} {:<15.2f} {:<15.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(
                trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'), trade['entry_price'], trade['stop_loss'], trade['take_profit'],
                trade['direction'], trade['outcome'], trade['close_time'].strftime('%Y-%m-%d %H:%M:%S'), trade['crossover'], trade['crossover_value'],
                trade['open_position_pct'], trade['open_vs_prv_close'], trade['open_vs_mdn8t'], trade['rsi'], trade['atr'], trade['ema']
            ))

    ib.disconnect()

if __name__ == "__main__":
    main()
