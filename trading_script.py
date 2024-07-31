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
TIMEZONE = 'Europe/Zurich'  # Geneva is in the Zurich time zone (CET/CEST)
DAYS_TO_FETCH = 5  # Last 5 trading days
TICK_SIZE = 0.25
STOP_LOSS_TICKS = 75
TRADE_AMOUNT = 100  # USD per trade

# Paramètres de direction de trading
LONG_ON_RED_CROSSOVER = True  # True pour long sur bougie rouge, False pour short sur bougie rouge
SHORT_ON_GREEN_CROSSOVER = True  # True pour short sur bougie verte, False pour long sur bougie verte

# Utilisation de argparse pour gérer les paramètres de ligne de commande
def parse_args():
    parser = argparse.ArgumentParser(description="Trading Script")
    parser.add_argument("--risk_reward_ratio", type=int, default=2, help="Risk/Reward Ratio (e.g., 1:2 -> 2)")
    parser.add_argument("--enable_prev_high_crossover", action='store_true', help="Enable Previous High Crossover")
    parser.add_argument("--enable_prev_low_crossover", action='store_true', help="Enable Previous Low Crossover")
    parser.add_argument("--enable_prev_close_crossover", action='store_true', help="Enable Previous Close Crossover")
    parser.add_argument("--enable_premarket_high_crossover", action='store_true', help="Enable Premarket High Crossover")
    parser.add_argument("--enable_premarket_low_crossover", action='store_true', help="Enable Premarket Low Crossover")
    return parser.parse_args()

# Connect to IB
def connect_ib():
    ib = IB()
    try:
        ib.connect(LOCAL_IP, PORT, clientId=CLIENT_ID)
    except Exception as e:
        print(f"Connection error: {e}")
        return None
    return ib

# Fetch historical data
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

# Convert timezone
def convert_to_local(df, timezone):
    local_tz = pytz.timezone(timezone)
    df['date'] = pd.to_datetime(df['date']).dt.tz_convert(local_tz).dt.tz_localize(None)
    df.set_index('date', inplace=True)
    return df

# Calculate trading day data
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
            'opening_price': group.iloc[0]['open'],
            'closing_price': group.iloc[-1]['close'],
            'mrkt_open': mrkt_open,
            'mrkt_close': mrkt_close,
            'mrkt_high': group.between_time('15:30', '22:00')['high'].max(),
            'mrkt_low': group.between_time('15:30', '22:00')['low'].min()
        }
        trading_data.append(day_data)
    return pd.DataFrame(trading_data)

# Add previous day's key levels
def add_previous_day_levels(trading_data):
    trading_data['prev_m_high'] = trading_data['mrkt_high'].shift(1)
    trading_data['prev_m_low'] = trading_data['mrkt_low'].shift(1)
    trading_data['prev_mrkt_close'] = trading_data['mrkt_close'].shift(1)
    trading_data['prem_high'] = trading_data['prem_high'].shift(1)
    trading_data['prem_low'] = trading_data['prem_low'].shift(1)
    return trading_data

# Identify crossovers
def identify_crossovers(df, trading_data, args):
    crossovers = []
    for idx, day_data in trading_data.iterrows():
        date = day_data['date']
        prev_m_high = day_data['prev_m_high']
        prev_m_low = day_data['prev_m_low']
        prev_mrkt_close = day_data['prev_mrkt_close']
        prem_high = day_data['prem_high']
        prem_low = day_data['prem_low']
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
                    'color': color
                })
            if args.enable_prev_close_crossover and prev_mrkt_close and low <= prev_mrkt_close <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prev_mrkt_close',
                    'crossover_value': prev_mrkt_close,
                    'size': size,
                    'color': color
                })
            if args.enable_prev_low_crossover and prev_m_low and low <= prev_m_low <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prev_m_low',
                    'crossover_value': prev_m_low,
                    'size': size,
                    'color': color
                })
            if args.enable_premarket_high_crossover and prem_high and low <= prem_high <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prem_high',
                    'crossover_value': prem_high,
                    'size': size,
                    'color': color
                })
            if args.enable_premarket_low_crossover and prem_low and low <= prem_low <= high:
                crossovers.append({
                    'time': time,
                    'crossover': 'prem_low',
                    'crossover_value': prem_low,
                    'size': size,
                    'color': color
                })
    return crossovers

# Simulate trades
def simulate_trades(df, crossovers, risk_reward_ratio, lookback_period=10):
    trades = []
    in_trade = False  # Flag to indicate if we are currently in a trade
    last_close_time = None  # Track the close time of the last trade

    for crossover in crossovers:
        if in_trade:
            continue  # Skip if already in a trade

        entry_time = crossover['time']
        if last_close_time and entry_time <= last_close_time:
            continue  # Skip if the entry time is not after the last close time

        entry_price = df.loc[entry_time]['close']
        level = crossover['crossover_value']

        # Determine the trend using the lookback period
        lookback_candles = df.loc[df.index < entry_time].tail(lookback_period)
        if lookback_candles.empty:
            continue

        trend_up = lookback_candles[lookback_candles['close'] > lookback_candles['open']].shape[0]
        trend_down = lookback_candles[lookback_candles['close'] < lookback_candles['open']].shape[0]

        if trend_up > trend_down:
            direction = 'long'
            stop_loss = entry_price - (STOP_LOSS_TICKS * TICK_SIZE)
            take_profit = entry_price + (STOP_LOSS_TICKS * risk_reward_ratio * TICK_SIZE)
        elif trend_down > trend_up:
            direction = 'short'
            stop_loss = entry_price + (STOP_LOSS_TICKS * TICK_SIZE)
            take_profit = entry_price - (STOP_LOSS_TICKS * risk_reward_ratio * TICK_SIZE)
        else:
            continue  # Skip if the trend is not clear

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
                        'crossover_value': crossover['crossover_value']
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
                        'crossover_value': crossover['crossover_value']
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
                        'crossover_value': crossover['crossover_value']
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
                        'crossover_value': crossover['crossover_value']
                    })
                    in_trade = False
                    last_close_time = row.name
                    break
    return trades

# Calculate trading summary
def calculate_summary(trades, risk_reward_ratio, args):
    total_pnl = 0  # Profit and Loss
    total_invested = 0  # Total amount invested
    total_return = 0
    total_gains = 0
    total_losses = 0
    win_amounts = []
    loss_amounts = []
    trading_days = set()  # Set to store unique trading days

    for trade in trades:
        pnl = TRADE_AMOUNT * risk_reward_ratio if trade['outcome'] == 'win' else -TRADE_AMOUNT
        total_pnl += pnl
        total_return += pnl
        trading_days.add(trade['entry_time'].date())  # Add the date of the trade to the set
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

# Get the last trading days
def get_last_trading_days(days_to_fetch):
    cme_equity = mcal.get_calendar('CME_Equity')
    end_date = datetime.datetime.now()
    start_date = end_date - pd.Timedelta(days=30)  # Assuming there are at least 10 trading days in the last 30 days
    schedule = cme_equity.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index[-days_to_fetch:].tolist()
    return trading_days

# Main function
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

    # Etape 1: Calculer tous les key levels pour le nombre de jours ouvrables égal à DAYS_TO_FETCH
    trading_data = calculate_trading_data(df)

    # Etape 2: Ajouter tous les key levels de la veille pour chaque journée de trading
    trading_data = add_previous_day_levels(trading_data)

    # Etape 3: Calculer les croisements
    crossovers = identify_crossovers(df, trading_data, args)

    # Etape 4: Simuler les trades
    trades = simulate_trades(df, crossovers, args.risk_reward_ratio)

    # Etape 5: Calculer les win/loss avec les ratios PnL, investissement et tous les paramètres pertinents au résumé de la fin
    summary = calculate_summary(trades, args.risk_reward_ratio, args)

    # Affichage du résumé
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

    # Affichage des trades
    if trades:
        print("\nTrades:")
        header = "{:<20} {:<10} {:<10} {:<10} {:<10} {:<10} {:<20} {:<15} {:<10}".format(
            "EntryTime", "Price", "SL", "TP", "Direction", "Outcome", "CloseTime", "CrossLevel", "CrossValue"
        )
        print(header)
        print("-" * len(header))
        for trade in trades:
            print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10} {:<10} {:<20} {:<15} {:<10.2f}".format(
                trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'), trade['entry_price'], trade['stop_loss'], trade['take_profit'],
                trade['direction'], trade['outcome'], trade['close_time'].strftime('%Y-%m-%d %H:%M:%S'), trade['crossover'], trade['crossover_value']
            ))

    ib.disconnect()

if __name__ == "__main__":
    main()
