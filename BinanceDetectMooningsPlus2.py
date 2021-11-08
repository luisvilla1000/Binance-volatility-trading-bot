# use for environment variables
import os

# needed for the binance API and websockets
from binance.client import Client

# used for dates
from datetime import datetime, timedelta
import logging
import threading
import time

# used to repeatedly execute the code
from itertools import count

# used to store trades and sell assets
import json
import sys

# used for argument parsing. Rather than hardcoded values
import argparse

from dotenv import load_dotenv
import math as m
import numpy as np


load_dotenv()

ARG_ENV_HELP = '''
setting this argument to 'prod' will use REAL currencies (USE AT YOUR OWN RISK)
please use 'test' to run against the test network.
'''

# Get binance key and secret for TEST and MAINNET
# The keys below are pulled from environment variables using os.getenv
# Simply remove this and use the following format instead: api_key_test = 'YOUR_API_KEY'
api_key_test = os.getenv('binance_api_stalkbot_testnet')
api_secret_test = os.getenv('binance_secret_stalkbot_testnet')

api_key_live = os.getenv('binance_api_stalkbot_live')
api_secret_live = os.getenv('binance_secret_stalkbot_live')


####################################################
#                   USER INPUTS                    #
# You may edit to adjust the parameters of the bot #
####################################################

# select what to pair the coins to and pull all coins paied with PAIR_WITH
PAIR_WITH = 'USDT'

# Define the size of each trade, by default in USDT
QUANTITY = 15.00

# List of pairs to exlcude
# by default we're excluding the most popular fiat pairs
# and some margin keywords, as we're only working on the SPOT account
FIATS = ['EURUSDT', 'GBPUSDT', 'JPYUSDT', 'USDUSDT', 'DOWN', 'UP']

# the amount of time in SECONDS to calculate the differnce from the current price
BUY_TIME_DIFFERENCE = 60

# Time in seconds to query a sell
SELL_TIME_DIFFERENCE = 30

# the difference in % between the first and second checks for the price, by default set at 10 minutes apart.
CHANGE_IN_PRICE = 0.7

MAX_PRICES_SAMPLES = 12

# define in % when to sell a coin that's not making a profit
STOP_LOSS = 1
MAX_STOP_LOSS = 4

# define in % when to take profit on a profitable coin
TAKE_PROFIT = 6
MIN_TAKE_PROFIT = 1

# BINANCE cannot always sell all that it buys. Select in % how much to sell
# Set to True to toggle SELL_AMOUNT
CAPPED_SELL = True
SELL_AMOUNT = 100

# Use custom tickers.txt list for filtering pairs
CUSTOM_LIST = True

# Use log file for trades
LOG_TRADES = True
LOG_FILE = 'trades.txt'

####################################################
#                END OF USER INPUTS                #
#                  Edit with care                  #
####################################################


def load_arguments():
    '''Loads required args, and returns parse_args()'''
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument(
        '--env',
        '-e',
        choices=['prod', 'test'],
        required=True,
        help=ARG_ENV_HELP)
    return my_parser.parse_args()


def process_ticker(coin):
    price = float(coin['price'])
    if coin['symbol'] not in tickers_info:
        tickers_info[coin['symbol']] = {}
        tickers_info[coin['symbol']]['prices'] = []

    tickers_info[coin['symbol']]['prices'].append(price)
    tickers_info[coin['symbol']]['last_price'] = price

    if len(tickers_info[coin['symbol']]['prices']) == 1:
        tickers_info[coin['symbol']]['prices'].append(price)

    if len(tickers_info[coin['symbol']]['prices']) > MAX_PRICES_SAMPLES:
        tickers_info[coin['symbol']]['prices'].pop(0)

    prices = np.array(tickers_info[coin['symbol']]['prices'])

    #avg = np.average(prices)
    avg = np.average(prices[:-1])
    diff = difference(price, avg)

    tickers_info[coin['symbol']]['avg'] = avg
    tickers_info[coin['symbol']]['diff'] = diff


def get_price():
    '''Return the current price for all coins on binance'''
    prices = client.get_all_tickers()

    for coin in prices:

        if CUSTOM_LIST:
            if any(item + PAIR_WITH == coin['symbol'] for item in tickers) and all(item not in coin['symbol'] for item in FIATS):
                process_ticker(coin)
        else:
            if PAIR_WITH in coin['symbol'] and all(item not in coin['symbol'] for item in FIATS):
                process_ticker(coin)

    #print(tickers_info)
    
    # save the coins in a json file in the same directory
    with open(ticker_prices_file_path, 'w') as file:
        json.dump(tickers_info, file, indent=4)

    return tickers_info

def filter_gain_coins():
    volatile_coins = dict(filter(lambda item: len(item[1]['prices']) > (
        MAX_PRICES_SAMPLES/2) and item[1]['diff'] > CHANGE_IN_PRICE, tickers_info.items()))
    
    if len(volatile_coins) > 0:
        marklist = sorted(volatile_coins.items(),
                        key=lambda x: x[1]['diff'], reverse=True)
        volatile_coins = dict(marklist)

        for coin in volatile_coins.keys():
            print(f"{coin} gain {volatile_coins[coin]['diff']}%")

    else:
        print("Nothing to buy...")

    return volatile_coins


def convert_volume():
    '''Converts the volume given in QUANTITY from USDT to the each coin's volume'''

    # Empezamos a filtrar las monedas mas ganadoras
    volatile_coins = filter_gain_coins()

    coins = volatile_coins.keys()

    lot_size = {}
    volume = {}

    for coin in coins:
        # Find the correct step size for each coin
        # max accuracy for BTC for example is 6 decimal points
        # while XRP is only 1
        try:
            info = client.get_symbol_info(coin)
            step_size = info['filters'][2]['stepSize']
            lot_size[coin] = step_size.index('1') - 1

            if lot_size[coin] < 0:
                lot_size[coin] = 0

        except:
            pass

        # calculate the volume in coin from QUANTITY in USDT (default)
        volume[coin] = float(QUANTITY / volatile_coins[coin]['last_price'])

        # define the volume with the correct step size
        if coin not in lot_size:
            volume[coin] = float('{:.1f}'.format(volume[coin]))

        else:
            # if lot size has 0 decimal points, make the volume an integer
            if lot_size[coin] == 0:
                volume[coin] = int(volume[coin])
            else:
                volume[coin] = float('{:.{}f}'.format(
                    volume[coin], lot_size[coin]))

    return volume, volatile_coins


def buy():
    '''Place Buy market orders for each volatile coin found'''

    volume, last_price = convert_volume()
    orders = {}

    return orders, last_price, volume

    for coin in volume:

        # only buy if the there are no active trades on the coin
        if coin not in coins_bought:
            print(
                f"Preparing to buy {volume[coin]} {coin} at {last_price[coin]['last_price']}")

            if TESTNET:
                # create test order before pushing an actual order
                test_order = client.create_test_order(
                    symbol=coin, side='BUY', type='MARKET', quantity=volume[coin])

            # try to create a real order if the test orders did not raise an exception
            try:
                buy_limit = client.create_order(
                    symbol=coin,
                    side='BUY',
                    type='MARKET',
                    quantity=volume[coin]
                )

            # error handling here in case position cannot be placed
            except Exception as e:
                print(e)

            # run the else block if the position has been placed and return order info
            else:
                orders[coin] = client.get_all_orders(symbol=coin, limit=1)

                # binance sometimes returns an empty list, the code will wait here unti binance returns the order
                while orders[coin] == []:
                    print(
                        'Binance is being slow in returning the order, calling the API again...')

                    orders[coin] = client.get_all_orders(symbol=coin, limit=1)
                    time.sleep(1)

                else:
                    print('Order returned, saving order to file')

                    if LOG_TRADES:
                        write_log(
                            f"Buy : {volume[coin]} {coin} - {last_price[coin]['last_price']}")
        else:
            print(
                f'Signal detected, but there is already an active trade on {coin}')

    return orders, last_price, volume


def sell_coin(coins_bought, coin, coins_sold):
    if TESTNET:
        # create test order before pushing an actual order
        test_order = client.create_test_order(
            symbol=coin, side='SELL', type='MARKET', quantity=coins_bought[coin]['volume'])

    # try to create a real order if the test orders did not raise an exception
    try:

        sell_amount = coins_bought[coin]['volume']*SELL_AMOUNT/100
        #sell_amount = coins_bought[coin]['volume']
        #decimals = len(str(coins_bought[coin]['volume']).split("."))

        # convert to correct volume
        #sell_amount = float('{:.{}f}'.format(sell_amount, decimals))

        info = client.get_symbol_info(symbol=coin)
        f = [i["stepSize"]
             for i in info["filters"] if i["filterType"] == "LOT_SIZE"][0]
        precision = f.index("1") - 1

        if precision == -1:
            sell_amount = m.floor(sell_amount)
        else:
            sell_amount = round(sell_amount, precision)

        print(f"Selling {sell_amount} {coin}...")

        sell_coins_limit = client.create_order(
            symbol=coin,
            side='SELL',
            type='MARKET',
            quantity=sell_amount  # coins_bought[coin]['volume']
        )

    # error handling here in case position cannot be placed
    except Exception as e:
        print(e)

    # run the else block if coin has been sold and create a dict for each coin sold
    else:
        coins_sold[coin] = coins_bought[coin]

    return coins_sold


def sell_coins():
    '''sell coins that have reached the STOP LOSS or TAKE PROFIT thershold'''

    last_price = get_price()
    coins_sold = {}

    if (len(coins_bought) == 0):
        print(f"Nothing to sell...")

    for coin in list(coins_bought):
        must_sell = False
        # define stop loss and take profit
        TP = float(coins_bought[coin]['bought_at']) + \
            (float(coins_bought[coin]['bought_at']) * TAKE_PROFIT) / 100
        MTP = float(coins_bought[coin]['bought_at']) + \
            (float(coins_bought[coin]['bought_at']) * MIN_TAKE_PROFIT) / 100
        SL = float(coins_bought[coin]['bought_at']) - \
            (float(coins_bought[coin]['bought_at']) * STOP_LOSS) / 100
        MSL = float(coins_bought[coin]['bought_at']) - \
            (float(coins_bought[coin]['bought_at']) * MAX_STOP_LOSS) / 100

        LastPrice = float(last_price[coin]['last_price'])
        BuyPrice = float(coins_bought[coin]['bought_at'])
        PriceChange = float((LastPrice - BuyPrice) / BuyPrice * 100)

        if float(last_price[coin]['last_price']) > TP:
            print(
                f"TP {TP}, selling {coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} : {PriceChange:.2f}%")
            must_sell = True
        elif float(last_price[coin]['last_price']) > MTP:
            print(
                f"MTP {MTP}, selling {coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} : {PriceChange:.2f}%")
            must_sell = True
        elif float(last_price[coin]['last_price']) < MSL:
            print(
                f"mSL {MSL} reached, not selling {coin} for now... Last price at: {last_price[coin]['last_price']}")
        elif float(last_price[coin]['last_price']) < SL:
            print(
                f"SL {SL}, selling {coins_bought[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} : {PriceChange:.2f}%")
            must_sell = True
        else:
            print(
                f"mTP {MTP} or SL {SL} not yet reached, not selling {coin} for now... Last price at: {last_price[coin]['last_price']}")

        if must_sell:
            coins_sold = sell_coin(coins_bought, coin, coins_sold)
            # Log trade
            if LOG_TRADES:
                write_log(
                    f"Sell: {coins_sold[coin]['volume']} {coin} - {BuyPrice} - {LastPrice} : {PriceChange:.2f}%")

    return coins_sold


def update_porfolio(orders, last_price, volume):
    '''add every coin bought to our portfolio for tracking/selling later'''
    # print(orders)
    for coin in orders:

        coins_bought[coin] = {
            'symbol': orders[coin][0]['symbol'],
            'orderid': orders[coin][0]['orderId'],
            'timestamp': orders[coin][0]['time'],
            'bought_at': last_price[coin]['last_price'],
            'volume': volume[coin]
        }

        # save the coins in a json file in the same directory
        with open(coins_bought_file_path, 'w') as file:
            json.dump(coins_bought, file, indent=4)

        print(
            f'Order with id {orders[coin][0]["orderId"]} placed and saved to file')


def remove_from_portfolio(coins_sold):
    '''Remove coins sold due to SL or TP from portofio'''
    for coin in coins_sold:
        coins_bought.pop(coin)

        print(f"Reset {coin} history")
        tickers_info[coin] = {}
        tickers_info[coin]['prices'] = []

    with open(coins_bought_file_path, 'w') as file:
        json.dump(coins_bought, file, indent=4)


def sell_bot():
    logging.info("sell_bot: Starting")
    for i in count():
        time.sleep(SELL_TIME_DIFFERENCE)
        coins_sold = sell_coins()
        remove_from_portfolio(coins_sold)
        
def buy_bot():
    logging.info("buy_bot: Starting")
    for i in count():
        time.sleep(BUY_TIME_DIFFERENCE)
        orders, last_price, volume = buy()
        update_porfolio(orders, last_price, volume)


def fetch_bot():
    logging.info("fetch_bot: Starting")
    for i in count():
        tickers_info = get_price()
        time.sleep(15)


def write_log(logline):
    timestamp = datetime.now().strftime("%d/%m %H:%M:%S")
    with open(LOG_FILE, 'a+') as f:
        f.write(timestamp + ' ' + logline + '\n')


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def difference(x, b):
    try:
        d = (1 - (b / x)) * 100
    except ZeroDivisionError:
        d = 0
    return d


if __name__ == '__main__':
    print('Press Ctrl-Q to stop the script')

    args = load_arguments()
    # Set global dependant on the environment requested
    environment = args.env.lower()
    if environment == "prod":
        TESTNET = False
    else:
        TESTNET = True

    # Load custom tickerlist from file tickers.txt into array tickers *BNB must be in list for script to run.
    tickers = [line.strip() for line in open('tickers.txt')]

    # try to load all the coins bought by the bot if the file exists and is not empty
    coins_bought = {}

    tickers_info = {}

    # path to the saved coins_bought file
    coins_bought_file_path = 'coins_bought.json'
    
    ticker_prices_file_path = 'ticker_prices.json'

    # Authenticate with the client
    if TESTNET:
        coins_bought_file_path = 'testnet_' + coins_bought_file_path  # seperate files
        client = Client(api_key_test, api_secret_test)
        # The API URL needs to be manually changed in the library to work on the TESTNET
        client.API_URL = 'https://testnet.binance.vision/api'

    else:
        client = Client(api_key_live, api_secret_live)

    # if saved coins_bought json file exists and it's not empty then load it
    if os.path.isfile(coins_bought_file_path) and os.stat(coins_bought_file_path).st_size != 0:
        with open(coins_bought_file_path) as file:
            coins_bought = json.load(file)

    if not TESTNET:
        print('WARNING: You are using the Mainnet and live funds. As a safety measure, the script will start executing in 30 seconds.')
        time.sleep(3)

    threads = list()

    format = "[%(threadName)s] - %(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    x1 = threading.Thread(target=sell_bot, name='Sell')
    x2 = threading.Thread(target=buy_bot, name='Buy')
    x3 = threading.Thread(target=fetch_bot, name='Fetch')

    threads.append(x1)
    threads.append(x2)
    threads.append(x3)

    x1.start()
    x2.start()
    x3.start()
