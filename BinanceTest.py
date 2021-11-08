# use for environment variables
import os

# needed for the binance API and websockets
from binance.client import Client

# used for dates
from datetime import datetime, timedelta
import time

# used to repeatedly execute the code
from itertools import count

# used to store trades and sell assets
import json
import sys
import math as m
from statistics import mean
import numpy as np


# Switch between testnet and mainnet
# Setting this to False will use REAL funds, use at your own risk
# Define your API keys below in order for the toggle to work
TESTNET = False #True


# Get binance key and secret for TEST and MAINNET
# The keys below are pulled from environment variables using os.getenv
# Simply remove this and use the following format instead: api_key_test = 'YOUR_API_KEY'
api_key_test = "npwGL4pmBWLCUpNiKm95C8g88J4043AoloUv0ZEviwsxp6A9GnVPvKkcCGkG69o2" #os.getenv('binance_api_stalkbot_testnet')
api_secret_test = "5Pwdbcf5n16UnKXzrIH9cARL4YCaTzlhnSdn7YBcC3LZQHdeKM6LKZM0ottsz7s7" #os.getenv('binance_secret_stalkbot_testnet')

api_key_live = "xDT2x3AI14tVz0qehtGMcyFKyXd56JRlLIcJqwoZpF6X5i882C8lljsGVnymdZk8" #os.getenv('binance_api_stalkbot_live')
api_secret_live = "yBAjJ5JMXNZd1XeCKK787eDzHRpyAmUoenCbOGNWkMwQwxUTCviE88Aykxng8qb6" #os.getenv('binance_secret_stalkbot_live')


# Authenticate with the client
if TESTNET:
    client = Client(api_key_test, api_secret_test)

    # The API URL needs to be manually changed in the library to work on the TESTNET
    client.API_URL = 'https://testnet.binance.vision/api'

else:
    client = Client(api_key_live, api_secret_live)


PAIR_WITH = 'USDT'
FIATS = ['EURUSDT', 'GBPUSDT', 'JPYUSDT', 'USDUSDT', 'DOWN', 'UP']
QUANTITY = 15.0
CHANGE_IN_PRICE = 0.1
LOG_TRADES = False
MAX_PRICES_SAMPLES = 6
CUSTOM_LIST = False

tickers_info = {}

def process_ticker(coin):
    price = float(coin['price'])
    if coin['symbol'] not in tickers_info:
        tickers_info[coin['symbol']] = {}
        tickers_info[coin['symbol']]['prices'] = []
    
    tickers_info[coin['symbol']]['prices'].append(price)
    tickers_info[coin['symbol']]['last_price'] = price
    
    if len(tickers_info[coin['symbol']]['prices']) > MAX_PRICES_SAMPLES:
        tickers_info[coin['symbol']]['prices'].pop(0)

    prices = np.array(tickers_info[coin['symbol']]['prices'])                            
    
    avg = np.average(prices)
    diff = difference(price, avg)
                                                        
    tickers_info[coin['symbol']]['avg'] = avg
    tickers_info[coin['symbol']]['diff'] = diff  
    

def refresh_tickers():
    prices = client.get_all_tickers()

    for coin in prices:
        
        if CUSTOM_LIST:
            if any(item + PAIR_WITH == coin['symbol'] for item in tickers) and all(item not in coin['symbol'] for item in FIATS):
                process_ticker(coin)
        else:
            if PAIR_WITH in coin['symbol'] and all(item not in coin['symbol'] for item in FIATS):
                process_ticker(coin)
                       
    #print(tickers_info)
    
    return tickers_info

def filter_gain_coins():
    volatile_coins = dict(filter(lambda item: len(item[1]['prices']) > (MAX_PRICES_SAMPLES/2) and item[1]['diff'] > CHANGE_IN_PRICE, tickers_info.items()))
    marklist = sorted(volatile_coins.items(), key=lambda x: x[1]['diff'], reverse=True)
    volatile_coins = dict(marklist)
    
    for coin in volatile_coins.keys():
        print(f"{coin} gain {volatile_coins[coin]['diff']}%")
    
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
    
    coins_bought = {}

    for coin in volume:

        # only buy if the there are no active trades on the coin
        if coin not in coins_bought:
            print(f"Preparing to buy {volume[coin]} {coin} at {last_price[coin]['last_price']}")

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


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def difference(x, b):
    try:
        d = (1 - b / x) * 100
    except ZeroDivisionError:
        d = 0
    return d

def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return m.trunc(stepper * number) / stepper

if __name__ == '__main__':    
    
    #tickers_info = refresh_tickers()  
    info = client.get_symbol_info(symbol='DARUSDT')
    print(info)
    
    f = [i["stepSize"]
            for i in info["filters"] if i["filterType"] == "LOT_SIZE"][0]
    precision = f.index("1") - 1

    print(f"Coin precision: {precision}")
    sell_amount = 0.29
    
    if precision == -1:
        sell_amount = m.floor(sell_amount)
    else:
        #sell_amount = round(sell_amount, precision)
        sell_amount = truncate(sell_amount, precision)

    print(f"Selling {sell_amount}")
    
    #return False 

    #while True:      
    #    tickers_info = refresh_tickers()  
    #    #orders, last_price, volume = buy()        
    #    time.sleep(20)    
