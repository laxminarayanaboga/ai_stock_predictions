
from api.fyers_session_management import get_fyers_session



tradeEnabled = True


def place_order(data):
    fyers = get_fyers_session()
    if not tradeEnabled:
        print("Trade is disabled. Skipping order placement.")
        return
    response = fyers.place_order(data=data)
    print(f'Placing order for {data["symbol"]} with qty {data["qty"]}')
    print(response)
    if response['s'] != 'ok':
        print("Error:", response['message'])


def place_intraday_buy_market_order(symbol, qty):
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": 2,
        "side": 1,
        "productType": "INTRADAY",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "orderTag": "tag1"
    }
    place_order(data)


def place_intraday_sell_market_order(symbol, qty):
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": 2,
        "side": -1,
        "productType": "INTRADAY",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "orderTag": "tag1"
    }
    place_order(data)


def place_intraday_buy_limit_order(symbol, qty, limitPrice):
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": 1,
        "side": 1,
        "productType": "INTRADAY",
        "limitPrice": limitPrice,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "orderTag": "tag1"
    }
    place_order(data)


def place_intraday_sell_limit_order(symbol, qty, limitPrice):
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": 1,
        "side": -1,
        "productType": "INTRADAY",
        "limitPrice": limitPrice,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "orderTag": "tag1"
    }
    place_order(data)


def place_intraday_buy_market_cover_order(symbol, qty, stopLoss):
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": 2,
        "side": 1,
        "productType": "CO",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "stopLoss": stopLoss,
        "takeProfit": 0
    }
    place_order(data)


def place_intraday_sell_market_cover_order(symbol, qty, stopLoss):
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": 2,
        "side": -1,
        "productType": "CO",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "stopLoss": stopLoss,
        "takeProfit": 0
    }
    place_order(data)


def place_intraday_buy_bracket_order(symbol, qty, stopLoss, takeProfit):
    stopLoss = round_down_to_nearest_0_05(stopLoss)
    takeProfit = round_down_to_nearest_0_05(takeProfit)
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": 2,
        "side": 1,
        "productType": "BO",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "stopLoss": stopLoss,
        "takeProfit": takeProfit
    }
    # print(f'qty: {qty}, stopLoss: {stopLoss}, takeProfit: {takeProfit}')
    place_order(data)


def place_intraday_sell_bracket_order(symbol, qty, stopLoss, takeProfit):
    stopLoss = round_down_to_nearest_0_05(stopLoss)
    takeProfit = round_down_to_nearest_0_05(takeProfit)
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": 2,
        "side": -1,
        "productType": "BO",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": False,
        "stopLoss": stopLoss,
        "takeProfit": takeProfit
    }
    # print(f'qty: {qty}, stopLoss: {stopLoss}, takeProfit: {takeProfit}')
    place_order(data)

# stopLoss and takeProfit.. should be multiples of 0.05 only. 
def round_down_to_nearest_0_05(num):
    return round(num * 20) / 20

# place_intraday_buy_market_order("NSE:TATASTEEL-EQ", 1)
# place_intraday_sell_market_order("NSE:TATASTEEL-EQ", 1)

# place_intraday_buy_limit_order("NSE:TATASTEEL-EQ", 1, 141)
# place_intraday_sell_limit_order("NSE:TATASTEEL-EQ", 1, 139)

# place_intraday_buy_market_cover_order("NSE:COALINDIA-EQ", 1, 1)
# place_intraday_sell_market_cover_order("NSE:BEL-EQ", 1, 2)

# place_intraday_buy_bracket_order("NSE:APOLLOHOSP-EQ", 1, 0.5, 1)
# place_intraday_sell_bracket_order("NSE:WIPRO-EQ", 1, 0.5, 1)
