
from utilities.date_utilities import get_ndays_before_9am_ist_epoch_timestamp, get_today_9am_ist_epoch_timestamp, get_current_epoch_timestamp
from api.fyers_session_management import get_fyers_session

fyers = get_fyers_session()

#################################################################################################################

"""
DATA APIS : This includes following Apis(History,Quotes,MarketDepth)
"""


print("======== history - today's data =======")
data = {"symbol": "NSE:SBIN-EQ",
        "resolution": "60",
        "date_format": "0",
        "range_from": get_today_9am_ist_epoch_timestamp(),
        "range_to": get_current_epoch_timestamp(),
        "cont_flag": "1"}
print(fyers.history(data))


print("======== history - 5 days data =======")
data = {"symbol": "NSE:SBIN-EQ",
        "resolution": "60",
        "date_format": "0",
        "range_from": get_ndays_before_9am_ist_epoch_timestamp(5),
        "range_to": get_current_epoch_timestamp(),
        "cont_flag": "1"}
print(fyers.history(data))


print("======== quotes =======")
data = {"symbols": "NSE:SBIN-EQ"}
print(fyers.quotes(data))


print("======== Market Depth =======")
data = {"symbol": "NSE:SBIN-EQ", "ohlcv_flag": "1"}
print(fyers.depth(data))
