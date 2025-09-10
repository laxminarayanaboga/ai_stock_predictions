
from api.fyers_session_management import get_fyers_session


fyers = get_fyers_session()


########################################################################################################################

"""
2. Transaction Apis 
"""

print("======== Tradebook =======")
print(fyers.tradebook())

print("======== Orderbook =======")
tradebook = fyers.tradebook()
print(tradebook)
# print tradebook to file
with open("tradebook.json", "w") as f:
    f.write(str(tradebook))

print("======== Positions =======")
print(fyers.positions())

print("======== Holdings =======")
print(fyers.holdings())
