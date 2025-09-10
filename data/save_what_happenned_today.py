
from api.fyers_session_management import get_fyers_session
from pnl_calculator import charges_for_buy, charges_for_sell
import json
from datetime import datetime
import os


fyers = get_fyers_session()
# tradebook = fyers.tradebook()
# tradebook = tradebook["tradeBook"]


# Get the current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Create the directory if it doesn't exist
directory = f"archive/{current_date}"
os.makedirs(directory, exist_ok=True)

tradebook = fyers.tradebook()
# Save tradebook to file
with open(f"archive/{current_date}/tradebook.json", "w") as f:
    json.dump(tradebook, f, indent=4)
print("==== saved tradebook to file =======")


orders = fyers.orderbook()
# Save orders to file
with open(f"archive/{current_date}/orders.json", "w") as f:
    json.dump(orders, f, indent=4)
print("==== sved orderbook to file =======")


positions = fyers.positions()
# Save positions to file
with open(f"archive/{current_date}/positions.json", "w") as f:
    json.dump(positions, f, indent=4)
print("==== saved positions to file =======")


holdings = fyers.holdings()
# Save holdings to file
with open(f"archive/{current_date}/holdings.json", "w") as f:
    json.dump(holdings, f, indent=4)
print("==== saved holdings to file =======")

# calculate total charges for trades.
# if side is -1, it is a sell trade, else it is a buy trade.
tradebook = tradebook["tradeBook"]
total_charges = 0
total_buy_value = 0
total_sell_value = 0
for trade in tradebook:
    tradePrice = trade["tradePrice"]
    tradedQty = trade["tradedQty"]
    if trade["side"] == -1:
        total_charges += charges_for_sell(tradePrice, tradedQty)
        total_sell_value += tradePrice * tradedQty
    else:
        total_charges -= charges_for_buy(tradePrice, tradedQty)
        total_buy_value += tradePrice * tradedQty
pnl = total_sell_value - total_buy_value - total_charges

print(f"Total charges for trades: {total_charges}")
print(f"Todays pnl: {pnl}")

today_summary = {
    "total_charges": round(total_charges, 2),
    "pnl": round(pnl, 2)
}
with open(f"archive/{current_date}/today_summary.json", "w") as f:
    json.dump(today_summary, f, indent=4)