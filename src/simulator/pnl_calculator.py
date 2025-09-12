def calculate_charges(buy_price, sell_price, quantity):
    """
    Calculate FYERS charges for buying and selling stocks.

    Parameters:
        buy_price (float): The price at which the stock is bought.
        sell_price (float): The price at which the stock is sold.
        quantity (int): The number of stocks bought and sold.

    Returns:
        dict: A dictionary containing detailed charges and P/L calculation.
    """
    # Turnover (Buy + Sell value)
    turnover_buy = buy_price * quantity
    turnover_sell = sell_price * quantity
    turnover = turnover_buy + turnover_sell

    # Brokerage charges
    brokerage_buy = min(0.0003 * turnover_buy, 20)  # 0.03% of turnover, capped at Rs. 20
    brokerage_sell = min(0.0003 * turnover_sell, 20)  # 0.03% of turnover, capped at Rs. 20
    brokerage = brokerage_buy + brokerage_sell

    # Exchange Transaction Charges (0.00297% of turnover for NSE)
    exchange_transaction_charges = 0.0000297 * turnover

    # SEBI Charges (Rs. 10 per crore of turnover)
    sebi_charges = 0.000001 * turnover

    # GST (18% of Brokerage + Exchange Transaction Charges)
    gst = 0.18 * (brokerage + exchange_transaction_charges + sebi_charges)

    # STT (Securities Transaction Tax, 0.025% of Sell value) -- good
    stt = 0.00025 * sell_price * quantity

    # Stamp Duty (0.003% of Buy value, rounded to nearest paise)
    stamp_duty = round(0.00003 * buy_price * quantity, 2)

    # NSE IPFT Charges (Rs. 10 per crore of turnover)
    nse_ipft_charges = 0.000001 * turnover

    # Total charges
    total_charges = brokerage + exchange_transaction_charges + gst + stt + sebi_charges + stamp_duty + nse_ipft_charges

    # Net P/L
    net_pl = (sell_price - buy_price) * quantity - total_charges

    # Break-even points (charges per share to recover costs)
    break_even_points = total_charges / quantity

    return {
        "Turnover": round(turnover, 2),
        "Brokerage": round(brokerage, 2),
        "Exchange Transaction Charges": round(exchange_transaction_charges, 2),
        "GST": round(gst, 2),
        "STT": round(stt, 2),
        "SEBI Charges": round(sebi_charges, 2),
        "Stamp Duty": round(stamp_duty, 2),
        "NSE IPFT Charges": round(nse_ipft_charges, 2),
        "Total Charges": round(total_charges, 2),
        "Break Even Points": round(break_even_points, 2),
        "Net P/L": round(net_pl, 2)
    }

def charges_for_buy(buy_price, quantity):
    """
    Calculate the charges for buying stocks.

    Args:
        buy_price (float): The price at which the stock is bought.
        quantity (int): The number of stocks bought.

    Returns:
        dict: A dictionary containing detailed charges for buying stocks.
    """    # Turnover (Buy + Sell value)
    turnover = buy_price * quantity

    # Brokerage charges
    brokerage = min(0.0003 * turnover, 20)  # 0.03% of turnover, capped at Rs. 20

    # Exchange Transaction Charges (0.00297% of turnover for NSE)
    exchange_transaction_charges = 0.0000297 * turnover

    # SEBI Charges (Rs. 10 per crore of turnover)
    sebi_charges = 0.000001 * turnover

    # GST (18% of Brokerage + Exchange Transaction Charges)
    gst = 0.18 * (brokerage + exchange_transaction_charges + sebi_charges)

    # STT (Securities Transaction Tax, 0.025% of Sell value) -- good
    stt = 0 # 0.00025 * sell_price * quantity

    # Stamp Duty (0.003% of Buy value, rounded to nearest paise)
    stamp_duty = round(0.00003 * buy_price * quantity, 2)

    # NSE IPFT Charges (Rs. 10 per crore of turnover)
    nse_ipft_charges = 0.000001 * turnover

    # Total charges
    total_charges = brokerage + exchange_transaction_charges + gst + stt + sebi_charges + stamp_duty + nse_ipft_charges

    return total_charges

def charges_for_sell(sell_price, quantity):
    """
    Calculate the charges for selling stocks.

    Args:
        sell_price (float): The price at which the stock is sold.
        quantity (int): The number of stocks sold.

    Returns:
        dict: A dictionary containing detailed charges for selling stocks.
    """
    # Turnover (Buy + Sell value)
    turnover = sell_price * quantity

    # Brokerage charges
    brokerage = min(0.0003 * turnover, 20)  # 0.03% of turnover, capped at Rs. 20

    # Exchange Transaction Charges (0.00297% of turnover for NSE)
    exchange_transaction_charges = 0.0000297 * turnover

    # SEBI Charges (Rs. 10 per crore of turnover)
    sebi_charges = 0.000001 * turnover

    # GST (18% of Brokerage + Exchange Transaction Charges)
    gst = 0.18 * (brokerage + exchange_transaction_charges + sebi_charges)

    # STT (Securities Transaction Tax, 0.025% of Sell value) -- good
    stt = 0.00025 * sell_price * quantity

    # Stamp Duty (0.003% of Buy value, rounded to nearest paise)
    stamp_duty = 0 # round(0.00003 * sell_price * quantity, 2)

    # NSE IPFT Charges (Rs. 10 per crore of turnover)
    nse_ipft_charges = 0.000001 * turnover

    # Total charges
    total_charges = brokerage + exchange_transaction_charges + gst + stt + sebi_charges + stamp_duty + nse_ipft_charges

    return total_charges
    

def get_npl(buy_price, sell_price, quantity):
    """
    Calculate the net_profit/Loss (P/L) for a trade.
    Works for both LONG (buy then sell) and SHORT (sell then buy) positions.

    Args:
        buy_price (float): The price at which the asset was bought.
        sell_price (float): The price at which the asset was sold.
        quantity (int): The number of units traded.

    Returns:
        float: The net_profit/Loss after accounting for charges.
    """
    # For both LONG and SHORT, we calculate charges on both legs
    buy_charges = charges_for_buy(buy_price, quantity)
    sell_charges = charges_for_sell(sell_price, quantity)
    total_charges = buy_charges + sell_charges    
    
    # Net P/L = Price difference - Total charges
    # This works for both LONG and SHORT:
    # LONG: (sell_price - buy_price) * quantity - charges
    # SHORT: (buy_price - sell_price) * quantity - charges (same as LONG with swapped prices)
    npl = (sell_price - buy_price) * quantity - total_charges
    return npl

# Example usage
# charges = calculate_charges(buy_price=1100, sell_price=1100, quantity=100)
# for key, value in charges.items():
#     print(f"{key}: {value}")

# # Example usage
# npl = get_npl(buy_price=1758.7, sell_price=1749.9065, quantity=100)
# print(f"Net P/L: {npl}")

# npl = get_npl(buy_price=1309.6, sell_price=1308.15, quantity=100)
# print(f"Net P/L: {npl}")

# # # Example usage
# buy_price = 1100
# sell_price = 1100
# quantity = 100
# buy_charges = charges_for_buy(buy_price, quantity)
# sell_charges = charges_for_sell(sell_price, quantity)
# npl = buy_price * quantity - buy_charges - sell_price * quantity - sell_charges
# print(f"npl with new calculation: {npl}")
