
from api.fyers_session_management import check_access_token_expiry_and_get_fyers_session, get_fyers_session


fyers = check_access_token_expiry_and_get_fyers_session()


####################################################################################################################
"""
1. User Apis 
"""

print("======== Profile =======")
print(fyers.get_profile())

print("======== Funds =======")
print(fyers.funds())
