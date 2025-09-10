from configparser import ConfigParser
from fyers_apiv3 import fyersModel
import requests
import json

# Read config.ini
config = ConfigParser()
config.read('config.ini')

# Get values from config.ini
client_id = config.get('Fyers_APP', 'client_id')
try:
    access_token = config.get('Fyers_APP', 'access_token')
except Exception as e:
    access_token = None
    # Handle the missing access_token case, e.g., log an error or raise a custom exception
    print("Error: access_token is missing in the configuration.")
refresh_token = config.get('Fyers_APP', 'refresh_token')
app_id_hash = config.get('Fyers_APP', 'app_id_hash')  # Fetch appIdHash from config
pin = config.get('Fyers_APP', 'pin')  # Fetch PIN from config

# Function to get a new access token using refresh token
def refresh_access_token():
    url = "https://api-t1.fyers.in/api/v3/validate-refresh-token"
    payload = {
        "grant_type": "refresh_token",
        "appIdHash": app_id_hash,
        "refresh_token": refresh_token,
        "pin": pin
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("s") == "ok":
            new_access_token = data.get("access_token")
            # Update config.ini with new access token
            config.set('Fyers_APP', 'access_token', new_access_token)
            with open('config.ini', 'w') as configfile:
                config.write(configfile)
            return new_access_token
        else:
            print("Error refreshing token:", data.get("message"))
            return None
    else:
        print("HTTP Error:", response.status_code, response.text)
        return None

# Function to check if access token is valid and get fyers session
def check_access_token_expiry_and_get_fyers_session():
    global access_token
    fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="")
    
    response = fyers.get_profile()
    if isinstance(response, dict) and response.get("s") == "ok":
        return fyers
    else:
        print("Access token expired. Refreshing...")
        new_access_token = refresh_access_token()
        if not new_access_token:
            raise Exception("Failed to refresh access token")
        access_token = new_access_token
        return fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="")

# Ensure we have a valid access token
# if not access_token or access_token == "expired":  # Add a condition based on your token expiration logic
#     access_token = refresh_access_token()
#     if not access_token:
#         raise Exception("Failed to refresh access token")

# Function to get fyers session
def get_fyers_session():
    return fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="")

# check_access_token_expiry_and_get_fyers_session()