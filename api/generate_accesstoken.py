#  read the Fyers_APP variables from the config.ini FileExistsError

from configparser import ConfigParser
from fyers_apiv3 import fyersModel
import webbrowser


# read config.ini
config = ConfigParser()
config.read('config.ini')


# get the values from the config.ini
redirect_uri = config.get('Fyers_APP', 'redirect_uri')
client_id = config.get('Fyers_APP', 'client_id')
secret_key = config.get('Fyers_APP', 'secret_key')
grant_type = config.get('Fyers_APP', 'grant_type')
response_type = config.get('Fyers_APP', 'response_type')
state = config.get('Fyers_APP', 'state')


# Connect to the sessionModel object here with the required input parameters
appSession = fyersModel.SessionModel(client_id=client_id, redirect_uri=redirect_uri,
                                     response_type=response_type, state=state, secret_key=secret_key, grant_type=grant_type)


def step1():

    appSession = fyersModel.SessionModel(client_id=client_id, redirect_uri=redirect_uri,
                                     response_type=response_type, state=state, secret_key=secret_key, grant_type=grant_type)

    # ## Make  a request to generate_authcode object this will return a login url which you need to open in your browser from where you can get the generated auth_code
    generateTokenUrl = appSession.generate_authcode()

    """ STEP 1 === There are two method to get the Login url if  you are not automating the login flow
    1. Just by printing the variable name 
    2. There is a library named as webbrowser which will then open the url for you without the hasel of copy pasting
    both the methods are mentioned below"""
    print((generateTokenUrl))
    webbrowser.open(generateTokenUrl, new=1)


def step2():
    # STEP 2 === After succesfull login the user can copy the generated auth_code over here and make the request to generate the accessToken
    auth_code = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhcHBfaWQiOiJGWDJGV1lGWDMzIiwidXVpZCI6IjBiMzVmYWFhZjZiMTRhYzVhMGQ5Njc4MDgzZmYwOGQyIiwiaXBBZGRyIjoiIiwibm9uY2UiOiIiLCJzY29wZSI6IiIsImRpc3BsYXlfbmFtZSI6IllCMDc5NTQiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI1MGE2YTRhNWVjZmFiZGNlZDhjYTgxYmE3ZDMyYWI1M2JjMWMwYmRjNDk4ZWI3YjEzMjRjMGU3ZSIsImlzRGRwaUVuYWJsZWQiOiJZIiwiaXNNdGZFbmFibGVkIjoiTiIsImF1ZCI6IltcImQ6MVwiLFwiZDoyXCIsXCJ4OjBcIixcIng6MVwiLFwieDoyXCJdIiwiZXhwIjoxNzU3Mjk4ODkxLCJpYXQiOjE3NTcyNjg4OTEsImlzcyI6ImFwaS5sb2dpbi5meWVycy5pbiIsIm5iZiI6MTc1NzI2ODg5MSwic3ViIjoiYXV0aF9jb2RlIn0.OUPzI1XccmPEsbvvv0gyhsm4Z_tKgUrO_bBdXL4lVhg"

    appSession.set_token(auth_code)
    response = appSession.generate_token()
    print(response)

    access_token = None

    # There can be two cases over here you can successfully get the acccessToken over the request or you might get some error over here. so to avoid that have this in try except block
    try:
        access_token = response["access_token"]
    except Exception as e:
        # This will help you in debugging then and there itself like what was the error and also you would be able to see the value you got in response variable. instead of getting key_error for unsuccessfull response.
        print(e, response)

    # This is the access_token which you can use to make further requests to the API
    print(access_token)


# step1()
step2()
