import datetime
import pytz
from datetime import timedelta

def get_current_epoch_timestamp():
    """Get current epoch timestamp"""
    return int(datetime.datetime.now().timestamp())

def get_today_9am_ist_epoch_timestamp():
    """Get 9 AM IST epoch timestamp for today"""
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.datetime.now(ist).date()
    today_9am = datetime.datetime.combine(today, datetime.time(9, 0))
    today_9am_ist = ist.localize(today_9am)
    return int(today_9am_ist.timestamp())

def get_ndays_before_9am_ist_epoch_timestamp(n_days):
    """Get 9 AM IST epoch timestamp for n days before today"""
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.datetime.now(ist).date()
    target_date = today - timedelta(days=n_days)
    target_9am = datetime.datetime.combine(target_date, datetime.time(9, 0))
    target_9am_ist = ist.localize(target_9am)
    return int(target_9am_ist.timestamp())

def get_epoch_timestamp_from_datetime_ist_string(datetime_string, format_string="%Y-%m-%d %H:%M:%S"):
    """Convert datetime string to epoch timestamp (IST timezone)"""
    ist = pytz.timezone('Asia/Kolkata')
    dt = datetime.datetime.strptime(datetime_string, format_string)
    dt_ist = ist.localize(dt)
    return int(dt_ist.timestamp())

def get_n_years_ago_epoch_timestamp(n_years):
    """Get epoch timestamp for n years ago from today at 9 AM IST"""
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.datetime.now(ist).date()
    target_date = today.replace(year=today.year - n_years)
    target_9am = datetime.datetime.combine(target_date, datetime.time(9, 0))
    target_9am_ist = ist.localize(target_9am)
    return int(target_9am_ist.timestamp())

def epoch_to_datetime_ist(epoch_timestamp):
    """Convert epoch timestamp to IST datetime"""
    ist = pytz.timezone('Asia/Kolkata')
    dt_utc = datetime.datetime.fromtimestamp(epoch_timestamp, tz=pytz.UTC)
    dt_ist = dt_utc.astimezone(ist)
    return dt_ist
