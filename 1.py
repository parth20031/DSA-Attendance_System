from datetime import datetime


def today_date():
    today_date = datetime.now().strftime("%Y-%m-%d")  # Get the current date in YYYY-MM-DD format
    return today_date

today_date=today_date()
print(today_date)