import calendar
from datetime import time,datetime, timedelta
import pytz
import pandas as pd
import snscrape.modules.twitter as sntwitter


def GetSundayDate(date,month,nth_week):
    year = date.year
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    my_calendar = c.monthdatescalendar(year, month)
    return [day for week in my_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == month][nth_week-1]

def stock

summer_start_date = GetSundayDate(3, 2)
summer_end_date = GetSundayDate(11, 1)

timezone = pytz.timezone('UTC')
summer_start_time = timezone.localize(time(13,30,0))
summer_end_time = timezone.localize(time(20,0,0))
winter_start_time = timezone.localize(time(14,30,0))
winter_end_time = timezone.localize(time(21,0,0))

winter_start_time - timedelta(hours=2)

today_date = datetime.today().strftime('%Y-%m-%d')
next_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

df = pd.read_parquet('20211126tweet.parquet')

df['season'] = df['Datetime'].map(lambda x: )
