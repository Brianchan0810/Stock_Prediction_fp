import calendar
from datetime import time,datetime, timedelta, date
import pytz
import pandas as pd
import os
import yfinance as yf

summer_start_time, summer_end_time = time(13, 30, 0), time(20, 0, 0)
winter_start_time, winter_end_time = time(14, 30, 0), time(21, 0, 0)

def GetSundayDate(date,month,nth_week):
    year = date.year
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    my_calendar = c.monthdatescalendar(year, month)
    return [day for week in my_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == month][nth_week-1]

def MarketSeason(df0):
    if df0['market_date'] > df0['summer_start_date'] and df0['market_date'] < df0['summer_end_date']:
        return 'summer'
    else:
        return 'winter'

def WithinMarketPeriod(df0):
    if df0['market_season'] == 'summer':
        return df0['market_time'] >= summer_start_time and df0['market_time'] <= summer_end_time
    else:
        return df0['market_time'] >= winter_start_time and df0['market_time'] <= winter_end_time

def MarketSession(df0):
    if df0['market_season'] == 'summer':
        return (datetime.combine(date.min, df0['market_time']) - datetime.combine(date.min, summer_start_time)).seconds // (60*60*2)
    else:
        return (datetime.combine(date.min, df0['market_time']) - datetime.combine(date.min, winter_start_time)).seconds // (60*60*2)

h = pd.read_csv('HolidayTable.csv')
h.fillna('',inplace=True)
h = h.applymap(lambda x: datetime.strptime(x,'%B %d %Y') if len(str(x))>0 else '')
h


t = pd.read_parquet('AllHIsTweet.parquet')
t['t+2hr'] = t['Datetime'] + timedelta(hours=2) #offset 2hr as we are predicting the trend 2hr later
t['market_date'] = t['t+2hr'].dt.date
t['market_time'] = t['t+2hr'].dt.time
t['summer_start_date'] = t['market_date'].map(lambda x: GetSundayDate(x,3,2))
t['summer_end_date'] = t['market_date'].map(lambda x: GetSundayDate(x,11,1))
t['market_season'] = t.apply(MarketSeason,axis=1)
t['in_market_time'] = t.apply(WithinMarketPeriod, axis=1)
t = t[t['in_market_time']]
t['market_session'] = t.apply(MarketSession, axis=1)
t.loc[t['market_session'] == 3, 'market_session'] = 2

t

msft = yf.Ticker("MSFT")
msft.news

data = yf.download(tickers='TSLA',start='2021-01-01',end='2021-11-30',interval='1h',prepost=True)

data.tail(60)