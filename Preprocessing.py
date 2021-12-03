import calendar
from datetime import time, datetime, timedelta, date
import pandas as pd
import yfinance as yf
import re

summer_start_time, summer_end_time = time(13, 30, 0), time(20, 0, 0)
winter_start_time, winter_end_time = time(14, 30, 0), time(21, 0, 0)

def GetSundayDate(date,month,nth_week):
    year = date.year
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    my_calendar = c.monthdatescalendar(year, month)
    return [day for week in my_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == month][nth_week-1]

def MarketSeason(df0):
    if df0['summer_start_date'] < df0['market_date'] < df0['summer_end_date']:
        return 'summer'
    else:
        return 'winter'

def WithinMarketPeriod(df0):
    if df0['market_season'] == 'summer':
        return summer_start_time <= df0['market_time'] <= summer_end_time
    else:
        return winter_start_time <= df0['market_time'] <= winter_end_time

def MarketSession(df0, market_interval):
    if df0['market_season'] == 'summer':
        return (datetime.combine(date.min, df0['market_time'])
                - datetime.combine(date.min, summer_start_time)).seconds // (60 * 60 * market_interval)
    else:
        return (datetime.combine(date.min, df0['market_time'])
                - datetime.combine(date.min, winter_start_time)).seconds // (60 * 60 * market_interval)

def DataProcessing(df0, datetime_series, market_interval):
    df0['market_date'] = df0[f'{datetime_series}'].dt.date
    df0['market_time'] = df0[f'{datetime_series}'].dt.time
    df0['summer_start_date'] = df0['market_date'].map(lambda x: GetSundayDate(x, 3, 2))
    df0['summer_end_date'] = df0['market_date'].map(lambda x: GetSundayDate(x, 11, 1))
    df0['market_season'] = df0.apply(MarketSeason, axis=1)
    df0['in_market_time'] = df0.apply(WithinMarketPeriod, axis=1)
    df0 = df0[df0['in_market_time']]
    df0['market_session'] = df0.apply(MarketSession, args=(market_interval,), axis=1)
    if 6.5 % market_interval != 0:
        print('enter if case')
        last_interval = 6.5 // market_interval
        df0.loc[df0['market_session'] == last_interval, 'market_session'] = last_interval - 1
    return df0

def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def TrendLable(pct_change):
    if -0.1 <= pct_change <= 0.1:
        return 0
    elif pct_change > 0.1:
        return 1
    else:
        return -1

h = pd.read_csv('HolidayTable.csv')
h.fillna('',inplace=True)
h = h.applymap(lambda x: datetime.strptime(x, '%B %d %Y') if len(str(x)) > 0 else '')


t = pd.read_parquet('AllHIsTweet.parquet')
t = t.sort_values(by='Datetime')
t['t+2hr'] = t['Datetime'] + timedelta(hours=2)  # offset 2hr as we are predicting the trend 2hr later
t = DataProcessing(t, 't+2hr', 2)
t['is_holiday'] = t['market_date'].apply(lambda x: datetime.combine(x,time.min) in h[str(x.year)].dropna().to_list())
t = t[~(t['is_holiday'])]

t.drop_duplicates(inplace=True)


start_date = datetime.strftime(t.iloc[0]['market_date'], '%Y-%m-%d')
end_date = datetime.strftime((t.iloc[-1]['market_date'] + timedelta(days=1)), '%Y-%m-%d')

s = yf.download(tickers='TSLA', start=f'{start_date}', end='2021-11-30', interval='1h', prepost=True)
s['utc_datetime'] = s.index.tz_convert('UTC')
s = DataProcessing(s, 'utc_datetime', 2)
s.loc[s.groupby(['market_date'])['market_time'].rank(method='min', ascending=False) == 1, 'market_session'] += 1

stock_trend = pd.DataFrame(s.groupby(['market_date','market_session'])['Open'].first().reset_index())
stock_trend['next_session_open'] = stock_trend.groupby('market_date')['Open'].shift(-1)
stock_trend['price_diff'] = stock_trend['next_session_open'] - stock_trend['Open']
stock_trend['pct_diff'] = stock_trend['price_diff'] * 100 / stock_trend['Open']
stock_trend.dropna(inplace=True)
stock_trend['label'] = stock_trend['pct_diff'].apply(TrendLable)
