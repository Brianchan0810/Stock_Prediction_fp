import calendar
from datetime import time, datetime, timedelta, date
import pandas as pd
import yfinance as yf
import re
import matplotlib.pyplot as plt

summer_start_time, summer_end_time = time(13, 30, 0), time(20, 0, 0)
winter_start_time, winter_end_time = time(14, 30, 0), time(21, 0, 0)

def get_sunday_date(date,month,nth_week):
    year = date.year
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    my_calendar = c.monthdatescalendar(year, month)  
    return [day for week in my_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == month][nth_week-1]

def market_season(df0):
    if df0['summer_start_date'] < df0['market_date'] < df0['summer_end_date']:
        return 'summer'
    else:
        return 'winter'

def within_market_period(df0):
    if df0['market_season'] == 'summer':
        return summer_start_time <= df0['market_time'] <= summer_end_time
    else:
        return winter_start_time <= df0['market_time'] <= winter_end_time

def market_session(df0, market_interval):
    if df0['market_season'] == 'summer':
        return (datetime.combine(date.min, df0['market_time'])
                - datetime.combine(date.min, summer_start_time)).seconds // (60 * 60 * market_interval)
    else:
        return (datetime.combine(date.min, df0['market_time'])
                - datetime.combine(date.min, winter_start_time)).seconds // (60 * 60 * market_interval)

def data_processing(df, datetime_series, market_interval):
    df0 = df.copy()
    df0['market_date'] = df0[f'{datetime_series}'].dt.date
    df0['market_time'] = df0[f'{datetime_series}'].dt.time
    df0['summer_start_date'] = df0['market_date'].map(lambda x: get_sunday_date(x, 3, 2))
    df0['summer_end_date'] = df0['market_date'].map(lambda x: get_sunday_date(x, 11, 1))
    df0['market_season'] = df0.apply(market_season, axis=1)
    df0['in_market_time'] = df0.apply(within_market_period, axis=1)
    df0 = df0[df0['in_market_time']]
    df0['market_session'] = df0.apply(market_session, args=(market_interval,), axis=1)
    if 6.5 % market_interval != 0:
        print('enter if case')
        last_interval = 6.5 // market_interval
        df0.loc[df0['market_session'] == last_interval, 'market_session'] = last_interval - 1   
    return df0

def clean_tweet(df,tweet_column):
    df0 = df.copy()
    df0.drop_duplicates(subset=f'{tweet_column}',inplace=True)
    df0['symbol_count'] = df0[f'{tweet_column}'].apply(lambda x: len(re.findall(r'\$[a-zA-Z0-9]+', x)))
    df0 = df0[~(df0['symbol_count'] >= 7)]
    df0['cleaned_tweet'] = df0[f'{tweet_column}'].apply(lambda x: re.sub(r'(\$tsla)|(\$TSLA)', 'tsla', x))
    df0['cleaned_tweet'] = df0['cleaned_tweet'].apply(
        lambda x: re.sub(r'(\$[A-Za-z0-9]+)|(@[A-Za-z0-9_]+)|(https?:\/\/\S+)|(\w+:\/\/\S+)', '', x))
    df0['cleaned_tweet'] = df0['cleaned_tweet'].apply(lambda x: re.sub(r'\n\n', ' ', x))
    df0['cleaned_tweet'] = df0['cleaned_tweet'].apply(lambda x: re.sub(r'[^0-9a-zA-Z \t]', '', x))
    df0['cleaned_tweet'] = df0['cleaned_tweet'].apply(lambda x: x.strip())
    return df0

def trend_label(pct_change):
    if -0.1 <= pct_change <= 0.1:
        return 0
    elif pct_change > 0.1:
        return 1
    else:
        return -1

h = pd.read_csv('HolidayTable.csv')
h.fillna('',inplace=True)
h = h.applymap(lambda x: datetime.strptime(x, '%B %d %Y') if len(str(x)) > 0 else '')

#processing tweet data
t = pd.read_parquet('AllHIsTweet.parquet')
t = t.sort_values(by='Datetime')
t['t+2hr'] = t['Datetime'] + timedelta(hours=2)  # offset 2hr as we are predicting the trend 2hr later
t = data_processing(t, 't+2hr', 2)
t['is_holiday'] = t['market_date'].apply(lambda x: datetime.combine(x, time.min) in h[str(x.year)].dropna().to_list())
t = t[~(t['is_holiday'])]

cleaned_tweet = clean_tweet(t, 'Text')
cleaned_tweet.to_csv('cleaned_tweet.csv')

#processing stock price data
start_date = datetime.strftime(t.iloc[0]['market_date'], '%Y-%m-%d')
end_date = datetime.strftime((t.iloc[-1]['market_date'] + timedelta(days=1)), '%Y-%m-%d')

s = yf.download(tickers='TSLA', start=f'{start_date}', end='2021-11-30', interval='1h', prepost=True)
s['utc_datetime'] = s.index.tz_convert('UTC')
s = data_processing(s, 'utc_datetime', 2)
s.loc[s.groupby(['market_date'])['market_time'].rank(method='min', ascending=False) == 1, 'market_session'] += 1

stock_trend = pd.DataFrame(s.groupby(['market_date', 'market_session'])['Open'].first().reset_index())
stock_trend['next_session_open'] = stock_trend.groupby('market_date')['Open'].shift(-1)
stock_trend['price_diff'] = stock_trend['next_session_open'] - stock_trend['Open']
stock_trend['pct_diff'] = stock_trend['price_diff'] * 100 / stock_trend['Open']
stock_trend.dropna(inplace=True)
stock_trend['label'] = stock_trend['pct_diff'].apply(trend_label)
stock_trend.to_csv('stock_data.csv')
