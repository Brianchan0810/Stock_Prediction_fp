import calendar
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import re
from nltk.tokenize import word_tokenize
import os
from sqlalchemy import create_engine

# utc timezone adopted throughout the study

# collect 2hr tweets before market start and collect 3hr tweets after market end
be_start_hr, af_start_hr = 12, 21

def get_sunday_date(date, month, nth_week):
    year = date.year
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    my_calendar = c.monthdatescalendar(year, month)
    return [day for week in my_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == month][
        nth_week - 1]

def datetime_processing(df0, holiday_list, datetime_column, fil_by_date=False):
    df = df0.copy()
    df['year'] = df[f'{datetime_column}'].dt.year
    df['date'] = df[f'{datetime_column}'].dt.date
    df['hour'] = df[f'{datetime_column}'].dt.hour
    df['weekday'] = df[f'{datetime_column}'].dt.weekday
    df['date'] = pd.to_datetime(df['date'])
    if fil_by_date:
        try:
            df = df[(df['date'] <= end_date) & (df['date'] >= start_date)]
        except:
            pass
    df['is_holiday'] = df['date'].isin(holiday_list)
    df = df[~((df['is_holiday']) | (df['weekday'] > 4))]
    return df

def split_session(hour):
    if be_start_hr <= hour <= (be_start_hr + 2):
        return 'before market'
    elif af_start_hr <= hour <= (af_start_hr + 3):
        return 'after market'
    else:
        return 'out of scope'


# Load market holiday date list
sql_engine = create_engine(f'mysql+pymysql://mypc:mypc@3.138.187.239', pool_recycle=3600)
db_connection = sql_engine.connect()
h = pd.read_sql("select * from myfp.holiday", db_connection)
holiday_list = h['date'].to_list()

# Load summer start and end date
season = pd.read_sql("select * from myfp.season", db_connection)
season['year'] = season['summer_start'].apply(lambda x: x.year)


stock_symbol = 'AAPL'

# processing tweet data
os.chdir(f'.\\Data\\{stock_symbol}_analysis')
t = pd.read_parquet(f'combined_{stock_symbol}_t.parquet')
t = datetime_processing(t, holiday_list, datetime_column='Datetime')
t = pd.merge(t, season, how='left', on='year')
t['is_summer'] = (t['date'] > t['summer_start']) & (t['date'] < t['summer_end'])
t.loc[t['is_summer'], 'hour'] += 1  # offset one hour for summer time
t['session'] = t['hour'].apply(split_session)
t = t[t['session'] != 'out of scope']

cleaned_t = t.copy()
cleaned_t['stock_symbol_count'] = cleaned_t['Text'].apply(lambda x: len(re.findall(r'\$\w+', x)))
cleaned_t = cleaned_t[~(cleaned_t['stock_symbol_count'] >= 5)]
cleaned_t['cleaned'] = cleaned_t['Text'].apply(lambda x: re.sub(r'([$#@]\w+)|(\w+:\/\/\S+)', '', x))
cleaned_t['cleaned'] = cleaned_t['cleaned'].apply(lambda x: re.sub(r'\n\n', ' ', x))
cleaned_t['cleaned'] = cleaned_t['cleaned'].apply(lambda x: re.sub(r'[^0-9a-zA-Z .,!?:%\t]', '', x))
cleaned_t['tokenized'] = cleaned_t['cleaned'].apply(lambda x: word_tokenize((re.sub(r'[^a-zA-Z \t]', '', x)).lower()))
cleaned_t.drop_duplicates(subset='cleaned', inplace=True)
cleaned_t.dropna(inplace=True, subset=['cleaned'])

start_date = datetime.strftime(cleaned_t['date'].min() - timedelta(days=1), '%Y-%m-%d')
end_date = datetime.strftime(cleaned_t['date'].max(), '%Y-%m-%d')

# processing news
n = pd.read_parquet(f'combined_{stock_symbol}_n.parquet')
n['converted_dt'] = n['datetime'].apply(lambda x: datetime.utcfromtimestamp(x))
n['converted_dt+3h'] = n['converted_dt'] + timedelta(hours=3)
n = datetime_processing(n, holiday_list, datetime_column='converted_dt+3h', fil_by_date=True)
n = n[n['hour'] < 12]
n['content'] = n['headline'] + ' - ' + n['summary']
n['tokenized'] = n['headline'].apply(lambda x: word_tokenize((re.sub(r'[^a-zA-Z \t]', '', x)).lower()))
n.drop_duplicates(subset='headline', inplace=True)

# processing stock price
p = yf.download(tickers=stock_symbol, start=start_date, end='2021-12-29', interval='1d')
p['date'] = p.index
p['stock_symbol'] = stock_symbol
p['ud'] = p['Close'] > p['Open']
p['pm_ud'] = p['Open'] > p['Close'].shift(1)
p['last_ud'] = p['ud'].shift(1)
p['last_ud_rolling3d'] = p['ud'].rolling(3).sum().apply(lambda x: 1 if x == 3 else (-1 if x == 0 else 0)).shift(1)
p['pct_diff'] = (p['Close'] - p['Open']) * 100 / p['Open']
p['next_pct_diff'] = p['pct_diff'].shift(-1)

p[['date', 'ud', 'last_ud_rolling3d', 'pm_ud', 'last_ud', 'stock_symbol']].to_parquet(
    f'processed_{stock_symbol}_price.parquet', index=False)

pd.merge(cleaned_t[['date', 'session', 'Text', 'cleaned', 'tokenized']],
         p[['date', 'stock_symbol', 'pct_diff', 'next_pct_diff']], how='left', on='date').to_parquet(
    f'processed_{stock_symbol}_t.parquet', index=False)

pd.merge(n[['date', 'summary', 'headline', 'content', 'tokenized']],
         p[['date', 'stock_symbol', 'pct_diff']], how='left', on='date').to_parquet(
    f'processed_{stock_symbol}_news.parquet', index=False)

os.chdir('..\\..')

datetime.today().date() + timedelta(days=1)
datetime.strftime(datetime.today().date(),'%Y-%m-%d')