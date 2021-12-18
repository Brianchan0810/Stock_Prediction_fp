import calendar
from datetime import time, datetime, timedelta, date
import pandas as pd
import yfinance as yf
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import gensim
from sklearn.preprocessing import MinMaxScaler
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os

#utc timezone adopted throughout the study

# collect 2hr tweets before market start
summer_be_start, winter_be_start = time(11, 0, 0), time(12, 0, 0)
summer_be_end, winter_be_end = time(12, 59, 59), time(13, 59, 59)

# collect 3hr tweets after market end
summer_at_start, winter_at_start = time(20, 00, 0), time(21, 0, 0)
summer_at_end, winter_at_end = time(22, 59, 59), time(23, 59, 59)

# nltk.download('vader_lexicon')

def get_sunday_date(date, month, nth_week):
    year = date.year
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    my_calendar = c.monthdatescalendar(year, month)
    return [day for week in my_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == month][
        nth_week - 1]

def season(df):
    if df['summer_start_date'] < df['date'] < df['summer_end_date']:
        return 'summer'
    else:
        return 'winter'

def datetime_processing(main_df, holiday_df,datetime_column):
    df0 = main_df.copy()
    df0['date'] = df0[f'{datetime_column}'].dt.date
    df0['time'] = df0[f'{datetime_column}'].dt.time
    df0['date'] = pd.to_datetime(df0['date'])
    df0['weekday'] = df0[f'{datetime_column}'].dt.weekday
    df0['is_holiday'] = df0['date'].apply(lambda x: x in holiday_df[str(x.year)].dropna().to_list())
    df0 = df0[~((df0['is_holiday']) | (df0['weekday'] > 4))]
    df0['summer_start_date'] = df0['date'].apply(lambda x: get_sunday_date(x, 3, 2))
    df0['summer_end_date'] = df0['date'].apply(lambda x: get_sunday_date(x, 11, 1))
    df0['season'] = df0.apply(season, axis=1)
    return df0

def scan_target_period(df):
    if df['season'] == 'summer':
        if summer_be_start <= df['time'] <= summer_be_end:
            return 'before market'
        elif summer_at_start <= df['time'] <= summer_at_end:
            return 'after market'
        else:
            return 'out of scope'
    else:
        if winter_be_start <= df['time'] <= winter_be_end:
            return 'before market'
        elif winter_at_start <= df['time'] <= winter_at_end:
            return 'after market'
        else:
            return 'out of scope'

def est_to_edt(df, datetime_column):
    if df['season'] == 'summer':
        return df[f'{datetime_column}'] - timedelta(hours=1)
    else:
        return df[f'{datetime_column}']

def scan_target_period_for_news(df, datetime_column):
    if df['season'] == 'summer':
        return time(1, 0, 0) <= df[f'{datetime_column}'].time() <= time(13, 0, 0)
    else:
        return time(2, 0, 0) <= df[f'{datetime_column}'].time() <= time(14, 0, 0)

def news_processing(df0, holiday_df, start_date, end_date):
    df = df0.copy()
    df.drop_duplicates(subset='headline', inplace=True)
    df['converted_dt'] = df['datetime'].apply(lambda x: datetime.utcfromtimestamp(x))
    df['converted_dt+5h'] = df['converted_dt'] + timedelta(hours=5)
    df = datetime_processing(df, holiday_df, 'converted_dt+5h')
    df.sort_values(by='date', inplace=True)
    df = df[(df['date'] <= end_date) & (df['date'] >= start_date)]
    df['utc_dt'] = df.apply(est_to_edt, args=('converted_dt+5h',), axis=1)
    df = df[df.apply(scan_target_period_for_news, args=('utc_dt',), axis=1)]
    df['content'] = df['headline'] + ' - ' + df['summary']
    df['tokenized'] = df['headline'].apply(lambda x: word_tokenize((re.sub(r'[^a-zA-Z \t]', '', x)).lower()))
    return df

def sentiment_analysis(df, target_column, indicator_threshold, addition_word_dict={}):
    df0 = df.copy()
    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(addition_word_dict)
    scores = df0[f'{target_column}'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    df0 = pd.concat([df0.reset_index(drop=True), scores_df], axis=1)
    df0['positive'] = df0['compound'].apply(lambda x: 1 if x > indicator_threshold else 0)
    df0['negative'] = df0['compound'].apply(lambda x: 1 if x < -indicator_threshold else 0)
    return df0

def bag_of_words(df, tokenized_column, indicator_column, indicator_threshold):
    df0 = df.copy()
    df0['sw_dropped'] = df0[f'{tokenized_column}'].apply(lambda row: [word for word in row if word not in stop_words])
    pos_word_list = df0[df0[f'{indicator_column}'] >= indicator_threshold]['sw_dropped'].sum()
    neg_word_list = df0[df0[f'{indicator_column}'] <= -indicator_threshold]['sw_dropped'].sum()
    return pos_word_list, neg_word_list

def word_list_comparison(pos_list, neg_list, top_words):
    pos_dict = dict((k, v) for k, v in Counter(pos_list).most_common(top_words))
    neg_dict = dict((k, v) for k, v in Counter(neg_list).most_common(top_words))

    positive_key_dict = {}
    for k in pos_dict.keys():
        if neg_dict.get(k, 'not exit') == 'not exit' or pos_dict[k] / neg_dict[k] > 1.5:
            positive_key_dict[k] = pos_dict[k]

    negative_key_dict = {}
    for k in neg_dict.keys():
        if pos_dict.get(k, 'not exit') == 'not exit' or neg_dict[k] / pos_dict[k] > 1.5:
            negative_key_dict[k] = neg_dict[k]
    return positive_key_dict, negative_key_dict

def aggregation(df, groupby_list, column_prefix, indicator_threshold):
    df0 = df.copy()
    df0_gb = df0[~((df0['compound'] < indicator_threshold) & (df0['compound'] > -indicator_threshold))]\
        .groupby(groupby_list).agg({'compound': ['mean', 'count'], 'positive': 'sum', 'negative': 'sum'})
    df0_gb.columns = ['_'.join(col) for col in df0_gb.columns.values]
    df0_gb['positive_pct'] = df0_gb['positive_sum'] * 100 / df0_gb['compound_count']
    df0_gb['negative_pct'] = df0_gb['negative_sum'] * 100 / df0_gb['compound_count']
    df0_gb = df0_gb.add_prefix(f'{column_prefix}')
    df0_gb.reset_index(inplace=True)
    return df0_gb

def data_normalization(df, count_threshold, list_of_stock, prefix):
    list_of_processed_df = []
    for stock in list_of_stock:
        scaler = MinMaxScaler()
        df0 = df[df['stock_symbol'] == stock]
        df0.loc[df0[f'{prefix}_compound_count'] < count_threshold, f'{prefix}_compound_mean'] \
            = df0[df0[f'{prefix}_compound_count'] > count_threshold][f'{prefix}_compound_mean'].mean()
        df0[[f'{prefix}_compound_count', f'{prefix}_compound_mean']] \
            = scaler.fit_transform(df0[[f'{prefix}_compound_count', f'{prefix}_compound_mean']])
        list_of_processed_df.append(df0)
    return pd.concat(list_of_processed_df, axis=0)

def tagged_document(list_of_list_of_words):
   for i, list_of_words in enumerate(list_of_list_of_words):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


#Load market holiday date list
h = pd.read_csv('Holiday_list.csv')
h.set_index('Holiday', inplace=True)
h = h.applymap(lambda x: pd.to_datetime(x))


stock_symbol = 'AMZN'

# processing tweet data
t = pd.read_parquet(f'{stock_symbol}_tweet.parquet')
t = datetime_processing(t, h, datetime_column='Datetime')
t['session'] = t.apply(scan_target_period, axis=1)
t = t[t['session'] != 'out of scope']

cleaned_t = t.copy()
cleaned_t['stock_symbol_count'] = cleaned_t['Text'].apply(lambda x: len(re.findall(r'\$\w+', x)))
cleaned_t = cleaned_t[~(cleaned_t['stock_symbol_count'] >= 5)]
cleaned_t['cleaned'] = cleaned_t['Text'].apply(lambda x: re.sub(r'([$#@]\w+)|(\w+:\/\/\S+)', '', x))
cleaned_t['cleaned'] = cleaned_t['cleaned'].apply(lambda x: re.sub(r'\n\n', ' ', x))
cleaned_t['cleaned'] = cleaned_t['cleaned'].apply(lambda x: re.sub(r'[^0-9a-zA-Z .,!?:%\t]', '', x))
cleaned_t['tokenized'] = cleaned_t['cleaned'].apply(lambda x: word_tokenize((re.sub(r'[^a-zA-Z \t]', '', x)).lower()))
cleaned_t.drop_duplicates(subset='cleaned', inplace=True)
cleaned_t.dropna(subset='cleaned', inplace=True)

start_date = datetime.strftime(cleaned_t['date'].min(), '%Y-%m-%d')
end_date = datetime.strftime(cleaned_t['date'].max(), '%Y-%m-%d')


# processing news
n = pd.read_parquet(f'{stock_symbol}_news.parquet')
n = news_processing(n, h, start_date, end_date)


# processing stock price
s = yf.download(tickers=f'{stock_symbol}', start=f'{start_date}', end='2021-12-14', interval='1d', prepost=True)
s['ud'] = (s['Open'] - s['Close']) > 0
s['ud_rolling3d'] = s['ud'].apply(lambda x: 1 if x else -1).rolling(3).sum().apply(lambda x: 1 if x == 3 else (-1 if x == -3 else 0))
s['pm_ud'] = (s['Open'] - s['Close'].shift(1)) > 0
s['last_ud'] = s['ud'].shift(1)
s['price_diff'] = s['Close'] - s['Open']
s['pct_diff'] = s['price_diff'] * 100 / s['Open']
s['next_pct_diff'] = s['pct_diff'].shift(-1)
s['stock_symbol'] = stock_symbol
s['date'] = s.index
s[['date', 'ud', 'ud_rolling3d', 'pm_ud', 'last_ud', 'stock_symbol']].to_parquet(f'processed_{stock_symbol}_price.parquet', index=False)

pd.merge(cleaned_t[['date', 'session', 'Text', 'cleaned', 'tokenized']],
         s[['stock_symbol', 'pct_diff', 'next_pct_diff']], how='left', on='date').to_parquet(f'processed_{stock_symbol}_t.parquet', index=False)

pd.merge(n[['date', 'utc_dt', 'summary', 'headline', 'content', 'tokenized']],
         s[['stock_symbol', 'pct_diff']], how='left', on='date').to_parquet(f'processed_{stock_symbol}_news.parquet', index=False)


# study the three dataset together
list_of_stock = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
list_of_cleaned_t = []
list_of_n = []
list_of_s = []

for stock in list_of_stock:
    list_of_cleaned_t.append(pd.read_parquet(f'{stock}_t_processed.parquet'))
    list_of_n.append(pd.read_parquet(f'{stock}_news_processed.parquet'))
    list_of_s.append(pd.read_parquet(f'{stock}_price_processed.parquet'))

#len(list_of_cleaned_t) == len(list_of_n) == len(list_of_s)

cleaned_t = pd.concat(list_of_cleaned_t, axis=0)
n = pd.concat(list_of_n, axis=0)
s = pd.concat(list_of_s, axis=0)

be_t = cleaned_t[(cleaned_t['session'] == 'before market')]
af_t = cleaned_t[(cleaned_t['session'] == 'after market')]

stop_words = set(stopwords.words('english'))

be_pos_words, be_neg_words = bag_of_words(be_t, 'tokenized', 'pct_diff', 2)
af_pos_words, af_neg_words = bag_of_words(af_t, 'tokenized', 'next_pct_diff', 2)
news_pos_words, news_neg_words = bag_of_words(n, 'tokenized', 'pct_diff', 2)

tweet_pos_words = be_pos_words + af_pos_words
tweet_neg_words = be_neg_words + af_neg_words

tweet_pos_outstanding, tweet_neg_outstanding = word_list_comparison(tweet_pos_words, tweet_neg_words, 60)
news_pos_outstanding, news_neg_outstanding = word_list_comparison(news_pos_words, news_neg_words, 30)

# tweet_pos_outstanding
# tweet_neg_outstanding
# news_pos_outstanding
# news_neg_outstanding
# cleaned_t[cleaned_t['cleaned'].apply(lambda x: True if re.search('revenue', x, re.I) else False)]['stock_symbol'].value_counts()
# cleaned_t[cleaned_t['cleaned'].apply(lambda x: True if re.search('revenue', x, re.I) else False)]['Text'].tail(60)

extra_dict_for_tweet = {'raise': 3, 'raises': 3, 'raised': 3, 'buy': 3, 'buys': 3, 'brought': 3, 'up': 3, 'hold': 3,
                        'high': 3, 'highs': 3, 'higher': 3, 'split': 3, 'to the moon': 3, 'hit': 3, 'hits': 3,
                        'long': 3, 'ATH': 3, 'call': 3, 'calls': 3, 'short': -3, 'sell': -3, 'sells': -3, 'sold': -3,
                        'resistance': -3, 'resistances': -3}

be_t = sentiment_analysis(be_t, 'cleaned', 0.2, extra_dict_for_tweet)
af_t = sentiment_analysis(af_t, 'cleaned', 0.2, extra_dict_for_tweet)
n = sentiment_analysis(n, 'content', 0.2)

be_t_gb = aggregation(be_t, ['stock_symbol', 'date'], 'be_', 0.2)
af_t_gb = aggregation(af_t, ['stock_symbol', 'date'], 'af_', 0.2)
news_gb = aggregation(n, ['stock_symbol', 'date'], 'news_', 0.2)

be_t_gb2 = data_normalization(be_t_gb, 10, list_of_stock, 'be')
af_t_gb2 = data_normalization(af_t_gb, 10, list_of_stock, 'af')
news_gb2 = data_normalization(news_gb, 5, list_of_stock, 'news')

af_t_gb2['date'] = af_t_gb2.groupby('stock_symbol')['date'].shift(-1)
af_t_gb2.dropna(subset=['date'], inplace=True)

#Doc2Vec
headline_gb = n.groupby(['stock_symbol', 'date'])[['tokenized', 'headline']].agg({'tokenized': 'sum', 'headline': lambda x:' '.join(x)}).reset_index()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(headline_gb['headline'])]
model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=4)
vector = pd.DataFrame([list(model.infer_vector(row)) for row in headline_gb['tokenized']])
vector = vector.add_prefix('v_')
headline_gb = pd.concat([headline_gb, vector], axis=1)
news_gb3 = pd.merge(news_gb2, headline_gb, how='left', on=['stock_symbol', 'date'])

start_date = datetime.strftime(cleaned_t['date'].min(), '%Y-%m-%d')
end_date = datetime.strftime(cleaned_t['date'].max(), '%Y-%m-%d')

entire = pd.merge(
    pd.merge(
        pd.merge(s
                 , be_t_gb2, how='outer', on=['stock_symbol', 'date']),
        af_t_gb2, how='outer', on=['stock_symbol', 'date']),
    news_gb3, how='outer', on=['stock_symbol', 'date'])

entire = entire[((start_date <= entire['date']) & (entire['date'] <= end_date))].fillna(entire.mean())


#NASDAQ 100
start_date = datetime.strftime(entire['date'].min(), '%Y-%m-%d')
end_date = datetime.strftime(entire['date'].max(), '%Y-%m-%d')

nd_s = yf.download(tickers='^NDX', start='2020-12-15', end='2021-12-14', interval='1d', prepost=False)
nd_s['date'] = nd_s.index
nd_s['nd_ud'] = (nd_s['Close'] - nd_s['Open']) > 0
nd_s['nd_last_ud'] = nd_s['nd_ud'].shift(1)
nd_s['nd_ud'] = nd_s['nd_ud'].apply(lambda x: 1 if x else -1)
nd_s['nd_ud_rolling3d'] = nd_s['nd_ud'].rolling(3).sum().apply(lambda x: 1 if x == 3 else (-1 if x == -3 else 0))
nd_s['nd_cur_pm_ud'] = (nd_s['Open'] - nd_s['Close'].shift(1)) > 0

entire = pd.merge(entire, nd_s[['date', 'nd_last_ud', 'nd_cur_pm_ud', 'nd_ud_rolling3d']], how='left', on='date')

entire.to_csv('ready3.csv', index=False)
