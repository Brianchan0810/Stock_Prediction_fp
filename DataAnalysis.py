from datetime import datetime, timedelta, date
import pandas as pd
import yfinance as yf
from nltk.corpus import stopwords
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import gensim
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
from DataPreparation import df_to_db, get_value_from_db_column
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def sentiment_analysis(df, target_column, indicator_threshold, addition_word_dict={}):
    df0 = df.copy()
    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(addition_word_dict)
    scores = df0[f'{target_column}'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    df0 = pd.concat([df0.reset_index(drop=True), scores_df], axis=1)
    df0['positive'] = df0['compound'].apply(lambda x: 1 if x > indicator_threshold else 0)
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

def aggregation(df0, column_prefix, indicator_threshold, sent_to_db=False, table_name=''):
    df = df0.copy()
    df_gb = df[~((df['compound'] < indicator_threshold) & (df['compound'] > -indicator_threshold))] \
        .groupby('date').agg({'compound': ['mean', 'count'], 'positive': 'sum'})
    df_gb.columns = ['_'.join(col) for col in df_gb.columns.values]
    df_gb.reset_index(inplace=True)
    if sent_to_db:
        temp_df = df_gb.copy()
        last_date = get_value_from_db_column('myfp', table_name, 'date', 'max')
        if last_date is not None:
            last_date = datetime.combine(last_date, datetime.min.time())
            temp_df = temp_df[temp_df['date'] > last_date]
        df_to_db(temp_df, 'myfp', table_name)
    df_gb['positive_pct'] = df_gb['positive_sum'] * 100 / df_gb['compound_count']
    df_gb = df_gb.add_prefix(f'{column_prefix}_')
    df_gb.rename(columns={f'{column_prefix}_date': 'date'}, inplace=True)
    return df_gb

def data_normalization(df0, count_threshold, prefix, filename=None):
    df = df0.copy()
    scaler = MinMaxScaler()
    df.loc[df[f'{prefix}_compound_count'] < count_threshold, f'{prefix}_compound_mean'] \
        = df[df[f'{prefix}_compound_count'] > count_threshold][f'{prefix}_compound_mean'].mean()
    df[[f'{prefix}_compound_count', f'{prefix}_compound_mean']] \
        = scaler.fit_transform(df[[f'{prefix}_compound_count', f'{prefix}_compound_mean']])
    if filename is not None:
        joblib.dump(scaler, filename)
    return df

def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])


stock_symbol = 'AAPL'

os.chdir(f'.\\Data\\{stock_symbol}_analysis')

cleaned_t = pd.read_parquet(f'processed_{stock_symbol}_t.parquet')
n = pd.read_parquet(f'processed_{stock_symbol}_news.parquet')
p = pd.read_parquet(f'processed_{stock_symbol}_price.parquet')

be_t = cleaned_t[(cleaned_t['session'] == 'before market')]
af_t = cleaned_t[(cleaned_t['session'] == 'after market')]

# bag of word
stop_words = set(stopwords.words('english'))

be_pos_words, be_neg_words = bag_of_words(be_t, 'tokenized', 'pct_diff', 2)
af_pos_words, af_neg_words = bag_of_words(af_t, 'tokenized', 'next_pct_diff', 2)
news_pos_words, news_neg_words = bag_of_words(n, 'tokenized', 'pct_diff', 2)

tweet_pos_words = be_pos_words + af_pos_words
tweet_neg_words = be_neg_words + af_neg_words

tweet_pos_outstanding, tweet_neg_outstanding = word_list_comparison(tweet_pos_words, tweet_neg_words, 60)
news_pos_outstanding, news_neg_outstanding = word_list_comparison(news_pos_words, news_neg_words, 30)

# sentiment analysis
extra_dict_for_tweet = {'raise': 3, 'raises': 3, 'raised': 3, 'buy': 3, 'buys': 3, 'brought': 3, 'up': 3, 'hold': 3,
                        'high': 3, 'highs': 3, 'higher': 3, 'split': 3, 'to the moon': 3, 'hit': 3, 'hits': 3,
                        'long': 3, 'ATH': 3, 'call': 3, 'calls': 3, 'short': -3, 'sell': -3, 'sells': -3, 'sold': -3,
                        'resistance': -3, 'resistances': -3}

# nltk.download('vader_lexicon')

be_t = sentiment_analysis(be_t, 'cleaned', 0.2, extra_dict_for_tweet)
af_t = sentiment_analysis(af_t, 'cleaned', 0.2, extra_dict_for_tweet)
n = sentiment_analysis(n, 'content', 0.2)

be_t_gb = aggregation(be_t, 'be', 0.2, sent_to_db=True, table_name=f'{stock_symbol}_be_sent')
af_t_gb = aggregation(af_t, 'af', 0.2, sent_to_db=True, table_name=f'{stock_symbol}_af_sent')
news_gb = aggregation(n, 'news', 0.2, sent_to_db=True, table_name=f'{stock_symbol}_news_sent')

be_t_gb[['be_compound_mean', 'be_compound_count']].mean()
af_t_gb[['af_compound_mean', 'af_compound_count']].mean()
news_gb[['news_compound_mean', 'news_compound_count']].mean()

be_t_gb2 = data_normalization(be_t_gb, 10, 'be', f'{stock_symbol}_be_scaler.sav')
af_t_gb2 = data_normalization(af_t_gb, 10, 'af', f'{stock_symbol}_af_scaler.sav')
news_gb2 = data_normalization(news_gb, 5, 'news', f'{stock_symbol}_news_scaler.sav')

af_t_gb2['date'] = af_t_gb2['date'].shift(-1)
af_t_gb2.dropna(subset=['date'], inplace=True)

final = pd.merge(pd.merge(pd.merge(p
                 , be_t_gb2, how='outer', on='date'),
        af_t_gb2, how='outer', on='date'),
    news_gb2, how='outer', on='date')

start_date = datetime.strftime(cleaned_t['date'].min(), '%Y-%m-%d')
end_date = datetime.strftime(cleaned_t['date'].max(), '%Y-%m-%d')

final = final[((start_date <= final['date']) & (final['date'] <= end_date))].fillna(final.mean())


# NASDAQ 100
nd_p = yf.download(tickers='^NDX', start=start_date, end='2021-12-29', interval='1d')
nd_p['date'] = nd_p.index
nd_p['ud'] = nd_p['Close'] > nd_p['Open']
nd_p['pm_ud'] = nd_p['Open'] > nd_p['Close'].shift(1)
nd_p['last_ud'] = nd_p['ud'].shift(1)
nd_p['last_ud_rolling3d'] = nd_p['ud'].rolling(3).sum().apply(lambda x: 1 if x == 3 else (-1 if x == 0 else 0)).shift(1)
nd_p = nd_p.add_prefix('nd_')
nd_p.rename(columns={'nd_date': 'date'}, inplace=True)
final = pd.merge(final, nd_p[['date', 'nd_last_ud', 'nd_pm_ud', 'nd_last_ud_rolling3d']], how='left', on='date')

final.to_csv(f'{stock_symbol}_ready.csv', index=False)

os.chdir('..\\..')

dfs = []
list_of_stock = ['TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
for stock in list_of_stock:
   df = pd.read_csv(f'.\\Data\\{stock}_analysis\\{stock}_ready.csv') 
   dfs.append(df)
entire = pd.concat(dfs, axis=0)
entire.to_csv('entire_ready.csv')



list_of_cleaned_t = []
list_of_n = []
list_of_p = []

for stock_symbol in list_of_stock:
    os.chdir(f'.\\Data\\{stock_symbol}_analysis')
    list_of_cleaned_t.append(pd.read_parquet(f'processed_{stock_symbol}_t.parquet'))
    list_of_n.append(pd.read_parquet(f'processed_{stock_symbol}_news.parquet'))
    list_of_p.append(pd.read_parquet(f'processed_{stock_symbol}_price.parquet'))
    os.chdir(f'..\\..')


# Doc2Vec
# headline_gb = n.groupby(['stock_symbol', 'date'])[['tokenized', 'headline']].agg(
#     {'tokenized': 'sum', 'headline': lambda x: ' '.join(x)}).reset_index()
# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(headline_gb['headline'])]
# model = Doc2Vec(documents, vector_size=10, window=2, min_count=1, workers=4)
# vector = pd.DataFrame([list(model.infer_vector(row)) for row in headline_gb['tokenized']])
# vector = vector.add_prefix('v_')
# headline_gb = pd.concat([headline_gb, vector], axis=1)
# news_gb3 = pd.merge(news_gb2, headline_gb, how='left', on=['stock_symbol', 'date'])
