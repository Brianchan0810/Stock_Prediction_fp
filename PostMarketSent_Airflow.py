from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
import os

path = os.getcwd() + '/gcs/dags'

def post_market_task(path, execution_date):
    import sys
    sys.path.insert(0, path)

    import tweepy
    from sqlalchemy import create_engine
    import time
    import re
    import pandas as pd
    import nltk
    import yfinance as yf
    from datetime import datetime, timedelta
    from my_module.common_function import get_recent_tweet, sentiment_analysis, aggregation, df_to_db

    sql_engine = create_engine(f'mysql+pymysql://user:user@18.216.168.246', pool_recycle=3600)
    db_connection = sql_engine.connect()

    h = pd.read_sql("select * from myfp.holiday", db_connection)
    holiday_list = h['date'].to_list()

    season = pd.read_sql("select * from myfp.season", db_connection)
    season['year'] = season['summer_start'].apply(lambda x: x.year)

    db_connection.close()

    today_date = datetime.now().date()
    print(today_date)
    print(today_date.weekday())
    print(execution_date)

    if today_date not in holiday_list and today_date.weekday() < 5:

        in_summer = season[season['year'] == today_date.year].apply(lambda x: x['summer_start'] < today_date < x['summer_end'], axis=1).values[0]

        if not in_summer:
            time.sleep(3600)

        stock_symbol = 'GOOGL'

        # get premarket tweet
        consumer_key = 'dZl0SxZW22bTL1IItvi0T6w55'
        consumer_secret = 'VjpkXMrXjkdmu4m4CL1ugQMrXDTvfW694WMAQkjG8ughMIa8GB'
        access_token = '1459779615589736458-JlW3tQCMe7K1Q8ef6BFZAOtmRa29IB'
        access_token_secret = 'fSiS3ctTcOGMrN3hklUqj7SYTfNgIiErmDmFit99QSWq9'

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        api = tweepy.API(auth)

        af_t = pd.DataFrame(columns=['created_at', 'id', 'text'])

        counter = 0
        while counter < 9:
            af_t = pd.concat([af_t, get_recent_tweet(api, stock_symbol)], axis=0)
            af_t = pd.concat([af_t, get_recent_tweet(api, 'GOOG')], axis=0)
            if counter < 8:
                time.sleep(1200)
            counter += 1

        # data processing
        print(af_t.shape)
        af_t.drop_duplicates(subset='id', inplace=True)
        af_t['created_at'] = pd.to_datetime(af_t['created_at'])
        af_t['date'] = af_t['created_at'].dt.date
        af_t['hour'] = af_t['created_at'].dt.hour
        af_t = af_t[af_t['date'] == today_date]
        print(af_t.shape)

        if in_summer:
            af_t = af_t[af_t['hour'] > 20]
        else:
            af_t = af_t[af_t['hour'] > 21]

        print(af_t.shape)
        af_t['stock_symbol_count'] = af_t['text'].apply(lambda x: len(re.findall(r'\$[A-za-z]+', x)))
        af_t = af_t[~(af_t['stock_symbol_count'] >= 5)]
        af_t['cleaned'] = af_t['text'].apply(lambda x: re.sub(r'([$#@]\w+)|(\w+:\/\/\S+|(RT))', '', x))
        af_t['cleaned'] = af_t['cleaned'].apply(lambda x: re.sub(r'\n\n', ' ', x))
        af_t['cleaned'] = af_t['cleaned'].apply(lambda x: re.sub(r'[^0-9a-zA-Z .,!?:%\t]', '', x))
        af_t.drop_duplicates(subset='cleaned', inplace=True)
        af_t.dropna(subset=['cleaned'], inplace=True)

        extra_dict_for_tweet = {'raise': 3, 'raises': 3, 'raised': 3, 'buy': 3, 'buys': 3, 'brought': 3, 'up': 3, 'hold': 3,
                                'high': 3, 'highs': 3, 'higher': 3, 'split': 3, 'to the moon': 3, 'hit': 3, 'hits': 3,
                                'long': 3, 'ATH': 3, 'call': 3, 'calls': 3, 'short': -3, 'sell': -3, 'sells': -3,
                                'sold': -3, 'resistance': -3, 'resistances': -3}

        nltk.download('vader_lexicon')

        print(af_t.shape)
        af_t = sentiment_analysis(af_t, 'cleaned', 0.2, extra_dict_for_tweet)

        af_t_gb = aggregation(af_t, 'af', 0.2, 'myfp', f'{stock_symbol}_af_sent')

        # input last day price movement
        next_date = today_date + timedelta(days=1)
        p = yf.download(tickers=f'{stock_symbol}', start=next_date, end=next_date, interval='1d')
        if p.shape[0] == 0:
            print('get empty dataframe')
            time.sleep(7200)
            p = yf.download(tickers=f'{stock_symbol}', start=next_date, end=next_date, interval='1d')
        p = p.reset_index()[['Date', 'Open', 'Close']]
        p.columns = ['date', 'open', 'close']
        print(p)
        df_to_db(p, 'myfp', f'{stock_symbol}_price')

        print('finished whole task')

    else:
        print("doesn't do anything")


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['cxb2000abc@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2021, 12, 10),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}
with DAG(
        'PostMarketTask_test',
        default_args=default_args,
        description='get tweet and stock price on that day',
        start_date=datetime(2022, 1, 13),
        schedule_interval='20 20 * * *',
        catchup=False

) as dag:

    t1 = PythonVirtualenvOperator(
        task_id="task1",
        python_callable=post_market_task,
        # provide_context=False,
        requirements=[
            "pandas",
            "tweepy",
            "SQLAlchemy",
            "nltk",
            "yfinance",
            "mysql-connector-python",
            "pymysql"
        ],
        op_kwargs={"execution_date": "{{ds}}", "path": path},
        system_site_packages=False,
    )

    t1

























