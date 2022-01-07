from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
import os

path = os.getcwd() + '/gcs/dags'

def pre_market_task(path, execution_date):
    import sys
    sys.path.insert(0, path)

    import tweepy
    from sqlalchemy import create_engine
    import time
    import finnhub
    import re
    from datetime import datetime, timedelta
    import pandas as pd
    import nltk
    import joblib
    from my_module.common_function import get_recent_tweet, sentiment_analysis, aggregation, df_to_db, normalization

    sql_engine = create_engine(f'mysql+pymysql://user:user@18.216.168.246', pool_recycle=3600)
    db_connection = sql_engine.connect()

    h = pd.read_sql("select * from myfp.holiday", db_connection)
    holiday_list = h['date'].to_list()

    season = pd.read_sql("select * from myfp.season", db_connection)
    season['year'] = season['summer_start'].apply(lambda x: x.year)

    db_connection.close()

    today_date = datetime.now().date()
    print(today_date)
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

        be_t = pd.DataFrame(columns=['created_at', 'id', 'text'])

        counter = 0
        while counter < 6:
            be_t = pd.concat([be_t, get_recent_tweet(api, stock_symbol)], axis=0)
            be_t = pd.concat([be_t, get_recent_tweet(api, 'GOOG')], axis=0)
            if counter < 5:
                time.sleep(1200)
            counter += 1


        api_key = 'c6q4vh2ad3i891nj18e0'
        finnhub_client = finnhub.Client(api_key=api_key)

        ytd_str = datetime.strftime(today_date - timedelta(days=1), '%Y-%m-%d')
        today_str = datetime.strftime(today_date, '%Y-%m-%d')

        n = pd.DataFrame(finnhub_client.company_news(stock_symbol, _from=ytd_str, to=today_str))
        n = pd.concat([n, pd.DataFrame(finnhub_client.company_news('GOOG', _from=ytd_str, to=today_str))])

        # data processing
        print(be_t.shape)
        be_t.drop_duplicates(subset='id', inplace=True)
        be_t['created_at'] = pd.to_datetime(be_t['created_at'])
        be_t['date'] = be_t['created_at'].dt.date
        be_t['hour'] = be_t['created_at'].dt.hour
        be_t = be_t[be_t['date'] == today_date]
        print(be_t.shape)

        if in_summer:
            be_t = be_t[be_t['hour'] >= 11]
        else:
            be_t = be_t[be_t['hour'] >= 12]

        print(be_t.shape)
        be_t['stock_symbol_count'] = be_t['text'].apply(lambda x: len(re.findall(r'\$[A-za-z]+', x)))
        be_t = be_t[~(be_t['stock_symbol_count'] >= 5)]
        be_t['cleaned'] = be_t['text'].apply(lambda x: re.sub(r'([$#@]\w+)|(\w+:\/\/\S+|(RT))', '', x))
        be_t['cleaned'] = be_t['cleaned'].apply(lambda x: re.sub(r'\n\n', ' ', x))
        be_t['cleaned'] = be_t['cleaned'].apply(lambda x: re.sub(r'[^0-9a-zA-Z .,!?:%\t]', '', x))
        be_t.drop_duplicates(subset='cleaned', inplace=True)
        be_t.dropna(inplace=True, subset=['cleaned'])
        print(be_t.shape)

        print(n.shape)
        n['converted_dt'] = n['datetime'].apply(lambda x: datetime.utcfromtimestamp(x))
        n['converted_dt+3h'] = n['converted_dt'] + timedelta(hours=3)
        n['date'] = n['converted_dt+3h'].dt.date
        n = n[n['date'] == today_date]
        print(n.shape)
        n['content'] = n['headline'] + ' - ' + n['summary']
        n.drop_duplicates(subset='headline', inplace=True)
        print(n.shape)

        extra_dict_for_tweet = {'raise': 3, 'raises': 3, 'raised': 3, 'buy': 3, 'buys': 3, 'brought': 3, 'up': 3, 'hold': 3,
                                'high': 3, 'highs': 3, 'higher': 3, 'split': 3, 'to the moon': 3, 'hit': 3, 'hits': 3,
                                'long': 3, 'ATH': 3, 'call': 3, 'calls': 3, 'short': -3, 'sell': -3, 'sells': -3,
                                'sold': -3, 'resistance': -3, 'resistances': -3}

        nltk.download('vader_lexicon')

        be_t = sentiment_analysis(be_t, 'cleaned', 0.2, extra_dict_for_tweet)
        n = sentiment_analysis(n, 'content', 0.2)

        be_t_gb = aggregation(be_t, 'be', 0.2, f'{stock_symbol}_be_sent')
        news_gb = aggregation(n, 'news', 0.2, f'{stock_symbol}_news_sent')

        sql_engine = create_engine('mysql+pymysql://user:user@18.216.168.246', pool_recycle=3600)
        db_connection = sql_engine.connect()
        
        p = pd.read_sql(f"select * from myfp.{stock_symbol}_price order by date desc limit 3;", db_connection)
        p['last_ud'] = p['close'] > p['open']
        p['last_ud_rolling3d'] = p['last_ud'].rolling(3).sum().shift(-2)

        af_t_gb = pd.read_sql(f"select * from myfp.{stock_symbol}_af_sent order by date desc limit 1;", db_connection)
        af_t_gb = af_t_gb.add_prefix('af_')

        db_connection.close()

        mean_dict = {'be': {'be_compound_mean': 0.362744, 'be_compound_count': 23.880309}
                     , 'af': {'af_compound_mean': 0.362743, 'af_compound_count': 28.181467}
                     , 'news': {'news_compound_mean': 0.259703, 'news_compound_count': 12.921875}}

        be_t_gb2 = normalization(path, stock_symbol, mean_dict, be_t_gb, 'be', 10)
        af_t_gb2 = normalization(path, stock_symbol, mean_dict, af_t_gb, 'af', 10)
        news_gb2 = normalization(path, stock_symbol, mean_dict, news_gb, 'news', 5)

        final = pd.concat([be_t_gb2, af_t_gb2, news_gb2], axis=1)
        final = pd.merge(final, p, left_index=True, right_index=True)

        my_model = joblib.load(path + '/my_model.sav')
        result = my_model.predict(final[['last_ud', 'last_ud_rolling3d', 'be_compound_mean', 'be_compound_count',
                                         'af_compound_mean', 'af_compound_count', 'news_compound_mean', 'news_compound_count']])

        result_df = pd.DataFrame({'date': today_date, 'prediction': result})
        df_to_db(result_df, 'sandbox', f'{stock_symbol}_pred')


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
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
        'Prediction_test',
        default_args=default_args,
        description='predict the movement on that day',
        start_date=datetime(2022, 1, 1),
        schedule_interval='20 11 * * *',
        catchup=False

) as dag:

    t1 = PythonVirtualenvOperator(
        task_id="task1",
        python_callable=pre_market_task,
        # provide_context=False,
        requirements=[
            "pandas",
            "tweepy",
            "SQLAlchemy",
            "nltk",
            "yfinance",
            "mysql-connector-python",
            "finnhub-python",
            "pymysql"
        ],
        op_kwargs={"execution_date": "{{ds}}", "path": path},
        system_site_packages=False,
    )

    t1
























