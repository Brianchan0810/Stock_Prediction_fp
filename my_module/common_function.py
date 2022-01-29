def get_recent_tweet(api, stock_symbol):
    import pandas as pd
    tweets = api.search_tweets(f'${stock_symbol}', lang='en', count=150, result_type='recent')

    list_of_tweets = []
    for tweet in tweets:
        list_of_tweets.append(tweet._json)

    return pd.DataFrame(list_of_tweets)[['created_at', 'id', 'text']]

def sentiment_analysis(df0, target_column, indicator_threshold, addition_word_dict={}):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import pandas as pd

    df = df0.copy()
    vader = SentimentIntensityAnalyzer()
    vader.lexicon.update(addition_word_dict)
    scores = df[f'{target_column}'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
    df['positive'] = df['compound'].apply(lambda x: 1 if x > indicator_threshold else 0)
    return df

def df_to_db(df, db_name, table_name):
    from sqlalchemy import create_engine

    sql_engine = create_engine(f'mysql+pymysql://user:user@18.216.168.246/{db_name}', pool_recycle=3600)
    db_connection = sql_engine.connect()

    df.to_sql(table_name, db_connection, if_exists='append', index=False)

    db_connection.close()

    return 'Successfully added to database'

def get_value_from_db_column(database_name, table_name, column_name,how):
    import mysql.connector

    mydb = mysql.connector.connect(
        host="18.216.168.246",
        user="user",
        password="user",
        database=database_name
    )

    mycursor = mydb.cursor()
    if how == 'max':
        mycursor.execute(f"SELECT max({column_name}) FROM {table_name}")
    if how == 'min':
        mycursor.execute(f"SELECT min({column_name}) FROM {table_name}")
    return mycursor.fetchone()[0]

def aggregation(df0, prefix , threshold, db_name, table_name):
    from datetime import datetime
    df = df0
    df = df[~((df['compound'] < threshold) & (df['compound'] > -threshold))]
    df_gb = df.groupby('date').agg({'compound': ['mean', 'count'], 'positive': 'sum'})
    df_gb.columns = ['_'.join(col) for col in df_gb.columns.values]
    df_gb.reset_index(inplace=True)
    db_most_update_date = get_value_from_db_column(db_name, table_name, 'date', 'max')
    print(db_most_update_date)
    if db_most_update_date != datetime.now().date():
        df_to_db(df_gb, db_name, table_name)
        print('added to db')
    df_gb = df_gb.add_prefix(f'{prefix}_')
    return df_gb

def normalization(path, stock_symbol, mean_dict, df0_gb, prefix, threshold):
    from sklearn.preprocessing import MinMaxScaler
    import joblib

    df_gb = df0_gb.copy()
    if (df_gb[f'{prefix}_compound_count'] < threshold).values[0]:
        df_gb[f'{prefix}_compound_mean'] = mean_dict[prefix][f'{prefix}_compound_mean']
        df_gb[f'{prefix}_compound_count'] = mean_dict[prefix][f'{prefix}_compound_count']
    scaler = joblib.load(path + f'/{stock_symbol}_{prefix}_scaler.sav')
    df_gb[[f'{prefix}_compound_count', f'{prefix}_compound_mean']] \
        = scaler.transform(df_gb[[f'{prefix}_compound_count', f'{prefix}_compound_mean']])
    return df_gb
