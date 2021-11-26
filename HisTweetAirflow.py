from datetime import datetime, timedelta
import pandas as pd
import snscrape.modules.twitter as sntwitter

today_date = datetime.today().strftime('%Y-%m-%d')
next_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

tweets_list = []

for tweet in sntwitter.TwitterSearchScraper(f'$TSLA since:{today_date} until:{next_date} lang:en').get_items():

    tweets_list.append([tweet.date,tweet.content])

tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Text'])

date_format = datetime.today().strftime('%Y%m%d')
tweets_df.to_parquet(f'{date_format}tweet.parquet')
tweets_df.to_csv(f'{date_format}tweet.csv')