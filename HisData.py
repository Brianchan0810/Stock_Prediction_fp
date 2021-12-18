from datetime import datetime, timedelta
import pandas as pd
import snscrape.modules.twitter as sntwitter
import finnhub


today_date = datetime.today().strftime('%Y-%m-%d')
next_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

tweets_list = []

for tweet in sntwitter.TwitterSearchScraper(f'$TSLA since:{today_date} until:{next_date} lang:en').get_items():

    tweets_list.append([tweet.date, tweet.id, tweet.content])

tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'id','Text'])

date_format = datetime.today().strftime('%Y%m%d')
tweets_df.to_parquet(f'{date_format}tweet.parquet')
tweets_df.to_csv(f'{date_format}tweet.csv')


api_key = 'c6q4vh2ad3i891nj18e0'
finnhub_client = finnhub.Client(api_key=api_key)

fin_news = pd.DataFrame(finnhub_client.company_news('AMZN', _from="2021-02-03", to="2021-02-04"))
fin_news.shape
fin_news['headline'].head(60)
fin_news['converted_datetime'] = fin_news['datetime'].apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S'))

fin_news