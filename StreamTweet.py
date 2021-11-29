import tweepy
import pandas as pd

consumer_key = 'dZl0SxZW22bTL1IItvi0T6w55'
consumer_secret = 'VjpkXMrXjkdmu4m4CL1ugQMrXDTvfW694WMAQkjG8ughMIa8GB'
access_token = '1459779615589736458-JlW3tQCMe7K1Q8ef6BFZAOtmRa29IB'
access_token_secret = 'fSiS3ctTcOGMrN3hklUqj7SYTfNgIiErmDmFit99QSWq9'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets = api.search_tweets('$TSLA', lang='en', count=10, result_type='recent' )

list_of_tweets = []
for tweet in tweets:
    list_of_tweets.append(tweet._json)

df0 = pd.DataFrame(list_of_tweets)

df = df0[[]]

pd.to_datetime(df['created_at'])

df.info()

tweet._json[['created_at','id']]

df










class TwitterListener(tweepy.Stream):
    def __int__(self):
        self.mylist = []

    def on_data(self, raw_data):
        self.process_data(raw_data)

    def process_data(self, raw_data):

        print(raw_data)
        self.storage()

    def storage(self, raw_data):
        self.mylist.append(raw_data)

    def on_error(self, status_code):
        if status_code == 420:
            return False

    def disconnect(self):
        self.running = False


listener = TwitterListener(consumer_key, consumer_secret, access_token, access_token_secret)

listener.filter(track=['python'])

listener.disconnect()
