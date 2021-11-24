import calendar
from datetime import time,datetime, timedelta
import pytz
import pandas as pd
import snscrape.modules.twitter as sntwitter

c = calendar.Calendar(firstweekday=calendar.SUNDAY)

year = datetime.today().year
march = 3
november = 11

mar_calendar = c.monthdatescalendar(year,march)
summer_start_date = [day for week in mar_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == march][1]

nov_calendar = c.monthdatescalendar(year,november)
summer_end_date = [day for week in nov_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == november][0]

today_date = datetime.today().strftime('%Y-%m-%d')
next_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

timezone = pytz.timezone('UTC')
print(timezone.localize(time(9,0,0)))


#stock_start_time = {'summer': '2021, 'winter' }

tweets_list = []

for tweet in sntwitter.TwitterSearchScraper(f'$TSLA since:{today_date} until:{next_date}').get_items():
    if i > 100:
        break
    tweets_list.append([tweet.date, tweet.id, tweet.content])

tweets_df2 = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text'])

tweets_df2