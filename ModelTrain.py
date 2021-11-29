import calendar
from datetime import time,datetime, timedelta
import pytz
import pandas as pd
from google.oauth2 import service_account
from google.cloud import storage
import os


def GetSundayDate(date,month,nth_week):
    year = date.year
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    my_calendar = c.monthdatescalendar(year, month)
    return [day for week in my_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == month][nth_week-1]


key_path = 'gcs_key.json'
credentials = service_account.Credentials.from_service_account_file(
key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
bucket = storage_client.bucket('bootcamp-final-project')

blobs = bucket.list_blobs()

for a_blob in blobs:
    if '.parquet' in a_blob.name:
        file_name = a_blob.name.split('/')[1]
        bucket.blob(a_blob.name).download_to_filename(f'Data\\{file_name}')

os.chdir(os.getcwd()+'\\Data')

dfs = []
for file in os.listdir():
    if '.parquet' in file:
        df0 = pd.read_parquet(file)
        dfs.append(df0)

df = pd.concat(dfs)
df.reset_index(inplace=True)
df['Datetime'][0] > datetime(2021,3,5, 12, 44)



summer_start_date = GetSundayDate(3, 2)
summer_end_date = GetSundayDate(11, 1)

timezone = pytz.timezone('UTC')
summer_start_time = timezone.localize(time(13,30,0))
summer_end_time = timezone.localize(time(20,0,0))
winter_start_time = timezone.localize(time(14,30,0))
winter_end_time = timezone.localize(time(21,0,0))

winter_start_time - timedelta(hours=2)

today_date = datetime.today().strftime('%Y-%m-%d')
next_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

df = pd.read_parquet('20211126tweet.parquet')

df['season'] = df['Datetime'].map(lambda x: )
