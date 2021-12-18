from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import os


def combine_file(sort_column):
    dfs = []
    for file in os.listdir():
        if '.parquet' in file:
            df0 = pd.read_parquet(file)
            dfs.append(df0)

    df = pd.concat(dfs)
    df.sort_values(by=f'{sort_column}', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


stock_symbol = 'GOOG'

os.mkdir(f'.\\{stock_symbol}_tweet')
os.mkdir(f'.\\{stock_symbol}_news')
os.mkdir(f'.\\{stock_symbol}_analysis')

key_path = 'gcs_key.json'

credentials = service_account.Credentials.from_service_account_file(key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])

storage_client = storage.Client(credentials=credentials, project=credentials.project_id)

bucket = storage_client.bucket('for_bootcamp')

blobs = bucket.list_blobs()

for a_blob in blobs:
    if '.parquet' in a_blob.name and 'GOOG_' in a_blob.name:
        file_name = a_blob.name.split('/')[1]
        if 'news' in file_name:
            bucket.blob(a_blob.name).download_to_filename(f'{stock_symbol}_news\\{file_name}')
        elif 'tweet' in file_name:
            bucket.blob(a_blob.name).download_to_filename(f'{stock_symbol}_tweet\\{file_name}')


os.chdir(f'.\\{stock_symbol}_tweet')
df = combine_file('Datetime')

os.chdir('..')
df.to_parquet(f'{stock_symbol}_tweet.parquet')

os.chdir(f'.\\{stock_symbol}_news')
df = combine_file('datetime')

os.chdir('..')
df.to_parquet(f'{stock_symbol}_news.parquet')

