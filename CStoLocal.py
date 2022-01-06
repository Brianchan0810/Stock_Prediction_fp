from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import os
import re


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


stock_symbol = 'TSLA'

os.mkdir(f'.\\Data\\{stock_symbol}_tweet')
os.mkdir(f'.\\Data\\{stock_symbol}_news')
os.mkdir(f'.\\Data\\{stock_symbol}_analysis')

key_path = 'gcs_key.json'

credentials = service_account.Credentials.from_service_account_file(key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])

storage_client = storage.Client(credentials=credentials, project=credentials.project_id)

bucket = storage_client.bucket('for_bootcamp')

blobs = bucket.list_blobs()

local_t_list = os.listdir(f'.\\Data\\{stock_symbol}_tweet')
local_n_list = os.listdir(f'.\\Data\\{stock_symbol}_news')

t_pattern = re.compile(f'{stock_symbol}_his_tweet/tweet\d+.parquet')
n_pattern = re.compile(f'{stock_symbol}_his_news/news\d+.parquet')

for a_blob in blobs:
    if re.search(t_pattern, a_blob.name):
        if a_blob.name.split('/')[1] not in local_t_list:
            filename = a_blob.name.split('/')[1]
            bucket.blob(a_blob.name).download_to_filename(f'.\\Data\\{stock_symbol}_tweet\\{filename}')
    elif re.search(n_pattern, a_blob.name):
        if a_blob.name.split('/')[1] not in local_n_list:
            filename = a_blob.name.split('/')[1]
            bucket.blob(a_blob.name).download_to_filename(f'.\\Data\\{stock_symbol}_news\\{filename}')


os.chdir(f'.\\Data\\{stock_symbol}_tweet')

df = combine_file('Datetime')

df.to_parquet(f'..\\{stock_symbol}_analysis\\combined_{stock_symbol}_t.parquet')

os.chdir(f'..\\{stock_symbol}_news')

df = combine_file('datetime')

df.to_parquet(f'..\\{stock_symbol}_analysis\\combined_{stock_symbol}_n.parquet')

os.chdir('..\\..')
