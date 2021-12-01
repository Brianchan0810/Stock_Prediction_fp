from google.oauth2 import service_account
from google.cloud import storage
import pandas as pd
import os

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

os.chdir('.\\Data')

dfs = []
for file in os.listdir():
    if '.parquet' in file:
        df0 = pd.read_parquet(file)
        dfs.append(df0)

df = pd.concat(dfs)
df
df.reset_index(inplace=True,drop=True)
df
df.sort_values(by='Datetime',inplace=True)
df
os.chdir('..')

df.to_parquet('AllHIsTweet.parquet')