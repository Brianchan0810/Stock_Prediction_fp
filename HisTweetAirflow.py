from datetime import timedelta,datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator

def scrape_tweets(execution_date):
    from datetime import datetime, timedelta
    import pandas as pd
    import snscrape.modules.twitter as sntwitter
    from google.oauth2 import service_account
    from google.cloud import storage

    key_path = '/home/airflow/gcs/dags/gcs_key.json'
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
    bucket = storage_client.bucket('bootcamp-final-project')

    today_date = execution_date
    next_date = (datetime.strptime(execution_date,'%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    tweets_list = []

    for tweet in sntwitter.TwitterSearchScraper(f'$TSLA since:{today_date} until:{next_date} lang:en').get_items():
        tweets_list.append([tweet.date, tweet.content])

    tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Text'])

    date_format = execution_date.replace('-','')
    filename = 'tweet' + date_format + '.parquet'

    tweets_df.to_parquet(filename)
    bucket.blob('Historical-Tweets/'+filename).upload_from_filename(filename)

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
    #'end_date': datetime(2021, 11, 26),
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
    'myDAG',
    default_args=default_args,
    description='scrape historical tweets',
    schedule_interval='0 0 * * 1-5',
    start_date=datetime(2021, 1, 1),
    max_active_runs= 1,
    #tags=['example'],
    catchup=True
) as dag:

    t1 = PythonVirtualenvOperator(
    task_id="GetHisTweet",
    python_callable=scrape_tweets,
    #provide_context=False,
    requirements=[
        "pandas",
        "snscrape @ git+https://github.com/JustAnotherArchivist/snscrape.git",
        "fastparquet",
        "google-cloud-storage"
        ],
    op_kwargs={"execution_date":"{{ds}}"},
    system_site_packages=False,
    )

    t1