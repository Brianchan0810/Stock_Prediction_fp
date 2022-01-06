from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator


def get_hist_data(current_time, execution_date):
    from datetime import datetime, timedelta
    import pandas as pd
    import snscrape.modules.twitter as sntwitter
    import finnhub
    from google.oauth2 import service_account
    from google.cloud import storage

    print(current_time)

    stock_symbol = 'GOOGL'

    key_path = '/home/airflow/gcs/dags/gcs_key.json'
    credentials = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    storage_client = storage.Client(credentials=credentials, project=credentials.project_id)
    bucket = storage_client.bucket('for_bootcamp')

    today_date = execution_date
    ytd_date = (datetime.strptime(execution_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')

    #scrape historical tweet
    tweets_list = []

    for tweet in sntwitter.TwitterSearchScraper(f'${stock_symbol} since:{ytd_date} until:{today_date} lang:en').get_items():
        tweets_list.append([tweet.date, tweet.content])

    tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Text'])

    formated_ytd = ytd_date.replace('-', '')
    filename = 'tweet' + formated_ytd + '.parquet'

    tweets_df.to_parquet(filename)
    bucket.blob(f'{stock_symbol}_his_tweet/'+filename).upload_from_filename(filename)

    #get historical news
    api_key = 'c6q4vh2ad3i891nj18e0'
    finnhub_client = finnhub.Client(api_key=api_key)

    fin_news = pd.DataFrame(finnhub_client.company_news(f'{stock_symbol}', _from=f"{ytd_date}", to=f"{today_date}"))

    filename = 'news' + formated_ytd + '.parquet'

    fin_news.to_parquet(filename)
    bucket.blob(f'{stock_symbol}_his_news/' + filename).upload_from_filename(filename)


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
    'start_date': datetime(2021, 12, 12),
    #'end_date': datetime(2021, 12, 10),
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
    'myDAG_GOOGL',
    default_args=default_args,
    description='scrape historical tweets',
    schedule_interval='0 0 * * *',
    max_active_runs=5,
    #tags=['example'],
    catchup=True
) as dag:

    t1 = PythonVirtualenvOperator(
    task_id="GetHisData",
    python_callable=get_hist_data,
    #provide_context=False,
    requirements=[
        "pandas",
        "snscrape @ git+https://github.com/JustAnotherArchivist/snscrape.git",
        "fastparquet",
        "google-cloud-storage",
        " finnhub-python"
        ],
    op_kwargs={"execution_date": "{{ds}}", "current_time": "{{ ts }}"},
    system_site_packages=False,
    )

    t1
