Welcome to this repository! 

The main goal of this project is to predict daily price movement of a stock and ultimately a group of stocks. 

To do so, a model would be trained based on the sentiment of tweets, news and the historical price movement.

Regarding sentiment of tweets, data would be collected within two periods which are the pre-market period (2 hours before market open), the post-market period of the previous day (3 hours after market close)

Moreover, in order to expand the dataset, 5 stocks from Nasdaq-100 are picked which are TSLA, MSFT, AMZN, GOOGL and AAPL.

This project is divided into several parts:

1. **Data Collection ~ HisData_airflow.py**
    - Collect past 1 year tweets and news data
    - Data backfilling using Apache Airflow
    - A parquet file of each day would be sent to Cloud Storage.

2. **Data Transfer ~ CStoLocal.py**
    - Copy file from Cloud Storage to local

3. **Data input to DB ~ Datapreparation.py**
    - Basic information like market holiday date, summertime start and end date and past stock price records are stored into a relational database.

4. **Data Preprocessing ~ Preprocessing.py**
    - Perform data filtering and cleansing

5. **Data Analysis ~ DataAnalysis.py**
    - Sentiment of each records would be extracted
    - A general sentiment would be represented by the mean and numbers of effective records on that day.  
    - Try to adopt bag of word but no significant insight is generated

6. **Machine learning ~ ModelTrain.py**
    - General performance among 3 models (Linear SVM, Random Forest & k-nearest neighbors) are compared
    - Trained models are further applied to individual stock for selecting the best performance model

7. **Prediction Generation ~ (PostMarketSent_Airflow.py & Prediction_Airflow.py)**
    - Daily data collection and prediction generation is scheduled using Apache Airflow





