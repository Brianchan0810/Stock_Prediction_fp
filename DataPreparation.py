from sqlalchemy import create_engine
import pymysql
import calendar
import mysql.connector
from datetime import datetime, date, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import yfinance as yf


def df_to_db(df, db_name, table_name):
    sql_engine = create_engine(f'mysql+pymysql://user:user@18.216.168.246/{db_name}', pool_recycle=3600)
    db_connection = sql_engine.connect()

    try:
        df.to_sql(table_name, db_connection, if_exists='append', index=False)

    except ValueError as vx:
        return vx

    except Exception as ex:
        return ex

    finally:
        db_connection.close()

    return 'Successfully added to database'

def get_value_from_db_column(database_name, table_name, column_name,how):
    mydb = mysql.connector.connect(
        host="18.216.168.246",
        user="user",
        password="user",
        database=database_name
    )

    mycursor = mydb.cursor()
    if how == 'max':
        mycursor.execute(f"SELECT max({column_name}) FROM {table_name}")
    if how == 'min':
        mycursor.execute(f"SELECT min({column_name}) FROM {table_name}")
    return mycursor.fetchone()[0]

def get_sunday_date(year, month, nth_week):
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    my_calendar = c.monthdatescalendar(year, month)
    return [day for week in my_calendar for day in week if day.weekday() == calendar.SUNDAY and day.month == month][
        nth_week - 1]

def date_processing(string):
    if not pd.isnull(string):
        return datetime.strptime(re.search(pattern, string).group(0), '%B %d, %Y')

# prepare summer start and end date for each year
summer_start = []
summer_end = []

try:
    last_year = get_value_from_db_column('myfp', 'season', 'summer_start', 'max').year

except:
    last_year = 2018

for i in range(last_year, datetime.now().year + 1):
    summer_start.append(get_sunday_date(i, 3, 2))
    summer_end.append(get_sunday_date(i, 11, 1))

df = pd.DataFrame({'summer_start': summer_start, 'summer_end': summer_end})
df_to_db(df, 'myfp', 'season')


# Scrape stock market holidays
pattern = re.compile(r'[a-zA-Z]+\s[0-9]{1,2},\s[0-9]{4}')

URL = "https://www.insider-monitor.com/market-holidays.html"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")
tables = soup.select('table')

date_list = []

for table in tables:
    for row in table.select('tr')[1:]:
        date_list.append(row.select('td')[1].text)

df = pd.DataFrame(date_list, columns=['date'])
df['date'] = df['date'].apply(date_processing)

try:
    last_year = get_value_from_db_column('myfp', 'holiday', 'date', 'max').year
    df = df[df['date'].apply(lambda x: x.year > last_year)]

except:
    pass
df.drop_duplicates(inplace=True)
df_to_db(df, 'myfp', 'holiday')


# Backup stock price data
stock_db_dict = {'^NDX': 'ndx_price', 'GOOGL': 'GOOGL_price'}

for stock in stock_db_dict.keys():
    last_date = get_value_from_db_column('myfp', stock_db_dict[stock], 'date', 'max')

    if last_date is None:
        last_date = date(2019, 1, 1)

    last_date = last_date + timedelta(days=2)

    if last_date <= datetime.now().date():
        end_date = datetime.strftime(datetime.now(), '%Y-%m-%d')
        start_date = datetime.strftime(last_date, '%Y-%m-%d')

        df = yf.download(tickers=f'{stock}', start=start_date, end=end_date, interval='1d')
        df = df.reset_index()[['Date', 'Open', 'Close']]
        df.columns = ['date', 'open', 'close']
        df_to_db(df, 'myfp', stock_db_dict[stock])
