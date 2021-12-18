import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime

# Scrape stock market holidays

pattern = re.compile(r'([a-zA-Z]+\s[0-9]{1,2}),\s([0-9]{4})')


def date_processing(string):
    if not pd.isnull(string):
        return datetime.strptime(re.search(pattern, string).group(0), '%B %d, %Y')


URL = "https://www.insider-monitor.com/market-holidays.html"

page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

tables = soup.select('table')

dfs = []

for table in tables:

    name_list = []
    date_list = []

    for row in table.select('tr')[1:]:
        name_list.append(row.select('td')[0].text)
        date_list.append(row.select('td')[1].text)

    year = re.search(pattern, date_list[-1]).group(2)
    df = pd.DataFrame({'Holiday': name_list, f'{year}': date_list}).set_index('Holiday')

    dfs.append(df)

h = pd.concat(dfs, axis=1)
h.applymap(date_processing).to_csv('Holiday_list.csv')






