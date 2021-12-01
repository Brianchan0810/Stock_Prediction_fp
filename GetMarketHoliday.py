import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re

def DatePattern(string):
    pattern = re.compile('[a-zA-Z]+\s[0-9]{1,2}')
    if re.search(pattern, string) is not None:
        return re.search(pattern,string).group(0)
    else:
        return ''

def Fillzero(string):
    pattern = re.compile('[0-9]+')
    if len(string) > 0:
        date = re.search(pattern, string).group(0)
        if len(date)==1:
           return string.replace(f'{date}',f'0{date}')
        else: return string

URL = "https://www.nyse.com/markets/hours-calendars"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")

table = soup.select_one('div.stacked-section')

table_head = [item.text for item in table.select('table thead th')][1:]

table_content = []
for row in table.select('table tbody tr'):
    row_content = [DatePattern(item.text) for item in row.select('td')]
    table_content.append(row_content)

df = pd.DataFrame(data=np.array(table_content),columns=table_head)
df = df.applymap(Fillzero)
df = df.apply(lambda x :x + ' ' +x.index if len(str(x))>1 else 0,axis=1)
df = df.applymap(lambda x: x if len(str(x))>4 else '')

df.to_csv('HolidayTable.csv', index=False)

