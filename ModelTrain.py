import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
import os
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from DataPreparation import df_to_db

def split_by_date(df0, feature_list, label_column, test_size):
    df = df0.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['stock_symbol', 'date'])
    date_list = df['date'].unique()
    test_start_date = date_list[-round(len(date_list) * test_size)]
    train, test = df[df['date'] < test_start_date], df[df['date'] >= test_start_date]
    return train[feature_list], test[feature_list], train[label_column], test[label_column], test_start_date

# choices of features list
list0 = ['last_ud', 'last_ud_rolling3d', 'be_compound_mean', 'be_compound_count', 'af_compound_mean',
         'af_compound_count', 'news_compound_mean', 'news_compound_count', 'nd_last_ud', 'nd_last_ud_rolling3d']

list1 = list0[2:]

list2 = list0[:-2]

list3 = ['last_ud_rolling3d', 'pm_ud', 'last_ud', 'nd_last_ud', 'nd_pm_ud', 'nd_last_ud_rolling3d']

#split the whole df into training and testing dataset
entire = pd.read_csv('entire_ready.csv')

X_train, X_test, y_train, y_test, test_start_date = split_by_date(entire, list2, 'ud', 0.2)

# comparing performance among models

lin_svm = svm.LinearSVC()
rf = RandomForestClassifier(max_depth=4, random_state=42)
knn = KNeighborsClassifier(n_neighbors=2)

model_dict = {'linear_svm': lin_svm, 'random_forest': rf, 'k_neighbors': knn}

result = {}
for name, model in model_dict.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result[name] = [accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred)]

result_df = pd.DataFrame(result, index=['accuracy', 'recall', 'precision'])
result_df.reset_index(inplace=True)
result_df = pd.melt(result_df, id_vars='index')
sns.barplot(data=result_df, x='variable', y='value', hue='index')


# compare the performance of one model among different stock
dfs = []
stock_list = ['TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']
for stock in stock_list:
    df0 = pd.read_csv(f'.\\Data\\{stock}_analysis\\{stock}_ready.csv')
    dfs.append(df0)

model = svm.LinearSVC()
model.fit(X_train, y_train)

result = []
for df in dfs:
    df = df[pd.to_datetime(df['date']) > test_start_date]
    y_pred = model.predict(df[list2])
    result.append(accuracy_score(df['ud'], y_pred))
plt.bar(stock_list, result)
y_pred
#save model

filename = 'my_model.sav'
final_model = svm.LinearSVC()
final_model.fit(X_train, y_train)
final_model.feature_names = list(X_train.columns.values)
joblib.dump(final_model, filename)

# input the prediction to the database
my_model = joblib.load('my_model.sav')
individual = pd.read_csv('.\\Data\\GOOGL_analysis\\GOOGL_ready.csv')
individual = individual[pd.to_datetime(individual['date']) > test_start_date]
individual['prediction'] = my_model.predict(individual[list2])
df_to_db(individual[['date', 'prediction']], 'myfp', 'GOOGL_pred')



















