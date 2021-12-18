import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
import joblib

def split_by_date(df0, feature_list, label_column, test_size):
    df = df0.copy()

    date_list = df['date'].sort_values().unique()
    test_start_date = date_list[-round(len(date_list) * test_size)]
    train, test = df[df['date'] < test_start_date], df[df['date'] >= test_start_date]

    return train[feature_list], test[feature_list], train[label_column], test[label_column]

def model_evaluation(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred)


list0 = ['ud_rolling3d', 'pm_ud', 'last_ud', 'be_compound_mean', 'be_compound_count','af_compound_mean',
         'af_compound_count', 'news_compound_mean', 'news_compound_count', 'nd_last_ud', 'nd_cur_pm_ud',
         'nd_ud_rolling3d']

list1 = ['ud_rolling3d', 'last_ud', 'be_compound_mean', 'be_compound_count', 'af_compound_mean',
         'af_compound_count', 'news_compound_mean', 'news_compound_count', 'nd_last_ud', 'nd_ud_rolling3d',
         'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9']

list2 = ['ud_rolling3d', 'last_ud', 'be_compound_mean', 'be_compound_count', 'af_compound_mean',
         'af_compound_count', 'news_compound_mean', 'news_compound_count', 'nd_last_ud', 'nd_ud_rolling3d']

list3 = ['ud_rolling3d', 'pm_ud', 'last_ud', 'nd_last_ud', 'nd_cur_pm_ud', 'nd_ud_rolling3d']

df = pd.read_csv('ready3.csv')

X_train, X_test, y_train, y_test = split_by_date(df, list2, 'ud', 0.3)

lin_svm = svm.LinearSVC()
rf = RandomForestClassifier(max_depth=4, random_state=0)
knn = KNeighborsClassifier(n_neighbors=2)

for model in [lin_svm, rf, knn]:
    model_evaluation(model, X_train, X_test, y_train, y_test)

accuracy_list = []
recall_list = []
precision_list = []
for i in range(2, 10):
    rf = RandomForestClassifier(max_depth=i, random_state=42)
    accuracy, recall, precision = model_evaluation(rf, X_train, X_test, y_train, y_test)
    accuracy_list.append(accuracy)
    recall_list.append(recall)
    precision_list.append(precision)

plt.plot(range(2, 10), accuracy_list, label='accuracy')
plt.plot(range(2, 10), recall_list, label='recall')
plt.plot(range(2, 10), precision_list, label='precision')
plt.legend()

final_model = RandomForestClassifier(max_depth=4)
final_model.fit(X_train, y_train)
filename = 'my_model.pkl'
joblib.dump(final_model, filename)

my_model = joblib.load('my_model.sav')
model_evaluation(my_model, X_train, X_test, y_train, y_test)














