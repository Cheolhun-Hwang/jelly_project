import pandas as pd
import datetime as datetime

#loadData
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')
dataset = pd.read_csv('D:/data/csv/lstm_test.csv')
dataset.drop('No', axis=1, inplace=True)

print(dataset)