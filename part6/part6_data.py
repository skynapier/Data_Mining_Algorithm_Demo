import numpy as np
import random
import pandas as pd
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#Global values
root = '../datasets'
seed = 422
random.seed(seed)
np.random.seed(seed)

def load_balance_data():
    balance_title = []
    for i in range(0, 4):
        balance_title.append('feature%02.i'%i)
    balance_title.append('label')
    balance_dir = os.path.join(root, 'balance.data')
    df = pd.read_csv(balance_dir,names=balance_title)
    return df

def load_wine_data():
    wine_title = []
    for i in range(0, 13):
        wine_title.append('feature%02.i' % i)
    wine_title.append('label')
    wine_dir = os.path.join(root, 'wine.data')
    df = pd.read_csv(wine_dir, names=wine_title)
    return df

def data_preprocessing(df):
    data_full = df.copy()
    data = data_full.copy().drop(['label'], axis=1)
    labels = data_full['label']
    return data,labels

def get_wine_data():
    wine_df = load_wine_data()
    w_values, w_labels = data_preprocessing(wine_df)
    return w_values.values,w_labels.values

def get_balance_data():
    balance_df = load_balance_data()
    b_values, b_labels = data_preprocessing(balance_df)

    return b_values.values, b_labels.values

if __name__ == "__main__":
    balance_values, balance_labels = get_balance_data()

    NB = GaussianNB()
    NB.fit(balance_values, balance_labels)
    original = NB.score(balance_values, balance_labels)
    print('NB Balance original Accuracy: %0.2f' % original)

    DT = DecisionTreeClassifier()
    DT.fit(balance_values, balance_labels)
    original = DT.score(balance_values, balance_labels)
    print('DT Balance original Accuracy: %0.2f' % original)

    wine_values, wine_labels = get_wine_data()

    NB = GaussianNB()
    NB.fit(wine_values, wine_labels)
    original = NB.score(wine_values, wine_labels)
    print('NB wine original Accuracy: %0.2f' % original)

    DT = DecisionTreeClassifier()
    DT.fit(wine_values, wine_labels)
    original = DT.score(wine_values, wine_labels)
    print('DT wine original Accuracy: %0.2f' % original)