import pandas as pd


def load_data(x):
    m = pd.read_csv(x, sep=';')
    m.head()
    return m


def into_dataframe(x):
    m = pd.DataFrame(x)
    return m


def get_features(x, y):
    m = x.drop(y, axis=1)
    return m


def get_labels(x, y):
    m = x.pop(y)
    return m