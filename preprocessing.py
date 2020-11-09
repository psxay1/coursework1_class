import data_loader as dl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

df = dl.load_data('data/bank-full.csv')  # import data file
df.head()
df0 = dl.into_dataframe(df)
pd.set_option('display.max_columns', None)


df0_features = dl.get_features(df0, 'y')


df0_label = dl.get_labels(df0, 'y')


def encoder(m, column):
    for col in column:
        if m[col].dtype == np.dtype('object'):
            d = pd.get_dummies(m[col], prefix=col)
            d.head()
            m = pd.concat([m, d], axis=1)
            # drop the encoded column
            m.drop([col], axis=1, inplace=True)
    return m


x = df0_features

y = df0_label


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.45, random_state=45)
scaler = MinMaxScaler()
x_train = encoder(x_train, df0_features.head(0))
x_test = encoder(x_test, df0_features.head(0))
x_train[['balance', 'duration', 'pdays', 'previous']] = scaler.fit_transform(x_train[['balance', 'duration', 'pdays', 'previous']])
x_test[['balance', 'duration',  'pdays', 'previous']] = scaler.fit_transform(x_test[['balance', 'duration', 'pdays', 'previous']])

encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(pd.DataFrame(y_train))

# Dummy Transformation

y_train = encoder.transform(pd.DataFrame(y_train)).toarray()
y_test = encoder.transform(pd.DataFrame(y_test)).toarray()
s = x_train.shape[1]

print(x_train.shape)



