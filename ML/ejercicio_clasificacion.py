import requests
import zipfile


# descargamos el dataset
def getData(url, path):
    r = requests.get(url)
    with open(path, "wb") as f:
        f.write(r.content)
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall()


getData(
    "https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/titanic.zip",
    "titanic.zip",
)

import pandas as pd

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.head()

# missing values

# data exploration

# data process
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X], index=X.columns
        )
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# Support vector classifier

# Random forest classifier

# compare
