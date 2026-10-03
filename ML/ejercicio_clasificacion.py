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
train_data.info()

# data exploration
train_data.describe()

# data process
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline(
    [
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

num_pipeline.fit_transform(train_data)


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X], index=X.columns
        )
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline(
    [
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder()),
    ]
)

cat_pipeline.fit_transform(train_data)


from sklearn.pipeline import FeatureUnion

preprocess_pipeline = FeatureUnion(
    transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ]
)

X_train = preprocess_pipeline.fit_transform(train_data)
print(X_train)

# Support vector classifier
y_train = train_data["Survived"]

from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)

X_test = preprocess_pipeline.transform(test_data)
y_pred = svm_clf.predict(X_test)

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()

# Comparar ambos clasificadores
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([1] * 10, svm_scores, ".")
plt.plot([2] * 10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy, fontsize=14")
plt.show()
