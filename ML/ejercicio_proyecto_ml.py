import requests
import pandas as pd

# download the data


def getData(url, path):
    r = requests.get(url)
    with open(path, "wb") as f:
        f.write(r.content)


getData(
    "https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/housing_train.csv",
    "housing_train.csv",
)
getData(
    "https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/housing_test.csv",
    "housing_test.csv",
)

data = pd.read_csv("housing_train.csv")
data, labels = (
    data.drop(["median_house_value"], axis=1),
    data["median_house_value"].copy(),
)
num_attribs = list(data.drop(["ocean_proximity"], axis=1))
cat_attribs = ["ocean_proximity"]

test_data = pd.read_csv("housing_test.csv")
test_data, test_labels = (
    test_data.drop("median_house_value", axis=1),
    test_data["median_house_value"].copy(),
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

# prepare the data
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ]
)

data_prepared = full_pipeline.fit_transform(data)
test_data_prepared = full_pipeline.transform(test_data)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# modelo support vector machine
param_grid = [
    {"kernel": ["linear"], "C": [1.0, 10.0, 1000.0]},
    {"kernel": ["rbf"], "C": [1.0, 10.0, 100.0]},
    {"gamma": [0.01, 0.1, 1.0]},
]

svm_reg = SVR()
grid_search = GridSearchCV(
    svm_reg, param_grid, cv=3, scoring="neg_mean_squared_error", verbose=2
)
grid_search.fit(data_prepared, labels)


negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
print(rmse)

best_params = grid_search.best_params_
print(best_params)


# reemplazar gridsearchcv -> randomizedsearchcv
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

param_distribs = {
    "kernel": ["linear", "rbf"],
    "C": reciprocal(20, 200000),
    "gamma": expon(scale=1.0),
}

svm_reg = SVR()
rnd_search = RandomizedSearchCV(
    svm_reg,
    param_distributions=param_distribs,
    n_iter=5,
    cv=5,
    scoring="neg_mean_squared_error",
    verbose=2,
    random_state=42,
)
rnd_search.fit(data_prepared, labels)

negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
print(rmse)

best_params = rnd_search.best_params_
print(best_params)

feature_importances = np.array(
    [
        7.33442355e-02,
        6.29090705e-02,
        4.11437985e-02,
        1.46726854e-02,
        1.41064835e-02,
        1.48742809e-02,
        1.42575993e-02,
        3.66158981e-01,
        5.64191792e-02,
        1.08792957e-01,
        5.33510773e-02,
        1.03114883e-02,
        1.64780994e-01,
        6.02803867e-05,
        1.96041560e-03,
        2.85647464e-03,
    ]
)


# pipeline única
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]


k = 5

preparation_and_feature_selection_pipeline = Pipeline(
    [
        ("preparation", full_pipeline),
        ("feature_selection", TopFeatureSelector(feature_importances, k)),
    ]
)

data_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(
    data
)

data_prepared_top_k_features[0:3]

prepare_select_and_predict_pipeline = Pipeline(
    [
        ("preparation", full_pipeline),
        ("feature_selection", TopFeatureSelector(feature_importances, k)),
        ("svm_reg", SVR(**rnd_search.best_params_)),
    ]
)

prepare_select_and_predict_pipeline.fit(data, labels)

some_data = data.iloc[:4]
some_labels = labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))

# búsqueda de hiperparámetros de transformación
param_grid = [
    {
        "preparation__num__imputer__strategy": ["mean", "median", "most_frequent"],
        "feature_selection__k": list(range(1, len(feature_importances) + 1)),
    }
]

grid_search_prep = GridSearchCV(
    prepare_select_and_predict_pipeline,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    verbose=2,
)

grid_search_prep.fit(data, labels)

print(grid_search_prep.best_params_)
