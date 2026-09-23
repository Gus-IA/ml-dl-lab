import requests
import tarfile
import pandas as pd

URL = "https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/housing.tgz"
PATH = "housing.tgz"


# descargamos el dataset y lo descomprimimos
def getData(url=URL, path=PATH):
    r = requests.get(url)
    r.raise_for_status()

    with open(path, "wb") as f:
        f.write(r.content)

    with tarfile.open(path) as housing_tgz:
        housing_tgz.extractall()


getData()


# cargamos el dataset y mostramos algunos datos
def loadData(path="housing.csv"):
    return pd.read_csv(path)


data = loadData()
print(data.sample(10))


from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# separamos los datos en train y test
# y visualizamos los datos
train, test = train_test_split(data, test_size=0.2, random_state=42)

train["median_income"].hist(bins=50)
plt.show()

test["median_income"].hist(bins=50)
plt.show()

data["income_cat"] = pd.cut(
    data["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)
data["income_cat"].hist()
plt.show()

# usamos el stratify para que la separación sea equitativa
train, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["income_cat"]
)

train["median_income"].hist(bins=50)
plt.show()

test["median_income"].hist(bins=50)
plt.show()

# eliminamos la variable para la estritificación
for set_ in (train, test):
    set_.drop("income_cat", axis=1, inplace=True)

# guardamos los datos de train y test
train.to_csv("housing_train.csv", index=False)
test.to_csv("housing_test.csv", index=False)

import matplotlib as mpl

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# cargamos el dataset de entrenamiento
data = loadData("housing_train.csv")
print(data.sample(10))

data.info()

# instancias de cada clase
data_counts = data["ocean_proximity"].value_counts()
print(data_counts)

# estadísticas para valores numéricos
print(data.describe())

# estadísticas numéricas en gráficos
data.hist(bins=50, figsize=(20, 15))
plt.show()

# gráfico de dispersión - scatter
data.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=data["population"] / 100,
    label="population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    sharex=False,
),
plt.legend()
plt.show()


# buscando correlaciones
corr_matrix = data.drop(columns=["ocean_proximity"]).corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# feature engineering

data["rooms_per_household"] = data["total_rooms"] / data["households"]
data["bedrooms_per_room"] = data["total_bedrooms"] / data["total_rooms"]
data["population_per_household"] = data["population"] / data["households"]

# buscando correlaciones
corr_matrix = data.drop(columns=["ocean_proximity"]).corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# preparación datos

data = pd.read_csv("housing_train.csv")
print(data.head())

data, labels = (
    data.drop(["median_house_value"], axis=1),
    data["median_house_value"].copy(),
)
print(data.head())

print(labels.head())

# limpieza datos

print(data[data.isnull().any(axis=1)])

data.dropna(subset=["total_bedrooms"])


# separamos variables numéricas y categóricas

data_num = data.drop(["ocean_proximity"], axis=1)
print(data_num.head())

data_cat = data[["ocean_proximity"]]
print(data_cat.head())


from sklearn.impute import SimpleImputer

# rellenamos los missing values con la mediana
imputer = SimpleImputer(strategy="median")
imputer.fit(data_num)
imputer.statistics_

print(data_num.median().values)

# mostramos los datos ya rellenados
X = imputer.transform(data_num)
print(X)


from sklearn.preprocessing import OrdinalEncoder

# transformar datos a valores numéricos
ordinal_encoder = OrdinalEncoder()
data_cat_encoded = ordinal_encoder.fit_transform(data_cat)
print(data_cat_encoded[:10])

print(ordinal_encoder.categories_)


# flujo de transformación de datos
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

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


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
data_extra_attribs = attr_adder.transform(data.values)


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# creamos la pipeline aplicando el imputer y el escalador
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        # ('attribs_adder', CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

data_num_tr = num_pipeline.fit_transform(data_num)

from sklearn.compose import ColumnTransformer

# creamos otra pipeline para transformar los valores numéricos
num_attribs = list(data_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer(
    [
        ("num", num_pipeline, num_attribs),
        # ("cat", OneHotEncoder(), cat_attribs),
    ]
)

data_prepared = full_pipeline.fit_transform(data)


# Selección del modelo

from sklearn.linear_model import LinearRegression

# instancia del modelo de regressión lineal con los datos y las etiquetas
lin_reg = LinearRegression()
lin_reg.fit(data_prepared, labels)

# hacemos algunas predicciones
some_data = data.iloc[:5]
some_labels = labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error

# hacemos predicciones con el mean squared error
predictions = lin_reg.predict(data_prepared)
lin_mse = mean_squared_error(labels, predictions)
lin_mse = np.sqrt(lin_mse)
print(lin_mse)


from sklearn.tree import DecisionTreeRegressor

# instancia de álbol de decisión
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(data_prepared, labels)  # entrenamiento

predictions = tree_reg.predict(data_prepared)
tree_mse = mean_squared_error(labels, predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

from sklearn.model_selection import cross_val_score

# Validación cruzada
scores = cross_val_score(
    tree_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10
)
tree_rmse_scores = np.sqrt(-scores)


# mostramos los resultados
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


display_scores(tree_rmse_scores)


lin_scores = cross_val_score(
    lin_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10
)
lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)


from sklearn.ensemble import RandomForestRegressor

# random forest
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(data_prepared, labels)
predictions = forest_reg.predict(data_prepared)
forest_mse = mean_squared_error(labels, predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)


# resultados
forest_scores = cross_val_score(
    forest_reg, data_prepared, labels, scoring="neg_mean_squared_error", cv=10
)
forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)

# Finetuning

from sklearn.model_selection import GridSearchCV

# búsqueda de hiperparámetros
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(data_prepared, labels)

# mejores parámetros
print(grid_search.best_params_)

# mejor modelo
print(grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

# importancia de las características
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)


# calculamos las métricas finales
test_data = pd.read_csv("housing_test.csv")

final_model = grid_search.best_estimator_

X_test = test_data.drop("median_house_value", axis=1)
y_test = test_data["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(final_rmse)
