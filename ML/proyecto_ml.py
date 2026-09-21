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
