import requests

# Descargar dataset
url = "https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/country_stats.csv"

response = requests.get(url)
response.raise_for_status()

with open("country_stats.csv", "wb") as f:
    f.write(response.content)

print("Archivo descargado correctamente")


import pandas as pd

# Mostrar datos
data = pd.read_csv("country_stats.csv", index_col="Country")
print(data)

import matplotlib.pyplot as plt

# Visualizar datos
data.plot(kind="scatter", x="GDP per capita", y="Life satisfaction")
plt.show()


from sklearn import linear_model
import numpy as np

# Convertimos los datos en array numpy
# y los entrenamos usando el linear model
# para encontrar el mejor coeficiente

lin1 = linear_model.LinearRegression()
Xsample = np.c_[data["GDP per capita"]]
ysample = np.c_[data["Life satisfaction"]]
lin1.fit(Xsample, ysample)
t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
print(t0, t1)

# Visualización de datos

data.plot(kind="scatter", x="GDP per capita", y="Life satisfaction", figsize=(7, 5))
plt.xlabel("GDP per capita (USD)")
plt.axis([0, 60000, 0, 10])
X = np.linspace(0, 60000, 1000)
plt.plot(X, t0 + t1 * X, "b")
plt.show()

# Nueva predición con una nueva cifra
cyprus_gdp_per_capita = 22587
cyprus_predicted_life_satisfaction = lin1.predict([[cyprus_gdp_per_capita]])[0][0]
cyprus_predicted_life_satisfaction


# Visualizamos la nueva predición

data.plot(kind="scatter", x="GDP per capita", y="Life satisfaction", figsize=(7, 5))
plt.xlabel("GDP per capita (USD)")
X = np.linspace(0, 60000, 1000)
plt.plot(X, t0 + t1 * X, "b")
plt.axis([0, 60000, 0, 10])
plt.show()

# descargamos un dataset con datos que no reflejan la realidad

url = "https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/missing_data.csv"

response = requests.get(url)
response.raise_for_status()

with open("missing_data.csv", "wb") as f:
    f.write(response.content)

print("Archivo descargado correctamente")


missing_data = pd.read_csv("missing_data.csv", index_col="Country")
print(missing_data)

# visualizamos los nuevos datos


data.plot(kind="scatter", x="GDP per capita", y="Life satisfaction", figsize=(8, 3))
plt.axis([0, 110000, 0, 10])

for country in missing_data.iterrows():
    plt.plot(country[1]["GDP per capita"], country[1]["Life satisfaction"], "rs")

X = np.linspace(0, 110000, 1000)
plt.plot(X, t0 + t1 * X, "b:")

lin_reg_full = linear_model.LinearRegression()
Xfull = np.c_[np.r_[data["GDP per capita"], missing_data["GDP per capita"]]]
yfull = np.c_[np.r_[data["Life satisfaction"], missing_data["Life satisfaction"]]]
lin_reg_full.fit(Xfull, yfull)

t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]
X = np.linspace(0, 110000, 1000)
plt.plot(X, t0full + t1full * X, "k")
plt.xlabel("GDP per capita (USD)")

plt.show()


# Overfitting


full_data = pd.concat([data, missing_data])

full_data.plot(
    kind="scatter", x="GDP per capita", y="Life satisfaction", figsize=(8, 3)
)
plt.axis([0, 110000, 0, 10])

from sklearn import preprocessing
from sklearn import pipeline

poly = preprocessing.PolynomialFeatures(degree=20, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression()

pipeline_reg = pipeline.Pipeline([("poly", poly), ("scal", scaler), ("lin", lin_reg2)])
pipeline_reg.fit(Xfull, yfull)
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve)
plt.xlabel("GDP per capita (USD)")
plt.show()
