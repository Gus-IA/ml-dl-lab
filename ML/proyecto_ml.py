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
