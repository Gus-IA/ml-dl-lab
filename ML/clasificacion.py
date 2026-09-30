from sklearn.datasets import fetch_openml

# clasificación binaria

mnist = fetch_openml("mnist_784", version=1)
mnist.keys()

X, y = mnist["data"].values, mnist["target"].values
print(X.shape, y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt
import random

ix = random.randint(0, len(X) - 1)
some_digit = X[ix]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.title(y[ix])
plt.show()

# separamos en train y test

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# el modelo nos tiene que decir si la imagen es un 5
y_train_5 = y_train == "5"
y_test_5 = y_test == "5"

from sklearn.linear_model import SGDClassifier

# entrenamos un modelo SGD clasificador
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# mostramos una imagen aleatoria
ix = random.randint(0, len(X) - 1)
some_digit = X[ix]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.title(y[ix])
plt.show()
sgd_clf.predict([some_digit])

# métricas de clasificación

from sklearn.model_selection import cross_val_score

# validación cruzada
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

from sklearn.base import BaseEstimator
import numpy as np


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# matrix de confusión
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

import seaborn as sns
import pandas as pd

# Convert confusion matrix to DataFrame for better visualization
conf_matrix = confusion_matrix(y_train_5, y_train_pred)
conf_df = pd.DataFrame(
    conf_matrix, index=["False", "True"], columns=["Pred False", "Pred True"]
)

# Create a heatmap
sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)


# métricas de precisión y de recall
from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train_5, y_train_pred))

print(recall_score(y_train_5, y_train_pred))


# métricas de f1
from sklearn.metrics import f1_score

print(f1_score(y_train_5, y_train_pred))

# tradeoff
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)

threshold = 0
y_some_digit_pred = y_scores > threshold
print(y_some_digit_pred)

threshold = 8000
y_some_digit_pred = y_scores > threshold
print(y_some_digit_pred)

# clasificación multiclase

from sklearn.svm import SVC  # OvO

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])  # y_train, not y_train_5
svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores)

print(np.argmax(some_digit_scores))


from sklearn.multiclass import OneVsRestClassifier  # OvR

ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])
ovr_clf.predict([some_digit])

sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

some_digit_scores = sgd_clf.decision_function([some_digit])
print(some_digit_scores)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled[:3000], y_train[:3000], cv=3)
conf_mx = confusion_matrix(y_train[:3000], y_train_pred[:3000])

# dataframe de la matrix de confusión
conf_df = pd.DataFrame(
    conf_mx, index=[str(i) for i in range(10)], columns=[f"Pred {i}" for i in range(10)]
)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, fmt="d", cmap="Blues", cbar=True, square=True)
plt.title("Confusion Matrix for MNIST Classes")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# matriz de confusión normalizada
np.fill_diagonal(norm_conf_mx, 0)
plt.figure(figsize=(10, 8))
sns.heatmap(norm_conf_mx, cmap="gray_r", annot=True, fmt=".2f", cbar=True)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
