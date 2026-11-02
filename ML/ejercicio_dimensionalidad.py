from sklearn.datasets import fetch_openml
import numpy as np

# dataset mnist
mnist = fetch_openml("mnist_784", version=1)
mnist.target = mnist.target.astype(np.uint8)

X = mnist["data"].values
y = mnist["target"].values

X_train = mnist["data"][:60000]
y_train = mnist["data"][:60000]

X_test = mnist["data"][60000:]
y_test = mnist["data"][60000:]

# random forest classifier
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)

rnd_clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred)

# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)

# volver a entrenar con el dataset reducido

rnd_clf2 = RandomForestClassifier(n_estimators=100, random_state=42)

rnd_clf2.fit(X_train_reduced, y_train)

X_test_reduced = pca.transform(X_test)
y_pred = rnd_clf2.predict(X_test_reduced)
accuracy_score(y_test, y_pred)

# t-SNE
m = 1000
idx = np.random.permutation(60000)[:m]
X = mnist["data"].values[idx]
y = mnist["target"].values[idx]

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)

import matplotlib.pyplot as plt

plt.figure(figsize=(13, 10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis("off")
plt.colorbar()
plt.show()
