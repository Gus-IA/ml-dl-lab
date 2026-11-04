from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.model_selection import train_test_split

# dataset mnist
mnist = fetch_openml("mnist_784", version=1)
mnist.target = mnist.target.astype(np.uint8)

X_train, X_test, y_train, y_test = train_test_split(
    mnist["data"].values, mnist["target"].values, random_state=42
)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

n_labeled = 50
X_train_n, y_train_n = X_train[:n_labeled], y_train[:n_labeled]
log_reg = OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=50))
log_reg.fit(X_train_n, y_train_n)
print(log_reg.score(X_test, y_test))

from sklearn.cluster import KMeans

k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)

representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(
        X_representative_digit.reshape(28, 28), cmap="binary", interpolation="bilinear"
    )
    plt.axis("off")
plt.show()

y_representative_digits = np.array(
    [
        2,
        1,
        9,
        4,
        7,
        6,
        0,
        3,
        6,
        5,
        2,
        6,
        7,
        5,
        0,
        9,
        3,
        7,
        4,
        9,
        0,
        9,
        1,
        2,
        3,
        6,
        3,
        1,
        6,
        0,
        5,
        7,
        5,
        9,
        8,
        4,
        7,
        0,
        0,
        8,
        0,
        1,
        4,
        3,
        0,
        8,
        2,
        9,
        6,
        6,
    ]
)

log_reg = OneVsRestClassifier(
    LogisticRegression(solver="lbfgs", max_iter=5000, random_state=4)
)
log_reg.fit(X_representative_digits, y_representative_digits)
print(log_reg.score(X_test, y_test))

y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

X_prop = np.r_[X_representative_digits, X_train[:1000]]
y_prop = np.r_[y_representative_digits, y_train_propagated[:1000]]

log_reg = OneVsRestClassifier(LogisticRegression(solver="lbfgs", max_iter=5000))
log_reg.fit(X_prop, y_prop)
print(log_reg.score(X_test, y_test))

percentile_closest = 20

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = kmeans.labels_ == i
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = X_cluster_dist > cutoff_distance
    X_cluster_dist[in_cluster & above_cutoff] = -1

partially_propagated = X_cluster_dist != -1
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

X_prop2 = np.r_[X_representative_digits, X_train_partially_propagated[:1000]]
y_prop2 = np.r_[y_representative_digits, y_train_partially_propagated[:1000]]

log_reg = OneVsRestClassifier(
    LogisticRegression(solver="lbfgs", max_iter=5000, random_state=42)
)
log_reg.fit(X_prop2, y_prop2)
print(log_reg.score(X_test, y_test))

k = 50
probas = log_reg.predict_proba(X_prop2)
labels_ixs = np.argmax(probas, axis=1)
labels = np.array([proba[ix] for proba, ix in zip(probas, labels_ixs)])
sorted_ixs = np.argsort(labels)
print(labels[sorted_ixs[:k]])

X_lowest = X_prop2[sorted_ixs[:k]]
plt.figure(figsize=(15, 4))
for index, img in enumerate(X_lowest):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(img.reshape(28, 28), cmap="binary", interpolation="bilinear")
    plt.axis("off")
plt.show()

y_lowest = np.array(
    [
        7,
        7,
        6,
        7,
        7,
        5,
        1,
        7,
        7,
        4,
        4,
        3,
        7,
        5,
        2,
        9,
        9,
        5,
        4,
        7,
        2,
        0,
        4,
        5,
        9,
        9,
        3,
        3,
        7,
        1,
        1,
        4,
        5,
        9,
        2,
        5,
        6,
        3,
        3,
        5,
        0,
        7,
        5,
        9,
        3,
        4,
        3,
        4,
        7,
        1,
    ]
)

y_prop2_active = y_prop2.copy()
y_prop2_active[sorted_ixs[:k]] = y_lowest

log_reg = OneVsRestClassifier(
    LogisticRegression(solver="lbfgs", max_iter=5000, random_state=42)
)
log_reg.fit(X_prop2, y_prop2_active)
print(log_reg.score(X_test, y_test))
