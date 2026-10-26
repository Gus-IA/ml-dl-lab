import numpy as np


def plot_moons(X, y):
    plt.plot(X[y == 0][:, 0], X[y == 0][:, 1], ".r")
    plt.plot(X[y == 1][:, 0], X[y == 1][:, 1], ".b")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    plt.title(f"Samples: 0 - {np.sum(y == 0)} 1 - {np.sum(y == 1)} ")


# generar datos
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

plot_moons(X, y)
plt.show()

# separar train y test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

plot_moons(X_train, y_train)
plt.show()

# train decisionTree GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

params = {"max_leaf_nodes": list(range(2, 100)), "min_samples_split": [2, 3, 4]}
grid_search_cv = GridSearchCV(
    DecisionTreeClassifier(random_state=42), params, verbose=1, cv=5, n_jobs=-1
)

grid_search_cv.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)

# create decisionTree
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []


rs = ShuffleSplit(n_splits=n_trees, train_size=100, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

import random

ix = random.randint(0, n_trees)
print(f"Tree {ix}")
plot_moons(mini_sets[ix][0], mini_sets[ix][1])

# train decisionTree por subconjunto
from sklearn.base import clone

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)

    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)

# modelo ensamblado
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

from scipy.stats import mode

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
