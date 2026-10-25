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

# separar train y test

# train decisionTree

# create decisionTree

# train decisionTree por subconjunto

# modelo ensamblado
