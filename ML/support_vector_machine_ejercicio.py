# get the data

from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)  # Iris virginica

import matplotlib.pyplot as plt


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors="#FFAAAA")
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


# clasificador SVM

# SVM usando SGD Classifier

# BGD


from sklearn.base import BaseEstimator


class MyLinearSVC(BaseEstimator):
    def __init__(self, C=1, eta=0.1, n_epochs=1000):
        self.C = C
        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, y):
        # tu código aquí

        # PASOS
        # 1 - Inicializar pesos aleatorios
        # 2 - Bucle de entrenamiento
        #   2.1 - Calcular los vectores de soporte
        #   2.2 - Calcular la función de pérdida (hinge loss)
        #   2.3 - Calcular el gradiente de la función de pérdida con respecto a los parámetros
        #   2.4 - Actualizar los parámetros
        # 3 - Guardar los resultados finales en el formato de sklearn

        return self

    def decision_function(self, X):
        # tu código aquí
        return

    def predict(self, X):
        # tu código aquí
        return


svm_clf = MyLinearSVC(C=100, eta=0.0001, n_epochs=50000)
svm_clf.fit(X, y)
