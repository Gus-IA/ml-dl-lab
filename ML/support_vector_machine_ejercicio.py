# get the data

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1)  # Iris virginica


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    # Distance between the decision boundary and each margin
    margin = 1 / np.linalg.norm(w)
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors="#FFAAAA")
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


# clasificador SVM
from sklearn.svm import SVC

C = 2
svm_clf = SVC(kernel="linear", C=C)
svm_clf.fit(X, y.ravel())

print(svm_clf.intercept_, svm_clf.coef_)

plt.figure(figsize=(5.5, 3.2))
yr = y.ravel()

plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs")

plot_svc_decision_boundary(svm_clf, 4, 6)

plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.title("SVC", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])
plt.show()


# SVM usando SGD Classifier
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(
    loss="hinge", alpha=0.017, max_iter=1000, tol=1e-3, random_state=42
)
sgd_clf.fit(X, y.ravel())


m = len(X)
t = y * 2 - 1
X_b = np.c_[np.ones((m, 1)), X]


# BGD
from sklearn.base import BaseEstimator


class MyLinearSVC(BaseEstimator):
    def __init__(self, C=1, eta=0.1, n_epochs=1000):
        self.C = C
        self.eta = eta
        self.n_epochs = n_epochs

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1, 1)

        # Convert labels from 0/1 to -1/+1
        t = y * 2 - 1

        w = np.random.randn(X.shape[1], 1)
        b = 0.0

        m = len(X)
        X_t = X * t

        self.Js = []
        self.intercepts = []
        self.coefs = []
        self.support_vectors = []
        self.epochs = []

        for epoch in range(1, self.n_epochs + 1):
            support_vectors_idx = ((X_t.dot(w) + t * b) < 1).ravel()

            X_t_sv = X_t[support_vectors_idx]
            t_sv = t[support_vectors_idx]

            # Hinge loss + regularization
            J = 0.5 * np.sum(w * w) + self.C * np.sum(1 - X_t_sv.dot(w) - t_sv * b)
            self.Js.append(float(J))

            # Gradients
            w_gradient_vector = w - self.C * np.sum(X_t_sv, axis=0).reshape(-1, 1)
            b_derivative = -self.C * np.sum(t_sv)

            # Gradient descent
            w = w - self.eta * w_gradient_vector
            b = b - self.eta * b_derivative

            if epoch % 1000 == 0:
                self.epochs.append(epoch)
                self.intercepts.append(np.array([b]))
                self.coefs.append(w.T.copy())

                support_vectors_idx = ((X_t.dot(w) + t * b) < 1).ravel()
                self.support_vectors.append(X[support_vectors_idx])

        # Store parameters in sklearn-compatible format
        self.intercept_ = np.array([b])
        self.coef_ = w.T.copy()

        support_vectors_idx = ((X_t.dot(w) + t * b) < 1).ravel()
        self.support_vectors_ = X[support_vectors_idx]

        return self

    def decision_function(self, X):
        return X.dot(self.coef_.T).ravel() + self.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)


svm_clf = MyLinearSVC(C=100, eta=0.0001, n_epochs=50000)
svm_clf.fit(X, y)

plt.plot(range(1, svm_clf.n_epochs + 1), svm_clf.Js)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Training cost")
plt.show()

print(svm_clf.intercept_, svm_clf.coef_)

plt.figure(figsize=(5.5, 3.2))
plt.plot(X[:, 0][yr == 1], X[:, 1][yr == 1], "g^", label="Iris virginica")
plt.plot(X[:, 0][yr == 0], X[:, 1][yr == 0], "bs", label="Not Iris virginica")

plot_svc_decision_boundary(svm_clf, 4, 6)

plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.title("MyLinearSVC", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])
plt.legend(loc="upper left")
plt.show()
