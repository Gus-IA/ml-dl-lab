# regresión lineal
import numpy as np

# generamos números aleatorios en X e y
X = 2 * np.random.rand(100, 1)
y = 3 * X + np.random.rand(100, 1)

import matplotlib.pyplot as plt

# mostramos
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, -3, 10])
plt.show()

# añadimos x0 = 1
X_b = np.c_[np.ones((100, 1)), X]
w_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # pesos óptimos
print(w_best)

# mostramos
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(w_best)
plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 2, -3, 10])
plt.show()


# regresión lineal
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()  # instanciamos modelo
lin_reg.fit(X, y)  # entrenamos
print(lin_reg.intercept_, lin_reg.coef_)  # pesos


# descenso del gradiente
x = np.random.rand(20)
y = 2 * x + (np.random.rand(20) - 0.5) * 0.5

# mostramos
plt.plot(x, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.show()

from matplotlib import animation, rc

rc("animation", html="html5")


def init_fig(x, t, ws, cost_ws):
    """Initialise figure"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax2.plot(x, t, "bo", label="target: t")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 2])
    ax2.set_xlabel("input: $x$", fontsize=15)
    ax2.set_ylabel("target: $t$", fontsize=15)
    ax2.yaxis.set_label_position("right")
    ax2.set_title("Labelled data & model output", fontsize=18)
    (line2,) = ax2.plot([], [], "k-", label="fitted line: $y=x*p$")
    ax2.legend(loc=2)
    # Cost function plot
    ax1.plot(ws, cost_ws, "r-", label="cost")
    ax1.set_ylim([-2, 8])
    ax1.set_xlim([1, 3])
    ax1.set_xlabel("parameter: $p$", fontsize=15)
    ax1.set_ylabel("cost: $\sum |t-y|^2$", fontsize=15)
    cost_text = ax1.set_title("Cost at step {}".format(0), fontsize=18)
    (line1,) = ax1.plot([], [], "k:", label="derivative at $p$")
    (pc_dots,) = ax1.plot([], [], "ko")
    ax1.legend(loc=2)
    return fig, ax1, ax2, line1, line2, pc_dots, cost_text


def gradient(w, x, t):
    return np.sum(2.0 * x * (x * w - t))


def cost(y, t):
    return ((t - y) ** 2).sum()


ws = np.linspace(0, 4, num=100)
cost_ws = np.vectorize(lambda w: cost(x * w, y))(ws)
fig, ax1, ax2, line1, line2, pc_dots, cost_text = init_fig(x, y, ws, cost_ws)
plt.show()

# descenso por gradiente estocástico

w = 1
lr = 0.1
epochs = 2
weights = [(w, gradient(w, x, y), cost(x * w, y))]
N = x.shape[0]
ixs = np.arange(N)
for i in range(epochs):
    np.random.shuffle(ixs)
    for ix in ixs:
        _x, _y = x[ix], y[ix]
        dw = gradient(w, _x, _y)
        w = w - lr * dw
        weights.append((w, dw, cost(_x * w, _y)))

fig, ax1, ax2, line1, line2, pc_dots, cost_text = init_fig(x, y, ws, cost_ws)
plt.show()

# descenso por gradiente mini-lotes

w = 1
lr = 0.01
epochs = 10
batch_size = 10
weights = [(w, gradient(w, x, y), cost(x * w, y))]
ixs = np.arange(x.shape[0])
batches = x.shape[0] // batch_size
for i in range(epochs):
    np.random.shuffle(ixs)
    for i in range(batches):
        _x, _y = (
            x[ixs[i * batch_size : (i + 1) * batch_size]],
            y[ixs[i * batch_size : (i + 1) * batch_size]],
        )
        dw = gradient(w, _x, _y)
        w = w - lr * dw
        weights.append((w, dw, cost(_x * w, _y)))

fig, ax1, ax2, line1, line2, pc_dots, cost_text = init_fig(x, y, ws, cost_ws)
plt.show()

# regresión polinómica
import numpy.random as rnd

# generamos datos sintéticos aleatorios
np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.rand(m, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()


from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0], X_poly[0]


# modelo polinómico
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0], X_poly[0]

# creamos la regresión lineal con los nuevos datos
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

# visualizamos el resultado
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.axis([-3, 3, 0, 10])
plt.show()


# visualizamos los 3 modelos
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

for style, width, degree in (("g-", 1, 40), ("b--", 2, 2), ("r-+", 2, 1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline(
        [
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ]
    )
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

plt.plot(X, y, "b.", linewidth=3)
plt.legend(loc="upper left")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([-3, 3, 0, 10])
plt.show()


# regresión logística
t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])
plt.show()

from sklearn import datasets

# cargamos el dataset iris
iris = datasets.load_iris()

X = iris["data"][:, 3:]  # ancho pétalo
y = (iris["target"] == 2).astype(int)  # clase 2

from sklearn.linear_model import LogisticRegression

# instancia modelo regresión Logística
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X, y)  # entrenamiento

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

# visualizamos los resultados
plt.figure(figsize=(8, 3))
plt.plot(X[y == 0], y[y == 0], "bs")
plt.plot(X[y == 1], y[y == 1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
plt.text(
    decision_boundary + 0.02,
    0.15,
    "Decision  boundary",
    fontsize=14,
    color="k",
    ha="center",
)
plt.annotate(
    "",
    xy=(decision_boundary - 0.3, 0.08),
    xytext=(decision_boundary, 0.08),
    arrowprops=dict(facecolor="b", edgecolor="b", width=2, headwidth=10, headlength=10),
)
plt.annotate(
    "",
    xy=(decision_boundary + 0.3, 0.92),
    xytext=(decision_boundary, 0.92),
    arrowprops=dict(facecolor="g", edgecolor="g", width=2, headwidth=10, headlength=10),
)
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()


from sklearn.linear_model import LogisticRegression

# entrenamos un segundo modelo con el ancho y largo del pétalo
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(int)

log_reg = LogisticRegression(
    solver="lbfgs", C=10**10, random_state=42
)  # instancia modelo
log_reg.fit(X, y)  # entrenamiento


from matplotlib.colors import ListedColormap


# visualizamos el resultado
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor="black",
        )

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c="",
            edgecolor="black",
            alpha=1.0,
            linewidth=1,
            marker="o",
            s=100,
            label="test set",
        )


plot_decision_regions(X, y, log_reg)
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.legend(loc="upper left")
plt.show()
