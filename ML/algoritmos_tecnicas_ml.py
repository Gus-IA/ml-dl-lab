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
