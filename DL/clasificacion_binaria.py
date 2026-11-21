import numpy as np
import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# funciones de pérdida


def mse(y, y_hat):
    return np.mean((y_hat - y) ** 2)


def bce(y, y_hat):
    return -np.mean(y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))


# funciones de activación


def linear(x):
    return x


def step(x):
    return x > 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Perceptrón


class Perceptron:
    def __init__(self, size, activation, loss):
        self.w = np.random.randn(size)
        self.ws = []
        self.activation = activation
        self.loss = loss

    def __call__(self, w, x):
        return self.activation(np.dot(x, w))

    def fit(self, x, y, epochs, lr):
        x = np.c_[np.ones(len(x)), x]
        for epoch in range(epochs):
            # Batch Gradient Descent
            y_hat = self(self.w, x)
            # función de pérdida
            l = self.loss(y, y_hat)
            # derivadas
            dldh = y_hat - y
            dhdw = x
            dldw = np.dot(dldh, dhdw)
            # actualizar pesos
            self.w = self.w - lr * dldw
            # guardar pesos para animación
            self.ws.append(self.w.copy())


from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(int)  # clsificación binaria

print(X.shape, y.shape)


plt.plot(X[y == 1, 0], X[y == 1, 1], "s", label="Iris Setosa")
plt.plot(X[y == 0, 0], X[y == 0, 1], "x", label="No Iris Setosa")
plt.grid()
plt.legend()
plt.xlabel("petal length", fontsize=14)
plt.ylabel("petal width", fontsize=14)
plt.title("Iris dataset", fontsize=14)
plt.show()


np.random.seed(42)

perceptron = Perceptron(3, sigmoid, bce)
epochs, lr = 20, 0.01
perceptron.fit(X, y, epochs, lr)

from matplotlib.colors import ListedColormap
from matplotlib import animation, rc

rc("animation", html="html5")


def plot(epoch, w):
    ax.clear()
    tit = ax.set_title(f"Epoch {epoch+1}", fontsize=14)
    axes = [0, 5, 0, 2]
    x0, x1 = np.meshgrid(
        np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
        np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    X_new = np.c_[np.ones(len(X_new)), X_new]
    y_predict = perceptron(w, X_new)
    zz = y_predict.reshape(x0.shape)

    ax.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris-Setosa")
    ax.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris-Setosa")
    custom_cmap = ListedColormap(["#9898ff", "#fafab0"])

    ax.contourf(x0, x1, zz, cmap=custom_cmap)
    ax.set_xlabel("Petal length", fontsize=14)
    ax.set_ylabel("Petal width", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.axis(axes)
    plt.close()
    return [tit]


def get_anim(fig, ax, ws):
    def anim(i):
        return plot(i, ws[i])

    return anim


fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111, autoscale_on=False)

animate = get_anim(fig, ax, perceptron.ws)

anim = animation.FuncAnimation(
    fig, animate, frames=len(perceptron.ws), interval=100, blit=False, repeat=False
)

plt.show()


w = perceptron.ws[-1]
print(w)

x_new = [1, 2, 0.5]
y_pred = perceptron(w, x_new)
y_pred  # Iris Setosa

x_new = [1, 1, 0.5]
y_pred = perceptron(w, x_new)
y_pred  # Iris Setosa


x_new = [1, 3, 0.5]
y_pred = perceptron(w, x_new)
y_pred  # No Iris Setosa


x_new = [1, 4, 0.5]
y_pred = perceptron(w, x_new)
y_pred  # No Iris Setosa
