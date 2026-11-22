import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)


# aplica softmax + cross entropy de manera estable


def crossentropy(y, y_hat):
    logits = y_hat[np.arange(len(y_hat)), y]
    entropy = -logits + np.log(np.sum(np.exp(y_hat), axis=-1))
    return entropy.mean()


# solo si usamos softmax
def grad_crossentropy(y, y_hat):
    answers = np.zeros_like(y_hat)
    answers[np.arange(len(y_hat)), y] = 1
    return (-answers + softmax(y_hat)) / y_hat.shape[0]


class Perceptron:
    def __init__(self, inputs, outputs, activation, loss, grad_loss):
        inputs = inputs + 1
        self.w = np.random.normal(
            loc=0.0, scale=np.sqrt(2 / (inputs + outputs)), size=(inputs, outputs)
        )
        self.ws = []
        self.activation = activation
        self.loss = loss
        self.grad_loss = grad_loss

    def __call__(self, w, x):
        return self.activation(np.dot(x, w))

    def fit(self, x, y, epochs, lr, batch_size=None, verbose=True, log_each=1):
        if batch_size == None:
            batch_size = len(x)
        x = np.c_[np.ones(len(x)), x]
        batches = len(x) // batch_size
        for epoch in range(1, epochs + 1):
            # Mini-Batch Gradient Descent
            for b in range(batches):
                _x = x[b * batch_size : (b + 1) * batch_size]
                _y = y[b * batch_size : (b + 1) * batch_size]
                y_hat = self(self.w, _x)
                # print(y_hat.shape)
                # función de pérdida
                l = self.loss(_y, y_hat)
                # derivadas
                dldh = self.grad_loss(_y, y_hat)
                dhdw = _x
                dldw = np.dot(dhdw.T, dldh)
                # actualizar pesos
                self.w = self.w - lr * dldw
            # guardar pesos para animación
            self.ws.append(self.w.copy())
            # print loss
            if verbose and not epoch % log_each:
                print(f"Epoch {epoch}/{epochs} Loss {l}")

    def predict(self, x):
        x = np.c_[np.ones(len(x)), x]
        return self(self.w, x)


# Mean Square Error -> usada para regresión (con activación lineal)
def mse(y, y_hat):
    return np.mean((y_hat - y.reshape(y_hat.shape)) ** 2)


# Binary Cross Entropy -> usada para clasificación binaria (con sigmoid)
def bce(y, y_hat):
    return -np.mean(
        y.reshape(y_hat.shape) * np.log(y_hat)
        - (1 - y.reshape(y_hat.shape)) * np.log(1 - y_hat)
    )


# solo si usamos activación lineal
def grad_mse(y, y_hat):
    return y_hat - y.reshape(y_hat.shape)


# solo si utilizamos sigmoid
def grad_bce(y, y_hat):
    return y_hat - y.reshape(y_hat.shape)


def linear(x):
    return x


def step(x):
    return x > 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinearRegression(Perceptron):
    def __init__(self, inputs, outputs=1):
        super().__init__(inputs, outputs, linear, mse, grad_mse)


class LinearClassification(Perceptron):
    def __init__(self, inputs, outputs=1):
        super().__init__(inputs, outputs, step, mse, grad_mse)


class LogisticRegression(Perceptron):
    def __init__(self, inputs, outputs=1):
        super().__init__(inputs, outputs, sigmoid, bce, grad_bce)


class SoftmaxRegression(Perceptron):
    def __init__(self, inputs, outputs):
        # usamos activación lineal porque `crossentropy` ya incluye la softmax
        super().__init__(inputs, outputs, linear, crossentropy, grad_crossentropy)


iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_norm = (X - X_mean) / X_std  # normalización

y = iris.target

print(X.shape, y.shape)

plt.plot(X[y == 0, 0], X[y == 0, 1], "s", label="Iris Setosa")
plt.plot(X[y == 1, 0], X[y == 1, 1], "x", label="Iris Versicolor")
plt.plot(X[y == 2, 0], X[y == 2, 1], "o", label="Iris Virginica")
plt.grid()
plt.legend()
plt.xlabel("petal length", fontsize=14)
plt.ylabel("petal width", fontsize=14)
plt.title("Iris dataset", fontsize=14)
plt.show()

perceptron = SoftmaxRegression(2, 3)
epochs, lr = 50, 1
perceptron.fit(X_norm, y, epochs, lr, log_each=10)


def plot_multiclass(perceptron, N=4):
    fig, axes = plt.subplots(2, N // 2, figsize=(10, 8), squeeze=False)
    for n in range(N):
        if n == 0 or n == N - 1:
            t = n + 1 if n == 0 else len(perceptron.ws) - 1
        else:
            t = int(n * len(perceptron.ws) / (N - 1))
        ax = axes[n // (N // 2), n % (N // 2)]
        resolution = 0.02
        ax.set_title(f"Epoch {t}", fontsize=14)
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
        X_new = (np.array([xx1.ravel(), xx2.ravel()]).T - X_mean) / X_std
        X_new = np.c_[np.ones(len(X_new)), X_new]
        w = perceptron.ws[t]
        Z = perceptron(w, X_new)
        Z = np.argmax(softmax(Z), axis=1)
        Z = Z.reshape(xx1.shape)
        ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        ax.set_xlim(xx1.min(), xx1.max())
        ax.set_ylim(xx2.min(), xx2.max())
        ax.set_xlabel("petal length", fontsize=12)
        ax.set_ylabel("petal width", fontsize=12)
        classes = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]
        for idx, cl in enumerate(np.unique(y)):
            ax.scatter(
                x=X[y == cl, 0],
                y=X[y == cl, 1],
                alpha=0.8,
                c=colors[idx],
                marker=markers[idx],
                label=classes[cl],
                edgecolor="black",
            )
        if n == N - 1:  # Only add legend to the last plot
            ax.legend(loc="upper left", fontsize=12)
    plt.tight_layout()
    plt.show()


plot_multiclass(perceptron)


#  nuevo punto
X_new = [[2, 0.5]]
# normalizamos
X_new_norm = (X_new - X_mean) / X_std
# salida del perceptron
y = perceptron.predict(X_new_norm)
print(y)

y_probas = softmax(y)
print(y_probas)

plt.bar(list(range(1, 4)), y_probas[0])
plt.grid(True)
plt.show()


y_probas.sum(axis=1)

y_pred = np.argmax(y_probas, axis=1)


def evaluate(x):
    x = (x - X_mean) / X_std
    y = perceptron.predict(x)
    y_probas = softmax(y)
    return np.argmax(y_probas, axis=1)


X = [[5, 1], [2, 0.5], [7, 3]]
evaluate(X)
