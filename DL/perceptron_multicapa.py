import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)


def reluPrime(x):
    return x > 0


# Mean Square Error -> usada para regresión (con activación lineal)
def mse(y, y_hat):
    return np.mean((y_hat - y.reshape(y_hat.shape)) ** 2)


# solo si usamos activación lineal
def grad_mse(y, y_hat):
    return y_hat - y.reshape(y_hat.shape)


class MLP:
    def __init__(self, D_in, H, D_out):
        self.w1, self.b1 = np.random.normal(
            loc=0.0, scale=np.sqrt(2 / (D_in + H)), size=(D_in, H)
        ), np.zeros(H)
        self.w2, self.b2 = np.random.normal(
            loc=0.0, scale=np.sqrt(2 / (H + D_out)), size=(H, D_out)
        ), np.zeros(D_out)
        self.ws = []
        self.loss = mse
        self.grad_loss = grad_mse

    def __call__(self, x):
        self.h_pre = np.dot(x, self.w1) + self.b1
        self.h = relu(self.h_pre)
        y_hat = np.dot(self.h, self.w2) + self.b2
        return y_hat

    def fit(
        self, X, Y, epochs=100, lr=0.001, batch_size=None, verbose=True, log_each=1
    ):
        batch_size = len(X) if batch_size == None else batch_size
        batches = len(X) // batch_size
        l = []
        for e in range(1, epochs + 1):
            # Mini-Batch Gradient Descent
            _l = []
            for b in range(batches):
                x = X[b * batch_size : (b + 1) * batch_size]
                y = Y[b * batch_size : (b + 1) * batch_size]
                y_pred = self(x)
                loss = self.loss(y, y_pred)
                _l.append(loss)
                # Backprop
                dldy = self.grad_loss(y, y_pred)
                grad_w2 = np.dot(self.h.T, dldy)
                grad_b2 = dldy.mean(axis=0)
                dldh = np.dot(dldy, self.w2.T) * reluPrime(self.h_pre)
                grad_w1 = np.dot(x.T, dldh)
                grad_b1 = dldh.mean(axis=0)
                # Update (GD)
                self.w1 = self.w1 - lr * grad_w1
                self.b1 = self.b1 - lr * grad_b1
                self.w2 = self.w2 - lr * grad_w2
                self.b2 = self.b2 - lr * grad_b2
            l.append(np.mean(_l))
            self.ws.append(
                (self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())
            )
            if verbose and not e % log_each:
                print(f"Epoch: {e}/{epochs}, Loss: {np.mean(l):.5f}")

    def predict(self, ws, x):
        w1, b1, w2, b2 = ws
        h = relu(np.dot(x, w1) + b1)
        y_hat = np.dot(h, w2) + b2
        return y_hat


m = 100
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + np.random.randn(m, 1)

plt.plot(x, y, "b.")
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$y$", rotation=0, fontsize=14)
plt.grid(True)
plt.show()

model = MLP(D_in=1, H=5, D_out=1)
epochs, lr = 50, 0.0002
model.fit(x, y, epochs, lr, batch_size=1, log_each=10)

N = 3
fig, axs = plt.subplots(N, 1, figsize=(4, 8))
for n in range(N):
    if n == 0 or n == N - 1:
        t = n + 1 if n == 0 else len(model.ws) - 1
    else:
        t = int(n * len(model.ws) / (N - 1))
    ax = axs[n]
    ax.plot(x, y, "b.")
    _x_new = np.linspace(-3, 3, 100)
    x_new = _x_new.reshape(len(_x_new), 1)
    w = model.ws[t]
    y_pred = model.predict(w, x_new)
    ax.plot(_x_new, y_pred, "-k")
    ax.set_xlabel("$x_1$", fontsize=14)
    ax.set_ylabel("$y$", rotation=0, fontsize=14)
    ax.set_title(f"Iteración: {t}")
    ax.grid(True)
    ax.axis([-3, 3, 0, 10])
plt.tight_layout()
plt.show()
