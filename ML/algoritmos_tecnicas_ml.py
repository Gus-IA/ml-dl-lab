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
