import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10)
y = x

plt.plot(x, y)
plt.grid(True)
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x) = x", fontsize=14)
plt.title("Función de activación lineal", fontsize=14)
plt.plot(x, np.zeros(len(x)), "--k")
plt.show()

np.random.seed(42)

x = np.random.rand(20)
y = 2 * x + (np.random.rand(20) - 0.5) * 0.5

plt.plot(x, y, "b.")
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$y$", rotation=0, fontsize=14)
plt.grid(True)
plt.show()

plt.plot(x, y, "b.")
plt.plot(x, 2 * x, "k")
plt.plot(0.5, 2 * 0.5, "sr")
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$y$", rotation=0, fontsize=14)
plt.grid(True)
plt.show()
