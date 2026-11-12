# valor escalar
X = 1

# vector
X = [1, 2, 3, 4]
print(X)

# matriz
A = [[1, 2], [3, 4]]
print(A)

# tensor 0 dim
X = 1
print(X)

# tensor 1 dim
X = [1, 2, 3, 4]
print(X)

# tensor 2 dim
A = [[1, 2], [3, 4]]

import numpy as np

a = np.array(X)
print(a)

a = np.array(A)
print(a)

l = [i for i in range(10000000)]
a = np.array(l)
print(a)

a2 = 2 * a


# crear vector ceros
np.zeros(5)

# matriz ceros
np.zeros((3, 4))

# tensor unos
a = np.ones((3, 4, 2))
print(a)

# vector de int en rango
np.arange(1, 5)

# vector de float en rango
np.linspace(0, 1, 10)

# matriz de número aleatorios
np.random.rand(3, 4)

# número elementos de cada dimensión
a.shape
print(a)

# longitud array
print(a.ndim)

# elementos totales en el array
print(a.size)

# tipo de datos
print(a.dtype)

# indicar inicialización
a = np.arange(1, 5, dtype=np.uint8)
