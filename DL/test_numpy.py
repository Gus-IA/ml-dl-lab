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


def fast(n=1000):
    p = np.linspace(-1, 1, n)
    x, y = np.meshgrid(p, p)
    z = np.sqrt(x**2 + y**2)
    return x, y, z


# vectorización

# calcular valor medio de cada columna
print(a.mean(axis=0))

a = np.arange(5).reshape(1, 1, 5)

print(a.shape)

b = np.arange(5)

print(b.shape)

print(a + b)

a = np.arange(6).reshape(2, 3)

print(a.shape)

b = np.array([[100], [200]])

print(b.shape)

print(a + b)

# multiplicación

A = np.array([[1, 2, 1], [0, 1, 0], [2, 3, 4]])

B = np.array([[2, 5], [6, 7], [1, 8]])

C = A.dot(B)

C = A @ B

print(C.shape)

import torch

# array numpy
x = np.array([1, 2, 3])

# tensor pytorch
x = torch.tensor([1, 2, 3])
