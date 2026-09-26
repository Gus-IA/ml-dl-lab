from sklearn.datasets import fetch_openml

# clasificación binaria

mnist = fetch_openml("mnist_784", version=1)
mnist.keys()

X, y = mnist["data"].values, mnist["target"].values
print(X.shape, y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt
import random

ix = random.randint(0, len(X) - 1)
some_digit = X[ix]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.title(y[ix])
plt.show()

# separamos en train y test

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# el modelo nos tiene que decir si la imagen es un 5
y_train_5 = y_train == "5"
y_test_5 = y_test == "5"

from sklearn.linear_model import SGDClassifier

# entrenamos un modelo SGD clasificador
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# mostramos una imagen aleatoria
ix = random.randint(0, len(X) - 1)
some_digit = X[ix]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")
plt.title(y[ix])
plt.show()
sgd_clf.predict([some_digit])
