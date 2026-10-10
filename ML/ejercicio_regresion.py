from sklearn import datasets

# dataset iris con dos características
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

import numpy as np

X_with_bias = np.c_[np.ones([len(X), 1]), X]  # add the bias term (x0 = 1)

test_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
train_size = total_size - test_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]

X_train.shape, X_test.shape


# one hot encoding
def to_one_hot(y):
    one_hot = np.zeros((y.size, 3))
    # tu código aquí
    return one_hot


a = np.array([0, 1, 0, 1, 1, 1, 1, 0, 2, 0])
a_one_hot = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)

assert np.allclose(a_one_hot, to_one_hot(a))

Y_train_one_hot = to_one_hot(y_train)
Y_test_one_hot = to_one_hot(y_test)


# softmax


def softmax(logits):
    # tu código aquí
    return


a = np.array(
    [
        [-1.1005929, -4.40007828, -1.34103465],
        [-3.4269555, -11.18871295, -3.49347319],
        [-0.71891006, -3.55440292, -1.11117902],
    ]
)

a_softmax = np.array(
    [
        [5.48491412e-01, 2.02405142e-02, 4.31268074e-01],
        [5.16509697e-01, 2.19882186e-04, 4.83270420e-01],
        [5.76630772e-01, 3.38422251e-02, 3.89527003e-01],
    ]
)

assert np.allclose(a_softmax, softmax(a))

# entrenamiento


n_features = X_train.shape[1]  # == 3 (2 features plus the bias term)
n_outputs = len(np.unique(y_train))  # == 3 (3 iris classes)

eta = 0.01
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7

w = np.random.randn(n_features, n_outputs)

for iteration in range(n_iterations):
    pass

# visualizacion fronteras de decisión

# evaluación en test
