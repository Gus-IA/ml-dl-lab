from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml("mnist_784", version=1)
X, y = mnist["data"].values.astype(np.float32), mnist["target"].values.astype(int)

from torch.nn import Sequential as S
from torch.nn import Linear as L
from torch.nn import ReLU as R

model = S(L(784, 128), R(), L(128, 10))

from tqdm import tqdm
import torch

X_train, X_test = torch.from_numpy(X[:60000] / 255.0), torch.from_numpy(
    X[60000:] / 255.0
)
y_train, y_test = torch.from_numpy(y[:60000]), torch.from_numpy(y[60000:])
bs = 32
num_batches = len(X_train) // bs

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for b in tqdm(range(num_batches)):
        x = X_train[b * bs : (b + 1) * bs]
        y = y_train[b * bs : (b + 1) * bs]
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} loss: {loss.item():.3f}")


def evaluate(model, X_test, y_test):
    acc = 0
    with torch.no_grad():
        for b in range(num_batches):
            x = X_test[b * bs : (b + 1) * bs]
            y = y_test[b * bs : (b + 1) * bs]
            y_hat = model(x)
            acc += torch.sum(torch.argmax(y_hat, dim=1) == y).item()
    return acc


acc = evaluate(model, X_test, y_test)
print(f"Accuracy: {acc} / {len(X_test)}")


torch.onnx.export(
    model.cpu(),
    torch.randn(10, 784),
    "mnist/model.onnx",
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
