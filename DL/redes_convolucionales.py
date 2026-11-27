from torch import torch
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml("mnist_784", version=1)
X, Y = mnist["data"].values.astype(np.float32) / 255.0, mnist["target"].values.astype(
    int
)


def block(c_in, c_out, k=3, p=1, s=1, pk=2, ps=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, c_out, k, padding=p, stride=s),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(pk, stride=ps),
    )


class CNN(torch.nn.Module):
    def __init__(self, n_channels=1, n_outputs=10):
        super().__init__()
        self.conv1 = block(n_channels, 64)
        self.conv2 = block(64, 128)
        self.fc = torch.nn.Linear(128 * 7 * 7, n_outputs)

    def forward(self, x):
        print("Dimensiones:")
        print("Entrada: ", x.shape)
        x = self.conv1(x)
        print("conv1: ", x.shape)
        x = self.conv2(x)
        print("conv2: ", x.shape)
        x = x.view(x.shape[0], -1)
        print("pre fc: ", x.shape)
        x = self.fc(x)
        print("Salida: ", x.shape)
        return x


model = CNN()

output = model(torch.randn(64, 1, 28, 28))


class CNN(torch.nn.Module):
    def __init__(self, n_channels=1, n_outputs=10):
        super().__init__()
        self.conv1 = block(n_channels, 64)
        self.conv2 = block(64, 128)
        self.fc = torch.nn.Linear(128 * 7 * 7, n_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).long()

    def __len__(self):
        return len(self.X)

    # devolvemos las imágenes con dimensiones (C, H, W)
    def __getitem__(self, ix):
        return self.X[ix].reshape(1, 28, 28), self.Y[ix]


model = CNN()


def train(model, epochs=5, batch_size=1000):
    dataset = {
        "train": Dataset(X[:60000], Y[:60000]),  # 60.000 imágenes para entrenamiento
        "val": Dataset(X[60000:], Y[60000:]),  # 10.000 imágenes para validación
    }
    dataloader = {
        "train": torch.utils.data.DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        ),
        "val": torch.utils.data.DataLoader(
            dataset["val"], batch_size=batch_size, num_workers=4, pin_memory=True
        ),
    }
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for e in range(1, epochs + 1):
        print(f"epoch: {e}/{epochs}")
        # entrenamiento
        model.train()
        for batch_ix, (x, y) in enumerate(dataloader["train"]):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(x)
                loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if batch_ix % 10 == 0:
                loss, current = loss.item(), (batch_ix + 1) * len(x)
                print(f"loss: {loss:.4f} [{current:>5d}/{len(dataset['train']):>5d}]")
        # validación
        model.eval()
        val_loss, val_acc = [], []
        with torch.no_grad():
            for batch_ix, (x, y) in enumerate(dataloader["val"]):
                x, y = x.cuda(), y.cuda()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(x)
                    loss = criterion(outputs, y)
                val_loss.append(loss.item())
                val_acc.append((outputs.argmax(1) == y).float().mean().item())
        print(f"val_loss: {np.mean(val_loss):.4f} val_acc: {np.mean(val_acc):.4f}")


train(model)
