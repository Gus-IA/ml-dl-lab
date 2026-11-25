import torchvision
from torch import torch

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

import numpy as np

train_imgs, train_labels = np.array([np.array(i[0]) for i in trainset]), np.array(
    [i[1] for i in trainset]
)
test_imgs, test_labels = np.array([np.array(i[0]) for i in testset]), np.array(
    [i[1] for i in testset]
)

train_imgs.shape, test_imgs.shape


import numpy as np
import scipy.signal
from skimage import color
from skimage import exposure
import random
import matplotlib.pyplot as plt

ix = random.randint(0, len(train_imgs))
img, label = train_imgs[ix], train_labels[ix]

plt.imshow(img)
plt.title(classes[label])
plt.show()

# convertir la imágen en tensor con dimensiones (N, C_in, H, W)

img_tensor = torch.from_numpy(img / 255.0).unsqueeze(0)
img_tensor = img_tensor.permute(0, 3, 1, 2).float()

img_tensor.shape, img_tensor.dtype

# aplicamos 10 filtros de tamaño 3x3

conv = torch.nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)

output = conv(img_tensor)

# dimensiones: (N, #filtros, H', W')
output.shape

conv = torch.nn.Conv2d(
    in_channels=3, out_channels=10, kernel_size=3, padding=1, stride=1
)

output = conv(img_tensor)

# dimensiones: (N, #filtros, H', W')
output.shape

conv = torch.nn.Conv2d(
    in_channels=3, out_channels=10, kernel_size=3, padding=1, stride=2
)

output = conv(img_tensor)

# dimensiones: (N, #filtros, H', W')
output.shape
