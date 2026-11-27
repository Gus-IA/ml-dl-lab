import torch


# lenet-5
def block(c_in, c_out, k=3, p=1, s=1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(c_in, c_out, k, padding=p, stride=s),
        torch.nn.Tanh(),
        torch.nn.AvgPool2d(2, stride=2),
    )


def block2(c_in, c_out):
    return torch.nn.Sequential(torch.nn.Linear(c_in, c_out), torch.nn.ReLU())


class LeNet5(torch.nn.Module):
    def __init__(self, n_channels=1, n_outputs=10):
        super().__init__()
        # self.pad = torch.nn.ConstantPad2d(2, 0.)
        self.conv1 = block(n_channels, 6, 5, p=0)
        self.conv2 = block(6, 16, 5, p=0)
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 120, 5, padding=0), torch.nn.Tanh()
        )
        self.fc1 = block2(120, 84)
        self.fc2 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


lenet5 = LeNet5()
output = lenet5(torch.randn(64, 1, 32, 32))
output.shape

# alexnet
import torchvision

alexnet = torchvision.models.AlexNet()
print(alexnet)

output = alexnet(torch.randn(64, 3, 224, 224))
output.shape

# VGG
vgg16 = torchvision.models.vgg16()
print(vgg16)

output = vgg16(torch.randn(64, 3, 224, 224))
output.shape

# RESNET
resnet34 = torchvision.models.resnet34()
print(resnet34)

output = resnet34(torch.randn(64, 3, 224, 224))
output.shape
