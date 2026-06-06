import torch

# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

import torch.nn.functional as F

from torch import nn
import torchvision.models as models

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
    target_transform=v2.Lambda(
        lambda y: F.one_hot(torch.tensor(y), num_classes=10).float()
    ),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # print(x.shape) # x has shape [64, 1, 28, 28]
        x = self.cnn_stack(x)
        # print(x.shape) # [64, 16, 4, 4]
        x = self.flatten(x)
        # print(x.shape) # [64, 16*4*4]
        # x = torch.flatten(x, 1)
        logits = self.linear_relu_stack(x)
        return logits


model = CNN().to(device)
print(model)

# setting hyperparameters
learning_rate = 5e-3
batch_size = 64

print(len(next(iter(train_dataloader))[1]))
# train_dataloader contains (img, label), both are huge tensors with all the images.
# the img tensor has shape [64, 1, 28, 28], and the label tensor has shape [64, 10], 10 is because of one-hot encoding.


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)  # 60000
    model.train()
    for batch, (X, y) in enumerate(dataloader):  # there are 938 iterations = 60000/64
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (
            batch % 100 == 0
        ):  # print loss every 100 iterations, when it's the 0th batch, 100th batch, ... up to 900th batch.
            loss = loss.item()
            current = batch * batch_size + len(X)  # index of the image
            # len(X) = 64. X has shape [64, 1, 28, 28]
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 10

for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model_weights_cnn.pth")


# model = models.vgg16()
# model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
# model.eval()
