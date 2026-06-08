import torch

# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

import torch.nn.functional as F

from torch import nn
import torchvision.models as models


# checking commit from using runpod
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # print(x.shape) # x has shape [32, 1, 28, 28]
        x = self.cnn_stack(x)
        # print(x.shape) # [32, 32, 10, 10]
        x = self.flatten(x)
        # print(x.shape) # [32, 32*10*10]
        logits = self.linear_relu_stack(x)
        return logits


# print(len(next(iter(train_dataloader))[1]))
# train_dataloader contains (img, label), both are huge tensors with all the images.
# the img tensor has shape [64, 1, 28, 28], and the label tensor has shape [64, 10], 10 is because of one-hot encoding.


def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
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


def test_loop(dataloader, model, loss_fn, loss_t, accuracy_t, device):
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
    loss_t.append(test_loss)
    accuracy_t.append(100 * correct)


def main():
    # setting hyperparameters
    learning_rate = 1e-3
    batch_size = 32
    epochs = 30

    # load data
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

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # set device
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # set model, loss function and optimizer
    model = CNN().to(device)
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # for logging.
    loss_t = []
    accuracy_t = []

    # the main trianing and testing loop
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device, batch_size)
        test_loop(test_dataloader, model, loss_fn, loss_t, accuracy_t, device)
    print("Done!")

    # model = models.vgg16(weights="IMAGENET1K_V1")
    # torch.save(model.state_dict(), "model_weights_cnn.pth")

    # model = models.vgg16()
    # model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
    # model.eval()

    # --- Append this plotting code ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Test Loss
    ax1.plot(range(1, epochs + 1), loss_t, color="red", marker="o", linewidth=2)
    ax1.set_title(f"Test Loss. Final loss {loss_t[-1]:.4f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)

    # Plot Test Accuracy
    ax2.plot(range(1, epochs + 1), accuracy_t, color="blue", marker="o", linewidth=2)
    ax2.set_title(f"Test Accuracy. Final accuracy {accuracy_t[-1]:.4f}%")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True)

    # Prevent overlapping labels and save
    plt.tight_layout()
    plt.savefig(
        f"training_progress_cnn_learning_rate_{learning_rate}_batch_size_{batch_size}_optimizer_{type(optimizer).__name__}_vgg_like.png"
    )


if __name__ == "__main__":
    main()
