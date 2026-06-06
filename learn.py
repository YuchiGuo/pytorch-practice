# import torch
# import numpy as np

# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)
# print(x_data)


# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)
# print(x_np)

# x_ones = torch.ones_like(x_data)
# x_rand = torch.rand_like(x_data, dtype=torch.float)

# tensor = torch.rand(3, 4)

# print(tensor.shape)
# print(tensor.dtype)
# print(tensor.device)

# if torch.accelerator.is_available():
#     tensor = tensor.to(torch.accelerator.current_accelerator())

# print(tensor.device)

# tensor = torch.ones(4, 4)


# # Dataset and Dataloader. Dataloader wraps an iterable around the Dataset.
# # Built-in datasets are subclasses of Dataset. They can all be passed to Dataloader.


import torch

# from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

import torch.nn.functional as F

from torch import nn

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

# # Visualising some of the training data
# labels_map = {
#     0: "T-shirt/top",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle boot",
# }
# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


# # train_features, train_labels = next(
# #     iter(train_dataloader)
# # )  # iter() returns an interator object.
# train_features, train_labels = next(iter(train_dataloader))
# print(train_features.size())  # size [64, 1, 28, 28] = B, C, H, W
# print(train_labels.size())  # size [64] = B


# img = train_features[0].squeeze()
# label = train_labels[0]
# print(f"Label: {label}")
# plt.imshow(img, cmap="gray")
# plt.show()


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()  # .to(device)
print(model)


# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# print(logits)
# print(logits.shape)
# pred_probab = nn.Softmax(dim=1)(
#     logits
# )  # nn.Softmax(dim=1) returns a function. And values sum to 1 along the 1st dimension.
# print(pred_probab)
# y_pred = pred_probab.argmax(1)
# print(f"y_pred", y_pred)

# x_np = X.detach().cpu().numpy()
# figure = plt.figure(figsize=(8, 8))
# figure.add_subplot(1, 1, 1)
# plt.title("X")
# plt.axis("off")
# plt.imshow(x_np.squeeze(), cmap="gray")
# plt.show()

# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# print(next(iter(model.parameters())))


# setting hyperparameters
learning_rate = 1e-3
batch_size = 64


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
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

import torchvision.models as models

model = models.vgg16(weights="IMAGENET1K_V1")
torch.save(model.state_dict(), "model_weights.pth")


model = models.vgg16()
model.load_state_dict(torch.load("model_weights.pth", weights_only=True))
model.eval()
