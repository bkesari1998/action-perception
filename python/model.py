import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self, num_locations):
        super(CNN, self).__init__()
         # Define the layers of the CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=8)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=8)
        self.flatten = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(in_features=8700, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_locations * 2)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        # Define the forward pass of the CNN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def loss(self, x, y):
        # Define the loss function
        return F.cross_entropy(x, y)

    def optimizer(self, lr):
        # Define the optimizer
        return optim.Adam(self.parameters(), lr=lr)

    def accuracy(self, x, y):
        # Define the accuracy function
        return (x.argmax(dim=1) == y).float().mean()
    
    def predict(self, x):
        # Define the predict function
        return x.argmax(dim=1)

    def train(self, x, y, lr, epochs):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        if type(y) != torch.Tensor:
            y = torch.tensor(y, dtype=torch.float32)
        # Define the train function
        optimizer = self.optimizer(lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {self.accuracy(y_pred, y)}")
    
    def update_model(self, x, y):
        # Define the update function
        pass
