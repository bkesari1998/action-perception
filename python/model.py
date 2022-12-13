import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self, num_locations):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=6),

            # next layer
            nn.Conv2d(16, 32, kernel_size=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=6),

            # flatten
            nn.Flatten(2, -1),
            nn.AvgPool1d(5),
            nn.Flatten(1, -1),

            # fully connected layer
            nn.LazyLinear(64),
            nn.LazyLinear(num_locations),

            # softmax
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        # Define the forward pass of the CNN
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.model.forward(x)

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

    def train(self, x, y, lr=0.0005):
        # Convert to tensors
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        if type(y) != torch.Tensor:
            y = torch.tensor(y, dtype=torch.float32)

        # Define the train function
        optimizer = self.optimizer(lr)

        optimizer.zero_grad()
        y_pred = self.forward(x)
        loss = self.loss(y_pred, y)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss}, Accuracy: {self.accuracy(y_pred, y)}")
        return loss, self.accuracy(y_pred, y)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
