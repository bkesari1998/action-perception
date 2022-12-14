# Create a model that combines the CNN and LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # nn.Softmax(dim=1)
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



class LSTM(nn.Module):
    def __init__(self, num_locations):
        super(LSTM, self).__init__()
        self.model = nn.Sequential(
            nn.LSTM(64, 64, 2, batch_first=True),
            nn.Flatten(1, -1),
            nn.LazyLinear(num_locations),
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

    def accuracy(self, x, y):

        # Define the accuracy function
        return (x.argmax(dim=1) == y).float().mean()
    
    def predict(self, x):

        # Define the predict function
        return x.argmax(dim=1)


class CNN_LSTM(nn.Module):
    def __init__(self, num_locations):
        super(CNN_LSTM, self).__init__()
        self.cnn = CNN(num_locations)
        self.lstm = LSTM(num_locations)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        # Define the forward pass of the CNN
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.cnn.forward(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        return self.lstm.forward(x)

    def loss(self, x, y):

        # Define the loss function
        return F.cross_entropy(x, y)

    def accuracy(self, x, y):

        # Define the accuracy function
        return (x.argmax(dim=1) == y).float().mean()
    
    def predict(self, x):

        # Define the predict function
        return x.argmax(dim=1)