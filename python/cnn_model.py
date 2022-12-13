# CNN to predict initial location of an agent given an image of the environment. There are 10 possible locations.

import torch
import torch.nn as nn

# Define the CNN
class CNN(nn.Module):
    def __init__(self, num_locations=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=6),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=6),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=6),
        )
        # flatten
        self.flatten = nn.Flatten(2, -1)
        self.avgpool = nn.AvgPool1d(5)
        self.flatten2 = nn.Flatten(1, -1)

        # fully connected layer
        self.fc1 = nn.Linear(64, num_locations)

        # softmax layer
        self.softmax = nn.Softmax(dim=1)

        # loss function
        self.loss = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # Define the forward pass of the CNN
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(x)
        x = self.avgpool(x)
        x = self.flatten2(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x
    
    def train(self, x, y, epochs=10):
        # Define the train function
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

    
        


