# Create a CNN that takes in an image and gives a distribution over 10 classes using Pytorch

import torch
import torch.nn as nn

# Define the convolutional neural network
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(8*8*32, num_classes)
        self.sm = nn.Softmax(dim=1)

        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.sm(out)
        return out

    def train(self, x, y):
        y_pred = self.forward(x)
        loss = self.kl_div(y_pred, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item(), self.accuracy(y_pred, y)

    def accuracy(self, y_pred, y):
        y_pred = torch.argmax(y_pred, dim=1)
        y = torch.argmax(y, dim=1)
        return torch.sum(y_pred == y).item() / y.size(0)
    
    def test(self, x, y):
        y_pred = self.forward(x)
        return self.accuracy(y_pred, y)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


