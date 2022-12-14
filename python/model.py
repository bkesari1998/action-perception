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
        )

        kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        # Define the forward pass of the CNN
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.model.forward(x)

    def get_probability(self, x):
        return F.softmax(x, dim=1)
    
    def get_log_probability(self, x):
        return F.log_softmax(x, dim=1)
    
    def loss(self, x, y):
        # return F.cross_entropy(x, y)
        return self.kl_div(x, y)
        
    def kl_div(self, x, y):
        
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        # Calculate softmax of x
        x = F.log_softmax(x, dim=1)

        loss = kl_loss(x, y)
        # print(loss)
        return loss

    def cross_entropy(self, x, y):

        # Define the loss function
        return F.cross_entropy(x, y)

    def optimizer(self, lr):

        # Define the optimizer
        return optim.Adam(self.parameters(), lr=lr)

    def accuracy(self, predicted, labels):
        if len(predicted.shape) == 1:
            predicted = predicted.unsqueeze(0)
        # Define the accuracy function
        if len(labels.shape) == 1:
            return (predicted.argmax(dim=1) == labels).float().mean()
        else:
            return (predicted.argmax(dim=1) == labels.argmax(dim=1)).float().mean()

    def train(self, x, y, lr=0.001, epoches=150):
        # Convert to tensors
        if type(x) != torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)
        if type(y) != torch.Tensor:
            y = torch.tensor(y, dtype=torch.float32)

        # Define the train function
        optimizer = self.optimizer(lr)

        optimizer.zero_grad()
        prev_loss = None
        for i in range(epoches):
            y_pred = self.forward(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0 or i == epoches - 1:
                print(f"      Epoch {i+1}, Training Loss: {loss}, Accuracy: {self.accuracy(y_pred, y)}, Cross Entropy: {F.cross_entropy(y_pred, y)}")
            
            # early stopping
            if prev_loss is not None and abs(prev_loss - loss) < 0.0001:
                print("        Early Stoping.")
                break
            prev_loss = loss
        return loss, self.accuracy(y_pred, y)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))
