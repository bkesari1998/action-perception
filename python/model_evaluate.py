import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os

dataset = ImageFolder(root=os.path.dirname(__file__) + \
                        os.path.sep + ".." + os.path.sep + "evaluation",
        transform=transforms.Compose([transforms.ToTensor()])
)
dataloader = DataLoader(dataset)


def evaluate_model(model, dataloader=dataloader):
    """
    Evaluate the model on the dataset.
    """
    with torch.no_grad():
        running_accuracy = 0
        running_loss = 0
        for x, y in dataloader:
            predicted = model.forward(x)[:, :10]
            running_accuracy += model.accuracy(predicted, y)
            running_loss += F.cross_entropy(predicted, y)
        return running_accuracy / len(dataloader), running_loss / len(dataloader)
