import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np

dataset = ImageFolder(root=os.path.dirname(__file__) + \
                        os.path.sep + ".." + os.path.sep + "evaluation",
        transform=transforms.Compose([transforms.ToTensor()])
)
dataloader = DataLoader(dataset, batch_size=10)

KL_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

def num_successful_actions(predicted, actual):
    """
    Calculate the number of successful actions. This is the number of actions that don't fail or we end up in the goal state at some point.
    We then calculate the ration of successful actions to the total number of actions in the optimal path.
    """
    if predicted == actual or actual == 9:
        return 1
    elif predicted == 0:
        if actual >= 7:
            return 1
        if actual == 1:
            return 1 / 8
        else:
            return 0
    elif predicted == 1:
        if actual == 0:
            return 1 / 9
        elif actual == 7:
            return 1 / 2
        elif actual == 8:
            return 1
        else:
            return 0
    elif predicted == 2:
        if actual < 2 or actual >= 7:
            return 0
        elif actual > 2:
            return (5 - (actual - predicted)) / ((5 - (actual - predicted)) + 2)
    elif predicted == 3:
        if actual < 2 or actual >= 7:
            return 0
        elif actual > 3:
            return (4 - (actual - predicted)) / ((4 - (actual - predicted)) + 2)
        elif actual == 2:
            return 4 / 7
    elif predicted == 4:
        if actual < 2 or actual >= 7:
            return 0
        elif actual > 4:
            return (3 - (actual - predicted)) / ((3 - (actual - predicted)) + 2)
        elif actual == 2:
            return 3 / 7
        elif actual == 3:
            return 3 / 6
    elif predicted == 5:
        if actual < 2 or actual >= 7:
            return 0
        elif actual > 5:
            return (2 - (actual - predicted)) / ((2 - (actual - predicted)) + 2)
        elif actual == 2:
            return 2 / 7
        elif actual == 3:
            return 2 / 6
        elif actual == 4:
            return 2 / 5
    elif predicted == 6:
        if actual < 2 or actual >= 7:
            return 0
        elif actual == 2:
            return 1 / 7
        elif actual == 3:
            return 1 / 6
        elif actual == 4:
            return 1 / 5
        elif actual == 5:
            return 1 / 4
    elif predicted == 7:
        if actual == 0:
            return 2 / 9
        elif actual == 1:
            return 1 / 8
        elif actual == 8:
            return 1
        else:
            return 0
    elif predicted == 8:
        if actual == 0:
            return 1 / 9
        elif actual == 1:
            return 1 / 8
        elif actual == 7:
            return 1 / 2
        else:
            return 0
        

def evaluate_model(model, dataloader=dataloader):
    """
    Evaluate the model on the dataset.
    """
    with torch.no_grad():
        running_accuracy = 0
        running_loss = 0
        for x, y in dataloader:
            predicted = model.forward(x)[:, :10]
            predicted_action = predicted.argmax(dim=1).numpy()
            print("Evaluation predicted:", predicted_action)
            running_accuracy += model.accuracy(predicted, y)

            # Calcuate the success rate of the plan
            y_np = y.detach().numpy()
            print(y_np.shape)
            success_rate = np.array([num_successful_actions(predicted_action[i], y_np[i]) for i in range(len(y_np))])
            mean_success_rate = np.mean(success_rate)
            print("Evaluation success rate:", mean_success_rate)

            running_loss += F.cross_entropy(predicted, y)
        return running_accuracy / len(dataloader), running_loss / len(dataloader), mean_success_rate