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

total_steps_in_test = 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1

def num_successful_actions(predicted, actual):
    """
    Calculate the number of successful actions. This is the number of actions that don't fail or we end up in the goal state at some point.
    We then calculate the ration of successful actions to the total number of actions in the optimal path.
    """
    if predicted == actual:
       return 9 - predicted
    else:
        if predicted == 0:
            if actual == 1 or actual == 8:
                return 1
            elif actual == 7:
                return 2
            else:
                return 0
        elif predicted == 1:
            if actual == 0 or actual == 7 or actual == 8:
                return 1
            else:
                return 0
        elif predicted == 2:
            if actual == 0 or actual == 1 or actual == 7 or actual == 8:
                return 0
            else:
                return 8 - actual
        elif predicted == 3:
            if actual == 0 or actual == 1 or actual == 7 or actual == 8:
                return 0
            elif actual > 3:
                return 7 - actual
            else:
                return 4 
        elif predicted == 4:
            if actual == 0 or actual == 1 or actual == 7 or actual == 8:
                return 0
            elif actual > 4:
                return 7 - actual
            else:
                return 3
        elif predicted == 5:
            if actual == 0 or actual == 1 or actual == 7 or actual == 8:
                return 0
            elif actual > 5:
                return 7 - actual
            else:
                return 2
        elif predicted == 6:
            if actual == 0 or actual == 1 or actual == 7 or actual == 8:
                return 0
            elif actual > 6:
                return 7 - actual
            else:
                return 1
        elif predicted == 7:
            if actual == 0:
                return 2
            elif actual == 1 or actual == 8:
                return 1
            else:
                return 0
        elif predicted == 8:
            if actual == 0 or actual == 1 or actual == 7:
                return 1
            else:
                return 0
        elif predicted == 9:
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
            successful_actions = np.sum([num_successful_actions(predicted_action[i], y_np[i]) for i in range(len(y_np))])
            success_rate = successful_actions / total_steps_in_test
            print("Evaluation success rate:", success_rate)

            running_loss += F.cross_entropy(predicted, y)
        return running_accuracy / len(dataloader), running_loss / len(dataloader), success_rate