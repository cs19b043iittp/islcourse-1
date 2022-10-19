from functools import total_ordering
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image


device = "cpu"

loss_function = nn.CrossEntropyLoss()


def adi():
    print('adi')

# Define a neural network YOUR ROLL NUMBER (all small letters) should prefix the classname


class cs19b001NN(nn.Module):
    def __init__(self):
        super(cs19b001NN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_function(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# sample invocation torch.hub.load(myrepo,'get_model',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)
def get_model(train_data_loader=None, n_epochs=10):
    model = None

    model = cs19b001NN().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data_loader, model, optimizer)

    # write your code here as per instructions
    # ... your code ...
    # ... your code ...
    # ... and so on ...
    # Use softmax and cross entropy loss functions
    # set model variable to proper object, make use of train_data

    print('Returning model... (rollnumber: cs19b001)')

    return model

# sample invocation torch.hub.load(myrepo,'get_model_advanced',train_data_loader=train_data_loader,n_epochs=5, force_reload=True)


def get_model_advanced(train_data_loader=None, n_epochs=10, lr=1e-4, config=None):
    model = None

    # write your code here as per instructions
    # ... your code ...
    # ... your code ...
    # ... and so on ...
    # Use softmax and cross entropy loss functions
    # set model variable to proper object, make use of train_data

    # In addition,
    # Refer to config dict, where learning rate is given,
    # List of (in_channels, out_channels, kernel_size, stride=1, padding='same')  are specified
    # Example, config = [(1,10,(3,3),1,'same'), (10,3,(5,5),1,'same'), (3,1,(7,7),1,'same')], it can have any number of elements
    # You need to create 2d convoution layers as per specification above in each element
    # You need to add a proper fully connected layer as the last layer

    # HINT: You can print sizes of tensors to get an idea of the size of the fc layer required
    # HINT: Flatten function can also be used if required

    print('Returning model... (rollnumber: xx)')

    return model

# sample invocation torch.hub.load(myrepo,'test_model',model1=model,test_data_loader=test_data_loader,force_reload=True)


def test_model(model1=None, test_data_loader=None):

    accuracy_val, precision_val, recall_val, f1score_val = 0, 0, 0, 0
    # write your code here as per instructions
    # ... your code ...
    # ... your code ...
    # ... and so on ...
    # calculate accuracy, precision, recall and f1score

    size = len(test_data_loader.dataset)
    num_batches = len(test_data_loader)
    model1.eval()
    test_loss, correct = 0, 0

    total_positive_pred = 0
    total_positive_actual = 0

    with torch.no_grad():
        for X, y in test_data_loader:
            X, y = X.to(device), y.to(device)
            pred = model1(X)
            total_positive_pred += pred.argmax(1)
            total_positive_actual += y
            print(pred.argmax(1), y)
            test_loss += loss_function(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    precision_val = correct / total_positive_pred
    recall_val = correct / total_positive_actual
    correct /= size

    print(
        f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    accuracy_val = 100*correct

    # f1score_val = 2*(recall_val * precision_val) / (recall_val + precision_val)
    print('Returning metrics... (rollnumber: cs19b001)')

    return accuracy_val, precision_val, recall_val, f1score_val
