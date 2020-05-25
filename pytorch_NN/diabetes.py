# A Neural Net classyfing if person has diabetes or no
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader as DL


# Loading the ds using Pandas:
data = pd.read_csv('diabetes.csv')

#x and y (using ".values" changes it into numpy array):
x = data.iloc[:, :-1].values
y_raw = data.iloc[:, -1].values

label_encoder = LabelEncoder()
z = label_encoder.fit_transform(y_raw)
z = z.reshape(len(z), 1)
#y = torch.from_numpy(z)
#y = y.type(torch.FloatTensor) 

# Feature normalization:
scaler = StandardScaler()
X = scaler.fit_transform(x)

X = torch.from_numpy(X)
y = torch.from_numpy(z)

class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    
dataset = Dataset(X, y)

# Loading Data:
train_loader = DL(dataset = dataset,
                  batch_size = 32,
                  shuffle = True)

# Neural Network:

class NeuralNet(nn.Module):
    def __init__(self, input_features, hidden_layer1, hidden_layer2, hidden_layer3, output_features):
        super(NeuralNet, self).__init__()     # inheriting from the parent class
    # layers:
        self.input_l = nn.Linear(input_features, hidden_layer1)
        self.hidden_l1 = nn.Linear(hidden_layer1, hidden_layer2)
        self.hidden_l2 = nn.Linear(hidden_layer2, hidden_layer3)
        self.hidden_l3 = nn.Linear(hidden_layer3, output_features)
    # activation functions:
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def forward(self, x):
        out = self.input_l(x)
        out = self.tanh(out)
        out = self.hidden_l1(out)
        out = self.tanh(out)
        out = self.hidden_l2(out)
        out = self.tanh(out)
        out = self.hidden_l3(out)
        out = self.sigmoid(out)
        return out
    # There is no need to program backprop as pytorch does it itself
    
net = NeuralNet(7,5,4,3,1)
 
# defining loss function: Binary Cross Entropy
loss_fn = nn.BCELoss(size_average = True)    
# this parameter implies that the losses are averaged over the size of mini batch

# Optimization: Stochastic gradient descent with momentum
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9)
     
# Training:
epochs = 2000

for epoch in range(2000):
    for inputs, labels in train_loader:
        inputs = inputs.float()
        labels = labels.float()
        # Forward:
        output = net.forward(inputs)
        # Loss calc:
        loss = loss_fn(output, labels)
        # Clear the gradient buffer:
        optimizer.zero_grad()
        # Calculate the gradient (Backprop):
        loss.backward()
        # Update weights:
        optimizer.step()
    
    # accuracy:
    # this changes the values bigger than it to one and smaller to zero in the matrix
    outputs = (output>0.5).float()  
    accuracy = (outputs == labels).float().mean() #This checks if the same
    # print statisctics:
    print("Epochs {}/{}, \n Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,epochs, loss, accuracy))
    
    







