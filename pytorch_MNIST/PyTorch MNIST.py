#!/usr/bin/env python
# coding: utf-8

# In[82]:


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data


# In[110]:


input_size = 784
hidden_size1 = 64
hidden_size2 = 44
out_size = 10
epochs = 10
batch_size = 100
learning_rate = 0.001


# In[111]:


train_dataset = datasets.MNIST(root='./data',
                              train = True,
                              transform = transforms.ToTensor(),
                              download = True)
test_dataset = datasets.MNIST(root='./data',
                             train = False,
                             transform = transforms.ToTensor()
                             )


# In[112]:


# Make data iterable by loading it to Loader. Shuffle the training data to make it independent of the order.
train_loader = data.DataLoader(dataset = train_dataset,
                              batch_size = batch_size,
                              shuffle = True)
test_loader = data.DataLoader(dataset = test_dataset,
                              batch_size = batch_size,
                              shuffle = True)


# In[113]:


class NeuralNet(nn.Module):
    def __init__(self, input_layer, hidden_layer1, hidden_layer2, output_layer):
        super(NeuralNet, self).__init__()
        self.first_l = nn.Linear(input_layer, hidden_layer1)
        self.second_l = nn.Linear(hidden_layer1, hidden_layer2)
        self.third_l = nn.Linear(hidden_layer2, output_layer)
        
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.first_l(x)
        out = self.relu(out)
        out = self.second_l(out)
        out = self.relu(out)
        out = self.third_l(out)
        return out


# In[114]:


the_net = NeuralNet(input_size, hidden_size1, hidden_size2, out_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(the_net.parameters(), lr = learning_rate)


# In[115]:


# resizing images:
for i, (images, lablels) in enumerate(train_loader):
    images = images.view(-1, 784)


# In[ ]:


# Training the net:
correct_train = 0
total_train = 0
for epoch in range (epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.view(-1, 28*28)
        outputs = the_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_train +=labels.size(0)
        correct_train += (predicted ==labels).sum()
        loss = loss_fn(outputs, labels)
        loss.backward()
        #print(loss.item())
        optimizer.step()
        if (i+1)% 100 == 0 :
            print('Epoch: {}/{}, Iteration: {}/{}, Training Loss: {}, Training Accuracy: {}%'.format(epoch+1,
                                                                                                     epochs,
                                                                                                     i+1,
                                                                                                     len(train_dataset)//batch_size,
                                                                                                     loss.item(),
                                                                                                     (100*correct_train//total_train)))
print("Done training!")            


# In[109]:


# testing:
correct = 0
total = 0
for images, labels in test_loader:
    images = images.view(-1, 28*28)
    outputs = the_net(images)
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct += (predicted ==labels).sum()
print("Final (test) accuracy: {} %%".format(correct/total))


# In[ ]:




