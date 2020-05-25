# PyTorch CNN
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

# Mean and starnadard deviation of all the MNIST images (precomputed):
mean_gray = 0.1307
stddev_gray = 0.3081 
# Transformations we need to make to the pics:
transformations = transforms.Compose([transforms.ToTensor(),                            # tranforming the pic to a tensor
                                 transforms.Normalize((mean_gray,), (stddev_gray,))])# normalizing the pic

# Downloading the dataset:
train_dataset = datasets.MNIST(root = './data',
                               train = True,
                               transform = transformations,
                               download = True)
test_dataset = datasets.MNIST(root = './data',
                               train = False,
                               transform = transformations,
                               )

batch_size = 100
train_loader = data.DataLoader(dataset = train_dataset,
                               batch_size = batch_size,
                               shuffle = True)
test_loader = data.DataLoader(dataset = test_dataset,
                              batch_size = batch_size,
                              shuffle = True)

# We will visualie a pic:
import matplotlib.pyplot as plt
'''
random_pic = train_dataset[6][0].numpy()
plt.imshow(random_pic.reshape(28,28), cmap = 'gray')
print(train_dataset[6][1])
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        ## FIRST CONVOLUTIONAL LAYER
        # we will use same padding, to preserve the input size
        # Same padding = (filter_size-1)/2
        # The output size of each of 8 feature maps:
        #output_size = [(input_size - filter_size + 2padding)/stride +1] = 
        #(28-3+2)/1 +1 = 28.
        self.cnn1 = nn.Conv2d( #The class from pytroch, resposnible for the convolutional layer
                            in_channels = 1,
                            out_channels = 8,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1)
        # Batch normalization:
        self.batchnorm1 = nn.BatchNorm2d(8)     #number of feature maps
        self.relu = nn.ReLU()
        ##
        
        ## FIRST MAX POOLING
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2)
        # the ouputsize: 28/2 = 14
        ##
        
        ## SECOND CONVOLUTIONAL LAYER
        # Same padding = (5-1)/2 = 2
        # output size = (14-5+4)/1 +1 = 14
        self.cnn2 = nn.Conv2d(in_channels = 8,    # (from the previous layer, and we have 8 feature maps)
                            out_channels = 32,
                            kernel_size = 5,
                            stride = 1,
                            padding = 2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        ##
        
        ## SECOND MAX POOLING
        # 14/2 = 7 - the size of the new feature maps
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2)
        ##
        
        ## FULLY CONNECTED LAYERS:
        # The number of input neurons of th fc1 is the number of pixels from 32 feature maps
        # from the second Max pooling; 7*7*32 = 1568
        self.fc1 = nn.Linear(1568, 600)     #600 is arbitrarly chosen
        # Dropout:
        self.dropout = nn.Dropout(p = 0.5)  # We will drop 50% of the neurons
        # Second layer:
        self.fc2 = nn.Linear(600, 10)                # because 10 digits
        
    def forward(self, x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        
        #Flattening:
        out = out.view(-1, 1568) #The same as writing (batch_size, 1568), as this is the same
        
        #Forward through FCL:
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

model = CNN()
loss_fn = nn.CrossEntropyLoss()         #Most suitable for softmax
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

iteration = 0
correct = 0
'''
for i, (inputs, labels) in enumerate (train_loader):
    print("For one iteration the following happens:")
    # Each tensor propagated through NN should be 4 dim (batch_size, channels, rows, columns)
    print("Input Shape: ", inputs.shape)
    print("Label Shape: ", labels.shape)
    output = model(inputs)
    print("Output Shape: ", output.shape)
    _, predicted = torch.max(output, 1)
    print("Predicted Shape: ", predicted.shape)
    print("Predicted Tensor:")
    print(predicted)
    correct += (predicted == labels).sum()
    break
'''
# Training the CNN
num_epochs = 10

# Storing loss and accuracy:
train_loss = []
train_accuracy = []

test_loss =[]
test_accuracy = []

for epoch in range (0, num_epochs):
    correct = 0
    iterations = 0
    iter_loss = 0.0
    
    model.train()
    
    for i, (inputs, labels) in enumerate (train_loader):
        # Forward:
        outputs = model(inputs)
        
        # Loss:
        loss = loss_fn(outputs, labels)
        iter_loss += loss.item()
        
        #Backprop:
        optimizer.zero_grad()           #Clearing the gradient
        loss.backward()                 #Backprop method in pytorch
        optimizer.step()                #Updating the weights
        
        # Accuracy:
        _, predicted = torch.max(outputs,1)     #we only need an index, not a value
        correct += (predicted ==labels).sum().item()
        iterations +=1
    
    train_loss.append(iter_loss/iterations)
    train_accuracy.append(correct / len(train_dataset))
    
    #Testing phase:
    test_iter_loss = 0.0
    correct = 0
    iterations = 0
    
    model.eval()     #This tells pytorch that we're starting the testing phase
    for i, (inputs, labels) in enumerate (test_loader):
        # Forward:
        outputs = model(inputs)
        
        # Loss:
        loss = loss_fn(outputs, labels)
        test_iter_loss += loss.item()
        
        # Accuracy:
        _, predicted = torch.max(outputs,1)     # we only need an index, not a value
        correct += (predicted ==labels).sum().item()
        iterations +=1
        
    test_loss.append(test_iter_loss/iterations)
    test_accuracy.append(correct / len(test_dataset))
    
    print('Epoch: {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f},Testing Loss: {}, Testing Acuracy: {}'.
          format(epoch+1,
                 num_epochs,
                 train_loss[-1],
                 train_accuracy[-1],
                 test_loss[-1],
                 test_accuracy[-1]))

#Plotting the loss:
f = plt.figure(figsize = (10,10))
plt.plot(train_loss, label = 'Training Loss')
plt.plot(test_loss, label = 'Testing Loss')
plt.legend()
plt.show()

#Plotting the accuracy:
f = plt.figure(figsize = (10,10))
plt.plot(train_accuracy, label = 'Training Accuracy')
plt.plot(test_accuracy, label = 'Testing Accuracy')
plt.legend()
plt.show()

# Let's check for some number:
img = test_dataset[10][0].resize_((1,1,28,28))
label = test_dataset[10][1]

model.eval()
outputs = model(img)
_, predicted = torch.max(outputs, 1)
print("Predictions: {}".format(predicted.item()))
print("Actual: {}".format(label))