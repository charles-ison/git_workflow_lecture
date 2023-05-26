import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import numpy as np
from torch.autograd import Variable

def get_data():
    training_data = torchvision.datasets.MNIST(root = 'data',
                                               train = True,
                                               transform = transforms.ToTensor(),
                                               download = True)

    testing_data = torchvision.datasets.MNIST(root = 'data',
                                              train = False,
                                              transform = transforms.ToTensor(),
                                              download=True)
    return training_data, testing_data

def get_loaders(training_data, testing_data, batch_size):
    training_loader = torch.utils.data.DataLoader(dataset = training_data,
                                                  batch_size = batch_size,
                                                  shuffle = True)
    testing_loader = torch.utils.data.DataLoader(dataset = testing_data,
                                                 batch_size = batch_size,
                                                 shuffle = True)
    return training_loader, testing_loader

def print_image(image_tensor):
    plt.imshow(image_tensor[0].detach(), cmap="gray")
    plt.show()

class CustomReLU:
    def run(self, input_tensor):
        return torch.where(input_tensor > 0, input_tensor, 0)

class CustomLeakyReLU:
    def run(self, input_tensor):
        return torch.where(input_tensor > 0, input_tensor, input_tensor * 0.01)

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer0 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.custom_relu = CustomReLU()
        self.custom_leaky_relu = CustomLeakyReLU()
        self.max_pool = nn.MaxPool2d(2) 
        self.conv_layer1 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=1, padding=2)                
        self.out = nn.Linear(5 * 7 * 7, num_classes)

    #def activation_function(self, x):
    #    return self.relu(x)
    
    def forward(self, x, flag):
        x = self.conv_layer0(x)

        if flag == 0:
            x = self.relu(x)
        elif flag == 1:
            x = self.custom_relu.run(x)
        elif flag == 2:
            x = self.custom_leaky_relu.run(x)

        x = self.relu(x)   
        x = self.max_pool(x)
        x = self.conv_layer1(x)

        if flag == 0:
            x = self.relu(x)
        elif flag == 1:
            x = self.custom_relu.run(x)
        elif flag == 2:
            x = self.custom_leaky_relu.run(x)

        x = self.relu(x)          
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output

#class CustomCNN(CNN):
#    def __init__(self, num_classes):
#        super(CustomCNN, self).__init__(num_classes)
#        self.custom_relu = CustomReLU()

#    def activation_function(self, x):
#        return self.custom_relu.run()

#class CustomLeakyCNN(CNN):
#    def __init__(self, num_classes):
#        super(CustomLeakyCNN, self).__init__(num_classes)
#        self.custom_leaky_relu = CustomLeakyReLU()

#    def activation_function(self, x):
#        return self.custom_leaky_relu.run()

def train(num_epochs, model, criterion, optimizer, training_loader):
    
    model.train()
    for epoch in range(num_epochs):
        num_correct = 0
        running_loss = 0.0
        for i, (images, labels) in enumerate(training_loader):
            optimizer.zero_grad() 
            output = model.forward(images, 1)
            loss = criterion(output, labels)
            running_loss += loss.item()
            
            _, predictions = torch.max(output.data, 1)
            num_correct += (predictions == labels).sum().item()
                      
            loss.backward()    
            optimizer.step()  
           
        loss = running_loss/len(training_loader.dataset) 
        accuracy = num_correct/len(training_loader.dataset)  
        print("\nTraining Loss: " + str(loss) + ", Training Accuracy: " + str(accuracy))

def test(model, criterion, testing_loader):
    model.eval()
    num_correct = 0
    running_loss = 0.0
       
    for i, (images, labels) in enumerate(testing_loader):
                
        output = model.forward(images, 1)
        loss = criterion(output, labels)
        running_loss += loss.item()
        
        _, predictions = torch.max(output.data, 1)
        num_correct += (predictions == labels).sum().item()
    
    loss = running_loss/len(testing_loader.dataset)
    accuracy = num_correct/len(testing_loader.dataset)
    print("Testing Loss:  " + str(loss) + ", Testing Accuracy:  " + str(accuracy))

training_data, testing_data = get_data()
print("Training data length: " + str(len(training_data)))
print("Test data length: " + str(len(testing_data)))

training_loader, testing_loader = get_loaders(training_data, testing_data, 20)

images, labels = next(iter(training_loader))
for image in images:
    print_image(image)

num_classes = 10
cnn = CNN(num_classes)
num_epochs = 1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.01)

train(num_epochs, cnn, criterion, optimizer, training_loader)
test(cnn, criterion, testing_loader)

images, labels = next(iter(testing_loader))
output = cnn.forward(images, 1)
_, predictions = torch.max(output.data, 1)

print("\nLabels:      " + str(labels))
print("Predictions: " + str(predictions))




