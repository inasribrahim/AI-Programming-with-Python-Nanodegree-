# import Libraries python 
import time
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models
import argparse 

# get user input value
def get_input_args():
  # Create Parse using ArgumentParser
   parser = argparse.ArgumentParser()
  # arg 1 - path to folder with default
  parser.add_argument('--dir' , type=str , default='image/' , help = 'it is the path folder u must contain the image default image ' )
  # arg 2 - path to CNN model Architecture for use for image classifacitons 
  parser.add_argument('--arch', type=str, default='cat_dog',
                        help='CNN model to use for image classification; default is cat_dog')
 
  return parser.parse_args()


args = get_input_args()

data_dir = args.data_dir
arch = args.arch

# user did not provide value, set default
if (arch == "densenet121"):
    input_size = 1024
    output_size = 102
else:
    print("Please select model architectures densenet121 Like cnn .")
    exit()
       
if(data_dir == None) or (arch == None) :
    print("data_dir, arch  cannot be none")
    exit()



# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['training', 'validation', 'testing']} 

# TODO: Build and train your network
if (arch == 'densenet121'):
    model = models.densenet121(pretrained=True)

model

# TODO: Do validation on the test set
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
hidden_layer = 512 
# Build a feed-forward network
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout1',nn.Dropout(0.2)),
                                        ('fc2', nn.Linear(hidden_units, output_size)),
                                        ('output', nn.LogSoftmax(dim=1))]))

# Put the classifier on the pretrained network
model.classifier = classifier

# Train a model with a pre-trained network
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
#model.to('cuda')
model.to(device)
print("Start training model")
for e in range(10):

    for dataset in ['training', 'validation']:
        if dataset == 'training':
            model.train()  
        else:
            model.eval()   
        
        running_loss = 0.0
        running_accuracy = 0
        
        for inputs, labels in dataloaders[dataset]:
            #inputs, labels = inputs.to('cuda'), labels.to('cuda')
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(dataset == 'training'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward 
                if dataset == 'training':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(preds == labels.data)
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'validation', 'testing']}
        epoch_loss = running_loss / dataset_sizes[dataset]
        epoch_accuracy = running_accuracy.double() / dataset_sizes[dataset]
        
        print("Epoch: {}/{}... ".format(e+1, epochs),
              "{} Loss: {:.4f}    Accurancy: {:.4f}".format(dataset, epoch_loss, epoch_accuracy))
        
# Do validation on the test set
def check_accuracy(test_loader):    
    correct = 0
    total = 0
    #model.to('cuda:0')
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            #images, labels = images.to('cuda'), labels.to('cuda')
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
check_accuracy(dataloaders['training'])

# TODO: Save the checkpoint 
save_dir = ''
model.class_to_idx = image_datasets['training'].class_to_idx
model.cpu()
torch.save({'model': arch,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            save_dir)
# 
print("Save model to:" + save_dir)