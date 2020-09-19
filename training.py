# import Libraries python 
import time
import torch
import os
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

# The default path of udacity
#path : /home/workspace/ImageClassifier/flowers

# get user input value
def input_argument():
    '''
    Argparse : create an object from argument 
    -- dir            : It is default path  
    -  arch           : Architecture Like resNet , vgg , ImagNet 
    -- Learning_rate  : it is hyper_parameter which the model can learn slow or fast 
    -- epohcs         :  a bunch of data can leran as in one iterater 
    -- Hidden units   : Number of hidden units  
    -- GPU            : Use GPU to speed up the training  
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir' , type=str , default='image/' , help = 'it is the path folder u must contain the image default image (required)' )
    parser.add_argument('--arch', type=str ,
                        help='CNN model to use for image classification, default is vgg16 ')
    parser.add_argument('--learning_rate', help='Learning rate it is hyperparameter to what the gradient move fast or slow')
    parser.add_argument('--hidden_units', help='Number of hidden_units')
    parser.add_argument('--save_dir', help='Save your network.')
    parser.add_argument('--epochs', help='Number of epochs')
    parser.add_argument('--gpu', help='Use GPU to speed up the training ')

    return parser.parse_args()
   
 
argument = input_argument()

arch          = argument.arch
hidden_units  = argument.hidden_units
path          = argument.save_dir


if(not os.path.isdir(argument.dir)):
    raise Exception('Error, Not found this directory !')

#Data dictory 

train_dir = argument.dir + '/train'
valid_dir = argument.dir + '/valid'
test_dir = argument.dir + '/test'


# user did not provide value, set default
if (arch == "densenet" or arch == "vgg" ):
    pass
else:
    print("Please select model between densenet or vgg.")
    exit()
       
if(argument.dir == None) or (arch == None):
    print("Miss the argument")
    exit()

if (argument.hidden_units is None):
    if (arch == "densenet121" ):
        hidden_units = 500 
    else:
        hidden_units = 4096
else:
    hidden_units = int(argument.hidden_units)

if(argument.learning_rate == None):
    learning_rate = 0.001
else:
    learning_rate = float(argument.learning_rate)
    
if (argument.gpu and not torch.cuda.is_available()):
     raise Exception("Error, gpu option enable but not detected ")

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir , transform=data_transforms),
    'valid': datasets.ImageFolder(valid_dir , transform=data_transforms),
    'test' : datasets.ImageFolder(test_dir  , transform=data_transforms)
}


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}  

# TODO: Build and train your network
if (arch == 'densenet121'):
    model = models.densenet121(pretrained=True)
    input_size=1024
    
elif (arch == 'vgg'):
    model = models.vgg19(pretrained=True)
    input_size=25088
    


output_size = 102
#Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


#Building and training the classifier

#to freeze part of the convolutional part of vgg16 model and train the rest
for param in model.parameters():
    param.requires_grad = False


# Define architecture
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1',nn.Dropout(0.3)),
                                            ('fc2', nn.Linear(hidden_units, output_size)),
                                            ('dropout2',nn.Dropout(0.2)),
                                            ('output', nn.LogSoftmax(dim=1))]))

model.classifier = classifier

# Convert model to be used on GPU
if torch.cuda.is_available():
    model.cuda()


#Define crit and gradient optimizer 
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
 
# define hyper_parameter epochs default = 18 
if argument.epochs == None:
    epochs = 18
else:
    epochs = int(argument.epochs)


# Train model 
for epoch in range(epochs):
    print("Epoch: {}/{}".format(epoch+1, epochs))
     
    model.train()
     
    # Loss and Accuracy within the epoch
    train_loss = train_acc = valid_loss = valid_acc = 0.0
 
    for i, (inputs, labels) in enumerate(dataloaders['train']):
 
        inputs , labels = inputs.to('cuda'),labels.to('cuda')
         
        # Clean existing gradients
        optimizer.zero_grad()
         
        # Forward pass - compute outputs on input data using the model
        outputs = model(inputs)
         
        # Compute loss
        loss = criterion(outputs, labels)
         
        # Backprop .. .. the gradients
        loss.backward()
         
        #Update the parameters
        optimizer.step()
         
        # Compute the total loss for the batch and add it to train_loss
        train_loss += loss.item() * inputs.size(0)
         
        # Compute the accuracy
        ret, predictions = torch.max(outputs.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
         
        # Convert correct_counts to float and then compute the mean
        accuracy = torch.mean(correct_counts.type(torch.FloatTensor))
         
        # Compute total accuracy in the whole batch and add to train_acc
        train_acc += accuracy.item() * inputs.size(0)
         
        print("Batch no: {:03d}, Loss on trainig: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), accuracy.item()))

print('The model is:'+ arch +',Finish traning') 
# Do validation on the test set
print('Check the Accuracy')
def check_accuracy(test_loader  , device='cpu' ):    
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    
check_accuracy(dataloaders['test'])
# TODO: Save the checkpoint 

if (path is None):
    save_path = 'checkpoint.pth'
else:
    save_path = path

model.class_to_idx = image_datasets['train'].class_to_idx

torch.save({ 'Arch' :arch,
             'hidden_units':hidden_units,
             'droupout1':0.3,
             'droupout2':0.2,
             'epochs':18,
             'state_dict'  :model.state_dict(),
             'class_to_idx':model.class_to_idx,
             'optim_dict'  :optimizer.state_dict()},
             save_path)
print("Save model to:" + save_path )