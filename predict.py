# import Libraries python 
import time
import os 
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
def input_argument():
    '''
    Argparse : create an object from argument 
    -- image_path     : It is  path of predict image   
    -  checkpoint     : it is leraning parameter file to use for predict image  
    -- top_k          : it is the top of classifcation category  
    -- category_names : it is category name of flowers  
    -- GPU            : Use GPU to speed up the training  
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path' , type=str , default='image/' , help = 'It is the directory of image to predict the image (required)' )
    parser.add_argument('--arch', type=str,
                        help=' CNN model to use for image classification ')
    parser.add_argument('--chk_point', type=str , help='It is model checkpoint useing to classify image ')
    parser.add_argument('--top_k', help='The Number of top classification')
    parser.add_argument('--category_names', help='File of name of category ')
    parser.add_argument('--gpu', help='Use GPU to speed up the training ')

    return parser.parse_args()

argument = input_argument()
chk_point = argument.chk_point
img_path = argument.image_path
arch = argument.arch

if(argument.category_names ==  None):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
else:
    filename, file_extension = os.path.splitext(argument.category_names)
    if file_extension != '.json':
        print("Please use file extension .json instead of " + argument.category_names + ".")
        exit()
    else:
        with open(argument.category_names, 'r') as f:
            cat_to_name = json.load(f)
print(chk_point)
if(chk_point == None):
    print("Please load checkpoint which using the model")
else:
    filename, file_extension = os.path.splitext(chk_point)
    if file_extension != '.pth':
        print("Please use file extension .pth instead of " + argument.checkpoint + ".")
        exit()
      
# TODO: Write a function that loads a checkpoint and rebuilds the model

def loading_the_chk_point(path=chk_point):
    if(arch == 'densenet'):
        model = models.densenet121(pretrained=True)
        input_size = 1024
        hidden_units = 500
        output_size = 102
    else:
        model = models.vgg19(pretrained=True)
        input_size = 25088
        hidden_units = 4096
        output_size = 102


    for param in model.parameters():
        param.requires_grad = False
        
        
    checkpoint = torch.load(path)
    model.class_to_idx = checkpoint['class_to_idx']

    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1',nn.Dropout(0.3)),
                                            ('fc2', nn.Linear(hidden_units, output_size)),
                                            ('dropout2',nn.Dropout(0.2)),
                                            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return model

model = loading_the_chk_point(chk_point)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
        
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    image_processing = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    
    image_tensor = image_processing(pil_image) # Edit image 
    processed_image = np.array(image_tensor) # Convert to array 
    processed_image = processed_image.transpose((0, 2, 1)) # Transpose 
    
    return processed_image
 



#Try 
#imshow(process_image("flowers/test/1/image_06752.jpg"))

def predict(image_path, model, topk = argument.top_k):    
    # TODO: Implement the code to predict the class from an image file

    device = tourch.device('cuda:0' if torch.cuda.is_available() else 'cpu ' )
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.to(device)
    img_torch = agumentation(image_path)
    # convert array from numpy to tensor 
    img_torch = torch.from_numpy(img_torch).type(torch.FloatTensor)
    img_torch = img_torch.unsqueeze(0)
    img_torch = img_torch.float()
    # Check what is cpu or GPU 
    with torch.no_grad():
        if device == "cpu":
            output = model.forward(img_torch.cpu())
        elif device == "cuda":
            output = model.forward(img_torch.cuda())
    # Using probability 
    probability = F.softmax(output.data,dim=1)
    probabilies = probability.topk(topk)
    score = np.array(probabilies[0][0])
    idx = 1
    flowers_list = [cat_to_name[str(idx + 1)] for idx in np.array(probabilies[1][0])]
    return score, flowers_list



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = ( std * image ) + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, top_k=argument.top_k , device = 'cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to(device)

    # Find discrete probabilities
    probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers




##image_path = "flowers/test/9/image_06413.jpg"
plt.figure(figsize = (6,10))
ax = plt.subplot(2,1,1)
flower_num = image_path.split('/')[2]
title_ = cat_to_name[flower_num]
img = process_image(img_path)
imshow(img, ax, title = title_);
probs, labs, flowers = predict(img_path, model) 
plt.subplot(2,1,2)
sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
plt.show()

## command : python predict.py --image_path flowers/test/9/image_06413.jpg --arch vgg  --chk_point checkpoint.pth --top_k 3 --gpu gpu 