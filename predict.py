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
  parser.add_argument('--arch', type=str, default='densenet121',
                        help='CNN model to use  default is densenet121')
 
  return parser.parse_args()


args = get_input_args()

data_dir = args.dir
arch = args.arch

with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
else:
    filename, file_extension = os.path.splitext(category_names)
    if file_extension != '.json':
        print("Please use file extension .json instead of " + category_names + ".")
        exit()
    else:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def loading_model(checkpoint_path):
    
    check_path = torch.load(checkpoint_path)
 
    if (arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
        input_size = 1024
        hidden_units = 500
        output_size = 102
            
    for param in model.parameters():
        param.requires_grad = False
        
    model.class_to_idx = check_path['class_to_idx']
                    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1',nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(hidden_units, output_size)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    model.load_state_dict(check_path['state_dict'])
    ####print("The model is loaded to" + save_dir)
    return model

model = loading_model('save_checkpoint.pth')

def agumentation(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # using PIL model 
    pil_image = Image.open(image)
    
    # Edit
    edit_image = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Dimension
    img_tensor = edit_image(pil_image) # Edit image 
    processed_image = np.array(img_tensor) # Convert to array 
    processed_image = processed_image.transpose((0, 2, 1)) # Transpose 
    
    return processed_image



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # using Undo preprocessing to get the orginal image 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = ( std * image ) + mean
    # trying the open image put the image not have the same display we need 
    # some modyifycation 
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
#Try 
#imshow(agumentation("flowers/test/1/image_06752.jpg"))

def predict(image_path, model, topk = top_k):
 
    
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


def print_top(image_path, model, save_result_dir):
   
    # Setting plot area
    plt.figure(figsize = (3,6))
    axis = plt.subplot(1,1,1)
    # Display test flower
    img = process_image(image_path)
    get_title  = image_path.split('/')
    print(cat_to_name[get_title[2]])
    
    imshow(img, axis, title = cat_to_name[get_title[2]]);
    
    # Making prediction
    score, flowers_list = predict(image_path, model) 
    fig,ax = plt.subplots(figsize=(10,10))
    sticks = np.arange(len(flowers_list))
    axis.barh(sticks, score, height=0.3, linewidth=2.0, align = 'center')
    axis.set_yticks(ticks = sticks)
    axis.set_yticklabels(flowers_list)
    #plt.show()
    plt.savefig(save_result_dir)

image_path = 'flowers/test/28/image_05277.jpg'
get_title  = image_path.split('/')
print("Test image:" + cat_to_name[get_title[2]])
save_result_dir = 'save_prediction_result_1'
display_top(image_path, model, save_result_dir)
print("Save prediction result to:" + save_result_dir)
print("Prediction result:")
score, flower_list = predict(image_path, model)
print(flower_list)
print(np.exp(score))
print("-------------------------------------------")
image_path = 'flowers/test/1/image_06752.jpg'
get_title  = image_path.split('/')
print("Test image:" + cat_to_name[get_title[2]])
save_result_dir = 'save_prediction_result_2'
display_top(image_path, model, save_result_dir)
print("Save prediction result to:" + save_result_dir)
print("Prediction result:")
score, flower_list = predict(image_path, model)
print(flower_list)
print(np.exp(score))
print("-------------------------------------------")
