import sys
sys.path.insert(1, '/Feature extraction/backbone')    #some important scripts in here, visualizations, custom datasets, etc
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision as tv
import numpy as np
import glob
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
import torchvision 
import torch
from IPython.display import display,clear_output
import backbone.VISUAL as viz
import backbone.Imagefolder as imf
import importlib
import pickle
import backbone.Custom as cust
from torch import Tensor
import backbone.Test as test
import time

import backbone.MiraBest as mb
importlib.reload(test)

#arguments


model_file = "models/best_resnet18"

Dir = "/directory"






#define device


device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")


model = tv.models.resnet18()

model.fc  = torch.nn.Linear(512, 100)

model.load_state_dict(torch.load(model_file,map_location = "cpu")['model_state_dict'])


model.train(mode = False)

model.fc = torch.nn.Identity()



dataset = torchvision.datasets.ImageFolder(Dir)
datasets = cust.train_val_dataset(dataset, val_split = 0.005)
names = [name[0].split('/')[-1] for name in dataset.imgs]
names = cust.train_val_dataset(names, val_split = 0.005)
val_names = names['train']




classes = imf.classes(Dir)
summ = sum(viz.groups(classes))
print(summ)


#initialize image dataset with appropriate transformations
#datasets = cust.train_val_dataset(dataset, val_split = 20)

importlib.reload(cust)
transformed_dataset = cust.Custom_labelled(datasets['train'],
                                           names = val_names,
                                           resize = resize,
                                           crop = 244,
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]
                                           )   

#transformed_dataset = mb.MBFRConfident(root='./batches', train=True, download=True, transform=transform) 

loader = DataLoader(transformed_dataset, 
                    batch_size, 
                    shuffle = True,
                    num_workers = 16)
time1 = time.time()
rep = []
labells = []
names = []
images = []
name = "_"
label = 0
i = 0
with torch.no_grad():
    for image,label,name in loader:                                   #name
        if i*batch_size > 300000:
            break;
        #images.append(image)
        image = image.to(device)
        rep.append(model(image))                 #Modify to model used here
        labells.append(label)

        names.append(name)                      #name
        i+=1

        clear_output(wait = True)
        display(i*batch_size)


        
#Unwrappping the data
rep2 = []
labells2 = []
rep2 = []
images2 = []
names2 = []



for i in range(len(rep)):
    for j in range(len(rep[i])):
        #images2.append(images[i][j].cpu().numpy()) #Images
        rep2.append(rep[i][j].cpu().numpy())        #Representations
        labells2.append(labells[i][j].item())
        names2.append(names[i][j])                  #Error here if no names
       
rep = rep2
#images = images2 
labels = labells2

names = names2
representations = [rep,labels]

print(len(rep2[1])) 