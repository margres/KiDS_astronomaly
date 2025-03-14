import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import importlib
import Features.backbone.Custom as Custom
from IPython.display import display,clear_output
import torchvision as tv
import pickle
import kornia.augmentation as K
import kornia
import Features.backbone.MiraBest as mb
import Features.backbone.Ini_resnet as inires


import Features.backbone.Test as test
import time
importlib.reload(Custom)
import random
import wandb
import matplotlib

if __name__ == '__main__': 
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22}

    #The code parameters:
    #augmentation probabilitiees
    g_p = 0.5
    v_p = 0.5
    h_p = 0.5
    g_r = 0.0
    r_r = 0.7
    r_c = 0.7
    valsplit = 0.1
    initial_weights = True
    continuation = False

    epochs = 50

    #Dataset arguments
    num_workers = 16
    batch_size = 32

    resize = 300
    
    dataset = "Mirabest"
    #training image floder for galaxy zoo
    galaxyzoo_dir = "/idia/projects/hippo/Koketso/galaxyzoo/galaxy_zoo"
    #galaxyzoo_dir = "/idia/projects/hippo/Koketso/galaxyzoo/galaxy_zoo_12"

    #Class validation image folder
    galaxyzooq_dir = "/idia/projects/hippo/Koketso/galaxyzoo/resized/galaxy_zoo_class_new"



    #Training arguments
    model_name = "ino_resnet"
    rep_layer = "avgpool"
    input_channel = 3
    
    patience = 5
    l_r = 1e-4
    best_loss = 5000000


    wandb.init(
        # set the wandb project where this run will be logged
        project="BYOL Mirabest test",
        resume = False,

        # track hyperparameters and run metadata
        config={
        "learning_rate": l_r,
        "architecture": model_name,
        "dataset": dataset,
        "augmentation (Rotationall360)":r_r,
        "augmentation (VFlip)":v_p,
        "augmentation (HFlip)":h_p,
        "augmentation (gblur)":g_p,
        "augmentation (crop)":r_c,

        "epochs": epochs,
        "patience":patience,
        "batch size":batch_size,
        "val_split":valsplit
        }
    )

    def rename_attribute(obj, old_name, new_name):
        obj._modules[new_name] = obj._modules.pop(old_name)



    #Define the model
    if initial_weights:
        model = tv.models.resnet18(weights = "IMAGENET1K_V1")

        
    if  model_name == "ino_resnet":
        
        #Obtain a version of network with the naming keys matching the ones they use in the state dictionary
        model = inires._resnet18()
        #Gray scale convolutional network with a smaller kernel on the first layer
        model.layers[0][0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)
        #Silence the extra layers
        model.finetuning_layers = torch.nn.Identity()
        #Load file
        file = torch.load("byol.ckpt",map_location=torch.device('cpu'))


        #Select only the weights corresponding to a Resnet online network

        New_dict = {k[8:]:file["state_dict"][k] for k in list(file["state_dict"])[1:121]}

        model.load_state_dict(New_dict)
        
        #move the last pooling layer outsilde one bracket
        rename_attribute(model.layers, '9', "features")
        
        
    if  model_name == "Resnet18_Myweights":
        model = tv.models.resnet18(weights = None)
        model.fc = torch.nn.Linear(512,100)
        model.fc.weight.data.normal_(0,0.01)

        model.load_state_dict(torch.load("Features/models_/Resnet18_ImNet.pt",map_location = "cpu")['model_state_dict'])
        print("Correct_model")
        
        
        
        
    
    #model.weight.data.normal_(0,0.01)
    #Dont need this in our analysis
    #model.fc = torch.nn.Identity()
    #print(model)



    



    # Define a feature extractor for classification based validation


    def features(loader,model,named = True):
        time1 = time.time()
        rep = []
        labells = []
        names = []
        images = []
        name = "_"
        label = 0
        i = 0
        with torch.no_grad():
            if named:
            
                for image,label in loader:                                   #name
                    if i*batch_size > 100000:
                        break
                    #images.append(image)
                    image = image.to(device)
                    rep.append(model(image, return_embedding = True)[1])
                    labells.append(label)

                    names.append(name)                      #name
                    i+=1
            else:
                
                for image,label in loader:                                   #name
                    if i*batch_size > 100000:
                        break
                    #images.append(image)
                    image = image.to(device)
                    rep.append(model(image, return_embedding = True)[1])
                    labells.append(label)

                    i+=1

        #Unwrappping the data
        rep2 = []
        labells2 = []
        rep2 = []
        images2 = []



        for i in range(len(rep)):
            for j in range(len(rep[i])):
                #images2.append(images[i][j].cpu().numpy()) #Images
                rep2.append(rep[i][j].cpu().numpy())        #Representations
                labells2.append(labells[i][j].item())

        rep = rep2
        #images = images2 
        labels = labells2

        return rep,labels


    #The datasets setup

    #training_validation
    if dataset == "Galaxy_zoo":
        print(dataset)

        dataset = Custom.dataset(galaxyzoo_dir)
        names = [name[0].split('/')[-1] for name in dataset.imgs]

        #classification validation

        classification_val_dataset = Custom.dataset(galaxyzooq_dir)

        datasets = Custom.train_val_dataset(dataset, val_split = valsplit)

        #Traning

        transformed_train_dataset = Custom.Custom(datasets['train'],
                                            names = names,
                                            resize = resize,
                                           crop = 244,
                                           )


        loader = DataLoader(transformed_train_dataset, 
                            batch_size, 
                            shuffle = True,
                            num_workers = num_workers)

        #validation

        transformed_val_dataset = Custom.Custom(datasets['val'],
                                            names = names,
                                            resize = resize,
                                           crop = 244,
                                           )

        val_loader = DataLoader(transformed_val_dataset, 
                            batch_size, 
                            shuffle = True,
                            num_workers = num_workers)


        #Classification validation

        transformed_classification_val_dataset = Custom.Custom_labelled(classification_val_dataset,
                                            names = names,
                                            resize = resize,
                                           crop = 244,
                                           )



        class_loader = DataLoader(transformed_classification_val_dataset, 
                            batch_size, 
                            shuffle = True,
                            num_workers = num_workers)
        
    elif dataset ==  "Mirabest":
        print(dataset)
        
        transform = tv.transforms.Compose([
                            tv.transforms.Resize((resize,resize)),
                            tv.transforms.CenterCrop(224),           # So they are compatible with the dnn models
                            tv.transforms.Grayscale(num_output_channels=1),
                            tv.transforms.ToTensor(),
                            tv.transforms.Grayscale(num_output_channels=1),
                            tv.transforms.Normalize(mean=[0.485],std=[0.229])
                            ])
        
        
        transformed_train_dataset = mb.MBFRFull(root='./batches', train=True, download=True, transform=transform) 
        transformed_val_dataset = mb.MBFRFull(root='./batches', train=False, download=True, transform=transform)
        transformed_classification_val_dataset = mb.MBFRConfident(root='./batches', train=True, download=True, transform=transform) 
        batch_size = 32


        loader = DataLoader(transformed_train_dataset, 
                    batch_size, 
                    shuffle = True,
                    num_workers = 15)

        val_loader = DataLoader(transformed_val_dataset, 
                    batch_size, 
                    shuffle = True,
                    num_workers = 15)


        class_loader = DataLoader(transformed_classification_val_dataset, 
                    batch_size, 
                    shuffle = True,
                    num_workers = 15)

        
    
    






    #define the augmentations to be used



    augment_fn = torch.nn.Sequential(



            Custom.RandomRotationWithCrop(degrees = [0,360],crop_size =200,p =r_r),
            kornia.augmentation.RandomVerticalFlip( p = v_p),
            kornia.augmentation.RandomHorizontalFlip( p = h_p),

            kornia.augmentation.RandomResizedCrop([244,244],scale =(0.7,1), p = r_c),
            K.RandomGaussianBlur(kernel_size = [3,3],sigma = [1,2], p =g_p)

    )



    #Define the learner
    print(model)

    learner = BYOL(
        model,
        image_size = 244,
        hidden_layer =  "layers.features",    # The final output of the network being used is our representations
        augment_fn = augment_fn

    )
    #BYOL_pytorch sends a mock tensor to initiate the learner, needs to be uodated for single input models

    # Send to gpu

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    learner = learner.to(device)

    opt = torch.optim.Adam(learner.parameters(), lr=l_r)


    #
    if continuation:

        try:
            epoch = torch.load("Features/models_/"+model_name+".pt",map_location = "cpu")['epoch']

            model.load_state_dict(torch.load("Features/models_/"+model_name+".pt",map_location = "cpu")['model_state_dict'])

            opt.load_state_dict(torch.load("Features/models_/"+model_name+".pt",map_location = "cpu")['optimizer_state_dict'])
            
            val_accuracies = torch.load("Features/models_/"+model_name+".pt",map_location = "cpu")['val_accuracies']
   
            print("Continuing from Checkpoint")
        except:
            epoch = 0
            val_accuracies = []
            # Self_supervised validation
            learner.eval()
            a,b = features(class_loader,learner)
            val_ac = test.KNN_accuracy(a,b)
            wandb.log({"Classification Validation":float(val_ac[0].item())})
            val_accuracies.append(val_ac)
            print("Retraining-from Scratch")

    else:
        epoch = 0
        val_accuracies = []
        # Self_supervised validation
        learner.eval()
        a,b = features(class_loader,learner)
        val_ac = test.KNN_accuracy(a,b)
        wandb.log({"Classification Validation":float(val_ac[0].item())})
        val_accuracies.append(val_ac)
        print("Retraining-from Scratch")




    print(model_name,"  ",dataset,"  Inputchannel",input_channel, " rep_layer", rep_layer)

    # Self_supervised fine_tunning
    while epoch <= epochs:

        loss_ = 0.0
        learner.train()
        for i,Images in enumerate(loader):
            Images = Images[0]
            #send imaged to device
            images = Images.to(device)
            #optain loss
            loss = learner(images)

            #optimization steps
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() #update moving average of target encoder
            loss_ += loss.item()
            loss_per_500 = loss_
            #display(progress)
            if i%5 ==0:
                print("Batch epoch :"+ str(epoch) + " Loss :" + str(loss.item()))


        #Implementing the early stopping

        # Validate
        learner.eval()
        clear_output(wait = True)
        print("Validating")
        with torch.no_grad():
            val_loss = 0
            for i, val_images in enumerate(val_loader):

                val_images = val_images[0]
                val_images = val_images.to(device)
                v_loss = learner(val_images)
                val_loss += v_loss.item() 
            wandb.log({"Validation epoch loss": val_loss})
            wandb.log({"Training epoch loss": loss_})


            print("Validation loss: ",val_loss)

        if val_loss < best_loss:

            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'augmentations': self.augment_fn,
                }, "./Features/models_/"+"best_"+model_name+".pt")


            counter = 0
        else:
            counter += 1


        #Classification validatation
        a,b = features(class_loader,learner)
        val_ac = test.KNN_accuracy(a,b)
        wandb.log({"Classification Validation":float(val_ac[0].item())})
        val_accuracies.append(val_ac)

        epoch +=1
        

        torch.save({
                'val_accuracies':val_accuracies,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss,
                'augmentations':self.augment_fn,
                'optimizer_state_dict': opt.state_dict(),
                }, "./Features/models_/"+model_name+".pt")  


    """    

        # Check if early stopping criteria are met
        if counter >= patience:
            print("Early stopping: No improvement in validation loss for {} epochs".format(patience))
            wandb.finish()
            break
    """        
