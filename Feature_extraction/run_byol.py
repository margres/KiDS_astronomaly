import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import importlib
import Features.backbone.Custom as Custom
import Features.backbone.utils_plot as utils_plot
from IPython.display import display,clear_output
import torchvision as tv
import kornia.augmentation as K
import kornia
import pandas as pd
from sklearn.metrics import classification_report
import umap
import sys
from sklearn.preprocessing import MinMaxScaler
import os
sys.path.append("./Features/")
import Features.backbone.MiraBest as mb
import seaborn
import Features.backbone.datasetkids as datasetkids
import Features.backbone.Test as test
import time
importlib.reload(Custom)
import wandb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import plotly.express as px
from astropy.stats import sigma_clip
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
sys.path.append('/users/grespanm/KiDS_astronomaly/astronomaly/preprocessing/')
import cv2
from astropy.stats import sigma_clipped_stats
try:
    import make_rgb
    sys.path.append('/users/grespanm/TEGLIE/teglie_scripts/')
except:
    sys.path.append('/home/grespanm/github/TEGLIE/teglie_scripts/')
from utils_byol import *
import os


pp = '/users/grespanm/KiDS_astronomaly/Feature_extraction'
path_results = os.path.join(pp, 'FE_results')

# Mapping of function objects to their names
function_name_mapping = {
    sigma_clip: 'sigmaclip',
    denoise_bilateral_img: 'denoisebilateral',
    sigma_clipping_gray: 'sigmaclip_gray',
    sigma_70pxsquare_clipping_gray : '50pxsigmaclip_gray'
}


def generate_clean_name(preprocessing_steps, var, scaler):
    preprocessing_names = [function_name_mapping[func] for func in preprocessing_steps]
    preprocessing_str = ','.join(preprocessing_names)
    return f'features&labels_prepbefore_[{preprocessing_str}]_PCA_var_{var}_scaler_{scaler}.npz'


class BYOLTEGLIETest:
    def __init__(self):

        self.g_p = 0.5
        self.v_p = 0.5
        self.h_p = 0.5
        self.g_r = 0.0
        self.r_r = 0.7
        self.r_c = 0.7
        self.valsplit = 0.1
        self.initial_weights = True
        self.continuation = False
        self.epoch_start = 0
        self.epochs = 50
        self.num_workers = 16
        self.batch_size = 32
        self.resize = 300
        self.dataset_name = "TEGLIE 51k"
        self.model_name = "Resnet18"
        self.rep_layer = "avgpool"
        self.input_channel = 3
        self.patience = 5
        self.l_r = 1e-4
        self.best_loss = 5000000
        self.model = tv.models.resnet18(weights="IMAGENET1K_V1")
        self.path_csv = '/users/grespanm/data/table_all_checked.csv'
        self.path_data = '/users/grespanm/data'
        self.path_models = os.path.join(pp, 'models')
        self.project_name = "BYOL TEGLIE test"
        self.tab_teglie = pd.read_csv(self.path_csv)
        self.continuation = False

        #self.wandb = self.initialize_wandb()
        self.augment_fn = self.initialize_augmentations()
        self.device = self.initialize_device()
        self.learner = self.initialize_BYOL()


    def initialize_BYOL(self):

        learner = BYOL(
            self.model,
            image_size = 244,
            hidden_layer =  "avgpool",    # The final output of the network being used is our representations
            augment_fn = self.augment_fn) 
        
        learner = learner.to(self.device)   
  
        return learner
    
    def initialize_augmentations(self):

        augment_fn = torch.nn.Sequential(

                Custom.RandomRotationWithCrop(degrees = [0,360],crop_size =200,p =self.r_r),
                kornia.augmentation.RandomVerticalFlip( p = self.v_p),
                kornia.augmentation.RandomHorizontalFlip( p = self.h_p),

                kornia.augmentation.RandomResizedCrop([244,244],scale =(0.7,1), p = self.r_c),
                K.RandomGaussianBlur(kernel_size = [3,3],sigma = [1,2], p =self.g_p) )

        
        return augment_fn


    def initialize_device(self):

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
            device = torch.device('cuda') 
        else:
            print("No GPU available. Training will run on CPU.")
            device =torch.device("cpu")
        return device

    def initialize_wandb(self):

        wandb.init(
            project=self.project_name,
            resume=False,
            config={
                "learning_rate": self.l_r,
                "architecture": self.model_name,
                "dataset": self.dataset_name,
                "augmentation (Rotationall360)": self.r_r,
                "augmentation (VFlip)": self.v_p,
                "augmentation (HFlip)": self.h_p,
                "augmentation (gblur)": self.g_p,
                "augmentation (crop)": self.r_c,
                "epochs": self.epochs,
                "patience": self.patience,
                "batch size": self.batch_size,
                "val_split": self.valsplit
            }
        )
        #return wandb
        
    def prepare_dataset(self, return_data=False):

        print('Preaparing dataset..')

        kids_id_list = self.tab_teglie['KIDS_ID'].values
        kids_tile_list = self.tab_teglie['KIDS_TILE'].values
        channels = ['r', 'i', 'g']
        label = self.tab_teglie['LABEL'].values

        dataset = datasetkids.KiDSDatasetloader(kids_id_list, kids_tile_list, checkpoint_path=self.path_data,  channels=channels, labels=label)

        none_indices = dataset.check_for_nones()
        if none_indices:
            print(f"Found None values at indices: {none_indices}")
        else:
            print("No None values found in the dataset")

        transformed_dataset = datasetkids.CustomTransformDataset(dataset.data, kids_id_list, labels=label)

        
        datasets = datasetkids.train_val_test_dataset(transformed_dataset)
        self.tot_dataset =  DataLoader(transformed_dataset, batch_size=self.batch_size, shuffle=False)

        self.train_loader = DataLoader(datasets['train'], batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(datasets['val'], batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(datasets['test'], batch_size=self.batch_size, shuffle=True)
        
        if return_data:
            return  DataLoader(self.transformed_dataset, batch_size=self.batch_size, shuffle=False)
    
    def extract_features(self, loader, model, preprocessing_after_byol=None):
         
        rep = []
        labels = []

        batch_size = loader.batch_size if loader.batch_size is not None else 1

        with torch.no_grad():
            i = 0
            for image, label in loader:
                image = image.to(self.device)
                #### here preprocessing before feature extraction
                if preprocessing_after_byol is not None:
                    image_tmp = image.copy()
                    for p in preprocessing_after_byol:
                        image_tmp = p(image_tmp) 
                    image = image_tmp
                rep.append(model(image, return_embedding=True)[1])
                labels.append(label)
                i += 1

        # Unwrapping the data
        rep_tmp = []
        label_tmp = []

        for i in range(len(rep)):
            for j in range(len(rep[i])):
                rep_tmp.append(rep[i][j].cpu().numpy())
                label_tmp.append(labels[i][j].item())

        rep = rep_tmp
        labels = label_tmp

        return rep, labels

    
    def plot_data(self, data, index):

        plt.imshow(data[index])
        plt.show()

    def print_shapes(self, dataloader_data):

        for batch in dataloader_data:
            images, labels = batch
            print(f"Batch image shape: {images.shape}")
            print(f"Batch labels shape: {labels.shape}")
            break  # Only check the first batch

    # Define Training Function
    def train_model(self, learner=None, train_loader=None, val_loader=None, test_loader=None, device=None):

        learner = learner or self.learner
        train_loader = train_loader or self.train_loader
        test_loader = test_loader or self.test_loader
        val_loader = val_loader or self.val_loader
        device = device or self.device

        opt = torch.optim.Adam(learner.parameters(), lr=self.l_r)


        if self.continuation:

            try:

                model_history = torch.load(os.path.join(self.path_models, self.model_name+".pt"),map_location = "cpu")
                self.epoch_start = model_history['epoch']
                self.model.load_state_dict(model_history['model_state_dict'])
                opt.load_state_dict(model_history['optimizer_state_dict'])
                val_accuracies = torch.load(model_history['classification_val_accuracies'])
                train_loss = torch.load(model_history['Training_loss'])
                val_loss = torch.load(model_history['val_loss'])
                print("Continuing from Checkpoint")

            except Exception as e:
                print(f'Exception {e}')
                self.continuation= False

        if not self.continuation:

            #self.epoch_start = 0
            val_accuracies = []
            val_loss_list = []
            train_loss = []
            

            # Self_supervised validation
            learner.eval()
            imgs_test,labels_test = self.extract_features(test_loader,learner)
            metrics = test.KNN_accuracy(imgs_test,labels_test)



            # Log metrics
            wandb.log({
                "Test Accuracy": float(metrics["accuracy"][0]),
                "Test F1 Score": float(metrics["f1"][0]),
                "Test Precision": float(metrics["precision"][0]),
                "Test Recall": float(metrics["recall"][0])
            })

            val_accuracies.append(metrics["accuracy"])

    

        print(self.model_name," Inputchannel",self.input_channel, " rep_layer", self.rep_layer)

        while self.epoch_start <= self.epochs:
             
            print(f'Epoch {self.epoch_start} / {self.epochs}' )

            loss_ = 0.0
            learner.train()
            
            #-----------------------------
            print('Training')

            for i, Images in enumerate(train_loader):
                Images = Images[0]
                # Send images to device
                images = Images.to(device)
                # Obtain loss
                loss = learner(images)

                # Optimization steps
                opt.zero_grad()
                loss.backward()
                opt.step()
                learner.update_moving_average()
                loss_ += loss.item()
                
                if i % 50 == 0:
                    print("Batch epoch :" + str(self.epoch_start) + " Training Loss :" + str(loss.item()))
            print(f'Finished {self.epoch_start} ')
            wandb.log({"Training epoch loss": loss_})
            # Record the training loss for the epoch
            train_loss.append(loss_ / len(train_loader))
            

            print( len(val_loader))
            #Validation step, if applicable
            if len(val_loader) != 0:
                print("Validating")
                val_loss = 0
                with torch.no_grad():
                    learner.eval()
                    for i, val_images in enumerate(val_loader):
                        val_images = val_images[0]
                        val_images = val_images.to(device)
                        v_loss = learner(val_images)
                        val_loss += v_loss.item()
                wandb.log({"Validation epoch loss": val_loss})
                print("Validation loss: ", val_loss)
                val_loss_list.append(val_loss/ len(val_loader))
    
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    torch.save({
                        #'classification_val_accuracies': val_accuracies,
                        'epoch': self.epoch_start,
                        'model_state_dict': self.model.state_dict(),
                        'Training_loss': loss,
                        'Validation_loss': val_loss,
                        'augmentations': self.augment_fn,
                        'optimizer_state_dict': opt.state_dict(),
                    }, os.path.join(self.path_models, "best_" + self.model_name + ".pt"))
                    counter = 0
                else:
                    counter += 1


            #-----------------------------
            # Classification validation
            print('KNN accuracy check')
            imgs_test,labels_test = self.extract_features(test_loader,learner)
            metrics = test.KNN_accuracy(imgs_test,labels_test)
            #print('Precision ')


            # Log metrics
            wandb.log({
                "Test Accuracy": float(metrics["accuracy"][0]),
                "Test F1 Score": float(metrics["f1"][0]),
                "Test Precision": float(metrics["precision"][0]),
                "Test Recall": float(metrics["recall"][0])
            })

            val_accuracies.append(metrics["accuracy"])

            self.epoch_start += 1
    
            torch.save({
                'classification_val_accuracies': val_accuracies,
                'epoch': self.epoch_start,
                'model_state_dict': self.model.state_dict(),
                'Training_loss': loss,
                'Validation loss': val_loss,
                'augmentations': self.augment_fn,
                'optimizer_state_dict': opt.state_dict(),
            }, os.path.join(self.path_models, self.model_name + ".pt"))
        

            
    def save_features_npz(self, features, labels, file_path='features_and_labels.npz'):
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels, dtype=int)
        
        # Save to npz file
        np.savez(file_path, features=features, labels=labels)

    def load_features_npz(self, file_path='features_and_labels.npz'):
        # Load from npz file
        data = np.load(file_path)
        features = data['features']
        labels = data['labels']
        
        return features, labels


    def get_features(self, model, file_path=None, variance_threshold = 0.90, use_scaler=False ):
        
        if file_path is None:
            file_path = '/users/grespanm/KiDS_astronomaly/Feature_extraction/features_and_labels.npz'
        else:
            file_path = os.path.join(path_results, file_path)
            
        if os.path.isfile(file_path):
            print('Features saved - loading it')
            features, labels = self.load_features_npz(file_path)
            print(f'The features have shape {np.shape(features)}')
        else:
            print('Features not found - running feature extraction...')
            self.prepare_dataset()
            tot_loader = self.tot_dataset
            features, labels = self.extract_features(tot_loader, model)
            #run pca
            pca_result = run_pca(features, variance_threshold, use_scaler=use_scaler)
            
            self.save_features_npz(pca_result, labels, file_path)
            print(f'Features extracted and reduced, saved in {file_path}')

        return features, labels

    def run_byol_training(self):
        self.initialize_wandb()
        self.prepare_dataset()
        self.train_model()

          
    def run_feature_extractor(self, variance_threshold=0.9, use_scaler=False, preprocessing_after_byol=None):
         
        checkpoint_path = os.path.join(self.path_models, "best_" + self.model_name + ".pt")
        checkpoint = torch.load(checkpoint_path, map_location = self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.train(mode=False)  # Set the model to evaluation mode
        self.model.fc = torch.nn.Identity()  # Replace the last FC layer with an Identity layer 
        
        name_features_fle =  generate_clean_name(preprocessing_after_byol, variance_threshold, use_scaler)
        print(f'File will be saved with name {name_features_fle}')
        
        self.learner.eval()
        features, labels = self.get_features(self.learner, file_path=name_features_fle, variance_threshold=variance_threshold,  use_scaler=use_scaler )

        #utils_plot.plot_UMAP(pca_result, labels)





if __name__ == "__main__":

    
    preprocessing_after_byol = [sigma_clipping_gray]  
    byol_test = BYOLTEGLIETest()
    #byol_test.run_byol_training()
    byol_test.run_feature_extractor(preprocessing_after_byol = preprocessing_after_byol)
    
    byol_test = BYOLTEGLIETest()
    byol_test.run_feature_extractor(preprocessing_after_byol = preprocessing_after_byol, variance_threshold=0.98)
    

