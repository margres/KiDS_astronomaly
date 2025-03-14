import torch
from byol_pytorch import BYOL
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import importlib
import Custom
# import utils_plot
# from IPython.display import display,clear_output
import torchvision as tv
import kornia.augmentation as K
import kornia
import pandas as pd
# from sklearn.metrics import classification_report
import sys
from sklearn.preprocessing import MinMaxScaler
import os, glob
import datasetkids as datasetkids
import Test as test
import wandb
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
# from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
# from astropy.stats import sigma_clipped_stats
from tqdm import tqdm
import logging
import psutil 
import gc
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm  # Import tqdm for progress bars

from collections import defaultdict



# Dynamically resolve the base path for the user
base_path = os.path.expanduser("~")  # Automatically resolves to '/home/<user>' or '/users/<user>'

# if 'home' in base_path:
#     base_path = '/home/astrodust/mnt/github/'
#     #os.path.join(base_path,'mnt', 'github')

## different paths in different machines
if 'home/grespanm' in base_path:
    base_path = os.path.join(base_path, 'github')
elif 'home/astrodust' in base_path:
    base_path = os.path.join(base_path, 'mnt','github')
print(base_path)

sys.path.append( os.path.join(base_path,'TEGLIE/teglie_scripts/'))

# #sys.path.append('/users/grespanm/KiDS_astronomaly/astronomaly/preprocessing/')
# try:
#     sys.path.append('/users/grespanm/TEGLIE/teglie_scripts/')
#     import make_rgb
# except:
#     sys.path.append('/home/grespanm/github/TEGLIE/teglie_scripts/')
#     base_path = os.path.join(base_path, 'github')

from utils_byol import *


pp = os.path.join(base_path,'KiDS_astronomaly/Feature_extraction')
path_results = os.path.join(pp, 'FE_results')

# Mapping of function objects to their names
function_name_mapping = {
    sigma_clip: 'sigmaclip',
    denoise_bilateral_img: 'denoisebilateral',
    sigma_clipping_gray: 'sigmaclipgray',
    sigma_70pxsquare_clipping_gray : '50pxsigmaclip_gray'
}


class BYOLTEGLIETest:

    def __init__(self, 
                path_csv=None,
                tab_kids=None, 
                preprocessing_after_byol=None,
                variance_threshold=0.97, 
                path_npz_rgb=None,
                dataset_name='BYOL'):
        """
        Initialize the BYOLTEGLIETest class.

        Parameters:
        - path_csv (str, optional): Path to the CSV or Parquet file containing KIDS data.
        - tab_kids (pd.DataFrame, optional): Directly pass a preloaded DataFrame instead of loading from a file.
        """

        # Hyperparameters and settings
        self.g_p = 0.5
        self.v_p = 0.5
        self.h_p = 0.5
        self.g_r = 0.0
        self.r_r = 0.7
        self.r_c = 0.7
        self.path_npz_rgb = path_npz_rgb
        self.variance_threshold = variance_threshold
        self.path_csv = path_csv
        self.preprocessing_after_byol =  preprocessing_after_byol
        self.valsplit = 0.1
        self.initial_weights = True
        self.continuation = False
        self.epoch_start = 0
        self.epochs = 50
        self.num_workers = 16
        self.batch_size = 32
        self.resize = 300
        self.dataset_name = dataset_name
        self.model_name = f"Resnet18_{self.dataset_name}"
        self.rep_layer = "avgpool"
        self.input_channel = 3
        self.patience = 5
        self.l_r = 1e-4
        self.best_loss = 5000000
        self.model = tv.models.resnet18(weights="IMAGENET1K_V1")

        # Paths
        self.path_folder_data = os.path.join(base_path, 'data')
        self.path_models = os.path.join(pp, 'models')
        self.project_name = self.dataset_name 


        # Load tab_kids: Either from argument or from a file
        self.tab_kids = self.load_tab_kids(path_csv, tab_kids)

        # Augmentations and Device Initialization
        self.continuation = False
        self.augment_fn = self.initialize_augmentations()
        self.device = self.initialize_device()
        self.learner = self.initialize_BYOL()

    def generate_clean_name(self, preprocessing_steps, var, scaler):
        preprocessing_names = [function_name_mapping[func] for func in preprocessing_steps]
        preprocessing_str = ','.join(preprocessing_names)
        return f'{self.dataset_name}try_features&labels_prepbefore_[{preprocessing_str}]_PCA_var_{var}_scaler_{scaler}.npz'



    def load_tab_kids(self, path_csv= None , tab_kids=None):
        """
        Loads the KIDS dataset from a provided DataFrame or from a CSV/Parquet file.

        Parameters:
        - path_csv (str, optional): Path to the CSV or Parquet file.
        - tab_kids (pd.DataFrame, optional): Preloaded DataFrame.

        Returns:
        - pd.DataFrame: The loaded KIDS dataset.
        """
        #print(tab_kids)
        if tab_kids is not None:
            print("Using provided DataFrame.")
            return tab_kids

        # Default path if not provided
        if path_csv is None:
            path_csv = self.path_csv 

                #os.path.join(base_path, 'data/big_dataframe.parquet')
        #os.path.join(base_path, '/users/grespanm/data/table_all_checked.csv')

        # Check file format and load accordingly
        if path_csv.endswith(".csv"):
            print(f"Loading CSV file: {path_csv}")
            return pd.read_csv(path_csv)
        elif path_csv.endswith(".parquet"):
            print(f"Loading Parquet file: {path_csv}")
            return pd.read_parquet(path_csv)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")


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
        
    def prepare_dataset(self, 
                    return_data=False, 
                    path_npz_rgb=None, 
                    kids_id_list=None, 
                    kids_tile_list=None, 
                    channels=None):

        '''
        it creates the dataset based on the given IDs
        '''

        print('Preparing dataset...')
        path_npz_rgb = path_npz_rgb or self.path_npz_rgb

        # Assign defaults if inputs are None
        if kids_id_list is None:
            try:
                kids_id_list = self.tab_kids['KIDS_ID'].values
            except KeyError:
                kids_id_list = self.tab_kids['ID'].values
        
        if kids_tile_list is None:
            kids_tile_list = self.tab_kids['KIDS_TILE'].values
        
        if channels is None:
            channels = ['r', 'i', 'g']
        
        if 'LABEL' not in self.tab_kids.columns:
            self.tab_kids['LABEL'] = -1

        label = self.tab_kids['LABEL'].values

        dataset = datasetkids.KiDSDatasetloader(df_kids =self.tab_kids, kids_id_list = kids_id_list, kids_tile_list=kids_tile_list, rgb_img_file_name= path_npz_rgb, checkpoint_path=self.path_folder_data,  channels=channels, labels=label)

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
    
    # def extract_features(self, loader, model, preprocessing_after_byol= None ):
         
    #     rep = []
    #     labels = []

    #     # batch_size = loader.batch_size if loader.batch_size is not None else 1

    #     if preprocessing_after_byol is None:
    #         preprocessing_after_byol = self.preprocessing_after_byol

    #     with torch.no_grad():
    #         for image, label in loader:
    #             #### here preprocessing before feature extraction
    #             if preprocessing_after_byol is not None:
    #                 image_tmp = image.cpu().numpy().copy()
    #                 print(np.shape(image))
    #                 print(np.shape(image_tmp))
    #                 for p in preprocessing_after_byol:
    #                     image_tmp = p(image_tmp) 
    #                 image = image_tmp
    #             rep.append(model(image, return_embedding=True)[1])
    #             labels.append(label)
        

    #     # Unwrapping the data
    #     rep_tmp = []
    #     label_tmp = []

    #     for i in range(len(rep)):
    #         for j in range(len(rep[i])):
    #             rep_tmp.append(rep[i][j].cpu().numpy())
    #             label_tmp.append(labels[i][j].item())

    #     rep = rep_tmp
    #     labels = label_tmp

    #     return rep, labels

    # def extract_features(self, data_loader, learner):
    #     """
    #     Extracts features from the data loader and applies preprocessing.

    #     Parameters:
    #         data_loader: PyTorch DataLoader object.
    #         learner: Model used for feature extraction.

    #     Returns:
    #         Extracted feature set and labels.
    #     """
    #     imgs_test, labels_test = [], []
    #     device = next(learner.parameters()).device  # Get the device of the model

    #     total_batches = len(data_loader)

    #     # Iterate over batches with progress bar
    #     for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting Features", unit="batch", total=total_batches)):
    #         images, labels = batch  # Unpack batch

    #         batch_features = []
    #         batch_labels = []

    #         # Iterate over each image in the batch
    #         for i in range(len(images)):  
    #             img = images[i].to(device)  # Send image to model's device
    #             label = labels[i]

    #             img_np = img.cpu().numpy()  # Convert to NumPy
    #             print(np.shape(img_np))
    #             # Apply sigma clipping only if image has 3 channels
    #             if img_np.shape[0] == 3:  
    #                 # If the first dimension is 3 or less, assume (C, H, W) format
    #                 if img_np.shape[0] <= 4:  
    #                     img_np = np.transpose(img_np, (1, 2, 0))  # Convert (C, H, W) → (H, W, C) for sigma clipping
    #                     print(f'Before Sigma Clip: {np.shape(img_np)}')


    #                 img_np = sigma_clipping_gray(img_np)  
    #                 print(f'After Sigma Clip: {np.shape(img_np)}')

    #                 # Convert back to (C, H, W) format for ResNet
    #                 img_np = np.transpose(img_np, (2, 0, 1))  
    #                 print(f'After Transpose Back for ResNet: {np.shape(img_np)}')

    #             # Convert back to tensor and add batch dimension for ResNet
    #             img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).to(device)
                
    #             print(f'After Transpose Back for ResNet: {np.shape(img_tensor)}')

    #             # Extract features using the learner (ResNet)
    #             with torch.no_grad():
    #                 features = learner(img_tensor)  # Forward pass

    #             print(f'Features Extracted Shape: {features.shape}')

    #             batch_features.append(features.cpu().numpy())  # Convert to NumPy and store
    #             batch_labels.append(label.cpu().numpy())

    #         imgs_test.extend(batch_features)
    #         labels_test.extend(batch_labels)

    #     # Convert lists to NumPy arrays and reshape appropriately
    #     imgs_test = np.array(imgs_test).reshape(len(imgs_test), -1)
    #     labels_test = np.array(labels_test).flatten()

    #     print(f"Final Extracted Features Shape: {imgs_test.shape}")
    #     print(f"Final Labels Shape: {labels_test.shape}")

    #     return imgs_test, labels_test
    import matplotlib.pyplot as plt

        

    
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
        """
        Trains the model with optional continuation from checkpoints.
        """
        # Assign default values if None
        learner = learner or self.learner
        train_loader = train_loader or self.train_loader
        test_loader = test_loader or self.test_loader
        val_loader = val_loader or self.val_loader
        device = device or self.device

        opt = torch.optim.Adam(learner.parameters(), lr=self.l_r)
        # **Load checkpoint if continuation is enabled**
        metric_history = defaultdict(list) 
        if self.continuation:
            print("Continuing training from checkpoint...")
            try:
                checkpoint_path = os.path.join(self.path_models, self.model_name + ".pt")
                model_history = torch.load(checkpoint_path, map_location="cpu")

                self.epoch_start = model_history['epoch']
                self.model.load_state_dict(model_history['model_state_dict'])
                # self.best_combined_f1 = model_history.get('best_combined_f1', 0.0)
                # opt.load_state_dict(model_history['optimizer_state_dict'])
                # val_accuracies = model_history.get('classification_val_accuracies', [])
                train_loss = model_history.get('Training_loss', [])
                val_loss_list = model_history.get('Validation_loss', [])
        
                metric_history = model_history.get('metric_history', defaultdict(list))
                print("Checkpoint loaded successfully.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Training from scratch.")
                self.continuation = False

        # **If no checkpoint, initialize training from scratch**
        if False:
            if not self.continuation:
                print("Training from scratch...")
                self.epoch_start = 0
                self.best_combined_f1 = 0.0
                
                learner.eval()
                imgs_test, labels_test = self.extract_features(test_loader, learner)
                metrics = test.KNN_accuracy(imgs_test, labels_test, n_neighbors =  np.tile(np.arange(5, 55, 5), 5), save_path=self.project_name, epoch=self.epoch_start)
    
                for key in metrics.keys():
                    metric_history[key].append(metrics[key][0])  
    
                wandb.log({key: metrics[key][0] for key in metrics.keys()})
        
        print(f"{self.model_name}: Input Channels={self.input_channel}, Rep Layer={self.rep_layer}")

                    
        # **Training loop**
        while self.epoch_start <= self.epochs:
            print(f"Epoch {self.epoch_start}/{self.epochs}")

            learner.train()
            epoch_loss = 0.0

            print("Training...")
            for i, (images, _) in enumerate(train_loader):
                images = images.to(device)

                # Compute loss and update model
                loss = learner(images)
                opt.zero_grad()
                loss.backward()
                opt.step()
                learner.update_moving_average()

                epoch_loss += loss.item()
                if i % 50 == 0:
                    print(f"Batch {i} | Training Loss: {loss.item():.6f}")

            # **Log and store training loss**
            avg_train_loss = epoch_loss / len(train_loader)
            metric_history["Training Loss"].append(avg_train_loss)
            wandb.log({"Training Epoch Loss": avg_train_loss})

            # **Validation step**
            if len(val_loader) > 0:
                print("Validating...")
                learner.eval()
                val_loss = 0.0

                with torch.no_grad():

                    for batch in val_loader:

                        if isinstance(batch, tuple) and len(batch) == 2:  
                            val_images, val_labels = batch
                        else:
                            val_images = batch 
                        
                        val_images = val_images[0]
                        #print(np.shape(val_images))
                        #print(val_labels)
                        val_images = val_images.to(device)
                        val_outputs = learner(val_images)
                        val_loss += val_outputs.item()


                avg_val_loss = val_loss / len(val_loader)
                metric_history["Validation Loss"].append(avg_val_loss)
                wandb.log({"Validation Epoch Loss": avg_val_loss})
                print(f"Validation Loss: {avg_val_loss:.6f}")

                # **Compute Updated Metrics (KNN Accuracy)**
                print("Running KNN accuracy check...")
                imgs_test, labels_test = self.extract_features(test_loader, learner)
                if len(imgs_test)==0:
                    print('No images laoded')
                if len(labels_test)==0:
                    print('No labels loaded')
                # metrics = test.KNN_accuracy(imgs_test, labels_test,  n_neighbors =  np.tile(np.arange(5, 55, 5), 5), save_path=self.project_name, epoch=self.epoch_start)

                # # **Store and log metrics**
                # for key in metrics.keys():
                #     metric_history[key].append(metrics[key][0])

                # wandb.log({key: metrics[key][0] for key in metrics.keys()})

                # print(f"Test Accuracy: {metrics['Accuracy'][0]:.4f}")
                # print(f"F1 (Macro): {metrics['F1 Macro'][0]:.4f} | F1 (Weighted): {metrics['F1 Weighted'][0]:.4f}")
                # print(f"F1 (Grade 1): {metrics['F1 Grade 1'][0]:.4f} | F1 (Grade 2): {metrics['F1 Grade 2'][0]:.4f}")

                # # **Compute Combined F1 Score for Model Saving**
                # combined_f1_score = metrics["F1 Macro"][0] + metrics["F1 Grade 1"][0] + metrics["F1 Grade 2"][0]

            #if combined_f1_score > self.best_combined_f1:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                # self.best_combined_f1 = combined_f1_score

                checkpoint_data = {
                    'epoch': self.epoch_start,
                    'model_state_dict': self.model.state_dict(),
                    'Training_loss': loss,
                    'Validation_loss': avg_val_loss,
                    # 'best_combined_f1': self.best_combined_f1,
                    # 'metric_history': metric_history,
                    # 'best_f1_macro': metrics["F1 Macro"][0],
                    # 'best_f1_weighted': metrics["F1 Weighted"][0],
                    # 'best_f1_grade_1': metrics["F1 Grade 1"][0],
                    # 'best_f1_grade_2': metrics["F1 Grade 2"][0],
                    # 'best_precision_macro': metrics["Precision Macro"][0],
                    # 'best_recall_macro': metrics["Recall Macro"][0],
                    # 'best_test_accuracy': metrics["Accuracy"][0],
                    'augmentations': self.augment_fn,
                    'optimizer_state_dict': opt.state_dict(),
                }

                torch.save(checkpoint_data, os.path.join(self.path_models, "best_" + self.model_name + ".pt"))

            self.epoch_start += 1

            # **Save latest model state (not necessarily best)**
            torch.save({
                'epoch': self.epoch_start,
                'model_state_dict': self.model.state_dict(),
                'Training_loss': loss,
                'Validation_loss': avg_val_loss,
                # 'metric_history': metric_history,
                # 'best_combined_f1': self.best_combined_f1,
                # 'last_f1_macro': metrics["F1 Macro"][0],
                # 'last_f1_weighted': metrics["F1 Weighted"][0],
                # 'last_f1_grade_1': metrics["F1 Grade 1"][0],
                # 'last_f1_grade_2': metrics["F1 Grade 2"][0],
                # 'last_precision_macro': metrics["Precision Macro"][0],
                # 'last_recall_macro': metrics["Recall Macro"][0],
                # 'last_test_accuracy': metrics["Accuracy"][0],
                'augmentations': self.augment_fn,
                'optimizer_state_dict': opt.state_dict(),
            }, os.path.join(self.path_models, self.model_name + ".pt"))

        

            
    def save_features_npz(self, features, labels, ids, file_path='features_and_labels.npz'):
        """
        Save features, labels, and IDs to an .npz file.

        Parameters:
            features (np.ndarray): Extracted features.
            labels (np.ndarray): Corresponding labels.
            ids (list or np.ndarray): Corresponding IDs.
            file_path (str): Path to save the .npz file.
        """
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels, dtype=int)
        ids = np.array(ids)

        # Save to npz file
        np.savez(file_path, features=features, labels=labels, ids=ids)
        print(f"Features, labels, and IDs saved to {file_path}")
     


    def load_features_npz(self, file_path='features_and_labels.npz'):
        """
        Load features, labels, and IDs from an .npz file.

        Parameters:
            file_path (str): Path to the .npz file.

        Returns:
            Tuple: (features, labels, ids)
        """
        # Load from npz file
        data = np.load(file_path,  allow_pickle=True)
        features = data['features']
        labels = data['labels']
        ids = data['ids']

        print(f"Loaded features, labels, and IDs from {file_path}")
        return features, labels, ids

    def run_byol_training(self, 
                        path_npz_rgb=None, 
                        kids_id_list=None, 
                        kids_tile_list=None, 
                        channels=None):
        """
        Runs BYOL training by preparing the dataset and training the model.

        Parameters:
            path_npz_rgb (str): Path to the RGB .npz file.
            kids_id_list (list or np.ndarray): List of KIDS IDs. If None, defaults to self.
            kids_tile_list (list or np.ndarray): List of KIDS tiles. If None, defaults to self.
            channels (list): List of channels (e.g., ['r', 'i', 'g']). If None, defaults to self or predefined values.
        """

        path_npz_rgb = path_npz_rgb or self.path_npz_rgb

        # Initialize logging or tracking
        self.initialize_wandb()

        # Prepare the dataset with flexible inputs
        self.prepare_dataset(
            path_npz_rgb=path_npz_rgb, 
            kids_id_list=kids_id_list, 
            kids_tile_list=kids_tile_list, 
            channels=channels
        )
        print('Training..........')
        # Train the BYOL model
        self.train_model()

    def create_dataloader(self, batch_ids, batch_size):
        """
        Create a DataLoader for the given batch IDs.

        Parameters:
            batch_ids (list): IDs for the current batch.
            batch_size (int): Size of the batch.

        Returns:
            DataLoader: DataLoader for the batch.
        """
        dataset = Subset(self.tot_dataset, batch_ids)  # Assuming tot_dataset is a PyTorch Dataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def generate_batch_file_path(self,batch_size, i):
        """
        Generates a file path based on batch size and the current index.

        Parameters:
        - batch_size (int): The size of each batch.
        - i (int): The current index in the dataset.

        Returns:
        - str: A formatted file path string.
        """
        return f"features_in_batches/batchsize_{batch_size}_features_{i // batch_size}_2_half.npz"

    def append_and_save_npz(self,files, output_file):
        """
        Append features, labels, and IDs from multiple .npz files and save them into one.

        Parameters:
            files (list): List of file paths to .npz files.
            output_file (str): Path to save the combined .npz file.
        """
        all_features = []
        all_labels = []
        all_ids = []

        for file in files:
            # Load data
            data = np.load(file, allow_pickle=True)
            features = data['features']
            labels = data['labels']
            ids = data['ids']

            # Append to lists
            all_features.append(features)
            all_labels.append(labels)
            all_ids.append(ids)

        # Concatenate all features, labels, and IDs
        combined_features = np.vstack(all_features)
        combined_labels = np.hstack(all_labels)
        combined_ids = np.hstack(all_ids)

        # Save the combined data to a new .npz file
        np.savez(output_file, features=combined_features, labels=combined_labels, ids=combined_ids)
        print(f"Combined features, labels, and IDs saved to {output_file}")

    def extract_features(self, data_loader, learner):
        """
        Efficiently extracts features from the data loader in batches.

        Parameters:
            data_loader: PyTorch DataLoader object.
            learner: Model used for feature extraction.

        Returns:
            Extracted feature set and labels as NumPy arrays.
        """
        device = next(learner.parameters()).device  # Get the device of the model

        all_features, all_labels = [], []
        total_batches = len(data_loader)

        if total_batches == 0:
            raise ValueError("❌ DataLoader is empty! No batches available for feature extraction.")

        # Iterate over batches with progress bar
        for batch_idx, (images, labels) in enumerate(tqdm(data_loader, desc="Extracting Features", unit="batch")):
            images = images.to(device)  # Move batch to device
            labels = labels.to(device)

            batch_size = images.shape[0]  # Get batch size
            #print(f"Processing Batch {batch_idx+1}/{total_batches} | Batch Size: {batch_size}")

            if batch_size == 0:
                print(f"⚠️ Warning: Batch {batch_idx+1} is empty! Skipping...")
                continue  # Skip empty batches

            # Apply sigma clipping if needed
            if images.shape[1] == 3:  # Ensure 3 channels (C, H, W)
                images_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) → (B, H, W, C)
                images_np = np.array([sigma_clipping_gray(img) for img in images_np])  # Apply sigma clipping
                images_np = images_np.transpose(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
                images = torch.tensor(images_np, dtype=torch.float32).to(device)

            # Extract features using the learner (ResNet)
            with torch.no_grad():
                learner.eval()  
                batch_features = learner(images, return_embedding= True)[1]
             


            # Check if `batch_features` is valid
            if batch_features is None or not isinstance(batch_features, torch.Tensor):
                print(f"⚠️ Warning: No features extracted for Batch {batch_idx+1}. Skipping...")
                continue

            # Ensure the output has a valid shape
            if batch_features.ndim == 0 or batch_features.shape[0] == 0:
                print(f"⚠️ Warning: Invalid feature shape {batch_features.shape} for Batch {batch_idx+1}. Skipping...")
                continue  

            # Convert to NumPy and store
            all_features.append(batch_features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Handle case where no features were extracted
        if len(all_features) == 0:
            raise ValueError("❌ No features were extracted! Check model output and input data.")

        # Concatenate all batches into a single array
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        print(f"✅ Final Extracted Features Shape: {all_features.shape}")
        print(f"✅ Final Labels Shape: {all_labels.shape}")

        return all_features, all_labels


    def get_features(self, model, 
                    file_path=None,
                    path_npz_rgb=None, 
                    kids_id_list=None, 
                    kids_tile_list=None, 
                    channels=None, 
                    overwrite=False):
        
        if path_npz_rgb is None:
            path_npz_rgb = self.path_npz_rgb


        if file_path is None:
            file_path = '/users/grespanm/KiDS_astronomaly/Feature_extraction/features_and_labels.npz'
        else:
            file_path = os.path.join(path_results, file_path)

        # Check if file exists and decide based on overwrite flag
        if os.path.isfile(file_path) and not overwrite:
            print('Features saved - loading it')
            features, labels = self.load_features_npz(file_path)
            print(f'The features have shape {np.shape(features)}')
        else:
            if os.path.isfile(file_path) and overwrite:
                print('Overwrite is enabled - recreating features...')
            else:
                print('Features not found - running feature extraction...')
            
            # Prepare the dataset
            self.prepare_dataset(
                path_npz_rgb=path_npz_rgb, 
                kids_id_list=kids_id_list, 
                kids_tile_list=kids_tile_list,
                channels=channels
            )
            tot_loader = self.tot_dataset

            # Extract features and labels
            features, labels = self.extract_features(tot_loader, model)

        return features, labels
    

    def run_feature_extractor(self, 
                              variance_threshold=None, 
                              use_scaler=False, 
                              preprocessing_after_byol=None, 
                              path_npz_rgb=None, 
                              kids_id_list=None, 
                              kids_tile_list=None, 
                              channels=None):
        """
        Runs the feature extractor without batch processing, applying dimensionality reduction and saving results.
    
        Parameters:
            variance_threshold (float): Variance threshold for PCA.
            use_scaler (bool): Whether to use a scaler during PCA.
            preprocessing_after_byol (str): Post-processing steps after BYOL.
            path_npz_rgb (str): Path to the RGB .npz file.
            kids_id_list (list or np.ndarray): List of KIDS IDs. If None, defaults to self.
            kids_tile_list (list or np.ndarray): List of KIDS tiles. If None, defaults to self.
            channels (list): List of channels (e.g., ['r', 'i', 'g']). If None, defaults to self or predefined values.
        """

        variance_threshold = variance_threshold or self.variance_threshold
        preprocessing_after_byol = preprocessing_after_byol or self.preprocessing_after_byol
        path_npz_rgb = path_npz_rgb or self.path_npz_rgb
            
        # Load the best model checkpoint
        checkpoint_path = os.path.join(self.path_models, "best_" + self.model_name + ".pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # self.model = tv.models.resnet18(weights="IMAGENET1K_V1")
        # self.model.eval()  # Set the model to evaluation mode
        self.model.fc = torch.nn.Identity()  # Replace the last FC layer with an Identity layer 

        # Generate a clean name for the feature file
        name_features_fle = self.generate_clean_name(preprocessing_after_byol, variance_threshold, use_scaler)
        print(f'File will be saved with name {name_features_fle}')

        features, labels = self.get_features(
            self.learner, 
            file_path=name_features_fle, 
            path_npz_rgb=path_npz_rgb,
            kids_id_list=kids_id_list, 
            kids_tile_list=kids_tile_list, 
            channels=channels,
            overwrite=True
        )

        features = np.array(features)
        labels = np.array(labels)
        # Remove invalid data (-1 rows)
        # valid_indices = [i for i, row in enumerate(features) if not np.array_equal(row, [-1])]
        # features = features[valid_indices]
        # labels = labels[valid_indices]
    
        # Apply PCA to reduce dimensionality
        print(f"Features shape before PCA: {features.shape}")
        pca_result = run_pca(features, variance_threshold, use_scaler=use_scaler)
    
        name_features_fle = os.path.join(path_results, name_features_fle)

        # Save the reduced features and labels
        self.save_features_npz(pca_result, labels, kids_id_list, name_features_fle)
    
        print(f'Features extracted and reduced, saved in {name_features_fle}')
    


        # Uncomment the following line if UMAP or PCA visualization is required
        # utils_plot.plot_UMAP(pca_result, labels)
 

    def run_feature_extractor_batches(self, 
                            variance_threshold=None, 
                            use_scaler=False, 
                            preprocessing_after_byol=None, 
                            path_npz_rgb=None, 
                            kids_id_list=None, 
                            kids_tile_list=None, 
                            channels=None, 
                            batch_size=25000,
                            overwrite=False,
                            incremental_pca=True): 
        #cahnge this to false if wou dont have enough memory!
        """
        Runs the feature extractor in batches, applying dimensionality reduction and saving results.

        Parameters:
            variance_threshold (float): Variance threshold for PCA.
            use_scaler (bool): Whether to use a scaler during PCA.
            preprocessing_after_byol (str): Post-processing steps after BYOL.
            path_npz_rgb (str): Path to the RGB .npz file.
            kids_id_list (list or np.ndarray): List of KIDS IDs. If None, defaults to self.
            kids_tile_list (list or np.ndarray): List of KIDS tiles. If None, defaults to self.
            channels (list): List of channels (e.g., ['r', 'i', 'g']). If None, defaults to self or predefined values.
            batch_size (int): Number of samples to process in each batch.
        """

        variance_threshold = variance_threshold or self.variance_threshold
        preprocessing_after_byol = preprocessing_after_byol or self.preprocessing_after_byol
        path_npz_rgb = path_npz_rgb or self.path_npz_rgb
        kids_id_list = kids_id_list or self.kids_id_list
        kids_tile_list = kids_tile_list or self.kids_tile_list
        channels = channels or self.channels

        # Initialize logger
        # Configure logger to write to a file and console
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("feature_extraction.log"),  # Save to a file
                logging.StreamHandler()  # Print to console
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("Starting feature extraction...")

        # Load the best model checkpoint
        checkpoint_path = os.path.join(self.path_models, "best_" + self.model_name + ".pt")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # Set the model to evaluation mode
        self.model.fc = torch.nn.Identity()  # Replace the last FC layer with an Identity layer 

        # Generate a clean name for the feature file
        name_features_fle = self.generate_clean_name(preprocessing_after_byol, variance_threshold, use_scaler)
        logger.info(f"File will be saved with name {name_features_fle}")

        # Evaluate the learner and extract features in batches
        # self.learner = self.initialize_BYOL(self.model)
        # self.learner.eval()


        num_samples = len(kids_id_list)
        logger.info(f"Processing {num_samples} samples in batches of {batch_size}...")

        for i in tqdm(range(0, num_samples, batch_size), desc="Feature extraction in batches"):
            batch_ids = kids_id_list[i:i+batch_size]
            batch_tiles = kids_tile_list[i:i+batch_size]
            batch_file_path = self.generate_batch_file_path(batch_size, i)
            
            # Check if file exists and skip if overwrite is False
            if not overwrite and os.path.exists(batch_file_path):
                print(f"File {batch_file_path} already exists. Skipping save as overwrite is set to False.")
                continue
    
            # Extract features and labels for the batch
            batch_features, batch_labels = self.get_features(
                self.model, 
                file_path=None,  # No need to specify file path for in-memory processing
                variance_threshold=variance_threshold, 
                use_scaler=use_scaler, 
                path_npz_rgb=path_npz_rgb,
                kids_id_list=batch_ids, 
                kids_tile_list=batch_tiles, 
                channels=channels,
                overwrite=True
            )

            # Save batch features and labels incrementally
          
            self.save_features_npz(batch_features, batch_labels, batch_ids, batch_file_path)
            logger.info(f"Batch {i//batch_size} saved to {batch_file_path}")
            
            del batch_features, batch_labels
            gc.collect() 
            # Monitor memory usage
            memory_info = psutil.virtual_memory()
            logger.info(f"Memory Usage: {memory_info.percent}%")

        # Combine and load all saved batches for PCA
        logger.info("Loading all batches for PCA...")

        if not incremental_pca:
        
            all_features,all_labels,all_ids = [], [], []
            for i in range(0, num_samples, batch_size):
                batch_file_path = self.generate_batch_file_path(batch_size, i)
            
                batch_file_path = self.generate_batch_file_path(batch_size, i)
                features, labels, ids = self.load_features_npz(batch_file_path)
                all_features.append(features)
                all_labels.append(labels)
                all_ids.append(ids)

            # Concatenate all features, labels, and IDs
            all_features = np.vstack(all_features)
            all_labels = np.hstack(all_labels)
            all_ids = np.hstack(all_ids)

            # Apply PCA to reduce dimensionality
            logger.info(f"Features shape before PCA: {all_features.shape}")
            pca_result = run_pca(all_features, variance_threshold, use_scaler=use_scaler)

            # Save the reduced features, labels, and IDs
            self.save_features_npz(pca_result, all_labels, all_ids, name_features_fle)
            logger.info(f"PCA results saved to {name_features_fle}")

            logger.info("Feature extraction completed.")
        else:
            batches_path = glob.glob('batchs*.npz')
            self.compute_pca_incrementally(batches_path, variance_threshold, batch_size=batch_size, output_folder="pca_results")
            logger.info("Feature extraction and PCA completed.")

            

    def compute_pca_incrementally(self, file_paths, n_components=0.98, batch_size=5000, output_folder="pca_results"):
        """
        Compute PCA incrementally over multiple files to avoid memory overflow.

        Parameters:
            file_paths (list of str): Paths to the input data files.
            n_components (float): Explained variance threshold for PCA.
            batch_size (int): Batch size for processing data incrementally.
            output_folder (str): Folder to save PCA-transformed results.

        Returns:
            None
        """
        os.makedirs(output_folder, exist_ok=True)

        # Initialize IncrementalPCA
        ipca = IncrementalPCA(n_components=n_components)
        all_labels, all_ids = [], []

        # First pass: Fit PCA incrementally
        logging.info("Fitting PCA incrementally...")
        for file_path in tqdm(file_paths, desc="Fitting PCA"):
            data = np.load(file_path, allow_pickle =True)
            features = data['features']
            labels = data['labels']
            ids = data['ids']

            all_labels.append(labels)
            all_ids.append(ids)

            for i in range(0, features.shape[0], batch_size):
                batch = features[i:i+batch_size]
                ipca.partial_fit(batch)

        logging.info("PCA fitting complete.")

        # Second pass: Transform data and save results
        transformed_features = []
        for file_idx, file_path in enumerate(tqdm(file_paths, desc="Transforming PCA")):
            data = np.load(file_path, allow_pickle =True)
            features = data['features']

            for i in range(0, features.shape[0], batch_size):
                batch = features[i:i+batch_size]
                transformed_batch = ipca.transform(batch)
                transformed_features.append(transformed_batch)

            # Clear memory after transforming each file
            gc.collect()

        # Save PCA-transformed features
        transformed_features = np.vstack(transformed_features)
        all_labels = np.hstack(all_labels)
        all_ids = np.hstack(all_ids)

        output_file = os.path.join(output_folder, "pca_transformed_all.npz")
        self.save_features_npz(transformed_features, all_labels, all_ids, output_file)
        logging.info(f"PCA-transformed data saved to {output_file}.")



if __name__ == "__main__":
    

    preprocessing_after_byol = [sigma_clipping_gray] 


    base_path = os.path.expanduser("~") 
    if 'home/grespanm' in base_path:
        base_path = os.path.join(base_path, 'github')
    elif 'home/astrodust' in base_path:
        base_path = os.path.join(base_path, 'mnt','github')
    file_path = os.path.join(base_path, "data/BGcut_TEGLIE_subsample.csv")

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    
    if df is None:
        raise ValueError

    byol_test = BYOLTEGLIETest( tab_kids= df,
                               preprocessing_after_byol = preprocessing_after_byol, 
                                variance_threshold=0.98, 
                                path_npz_rgb=os.path.join(base_path, "data/BGcut_TEGLIE_subsample_cropped.npz"),
                              dataset_name= '101x101BGTEGLIE')
    #byol_test.run_byol_training()
    byol_test.run_feature_extractor()
    '''
    ## if big df
    preprocessing_after_byol = [sigma_clipping_gray]  
    try:
        df = pd.read_parquet('/home/grespanm/github/KiDS_astronomaly/full_sample/big_dataframe_2_half.parquet')
    except:
        df = pd.read_parquet('/home/astrodust/mnt/github/KiDS_astronomaly/full_sample/big_dataframe_2_half.parquet')
    kids_id_list = df['ID'].values
    byol_test = BYOLTEGLIETest()
    byol_test.run_feature_extractor_batches(preprocessing_after_byol = preprocessing_after_byol, 
                                variance_threshold=0.95, path_npz_rgb='dr4_cutouts_2.h5',
                                kids_id_list=kids_id_list,
                                kids_tile_list= df['KIDS_TILE'].values,
                                incremental_pca=False )
    
     #byol_test.run_feature_extractor(preprocessing_after_byol = preprocessing_after_byol)

    # Specify the input files and output file
    npz_files = [
        "pca_results/DR4_features&labels_prepbefore_[sigmaclipgray]_PCA_var_0.95_scaler_False.npz",
        "pca_results/DR4_features&labels_prepbefore_[sigmaclipgray]_PCA_var_0.95_scaler_False_2.npz"
    ]
    output_file = "combined_features_labels_0.95.npz"

    # Append and save
    byol_test.append_and_save_npz(npz_files, output_file)
    '''
