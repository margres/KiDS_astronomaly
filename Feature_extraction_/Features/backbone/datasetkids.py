import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torchvision as tv
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
import sys
try:
    sys.path.append('/users/grespanm/TEGLIE/teglie_scripts/')
    import make_rgb, settings, utils
except:
    sys.path.append('/home/grespanm/github/TEGLIE/teglie_scripts/')
    import make_rgb, settings, utils
from preprocessing import scaling_clipping
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import time

class KiDSDatasetloader(Dataset):


    '''
    
    it returns numoy array
    
    '''
    def __init__(self, kids_id_list, 
                 kids_tile_list, 
                 checkpoint_path = None, 
                 channels = None,
                 path_dat = settings.path_to_save_imgs,
                 apply_preproc=True, 
                 transform=None,labels=None):
        
        self.path_dat = path_dat
        self.kids_id_list = kids_id_list
        self.kids_tile_list = kids_tile_list
        self.channels = channels
        self.checkpoint_path = checkpoint_path
        self.apply_preproc = apply_preproc
        self.transform = transform
        self.data =  self.dataset_to_rgb()
        self.labels = labels
      

    def load_image(self, i):
        try:
            if isinstance(self.path_dat, str):
                X_tmp = utils.from_fits_to_array(self.path_dat, self.kids_id_list[i], self.kids_tile_list[i], channels=self.channels)
            else:
                X_tmp = utils.from_fits_to_array(self.path_dat[i], self.kids_id_list[i], self.kids_tile_list[i], channels=self.channels)
        except FileNotFoundError:
            cutclass = gen_cutouts.Cutouts()
            try:
                gen_cutouts.cutout_by_name_tile(
                    self.kids_tile_list[i],
                    cutclass.getTableTile(self.kids_tile_list[i])[cutclass.getTableTile(self.kids_tile_list[i])['ID'] == self.kids_id_list[i]],
                    channels=self.channels, apply_preproc=self.apply_preproc, path_to_save=settings.path_to_save_imgs_alternative
                )
                X_tmp = utils.from_fits_to_array(settings.path_to_save_imgs_alternative, self.kids_id_list[i], self.kids_tile_list[i], channels=self.channels)
            except OSError:
                return None, i
        return X_tmp, i
    

    def save_checkpoint(self, X, valid_indices):

        start_time = time.time()
        # Save only the non-NaN parts
        X_valid = X#[valid_indices]
        print(np.shape(X_valid))
        np.savez(os.path.join(self.checkpoint_path,'all_imgs.npz'), X=X_valid, valid_indices=valid_indices)
        end_time = time.time()
        print(f"Checkpoint saved at {os.path.join(self.checkpoint_path,'all_imgs.npz')} in {end_time - start_time:.2f} seconds")


    
    def load_checkpoint(self):

        if os.path.exists(os.path.join(self.checkpoint_path,'all_imgs.npz')):
            checkpoint = np.load(os.path.join(self.checkpoint_path,'all_imgs.npz'))
            X = checkpoint['X']
            valid_indices = checkpoint['valid_indices'].tolist()
            if not np.isnan(X).any():
                print("Loaded complete dataset from checkpoint")
                return X, valid_indices
            else:
                print("Loaded incomplete dataset from checkpoint, resuming")
                return X, valid_indices
        else:
            return None, []
            


            
    def create_dataset(self, use_parallel=True):

        num_images = len(self.kids_id_list)

        X, valid_indices = self.load_checkpoint()

        if X is None:
            X = np.empty((num_images, 101, 101, len(self.channels)))
            X[:] = np.nan  # Pre-fill with NaN to identify invalid entries

        if len(valid_indices) == num_images and not np.isnan(X).any():
            print("Dataset is already complete, skipping processing")
            return X
        elif valid_indices[-1]>num_images-1:
            return X
        

        def process_and_store_image(i):
            X_tmp, idx = self.load_image(i)
            if X_tmp is not None:
                X_tmp = np.expand_dims(np.transpose(X_tmp, axes=(1, 2, 0)), axis=0)
                X[idx] = X_tmp
                valid_indices.append(idx)
            return idx

        if use_parallel:
            with ThreadPoolExecutor(max_workers=18) as executor:
                futures = {executor.submit(process_and_store_image, i): i for i in range(num_images) if i not in valid_indices}
                for future in tqdm(as_completed(futures), total=num_images, desc="Processing images"):
                    idx = future.result()
                    if len(valid_indices) % 10000 == 0:
                        self.save_checkpoint(X, valid_indices)
                        print(idx)
        else:
            for i in tqdm(range(num_images), desc="Processing images"):
                if i not in valid_indices:
                    process_and_store_image(i)
                    if len(valid_indices) % 10000 == 0:
                        self.save_checkpoint(X, valid_indices)

        self.save_checkpoint(X, valid_indices)
        X = X[valid_indices]
        #X = scaling_clipping(X)
        return X

    def dataset_to_rgb(self):

        #self.data_ugr = self.create_dataset()

        checkpoint_file = os.path.join(self.checkpoint_path, 'all_imgs_RGB.npz')

        if os.path.exists(checkpoint_file):
            #file with rgb images exists
            print(f"Loading data from {checkpoint_file}")
            data = np.load(checkpoint_file, allow_pickle=True)['data']
            print(f'File loaded, it has shape {np.shape(data)}')

        else:
            #it doesnt exist
            if os.path.exists(os.path.exists(os.path.join(self.checkpoint_path,'all_imgs.npz'))):
                    print('RGB images not found')
                    data_ugr = self.create_dataset()
                    print('Dataset to rgb')
                    data = []
                    for d in tqdm(data_ugr, desc="Converting dataset to RGB"):
                        data.append(make_rgb.make_rgb_one_image(d, return_img=True))
        
                    np.savez(checkpoint_file, data=np.asarray(data))
                    print(f"Data saved to {checkpoint_file}")

        return data

    def dataset_to_PIL(self,images):

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        pil_images = []

        for image in images:
            pil_image = Image.fromarray(image)
            pil_image = transform(pil_image)
            pil_images.append(pil_image)

        batch = torch.stack(pil_image)
        return batch
    

    def check_for_nones(self):
        none_indices = []
        for idx in range(len(self.data)):
            if self.data[idx] is None or self.labels[idx] is None:
                none_indices.append(idx)
        return none_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, test=False):
        sample = self.data[idx]

        print(np.shape(sample))

        if test:
            plt.imshow(sample[:,:,0])
            plt.show()
            #sys.exit()

        label = self.labels[idx] if self.labels is not None else None
        if self.transform:
            sample = self.transform(sample)
        return sample, label

    
class CustomTransformDataset(Dataset):
    def __init__(self, data, names, labels=None, resize=256, crop=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.data = data
        self.names = names
        self.labels = labels
        self.resize = resize
        self.crop = crop
        self.mean = mean
        self.std = std
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.crop),
            transforms.ToTensor(),
            #transforms.Grayscale(num_output_channels=3),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index, test=False):
        image = self.data[index]
        name = self.names[index]
        label = self.labels[index] if self.labels is not None else None
    
        if test:
            plt.imshow(image[:,:,0])
            plt.show()
            #sys.exit()

        if image is None or label is None:
            print(f"Found None at index: {index}")
        else:
            image = self.transform(image)
            return image, label


def train_val_test_dataset(dataset, val_split=0.20, test_split=0.10):
    n_samples = len(dataset)

    if n_samples < 3:
        raise ValueError("Dataset is too small to split into train, validation, and test sets.")
    
    # Ensure that there are enough samples for splitting
    train_val_idx, test_idx = train_test_split(list(range(n_samples)), test_size=test_split, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_split/(1-test_split), random_state=42)
    
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    
    return datasets
