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
sys.path.append('/home/grespanm/github/TEGLIE/teglie_scripts/')
import gen_cutouts, utils, settings, make_rgb
from preprocessing import scaling_clipping
import matplotlib.pyplot as plt

class KiDSDatasetloader(Dataset):


    '''
    
    it returns numoy array
    
    '''
    def __init__(self, kids_id_list, kids_tile_list, 
                 channels = settings.channels,
                 path_dat = settings.path_to_save_imgs,
                 apply_preproc=True, transform=None,labels=None,):
        self.path_dat = path_dat
        self.kids_id_list = kids_id_list
        self.kids_tile_list = kids_tile_list
        self.channels = channels
        self.apply_preproc = apply_preproc
        self.transform = transform
        self.data_ugr = self.create_dataset()
        self.data =  self.dataset_to_rgb(self.data_ugr)
        #self.data = self.dataset_to_PIL(self.data_rgb)
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

            
    def create_dataset(self):
        X = np.zeros((1, 101, 101, len(self.channels)))
        
        #idx_found_tiles = []
        for i in tqdm(range(len(self.kids_id_list)), desc="Processing images"):
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
                    continue

            X_tmp = np.expand_dims(np.transpose(X_tmp, axes=(1, 2, 0)), axis=0)
            #idx_found_tiles.append(i)
            X = np.vstack((X, X_tmp))
        X = X[1:]
        X = scaling_clipping(X)
        return X

    def dataset_to_rgb(self, data_ugr):

        print('Dataset to rgb')
         
        data = []
        for d in tqdm(data_ugr, desc="Converting dataset to RGB"):         
            data.append(make_rgb.make_rgb_one_image(d, return_img=True))

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

'''
# Example usage
path_dat = '/path/to/data'
kids_id_list = [...]  # Your list of IDs
kids_tile_list = [...]  # Your list of tile names
channels = [...]  # Your list of channels

transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transformations here if needed
])

dataset = KiDSDataset(path_dat, kids_id_list, kids_tile_list, channels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate over the DataLoader
for batch in dataloader:
    # Your training loop here
    pass
'''