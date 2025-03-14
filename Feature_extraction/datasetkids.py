import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torchvision as tv
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
import sys
import h5py

try:
    sys.path.append('/users/grespanm/TEGLIE/teglie_scripts/')
    import make_rgb, settings, utils
except:
    sys.path.append('/home/grespanm/github/TEGLIE/teglie_scripts/')
    sys.path.append('/home/astrodust/mnt/github/TEGLIE/teglie_scripts/')
    import make_rgb, settings, utils, gen_cutouts

from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
mpl.use('Agg')
import io
import cv2
from astropy.stats import sigma_clipped_stats


usr = '/'.join(os.getcwd().split('/')[:3])
## different paths in different machines
if 'home/grespanm' in usr:
    usr = os.path.join(usr, 'github')
elif 'home/astrodust' in usr:
    usr = os.path.join(usr, 'mnt','github')
print(usr)


def convert_array_to_image(arr, plot_cmap='viridis', interpolation='bicubic'):
    """
    Function to convert an array to a png image ready to be served on a web
    page.

    Parameters
    ----------
    arr : np.ndarray
        Input image
    plot_cmap : str, optional
        Which colourmap to use
    interpolation : str, optional
        Allows interpolation so low res images don't look so blocky

    Returns
    -------
    png image object
        Object ready to be passed directly to the frontend
    """
    with mpl.rc_context({'backend': 'Agg'}):
        #print(arr.shape)
        if len(arr.shape)==2:
            # if greyscale plot just that
            fig = plt.figure(figsize=(1, 1), dpi=4 * arr.shape[1])
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(arr, cmap=plot_cmap, origin='lower',
                    interpolation=interpolation)
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            plt.close(fig)
        else:
            #if 3 bands plot r band and rgb
            fig = plt.figure(figsize=(1, 1), dpi=6 * arr.shape[1])
            
            plt.imshow(make_rgb.make_rgb_one_image(arr, return_img=True),
                       cmap=plot_cmap, origin='lower', interpolation=interpolation)
            plt.axis('off')
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            plt.close(fig)

class KiDSDatasetloader(Dataset):

    def __init__(self,
                 df_kids = None, 
                 kids_id_list=None, 
                 kids_tile_list=None, 
                 checkpoint_path = None, 
                 channels = None,
                 path_dat = settings.path_to_save_imgs,
                 apply_preproc=True, 
                 transform=None,labels=None,
                 display_interpolation=None,
                 rgb_img_file_name= None,):
        
        self.display_interpolation = display_interpolation

        #this is used only if ID and tile are not given

        #self.df_kids = self.df_kids[:len(self.df_kids) // 2]
        #pd.read_csv(os.path.join(usr, 'data/table_all_checked.csv'))
        self.path_dat = path_dat
        
        # Assign default values if parameters are None
        if df_kids is not None:
            self.df_kids = df_kids  # Use provided dataframe
            self.kids_id_list = kids_id_list if kids_id_list is not None else self.df_kids['KIDS_ID'].values
            self.kids_tile_list = kids_tile_list if kids_tile_list is not None else self.df_kids['KIDS_TILE'].values

        elif kids_id_list is not None and kids_tile_list is not None:
            # Create a dataframe if only ID and Tile lists are provided
            self.df_kids = pd.DataFrame({'KIDS_ID': kids_id_list, 'KIDS_TILE': kids_tile_list})
            self.kids_id_list = kids_id_list
            self.kids_tile_list = kids_tile_list

        else:
            # Load from parquet if no other inputs are available
            self.df_kids = pd.read_parquet(os.path.join(usr, 'data/big_dataframe.parquet'))
            self.kids_id_list = self.df_kids['KIDS_ID'].values
            self.kids_tile_list = self.df_kids['KIDS_TILE'].values
        self.channels = channels if channels is not None else ['r', 'i', 'g']
        self.checkpoint_path = checkpoint_path if checkpoint_path is not None else os.path.join(usr,'data')
        self.apply_preproc = apply_preproc
        self.transform = transform
        self.labels = labels if labels is not None else self.df_kids['LABEL'].values 
        self.rgb_img_file_name = rgb_img_file_name if rgb_img_file_name is not None else 'dr4_cutouts.h5' #'all_imgs.npz'
        self.data = self.dataset_to_rgb(KiDS_IDs=kids_id_list)
        self.data_type = 'image'
        self.metadata = self.get_metadata()
        self.index = self.metadata.index.values

      


    def get_metadata(self,):
        
        if 'ID' in self.df_kids.columns and 'KIDS_TILE' in self.df_kids.columns :
            self.df_kids.rename(columns={"ID": "KIDS_ID"}, inplace=True)
        elif 'KIDS_ID' in self.df_kids.columns and 'KIDS_TILE'in self.df_kids.columns:
            pass
        else:
            print(self.df_kids.head())
            raise ValueError('Dataframe with columns ID or KIDS_ID and KIDS_TILE needed')

        #if 'FOLDER' not in self.df_kids.columns:
        #    raise ValueError('Dataframe with images path needed (FOLDER column)')

        if 'LABEL' in  self.df_kids.columns:
            pass
        else:
            self.df_kids['LABEL'] = np.ones(len(self.df_kids))*-1

        #label from active learning
        if 'LABEL_AL' in  self.df_kids.columns:
            pass
        else:
            self.df_kids['LABEL_AL'] = np.ones(len(self.df_kids))*-1
        
        inds =  self.df_kids['KIDS_ID'].values
        self.df_kids['filename'] = [tile +'__'+ id for id,tile in zip(self.df_kids['KIDS_ID'].values,self.df_kids['KIDS_TILE'].values)]

        #self.df_kids = self.df_kids.drop_duplicates(subset='KIDS_ID').reset_index(drop=True)
        #print(self.df_kids.head())
        self.metadata = self.df_kids.set_index(inds)
        return self.metadata

    

    def get_sample(self, idx):
        """
        Returns the data for a single sample in the dataset as indexed by idx.

        Parameters
        ----------
        idx : string
            Index of sample

        Returns
        -------
        nd.array
            Array of image CustomTransformDatasettout
        """

        return self.data[int(idx)]

    def get_display_data(self, idx):

        return convert_array_to_image(self.data[int(idx)], interpolation=self.display_interpolation)


      

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
        """
        Save the checkpoint data to a file, dynamically supporting .npz or .h5 formats.
        """
        start_time = time.time()
        # Save only the non-NaN parts
        X_valid = X[valid_indices]
        print(np.shape(X_valid))

        checkpoint_path = os.path.join(self.checkpoint_path, self.rgb_img_file_name)
        if checkpoint_path.endswith('.npz'):
            np.savez(checkpoint_path, X=X_valid, valid_indices=valid_indices)
            print(f"Checkpoint saved as .npz at {checkpoint_path}")
        elif checkpoint_path.endswith('.h5'):
            with h5py.File(checkpoint_path, 'w') as hf:
                hf.create_dataset('X', data=X_valid)
                hf.create_dataset('valid_indices', data=valid_indices)
            print(f"Checkpoint saved as .h5 at {checkpoint_path}")
        else:
            raise ValueError(f"Unsupported file format for checkpoint: {checkpoint_path}")

        end_time = time.time()
        print(f"Checkpoint saved in {end_time - start_time:.2f} seconds")



    def load_checkpoint(self):
        """
        Load dataset checkpoint from .npz or .h5 files.

        Returns:
            Tuple: (data array, valid_indices)
        """
        # Construct full path to the checkpoint file
        img_path = os.path.join(self.checkpoint_path, self.rgb_img_file_name)

        # Check the file extension and load accordingly

        if img_path.endswith('.npz'):
            if os.path.exists(img_path):
                checkpoint = np.load(img_path, mmap_mode='r')  # Lazy loading
                if indices is None:
                    data = checkpoint['X'][:]  # Load all data
                else:
                    # Load only the specified indices
                    data = checkpoint['X'][indices] if isinstance(indices, list) else checkpoint['X'][[indices]]
                print(f"Loaded checkpoint from {img_path} (npz format)")
            else:
                print(f"No .npz checkpoint found at {img_path}. Starting fresh.")
                return None

        elif img_path.endswith('.h5'):
            if os.path.exists(img_path):
                with h5py.File(img_path, 'r') as hf:
                    if ids is None:
                        data = hf['X'][:]  # Load all data
                    else:
                        data = {}
                        for id in ids:
                            if id in hf:
                                data[id] = hf[id][:]
                            else:
                                print(f"ID {id} not found in HDF5 file.")
                print(f"Loaded checkpoint from {img_path} (h5 format)")
            else:
                print(f"No .h5 checkpoint found at {img_path}. Starting fresh.")
                return None

        # Check for NaN values in the dataset
        if not np.isnan(X).any():
            print("Loaded complete dataset from checkpoint")
        else:
            print("Loaded incomplete dataset from checkpoint, resuming")

        return X, valid_indices


    def create_dataset(self, use_parallel=True):
        """
        Create the dataset for RGB images, load existing checkpoints if available, and save progress.
        """
        num_images = len(self.kids_id_list)

        # Load from checkpoint if available
        X, valid_indices = self.load_checkpoint()

        if X is None:
            # Initialize an empty dataset with NaN for invalid entries
            X = np.empty((num_images, 101, 101, len(self.channels)))
            X[:] = np.nan

        if len(valid_indices) == num_images and not np.isnan(X).any():
            print("Dataset is already complete, skipping processing")
            return X
        elif valid_indices and valid_indices[-1] > num_images - 1:
            return X

        # Define the processing function
        def process_and_store_image(i):
            X_tmp, idx = self.load_image(i)
            if X_tmp is not None:
                X_tmp = np.expand_dims(np.transpose(X_tmp, axes=(1, 2, 0)), axis=0)
                X[idx] = X_tmp
                valid_indices.append(idx)
            return idx

        # Process images
        if use_parallel:
            with ThreadPoolExecutor(max_workers=18) as executor:
                futures = {executor.submit(process_and_store_image, i): i for i in range(num_images) if i not in valid_indices}
                for future in tqdm(as_completed(futures), total=num_images, desc="Processing images"):
                    idx = future.result()
                    # Save checkpoint periodically
                    if len(valid_indices) % 10000 == 0:
                        self.save_checkpoint(X, valid_indices)
                        print(f"Checkpoint saved after processing {idx}")
        else:
            for i in tqdm(range(num_images), desc="Processing images"):
                if i not in valid_indices:
                    process_and_store_image(i)
                    # Save checkpoint periodically
                    if len(valid_indices) % 10000 == 0:
                        self.save_checkpoint(X, valid_indices)

        # Final checkpoint save
        self.save_checkpoint(X, valid_indices)
        X = X[valid_indices]
        return X

    def dataset_to_rgb(self, KiDS_IDs=None):
        """
        Convert dataset to RGB format, dynamically handling .npz and .h5 files.
        """
        checkpoint_path = os.path.join(self.checkpoint_path, self.rgb_img_file_name)

        # Dynamically check file extension
        if checkpoint_path.endswith('.npz'):
            if os.path.exists(checkpoint_path):
                print(f"Loading data from {checkpoint_path}")
                data = np.load(checkpoint_path, allow_pickle=True)['data']
                print(f"File loaded, shape: {np.shape(data)}")
            else:
                print('WARNING: check if given file path is correct!!')
                print(f"No .npz checkpoint found at {checkpoint_path}. Creating dataset...")
                data = self._generate_and_save_rgb_data(checkpoint_path, save_as_h5=False)

        elif checkpoint_path.endswith('.h5'):
            if os.path.exists(checkpoint_path):
                data = []
                print(f"Loading data from {checkpoint_path}")
                with h5py.File(checkpoint_path, 'r') as hf:

                    example_key = next(iter(hf.keys()))
                    placeholder_shape = hf[example_key].shape  # Use the shape of the first valid dataset
                    placeholder = np.zeros(placeholder_shape, dtype=np.float32)

                    # Iterate over all datasets (kids_id keys) and collect data
                    if KiDS_IDs is None: #load all 
                        for key in hf.keys():
                            data.append(hf[key][:])  # Load dataset by key
                    else:
                        for kids_id in KiDS_IDs:
                            if kids_id in hf:
                                data.append(hf[kids_id][:])  # Load the dataset corresponding to the ID
                            else:
                                print(f"Warning: ID {kids_id} not found in the HDF5 file.")
                                data.append(placeholder)
                    data = np.array(data)  # Convert list of arrays to a single NumPy array
                print(f"File loaded, shape: {data.shape}")
            else:
                print(f"No .h5 checkpoint found at {checkpoint_path}. Creating dataset...")
                data = self._generate_and_save_rgb_data(checkpoint_path, save_as_h5=True)

        else:
            raise ValueError(f"Unsupported file extension for {checkpoint_path}. Please use .npz or .h5.")

        return data



    def _generate_and_save_rgb_data(self, checkpoint_path, save_as_h5=False):
        """
        Generate RGB data from UGR dataset and save it to the specified checkpoint path.
        """
        data_ugr = self.create_dataset()
        data = []
        print("Converting to RGB...")
        for d in tqdm(data_ugr, desc="Converting dataset to RGB"):
            data.append(make_rgb.make_rgb_one_image(d, return_img=True))

        data = np.asarray(data)

        # Save as .npz or .h5 based on the flag
        if save_as_h5:
            with h5py.File(checkpoint_path, 'w') as hf:
                hf.create_dataset('data', data=data)
            print(f"Data saved as .h5 at {checkpoint_path}")
        else:
            np.savez(checkpoint_path, data=data)
            print(f"Data saved as .npz at {checkpoint_path}")

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

class SigmaClippingTransform:
    """Applies sigma clipping to an image while maintaining PyTorch tensor compatibility."""
    
    def __init__(self, sigma=2, central=True):
        self.sigma = sigma
        self.central = central

    def __call__(self, img):
        """
        Applies sigma clipping transformation.

        Parameters
        ----------
        img : torch.Tensor (C, H, W)
            Input image tensor (PyTorch format)

        Returns
        -------
        torch.Tensor
            Transformed image tensor with sigma-clipping applied
        """
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()  # Convert PyTorch tensor (C, H, W) -> NumPy (H, W, C)

        # Convert grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_img = np.nan_to_num(gray_img)  # Handle NaNs

        # Apply sigma clipping
        mean, median, std = sigma_clipped_stats(gray_img, sigma=self.sigma)
        thresh = std + median
        img_bin = (gray_img > thresh).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Determine the central contour
        x0, y0 = gray_img.shape[0] // 2, gray_img.shape[1] // 2
        selected_contour = None
        if self.central:
            for c in contours:
                if cv2.pointPolygonTest(c, (x0, y0), False) == 1:
                    selected_contour = c
                    break

        # Apply contour mask
        contour_mask = np.zeros_like(gray_img, dtype=np.uint8)
        if selected_contour is None:
            return torch.zeros_like(torch.from_numpy(img).permute(2, 0, 1))  # Return empty image if no contour

        cv2.drawContours(contour_mask, [selected_contour], 0, 1, -1)

        # Apply mask to each channel
        new_img = np.zeros_like(img)
        for i in range(3):
            new_img[:, :, i] = cv2.bitwise_and(img[:, :, i], img[:, :, i], mask=contour_mask)

        return torch.from_numpy(new_img).permute(2, 0, 1)  # Convert back to PyTorch tensor (C, H, W)

    
class CustomTransformDataset(Dataset):
    def __init__(self, data, names, labels=None, resize=254, crop=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], use_sigma_clipping=False, normalize =False):
        self.data = data
        self.names = names
        self.labels = labels
        self.resize = resize
        self.crop = crop
        self.mean = mean
        self.std = std
        transform = [
            transforms.ToPILImage(),
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.crop),
            transforms.ToTensor()]

        # Conditionally add SigmaClippingTransform
        if use_sigma_clipping:
            transform.append(SigmaClippingTransform(sigma=2, central=True))
        
        if normalize:
            # Add normalization at the end
            transform.append(transforms.Normalize(mean=self.mean, std=self.std))
        
        # Create the final transformation pipeline
        self.transform = transforms.Compose(transform)



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


def train_val_test_dataset(dataset, val_split=0.20, test_split=0.01, random_state=42):
    n_samples = len(dataset)

    if n_samples < 3:
        raise ValueError("Dataset is too small to split into train, validation, and test sets.")
    
    # Ensure that there are enough samples for splitting
    train_val_idx, test_idx = train_test_split(list(range(n_samples)), test_size=test_split, random_state=random_state)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_split/(1-test_split), random_state=random_state)
    
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['test'] = Subset(dataset, test_idx)
    
    return datasets
