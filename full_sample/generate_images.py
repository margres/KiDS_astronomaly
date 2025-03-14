import numpy as np
import pandas as pd
import h5py
import multiprocessing
import os
import pickle
import sys
from functools import partial
from tqdm import tqdm
import logging
import glob
from astropy.io import fits

base_path = os.path.expanduser("~")
if 'home/grespanm' in base_path:
    base_path = os.path.join(base_path, 'github')
elif 'home/astrodust' in base_path:
    base_path = os.path.join(base_path, 'mnt','github')
print(base_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{base_path}/data/log_generate_images_process.log"),  # Main log file
        logging.StreamHandler()                                 # Console output
    ]
)

# Add a dedicated error log x file
error_handler = logging.FileHandler(f"{base_path}/data/log_generate_images_error.log")  # Separate file for errors
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(error_handler)

sys.path.append(f'/{base_path}/TEGLIE/teglie_scripts/')
import gen_cutouts, make_rgb, settings

cuclass = gen_cutouts.Cutouts()

# Define paths
# pkl_file_path = f'{base_path}/data/image_data.pkl'
df_sample_path = f'{base_path}/data/big_dataframe.parquet'
hdf5_file_path = f'{base_path}/data/dr4_cutouts_1_part.h5'

# # Load the necessary files
# with open(pkl_file_path, 'rb') as file:
#     data = pickle.load(file)

df_sample_tosave = pd.read_parquet(df_sample_path)
df_sample_tosave = df_sample_tosave[:len(df_sample_tosave) // 2]


def change_specialcharacters(string):
    """
    Replace special characters in a string.

    Parameters:
    - string (str): Input string.

    Returns:
    - str: String with replaced special characters.
    """
    return string.replace('KIDS', 'KiDS_DR4.0').replace('.', 'p').replace('-', 'm').rstrip(' ')

def from_fits_to_array(folder_path, img_ID, tile_ID, channels=None, 
                       return_path=False):
    """
    Converts FITS files to a NumPy array.

    Parameters:
        folder_path (str): Path to the folder containing FITS files.
        img_ID (str): ID of the image.
        tile_ID (str): ID of the tile.
        channels (list): List of channels to consider (default: ['r', 'i', 'g', 'u']).
        path_to_save_imgs_alternative (str): Alternative path for FITS files.
        return_path (bool): If True, returns the paths of loaded FITS files.

    Returns:
        np.ndarray: A NumPy array containing image data from specified channels.
        list: (Optional) List of paths to the loaded FITS files.
    """
    if channels is None:
        channels = ['r', 'i', 'g', 'u']

    path_list = []
    img_data = []

    # Ensure tile_ID is formatted correctly
    tile_ID = change_specialcharacters(tile_ID)

    for f in channels:
        # Attempt to load the FITS file from the primary path
        fits_files = glob.glob(os.path.join(folder_path, f'{f}_band', f'*{tile_ID}', f'{img_ID}.fits'))
        # Open the FITS file and load the data
        hdu = fits.open(fits_files[0])
        img_data.append(hdu[0].data)
        path_list.append(fits_files[0])
        hdu.close()

    img_array = np.array(img_data)

    if return_path:
        return img_array, path_list
    return img_array

# Define bold printing for errors
def bold_print(message):
    """Prints the message in bold to the console."""
    sys.stderr.write(f"\033[1m{message}\033[0m\n")



def process_row(row, existing_ids):
    """Processes a single row and returns the processed image data and paths to delete."""
    path_to_delete = []
    kids_id = row['ID']
    kids_tile = row['KIDS_TILE']
    img_tmp = None  # Placeholder for image data

    if kids_id in existing_ids:
        logging.info(f"KIDS_ID {kids_id} already exists in HDF5. Skipping processing...")
        return None, None, []  # Skip further processing for this ID

    logging.info(f"Processing row with KIDS_ID {kids_id} and Tile {kids_tile}")

    # try:
    #     img_tmp = data.get(kids_id)
    #     if img_tmp is None:
    #         raise KeyError(f"ID {kids_id} not found in preloaded data.")
    #     logging.debug(f"Data found in preloaded data for {kids_id}.")
    # except KeyError:
    logging.debug(f"ID {kids_id} not found in preloaded data. Attempting to find FITS...")
    try:
        img_data = from_fits_to_array(settings.path_to_save_imgs, kids_id, kids_tile)
        img_tmp = make_rgb.make_rgb_one_image(img_data, display_plot=False, return_img=True)
        logging.debug(f"FITS found in {settings.path_to_save_imgs} for {kids_id}.")
    except (FileNotFoundError, IndexError, OSError):
        logging.debug(f"FITS file not found for ID {kids_id} at primary path. Trying alternative...")
        try:
            img_data = from_fits_to_array('/home/grespanm/mnt/HD_MG/KiDS_cutout', kids_id, kids_tile)
            img_tmp = make_rgb.make_rgb_one_image(img_data, display_plot=False, return_img=True)
            logging.debug(f"FITS found in {'/home/grespanm/mnt/HD_MG/KiDS_cutout'} for {kids_id}.")
        except (FileNotFoundError, IndexError, OSError):
            logging.debug(f"FITS file not found for ID {kids_id} at both paths. Generating cutouts...")
            try:
                path_list = gen_cutouts.cutout_by_name_tile(kids_tile, row, apply_preproc=False, return_paths=True)
                path_to_delete.extend(path_list)
                img_tmp = from_fits_to_array(settings.path_to_save_imgs, kids_id, kids_tile)
                img_tmp = make_rgb.make_rgb_one_image(img_array=img_tmp, display_plot=False, return_img=True)
                logging.debug(f"Generated cutout for KIDS_ID {kids_id}, Tile {kids_tile}")
            except Exception as e:
                logging.error(f"Failed to generate cutouts for KIDS_ID {kids_id}: {e}")
                return None, None, []

    except TypeError as e:
        logging.error(f"Buffer error while processing {row['ID']}: {e}")
        return None, None, []

    if img_tmp is not None:
        logging.info(f"Successfully processed image for {kids_id}. Shape: {img_tmp.shape}")
    return kids_id, img_tmp, path_to_delete



def write_to_hdf5(hdf5_path, output_queue, total_items):
    """Writes processed data to the HDF5 file."""
    try:
        with h5py.File(hdf5_path, 'a', libver='latest') as hdf5_file:
            with tqdm(total=total_items, desc="Writing to HDF5") as pbar:
                while True:
                    result = output_queue.get()
                    if result == "DONE":
                        break
                    kids_id, img_tmp, _ = result

                    if kids_id in hdf5_file:
                        logging.info(f"Dataset for {kids_id} already exists in HDF5. Skipping...")
                        pbar.update(1)
                        continue  # Skip creating the dataset again

                    logging.info(f"Writing image for {kids_id} to HDF5")
                    hdf5_file.create_dataset(kids_id, data=img_tmp, compression="gzip", compression_opts=9)
                    pbar.update(1)
    except Exception as e:
        logging.error(f"Error writing to HDF5: {e}")
        bold_print(f"Error writing to HDF5: {e}")
        pass

def process_chunk(chunk, output_queue, existing_ids):
    """Processes a chunk of rows."""
    for idx, row in chunk.iterrows():
        kids_id, img_tmp, path_to_delete = process_row(row, existing_ids)
        if kids_id and img_tmp is not None:
            logging.debug(f"Adding dataset for {kids_id} to the queue")
            output_queue.put((kids_id, img_tmp, path_to_delete))

        # Delete temporary files
        for file_path in path_to_delete:
            try:
                os.remove(file_path)
                logging.debug(f"Deleted temporary file: {file_path}")
            except OSError as e:
                logging.error(f"Error deleting file {file_path}: {e}")


def process_in_parallel(df, hdf5_path, num_workers=4, chunk_size=100):
    """Processes the DataFrame in parallel."""
    # Load existing IDs once
    existing_ids = get_existing_ids(hdf5_path)

    with multiprocessing.Manager() as manager:
        output_queue = manager.Queue()

        # Calculate total number of items for progress tracking
        total_items = len(df)

        # Split data into chunks
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        # Start the writer process
        writer_process = multiprocessing.Process(target=write_to_hdf5, args=(hdf5_path, output_queue, total_items))
        writer_process.start()

        # Start worker processes
        with multiprocessing.Pool(processes=num_workers) as pool:
            process_partial = partial(process_chunk, output_queue=output_queue, existing_ids=existing_ids)
            pool.map(process_partial, chunks)

        # Notify the writer process to terminate
        output_queue.put("DONE")
        writer_process.join()


def get_existing_ids(hdf5_path):
    """Retrieve the list of existing IDs in the HDF5 file."""
    if not os.path.exists(hdf5_path):
        logging.info("HDF5 file does not exist. Starting with an empty file.")
        return set()  # Return an empty set if the file doesn't exist
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        return set(hdf5_file.keys())

if __name__ == "__main__":


    process_in_parallel(
        df_sample_tosave,
        hdf5_file_path,
        num_workers=15,  # Adjust based on your system's CPU cores
        chunk_size=500 # Tune chunk size for optimal memory usage
    )
