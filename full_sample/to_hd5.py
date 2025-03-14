import numpy as np
import h5py
import os
import pandas as pd

# Define the paths
image_dir = '/home/grespanm/github/data'  # Directory containing the .npz file
image_path = 'all_imgs_RGB.npz'  # .npz file containing image arrays
full_image_path = os.path.join(image_dir, image_path)

hdf5_file = "images.h5"

def teglie_cand_to_h5():
    try:
        # Load the .npz file
        if not os.path.exists(full_image_path):
            raise FileNotFoundError(f"File '{full_image_path}' not found.")

        # Load image arrays from .npz
        data = np.load(full_image_path)
        if not isinstance(data, np.lib.npyio.NpzFile):
            raise ValueError(f"File '{full_image_path}' is not a valid .npz file.")
        
        # Iterate over all arrays in the .npz file and save to HDF5
        with h5py.File(hdf5_file, 'a') as f:  # 'a' mode to append to the file
            for key in data:
                dataset_name = key  # Use the .npz key as the dataset name
                image_array = data[key]

                if dataset_name in f:
                    print(f"Dataset '{dataset_name}' already exists in {hdf5_file}. Overwriting.")
                    del f[dataset_name]  # Delete existing dataset if overwriting
                f.create_dataset(dataset_name, data=image_array, compression="gzip", compression_opts=9)
                print(f"Array '{dataset_name}' saved to {hdf5_file} with compression.")

    except Exception as e:
        print(f"Error: {e}")


def merge_hdf5_files(file1, file2, output_file):
    """
    Merges two HDF5 files into a single HDF5 file.

    Parameters:
        file1 (str): Path to the first HDF5 file.
        file2 (str): Path to the second HDF5 file.
        output_file (str): Path to the output merged HDF5 file.

    Returns:
        None
    """
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Deleting...")
        os.remove(output_file)

    with h5py.File(output_file, 'w') as out_f:
        # Helper function to copy datasets
        def copy_datasets(source_file):
            with h5py.File(source_file, 'r') as f:
                def copy(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if name not in out_f:
                            print(f"Copying {name} from {source_file}")
                            f.copy(name, out_f, name=name)
                        else:
                            print(f"Skipping duplicate dataset: {name}")
                f.visititems(copy)

        # Copy from both files
        copy_datasets(file1)
        copy_datasets(file2)

    print(f"Successfully merged {file1} and {file2} into {output_file}")


def save_hdf5_keys_to_txt(hdf5_file_path, output_txt_path):
    """
    Reads an HDF5 file and saves all the dataset keys to a .txt file.

    Parameters:
        hdf5_file_path (str): Path to the input HDF5 file.
        output_txt_path (str): Path to the output .txt file where keys will be saved.
    """
    try:
        with h5py.File(hdf5_file_path, 'r') as hf:
            keys = list(hf.keys())
            print(f"Found {len(keys)} keys in the HDF5 file.")
            
            with open(output_txt_path, 'w') as txt_file:
                for key in keys:
                    txt_file.write(f"{key}\n")
            
            print(f"Keys have been saved to {output_txt_path}")
    except Exception as e:
        print(f"Error: {e}")




def filter_dataframe_by_ids(txt_file_path, dataframe, id_column='ID'):
    """
    Filters a DataFrame by checking if the IDs in a .txt file are present in a specified column.

    Parameters:
        txt_file_path (str): Path to the .txt file containing the IDs to filter by.
        dataframe (pd.DataFrame): The pandas DataFrame to filter.
        id_column (str): The column in the DataFrame containing the IDs.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only rows with IDs in the .txt file.
    """
    try:
        # Load IDs from the .txt file
        with open(txt_file_path, 'r') as file:
            valid_ids = set(line.strip() for line in file.readlines())
        
        print(f"Loaded {len(valid_ids)} IDs from {txt_file_path}")

        # Filter the DataFrame
        filtered_df = dataframe[dataframe[id_column].isin(valid_ids)]
        print(f"Filtered DataFrame contains {len(filtered_df)} rows (out of {len(dataframe)}).")
        
        return filtered_df

    except Exception as e:
        print(f"Error: {e}")
        return None


if  __name__ == "__main__":

    # File paths
    file1 = '/home/grespanm/github/data/dr4_cutouts.h5'
    file2 = '/home/grespanm/github/data/dr4_cutouts_2.h5'
    output_file = '/home/grespanm/merged_dr4_cutouts.h5'

    # Merge the files
    #merge_hdf5_files(file1, file2, output_file)
    #save_hdf5_keys_to_txt(file2, file1.split('.h5')[0]+'KIDS_ID_2.txt')

    # Load the DataFrame (replace with your actual DataFrame loading logic)
    df = pd.read_parquet('big_dataframe.parquet')

    # Filter the DataFrame
    filtered_df = filter_dataframe_by_ids('/home/grespanm/github/data/dr4_cutoutsKIDS_ID_2.txt', df, id_column='ID')

    # Save the filtered DataFrame to a new file
    if filtered_df is not None:
        filtered_df.to_parquet("big_dataframe_2_half.parquet", index=False)
        print("Filtered DataFrame saved ")