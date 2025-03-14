import sys
import pandas as pd
from tqdm import tqdm  # Progress bar
sys.path.append('/home/grespanm/github/TEGLIE/teglie_scripts/')
import gen_cutouts


def preprocess_lrg_candidates(df):
    """
    Preprocess dataframe to identify LRG candidates based on selection criteria.
    """
    c_par = 0.7 * df['g_min_r'] + 1.2 * (df['r_min_i'] - 0.18)
    c_perp = df['r_min_i'] - df['g_min_r'] / 4 - 0.18

    # Apply filters for LRG selection
    df_lrg = df[
        (abs(c_perp) < 0.2) & 
        (df['mag'] < (14 + c_par / 0.3)) & 
        (df['mag'] <= 20)
    ]
    return df_lrg


def BG_preproc(df):
    """
    Preprocess dataframe to filter for background sources based on selection criteria.
    """
    combined_condition = (
        (df['Flag'] < 4) &
        (df['IMAFLAGS_ISO'] == 0) &
        (df['MAG_AUTO'] <= 21) &
        (df['SG2DPHOT'] == 0)
    )
    return df[combined_condition]


def main():
    """
    Main function to process tiles, extract required data, and save the final dataframe.
    """
    # Initialize Cutouts object and filter catalogs for r-band
    cuclass = gen_cutouts.Cutouts()
    cats_only_one = cuclass.cats[cuclass.cats['Filter'] == 'r']

    # Define the required columns for processing
    required_columns = [
        'ID', 'KIDS_TILE', 'RAJ2000', 'DECJ2000', 'Z_B', 
        'MAG_AUTO', 'MAGERR_AUTO', 'EXTINCTION_g', 'EXTINCTION_r', 
        'COLOUR_GAAP_g_r', 'COLOUR_GAAP_r_i', 'Z_B_MIN', 'Z_B_MAX', 
        'Z_ML', 'SG2DPHOT', 'Flag', 'IMAFLAGS_ISO'
    ]

    # Initialize an empty list to store the processed dataframes
    all_df_bg = []

    # Process each tile with a progress bar
    for t in tqdm(cats_only_one['Tile name'], desc="Processing tiles"):
        try:
            # Load the table for the tile
            table = cuclass.getTableTile(t)

            # Handle missing columns
            missing_columns = [col for col in required_columns if col not in table.columns]
            if missing_columns:
                print(f"Warning: Missing columns in tile {t}: {missing_columns}")

            # Select only the available required columns
            available_columns = [col for col in required_columns if col in table.columns]
            df_tile = table[available_columns].to_pandas()

            # Preprocess the dataframe for background sources
            df_bg = BG_preproc(df_tile)

            # Append the processed dataframe
            all_df_bg.append(df_bg)

        except Exception as e:
            print(f"Error processing tile {t}: {e}")

    # Concatenate all processed dataframes into one
    big_df = pd.concat(all_df_bg, ignore_index=True)

    # Save the final dataframe to a compressed Parquet file
    big_df.to_parquet('big_dataframe.parquet', index=False, compression='snappy')
    print("Processing complete. Final dataframe saved to 'big_dataframe.parquet'.")



# import os
# import numpy as np
# import pandas as pd

# # Define paths
# usr = "/home/grespanm/github"
# table_file = os.path.join(usr, 'data/table_all_checked.csv')
# npz_file = os.path.join(usr, 'data', 'TEGLIE_cand_RGB.npz')
# output_csv = os.path.join(usr, "data/BGcut_TEGLIE_subsample.csv")
# output_npz = os.path.join(usr, "data/BGcut_TEGLIE_subsample_cropped.npz")

# # Load the DataFrame
# df = pd.read_csv(table_file)
# df_bg_teglie = BG_preproc(df).drop_duplicates(subset=['KIDS_ID'])

# # Load the image data
# data = np.load(npz_file, allow_pickle=True)['data']

# # Ensure KIDS_ID is set as the index
# df = df.set_index('KIDS_ID')

# # Filter DataFrame based on df_bg_teglie
# df_subsample = df[df.index.isin(df_bg_teglie['KIDS_ID'])]


# # Select the same subsample from `data`
# valid_indices = df.index.isin(df_bg_teglie['KIDS_ID'])
# data_subsample = data[valid_indices]

# # Function to center-crop a 101x101 image to 65x65
# def center_crop(img, crop_size=65):
#     original_size = img.shape[0]  # Assuming square images (101x101)
#     start = (original_size - crop_size) // 2  # Compute start index
#     return img[start:start+crop_size, start:start+crop_size]  # Crop

# # Apply cropping to all images
# cropped_data = np.array([center_crop(img) for img in data_subsample])

# # Save the subsampled and cropped images
# np.savez(output_npz, ID=df_subsample.index.values, data=cropped_data)

# print(f"Processed {len(cropped_data)} images with center cropping. Saved to {output_npz}")


if __name__ == "__main__":
    main()
