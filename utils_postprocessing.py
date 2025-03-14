from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import sys
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import math
import glob
# Display
from IPython.display import Image, display
import importlib
from astropy.table import Table, Column
import re
import random
import seaborn as sns
random.seed(42)
usr = '/'.join(os.getcwd().split('/')[:3])
## different paths in different machines
if 'home' in usr:
    usr = os.path.join(usr, 'github')
import logging
from typing import Tuple, List, Union
import itertools

logging.basicConfig(level=logging.INFO)

def add_labels(df_score, df_labels):
    '''
    this is to have one anomaly score and labels in the same notebook
    '''
    # Merge the dataframes on the common column 'KIDS_ID'
    merged_df = pd.merge(df_score, df_labels[['KIDS_ID', 'LABEL']], on='KIDS_ID', how='left')

    # Replace grade 2 and 1 with 5 if there are no 5s in the LABEL column
    if 5 not in merged_df['LABEL'].values:
        # i transform labels to interest score
        merged_df = to_interest_score(merged_df)

    return merged_df

def to_interest_score(df, column_name='LABEL'):
    df[column_name] = df[column_name].replace({1: 5, 2: 5})
    return df


def read_scores(folder: str, n_labelled: int, df_labels: pd.DataFrame) -> pd.DataFrame:
    
    csv_files = glob.glob(os.path.join(folder, f'ml_scores_{n_labelled}*.csv'))

    if not csv_files:
        logging.error(f"No matching files found in folder: {folder}")
        raise ValueError(f"No matching files found in folder: {folder}")
    # some file names differ for a letter 
    selected_file = next((file for file in csv_files if 'b' in os.path.basename(file).split('.csv')[0]), csv_files[0])

    # i read  the ml_score file
    df_score = pd.read_csv(selected_file)

    # when saved the index some time changes nime, index is source ID 
    rename_columns = {'Unnamed: 0': 'KIDS_ID', 'Index': 'KIDS_ID'}
    df_score.rename(columns=lambda col: rename_columns.get(col, col), inplace=True)
    
    # i add the 'true' labels and return the df 
    return add_labels(df_score, df_labels)



def generate_folder_names(
        dataset: Union[str, List[str]], 
        dimred_method: Union[str, List[str]], 
        var: Union[str, List[str]], 
        preprocessing: Union[str, List[str]], 
        gp_method: Union[str, List[str]], 
        tradeoff: Union[str, List[str]], 
        batches: Union[str, List[str]], 
        score: Union[str, List[str]]) -> List[str]:
    '''
    function that creates folder name given the parameters
    '''
    
    # Convert all inputs to lists if they are not already
    dataset = [dataset] if isinstance(dataset, str) else dataset
    dimred_method = [dimred_method] if isinstance(dimred_method, str) else dimred_method
    var = [var] if isinstance(var, str) else var
    preprocessing = [preprocessing] if isinstance(preprocessing, str) else preprocessing
    gp_method = [gp_method] if isinstance(gp_method, str) else gp_method
    tradeoff = [tradeoff] if isinstance(tradeoff, str) else tradeoff
    batches = [batches] if isinstance(batches, str) else batches
    score = [score] if isinstance(score, str) else score
    
    # Generate all combinations of the parameters using itertools.product
    combinations = itertools.product(dataset, dimred_method, var, preprocessing, gp_method, tradeoff, batches, score)
    
    folder_names = []
    
    # Template for the folder name
    template = "{dataset}_dimred_{dimred_method}_var_{var}_prepbefore_{preprocessing}_{gp_method}_tradeoff_{tradeoff}_batches_{batches}_score_{score}"
    
    # Iterate through the combinations and create folder names
    for combination in combinations:
        folder_name = template.format(
            dataset=combination[0],
            dimred_method=combination[1],
            var=combination[2],
            preprocessing=combination[3],
            gp_method =combination[4],
            tradeoff=combination[5],
            batches=combination[6],
            score=combination[7]
        )
        folder_names.append(folder_name)
    
    return folder_names



def recall(ml_score: pd.DataFrame, n_bin: int = 100, column: str = 'LABEL', sort_by: str = 'trained_score') -> Tuple[np.ndarray, List[float], List[int]]:

    '''
    function for calculating the recall 
    in the plots i use the number of TP, but I also calculate the recall in 0,1 range
    '''
    
    df_sorted = ml_score.sort_values(sort_by, ascending=False)
    
    #this is used to define the bins in which I calculate the TP and recall
    num_elements = np.arange(1, len(df_sorted) + 1, n_bin)

    recalls = []
    TP_list = []

    for i in num_elements:
        true_labels = df_sorted.iloc[:i][column]
        TP = (true_labels >= 3).sum()
        FN = (df_sorted.iloc[i:][column] >= 3).sum()

        recalls.append(TP / (TP + FN))
        TP_list.append(TP)

    return num_elements, recalls, TP_list


def recall_HQ(ml_score: pd.DataFrame, n_bin: int = 100, column: str = 'LABEL', sort_by: str = 'trained_score') -> Tuple[np.ndarray, List[float], List[int]]:
    df_sorted = ml_score.sort_values(sort_by, ascending=False)
    num_elements = np.arange(1, len(df_sorted) + 1, n_bin)

    '''
    same func as the one above but it considers only the 'high-quality candidates',
    namely the ones with grade >=4

    '''
    
    recalls = []
    TP_list = []

    for i in num_elements:
        true_labels = df_sorted.iloc[:i][column]
        TP = (true_labels >= 4).sum()
        FN = (df_sorted.iloc[i:][column] >= 4).sum()

        recalls.append(TP / (TP + FN))
        TP_list.append(TP)

    return num_elements, recalls, TP_list


def create_recall_files(folder_list, n_labelled_list, df_labels, recalls_filename, overwrite=True, sort_by= 'trained_score', HQ=True):
    """
    Generates recall files for each folder and number of labeled data points.

    Parameters:
    - folder_list: List of folders to process.
    - n_labelled_list: List of labeled data points to process.
    - df_labels: DataFrame containing labels.
    - recalls_filename: Base name for recall files.
    - overwrite: Boolean indicating whether to overwrite existing files.

    """
    for folder in folder_list:
        # Extract variables from the folder name
        try:
            img_prep, dim_reduction, variance, image_prep, AL_model, ei_tradeoff, batches, score = extract_variables(folder)
        except ValueError as e:
            logging.error(f"Error processing folder {folder}: {e}")
            continue

        for n in n_labelled_list:
            # Check if the .npz file already exists
            npz_file_path = os.path.join(folder, f'{recalls_filename}_{n}.npz')
            if os.path.exists(npz_file_path) and not overwrite:
                logging.info(f"{npz_file_path} already exists. Skipping...")
                continue

            # Read and process the selected file
            try:
                df_score = read_scores(folder, n, df_labels)
                
                # Calculate recall values
                if HQ==True:
                    num_elements, recalls, TP = recall_HQ(df_score, sort_by = sort_by)  # here i calcualte HQ recall
                else:
                    num_elements, recalls, TP = recall(df_score, sort_by = sort_by)  # here i calcualte recall
                
                # Save the recall and x-values as .npz file
                np.savez(npz_file_path, num_elements=num_elements, recalls=recalls, TP=TP)
            except Exception as e:
                logging.error(f"Error processing file for n={n} in folder {folder}: {e}")
                continue

def plot_from_different_folders_fixed_n(folder_list, n_labelled_list, recalls_filename='', y_axis='recalls'):
    """
    Plot recall values from different folders for the same n with an inset zoomed-in plot.
    """
    color_palette = sns.color_palette("tab20", len(folder_list))
    plt.figure(figsize=(10, 8))

    # Main plot
    ax = plt.gca()  # Get the current axes to plot the main plot

    # Create the inset for zoomed-in plot
    axins = inset_axes(ax, width="40%", height="40%", loc='lower right', borderpad=2)

    for idx, folder in enumerate(folder_list):
        for n in n_labelled_list:
            try:
                data = np.load(os.path.join(folder, f'{recalls_filename}{n}.npz'))
                # Extract label information from the folder
                id_lab = folder.split('/')[-1].split('_')[0]
                label = f'Tradeoff {extract_variables(folder)[5]}'
                if 'KiDS' in id_lab:
                    label = f'Tradeoff {extract_variables(folder)[5]} {id_lab}'
                elif 'old' in folder.split('/')[-1]:
                    label = f'Tradeoff {extract_variables(folder)[5]} old'
                label += f' batch={extract_variables(folder)[6]}, score={extract_variables(folder)[7]}'
                
                num_elements = data['num_elements']
                recalls = data[y_axis]

                # Plot on the main plot
                ax.plot(num_elements, recalls, label=label, color=color_palette[idx])

                # Plot on the inset (zoomed-in) plot
                axins.plot(num_elements, recalls, label=label, color=color_palette[idx])

            except FileNotFoundError:
                continue

    # Customize the main plot
    ax.set_xlabel('Index in ranked list', fontsize=14)
    ax.set_ylabel('Number of Lenses', fontsize=14)
    ax.legend(title=f'n labelled={n_labelled_list[0]}', fontsize=10, title_fontsize='11',loc='upper left')

    # Customize the inset (zoomed-in) plot
    axins.set_xlim(0, 2500)  # Set the x-axis limit for zoomed-in view
    axins.set_ylim(0,150) 
    axins.set_title("Zoom-in", fontsize=10)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'Recall_plots/Diff_folder_nlabelled={n_labelled_list[0]}_with_zoom_inset.png')
    plt.show()


def plot_from_same_folder(folder, n_labelled_list, recalls_filename='', y_axis='TP'):
    """
    Plot all recall values from the same folder for different n.
    """
    color_palette = sns.color_palette("tab20", len(n_labelled_list))
    plt.figure(figsize=(8, 6))

    for idx, n in enumerate(n_labelled_list):
        try:
            data = np.load(os.path.join(folder, f'{recalls_filename}{n}.npz'))
            num_elements = data['num_elements']
            recalls = data[y_axis]
            if idx%2==0:
                ls='-'
            else:
                ls= '--'
            plt.plot(num_elements, recalls, label=f'n = {n}', color=color_palette[idx], ls=ls)
        except FileNotFoundError:
            continue
            
    title = f'Tradeoff {extract_variables(folder)[5]} batch={extract_variables(folder)[6]}'
    plt.xlabel('Index in ranked list', fontsize=14)
    plt.ylabel('Number of Lenses', fontsize=14)
    plt.legend(title='n labelled', fontsize=10, title_fontsize='12')
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(f'Recall_plots/Same_folder_GP_tradeoff={extract_variables(folder)[5]}_nlabelled=={extract_variables(folder)[6]}')
    plt.show()



def extract_variables(
    folder_name: str, 
    pattern: str = r'([^_]+)_dimred_([^_]+)_var_([^_]+)_prepbefore_([^_]+)_([^_]+)_tradeoff_([^_]+)_batches_([^_]+)_score_([^_]+)'
) -> Tuple[str, str, str, str, str, str, str, str]:
    """
    Extract variables from a folder name using a general pattern.

    Parameters:
        folder_name (str): The folder name to parse.
        pattern (str): The regex pattern to match the folder name.

    Returns:
        Tuple[str, str, str, str, str, str, str, str]: Extracted variables from the folder name.

    Raises:
        ValueError: If the folder name does not match the expected pattern.
    """
    # Match the folder name with the pattern
    match = re.search(pattern, folder_name)
    if match:
        return match.groups()
    else:
        raise ValueError("Folder name does not match the expected pattern")

def generate_folder_names(
        prefix: Union[str, List[str]],
        dataset: Union[str, List[str]], 
        dimred_method: Union[str, List[str]], 
        var: Union[str, List[str]], 
        preprocessing: Union[str, List[str]], 
        gp_method: Union[str, List[str]], 
        tradeoff: Union[str, List[str]], 
        batches: Union[str, List[str]], 
        score: Union[str, List[str]]) -> List[str]:
    '''
    Function that creates folder names given the parameters.
    '''
    
    # Convert all inputs to lists if they are not already
    prefix = [prefix] if isinstance(prefix, str) else prefix
    dataset = [dataset] if isinstance(dataset, str) else dataset
    dimred_method = [dimred_method] if isinstance(dimred_method, str) else dimred_method
    var = [var] if isinstance(var, str) else var
    preprocessing = [preprocessing] if isinstance(preprocessing, str) else preprocessing
    gp_method = [gp_method] if isinstance(gp_method, str) else gp_method
    tradeoff = [tradeoff] if isinstance(tradeoff, str) else tradeoff
    batches = [batches] if isinstance(batches, str) else batches
    score = [score] if isinstance(score, str) else score
    
    # Generate all combinations of the parameters using itertools.product
    combinations = itertools.product(prefix, dataset, dimred_method, var, preprocessing, gp_method, tradeoff, batches, score)
    
    folder_names = []
    
    # Template for the folder name
    template = "{prefix}{dataset}_dimred_{dimred_method}_var_{var}_prepbefore_{preprocessing}_{gp_method}_tradeoff_{tradeoff}_batches_{batches}_score_{score}"
    
    # Iterate through the combinations and create folder names
    for combination in combinations:
        # If prefix is empty, set it as an empty string to avoid having "None" in the name
        prefix_value = f"{combination[0]}_" if combination[0] else ""
        
        folder_name = template.format(
            prefix=prefix_value,
            dataset=combination[1],
            dimred_method=combination[2],
            var=combination[3],
            preprocessing=combination[4],
            gp_method=combination[5],
            tradeoff=combination[6],
            batches=combination[7],
            score=combination[8]
        )
        folder_names.append(folder_name)
    
    return folder_names


def recall(ml_score: pd.DataFrame, n_bin: int = 10, column: str = 'LABEL', sort_by: str = 'trained_score') -> Tuple[np.ndarray, List[float], List[int]]:

    '''
    function for calculating the recall 
    in the plots i use the number of TP, but I also calculate the recall in 0,1 range
    '''
    
    df_sorted = ml_score.sort_values(sort_by, ascending=False)
    
    #this is used to define the bins in which I calculate the TP and recall
    num_elements = np.arange(1, len(df_sorted) + 1, n_bin)

    recalls = []
    TP_list = []

    for i in num_elements:
        true_labels = df_sorted.iloc[:i][column]
        TP = (true_labels >= 3).sum()
        FN = (df_sorted.iloc[i:][column] >= 3).sum()

        recalls.append(TP / (TP + FN))
        TP_list.append(TP)

    return num_elements, recalls, TP_list
