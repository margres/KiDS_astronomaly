import importlib.util
import importlib
import pandas as pd
from controller_module import Controller  # Import your Controller class
import os

import os
import pandas as pd
import sys
#sys.path.append("/Users/mrgr/Documents/GitHub/KiDS_astronomaly/")
sys.path.append('/home/grespanm/github/KiDS_astronomaly/')

from flask import Flask, render_template, request, Response
import json
from os.path import join
from astronomaly.frontend.interface import Controller

def save_ml_scores(ml_scores_path, output_dir):
    """
    Saves the current ml_scores to a backup file with the number of labels applied.
    
    Parameters:
    -----------
    ml_scores_path : str
        Path to the ml_scores.csv file.
    total_labels_applied : int
        Total number of labels applied so far.
    output_dir : str
        Directory where the backup file will be saved.
    """
    if os.path.exists(ml_scores_path):
        ml_scores_df = pd.read_csv(ml_scores_path)
        total_labels_applied = (ml_scores_df['human_label'] != -1).sum()
        # Create the backup filename with the total number of labels applied
        backup_filename = f'ml_scores_{total_labels_applied}.csv'
        backup_path = os.path.join(output_dir, backup_filename)
        # Save the ml_scores as a backup
        ml_scores_df.to_csv(backup_path, index=False)
        print(f"Backup saved: {backup_filename}")
    else:
        print(f"ml_scores.csv file not found at {ml_scores_path}")

    return total_labels_applied

def apply_labels_from_batch(controller, batch, df_human_labels, ml_scores_path, max_labels):
    """
    Applies human labels from df_human_labels to the given batch in sorted_ml_df.

    Parameters:
    -----------
    controller : Controller object
        The controller instance to interact with the pipeline.
    batch : pd.DataFrame
        A batch of the sorted_ml_df DataFrame to apply labels to.
    df_human_labels : pd.DataFrame
        DataFrame containing the KIDS_ID and FINAL_LABEL columns.
    ml_scores_path : str
        Path to the ml_scores.csv file.
    total_labels_applied : int
        Current number of labels applied.
    max_labels : int
        Maximum number of labels to apply before stopping.
    
    Returns:
    --------
    int
        Updated total_labels_applied after processing the batch.
    bool
        Whether the maximum labels limit has been reached.
    """

    for i, (idx, row) in enumerate(batch.iterrows()):
        kids_id = row.name  # Get KIDS_ID from the row's index

        # Get the corresponding human label from df_human_labels
        human_label_row = df_human_labels[df_human_labels['KIDS_ID'] == kids_id]

        if not human_label_row.empty:
            human_label = human_label_row['LABEL'].values[0]

            # Apply the human label to the corresponding KIDS_ID in ml_df
            controller.set_human_label(kids_id, human_label)
            #print(f"Label {human_label} applied to {kids_id}.")
            
            # Save after applying the first label in the batch
            if i == 0:
                total_labels_applied = save_ml_scores(ml_scores_path, controller.active_learning.output_dir)
        else:
            print(f"No human label found for KIDS_ID {kids_id} in df_human_labels.")

        # Stop if the label threshold is reached
        if total_labels_applied >= max_labels:
            print(f"Labeling threshold of {max_labels} reached.")
            total_labels_applied = save_ml_scores(ml_scores_path, controller.active_learning.output_dir)
            return total_labels_applied, True  # Stop labeling

    return total_labels_applied, False  # Continue labeling
        
def apply_labels_and_run_active_learning(controller, df_human_labels, batch_size, max_labels=50, by ='score'):
    """
    Apply initial labels in batches, run active learning, and then label remain}ing elements
    based on 'trained_score' in batches. Stops labeling after reaching max_labels.
    
    Parameters:
    controller : Controller object
        The controller instance to interact with the pipeline.
    df_human_labels : pd.DataFrame
        DataFrame containing KIDS_ID and FINAL_LABEL columns.
    initial_batch_size : int
        Number of labels to apply in the first round before running active learning.
    sorted_batch_size : int
        Number of labels to apply in batches after sorting by 'trained_score'.
    max_labels : int
        Maximum number of labels to apply before stopping the process.
    """

    # initial loading
    
    ml_df = controller.anomaly_scores

    # Check if 'human_label' column exists, if not, create it
    if 'human_label' not in ml_df.columns:
        ml_df['human_label'] = [-1] * len(ml_df)

    ml_scores_path = os.path.join(controller.active_learning.output_dir, 'ml_scores.csv')
    # initial sorting
    
    j=0
    try:
        sorted_ml_df = ml_df.sort_values(by=by, ascending=False)
    except KeyError:
        j+=1
        print('exception, sorting by score')
        sorted_ml_df = ml_df.sort_values(by='score', ascending=False)
        if j>1:
            raise Exception('Error with sorting value')

    
    for i in range(int(max_labels/batch_size)):
        #print(ml_df.head())
        controller.sort_ml_scores(column_to_sort_by=by, show_unlabelled_first=True)
        ml_df = controller.anomaly_scores
        #print(ml_df.head())
    
        batch = ml_df.iloc[0:batch_size]
    
        # Apply labels to the current batch
        total_labels_applied, stop_labeling = apply_labels_from_batch(
            controller, batch, df_human_labels, ml_scores_path, max_labels
        )
    
        if stop_labeling:
            return  
            
        # Run active learning after each batch
        print("Running active learning after labeling batch...")
        controller.run_active_learning()

    # Save the ml_scores.csv after each batch
    #save_ml_scores(ml_scores_path, total_labels_applied, controller.active_learning.output_dir)


# Function to load and reload the script module
def load_pipeline_module(module_path):
    spec = importlib.util.spec_from_file_location("pipeline_module", module_path)
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    return pipeline_module

# Function to set parameters dynamically in the module
def set_parameters(module, ei_tradeoff, batch, score):
    setattr(module, 'ei_tradeoff', ei_tradeoff)
    setattr(module, 'batch', batch)
    setattr(module, 'score', score)

# Define the path to the script
pipeline_file = '/home/grespanm/github/KiDS_astronomaly/astronomaly/scripts/astronomaly_scripted.py'

# Parameter combinations to test
ei_tradeoff_values = [0.2, 0.5, 1.0, 3]
batch_values = [10, 20, 50, 100]
score_options = ['acquisition']

# Load your DataFrame containing human labels
df_human_labels = pd.read_csv('mock_data_for_testing_continuos.csv')

# Loop through parameter combinations
for ei_tradeoff in ei_tradeoff_values:
    for batch in batch_values:
        for score in score_options:
            # Load the pipeline module
            pipeline_module = load_pipeline_module(pipeline_file)
            
            # Set parameters in the pipeline script
            set_parameters(pipeline_module, ei_tradeoff, batch, score)
            
            # Initialize and run the controller with new parameters
            controller = Controller(pipeline_file=pipeline_file)
            controller.run_pipeline()
            
            # Apply labels and run active learning
            print(f"Running with ei_tradeoff={ei_tradeoff}, batch={batch}, score={score}")
            apply_labels_and_run_active_learning(
                controller=controller, 
                df_human_labels=df_human_labels, 
                batch_size=batch, 
                max_labels=5+batch,
                by=score
            )

            # Optional: save results after each iteration
            output_file = f'results_ei_{ei_tradeoff}_batch_{batch}_score_{score}.csv'
            controller.anomaly_scores.to_csv(os.path.join('output', output_file), index=False)
            print(f"Saved results to {output_file}\n")
