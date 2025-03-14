import os
import pandas as pd
import sys
import importlib.util
import importlib
from flask import Flask, render_template, request, Response
from os.path import join
from astronomaly.frontend.interface import Controller
 
import time

user = os.path.expanduser("~") 
if 'home/grespanm' in user:
    base_path = os.path.join(user, 'github')
elif 'home/astrodust' in user:
    base_path = os.path.join(user, 'mnt','github') 


# Add the astronomaly module path
sys.path.append(f'{base_path}/KiDS_astronomaly/')

def save_ml_scores(ml_scores_path, output_dir):
    """
    Save the current ml_scores to a backup file with the number of labels applied.

    Parameters:
    -----------
    ml_scores_path : str
        Path to the ml_scores.csv file.
    output_dir : str
        Directory where the backup file will be saved.
    """
    if os.path.exists(ml_scores_path):
        ml_scores_df = pd.read_csv(ml_scores_path)
        total_labels_applied = (ml_scores_df['human_label'] != -1).sum()
        backup_filename = f'ml_scores_{total_labels_applied}.csv'
        backup_path = os.path.join(output_dir, backup_filename)
        ml_scores_df.to_csv(backup_path, index=False)
        print(f"Backup saved: {backup_filename}")
    else:
        print(f"ml_scores.csv file not found at {ml_scores_path}")
        total_labels_applied = 0

    return total_labels_applied

def apply_labels_from_batch(controller, batch, df_human_labels, ml_scores_path, max_labels):
    """
    Apply human labels from df_human_labels to the given batch in sorted_ml_df.
    """
    total_labels_applied = 0

    for i, (idx, row) in enumerate(batch.iterrows()):
        kids_id = row.name
        human_label_row = df_human_labels[df_human_labels['KIDS_ID'] == kids_id]

        if not human_label_row.empty:
            human_label = human_label_row['LABEL'].values[0]
            controller.set_human_label(kids_id, human_label)

            if i == 0:
                total_labels_applied = save_ml_scores(ml_scores_path, controller.active_learning.output_dir)
        else:
            print(f"No human label found for KIDS_ID {kids_id} in df_human_labels.")

        if total_labels_applied >= max_labels:
            print(f"Labeling threshold of {max_labels} reached.")
            total_labels_applied = save_ml_scores(ml_scores_path, controller.active_learning.output_dir)
            return total_labels_applied, True

    return total_labels_applied, False

def apply_labels_and_run_active_learning(controller, df_human_labels, batch_size, max_labels=50, by='score'):
    """
    Apply initial labels in batches, run active learning, and then label remaining elements.
    """
    ml_df = controller.anomaly_scores

    if 'human_label' not in ml_df.columns:
        ml_df['human_label'] = [-1] * len(ml_df)

    ml_scores_path = os.path.join(controller.active_learning.output_dir, 'ml_scores.csv')

    try:
        sorted_ml_df = ml_df.sort_values(by=by, ascending=False)
    except KeyError:
        print('Sorting key not found, falling back to "score"')
        sorted_ml_df = ml_df.sort_values(by='score', ascending=False)

    for i in range(int(max_labels / batch_size)):
        controller.sort_ml_scores(column_to_sort_by=by, show_unlabelled_first=True)
        ml_df = controller.anomaly_scores
        batch = ml_df.iloc[0:batch_size]

        total_labels_applied, stop_labeling = apply_labels_from_batch(
            controller, batch, df_human_labels, ml_scores_path, max_labels
        )
        if stop_labeling:
            return

        print("Running active learning after labeling batch...")
        controller.run_active_learning()

def load_pipeline_module(module_path):
    """
    Dynamically load a Python module from the given path.
    """
    spec = importlib.util.spec_from_file_location("pipeline_module", module_path)
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    return pipeline_module
'''
# Main execution logic
if __name__ == "__main__":
    pipeline_file = f'{base_path}/KiDS_astronomaly/astronomaly/scripts/astronomaly_scripted.py'
    pipeline_module = load_pipeline_module(pipeline_file)

    # Access variables directly from the pipeline script
    ei_tradeoff = pipeline_module.ei_tradeoff
    batch = pipeline_module.batch
    score = pipeline_module.score

    print(f"ei_tradeoff: {ei_tradeoff}, batch: {batch}, score: {score}")

    # Initialize the Controller with the script
    controller = Controller(pipeline_file=pipeline_file)
    controller.run_pipeline()

    # Sample DataFrame containing KIDS_ID and FINAL_LABEL (replace with your actual DataFrame)
    df_human_labels = pd.read_csv('final_labels_corrected_coord.csv')
    
    df_human_labels['FINAL_LABEL'] = df_human_labels['LABEL']
    df_human_labels['FINAL_LABEL'] = df_human_labels['FINAL_LABEL'].replace({0: 1})
    # Run the active learning process
    apply_labels_and_run_active_learning(
        controller=controller, 
        df_human_labels=df_human_labels.reset_index(), 
        batch_size=batch, 
        max_labels=1300,
        by=score
    )
'''
import itertools
import re

def update_pipeline_script(file_path, batch, ei_tradeoff, score):
    """
    Updates the pipeline script with the given batch, ei_tradeoff, and score values.
    """
    with open(file_path, 'r') as file:
        script_content = file.read()

    # Safely replace `batch`, `ei_tradeoff`, and `score` variables
    script_content = re.sub(r"batch\s*=\s*\d+", f"batch = {batch}", script_content)
    script_content = re.sub(r"ei_tradeoff\s*=\s*\d+", f"ei_tradeoff = {ei_tradeoff}", script_content)
    script_content = re.sub(r"score\s*=\s*['\"].*?['\"]", f"score = '{score}'", script_content)
    script_content = re.sub(r"name_features\s*=\s*['\"].*?['\"]", f"name_features = '{name_features}'", script_content)

    # Write the updated script back
    with open(file_path, 'w') as file:
        file.write(script_content)
        
def save_pipeline_config(ei_tradeoff, batch, score, output_dir, file_path='pipeline_config.txt', name_features = ''):
    """
    Saves the pipeline configuration to a text file.

    Parameters:
        ei_tradeoff (float): The exploration-exploitation tradeoff value.
        batch (int): The batch size.
        score (str): The scoring metric.
        output_dir (str): The output directory.
        file_path (str): Path to the file where configuration will be saved.
    """
    with open(file_path, 'w') as f:
        f.write(f"ei_tradeoff={ei_tradeoff}\n")
        f.write(f"batch={batch}\n")
        f.write(f"score={score}\n")
        f.write(f"output_dir={output_dir}\n")
        f.write(f"AL_model={AL_model}\n")
        f.write(f"dim_reduction={dim_reduction}\n")
        f.write(f"img_prep_list={img_prep_list}\n")  # Empty list will be written as []
        f.write(f"protege={protege}\n")
        f.write(f"variance={variance}\n")
        f.write(f"image_prep={image_prep}\n")
        f.write(f"name_features={name_features}\n")         

    print(f"Configuration saved to {file_path}")



# Define ranges or lists of values for each variable
batch_sizes = [10, 20,50]  # Example batch sizes
ei_tradeoffs = [2, 3, 4]    # Example tradeoff values
scores = ['score', 'acquisition']  # Example score keys


# Main execution logic
if __name__ == "__main__":

    pipeline_file = f'{base_path}/KiDS_astronomaly/astronomaly/scripts/astronomaly_scripted.py'
    # Root directory for data
    data_dir = '/home/astrodust/mnt/github'
    #'/home/astrodust/mnt/Data/KiDS_cutouts/GAMA_astronomaly_output'
    
    # Loop through all combinations of batch_sizes and scores
    for batch, score in itertools.product(batch_sizes, scores):
        if score == 'score':
            # Skip iterating over ei_tradeoff if score is 'score'
            ei_tradeoff_values = [None]  # Placeholder to avoid unnecessary iteration
        else:
            ei_tradeoff_values = ei_tradeoffs
        
        for ei_tradeoff in ei_tradeoff_values:
            print(f"Running with batch: {batch}, ei_tradeoff: {ei_tradeoff}, score: {score}")

            ### GP info
            AL_model =  'GP'
            dim_reduction= 'pca'
            img_prep_list = []
            protege = 'protege'
            variance= 0.99
            image_prep = 'sigmaclipgray'
            dataset = '101x101BGTEGLIE_0NOTreplaced'
            name_features = '101x101BGTEGLIE_features&labels_prepbefore_[sigmaclipgray]_PCA_var_0.99_scaler_False'#f'{dataset}_features&labels_prepbefore_[sigmaclipgray]_PCA_var_{variance}_scaler_False'            

            
            # Where output should be stored
            output_dir = os.path.join( data_dir, f'{dataset}{protege}{"_".join(img_prep_list)}_dimred_{dim_reduction}_var_{variance}_prepbefore_{image_prep.replace("_", "")}_{AL_model}_tradeoff_{ei_tradeoff}_batches_{batch}_score_{score}', '')
            
            # Update the pipeline script with new variables
            ei_tradeoff_value = ei_tradeoff if ei_tradeoff is not None else 3  # Default ei_tradeoff if skipped
            #update_pipeline_script(pipeline_file, batch, ei_tradeoff_value, score)
            os.makedirs(output_dir, exist_ok=True)
            file_path =  os.path.join(output_dir, 'pipeline_config.txt')

                                
            # Save the file_path to the pipeline configuration file
            with open('file_path.txt', 'w') as f:
                f.write(file_path)     
           
            save_pipeline_config(ei_tradeoff_value, batch, score, output_dir, file_path =  file_path, name_features= name_features)

            # Initialize the Controller with the updated script
            controller = Controller(pipeline_file=pipeline_file)
            pipeline_module = load_pipeline_module(pipeline_file)
            
            controller.run_pipeline()

            # Load human labels
            df_human_labels = pd.read_csv('final_labels_corrected_coord.csv')
            df_human_labels['FINAL_LABEL'] = df_human_labels['LABEL']
            df_human_labels['FINAL_LABEL'] = df_human_labels['FINAL_LABEL']#.replace({0: 1})

            # Run the active learning process for the current configuration
            apply_labels_and_run_active_learning(
                controller=controller,
                df_human_labels=df_human_labels.reset_index(),
                batch_size=batch,
                max_labels=1100,
                by=score
            
            )

            print(f"Completed batch: {batch}, ei_tradeoff: {ei_tradeoff}, score: {score}")

            import gc

            # Explicitly delete variables
            del controller
            gc.collect()  # Force garbage collection

