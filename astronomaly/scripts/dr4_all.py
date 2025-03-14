import sys
#import faulthandler
#faulthandler.enable()
import warnings
warnings.filterwarnings("ignore")


#sys.path.append('/home/grespanm/github/TEGLIE/teglie_scripts/')
#sys.path.append('/home/grespanm/github/KiDS_astronomaly/')

sys.path.append('/home/astrodust/mnt/github/TEGLIE/teglie_scripts/')
sys.path.append('/home/astrodust/mnt/github/KiDS_astronomaly/')

import preprocessing
from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
#from astronomaly.feature_extraction import shape_features, pretrained_cnn
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, gaussian_process, human_loop_learning
from astronomaly.visualisation import umap_plot
from astronomaly.visualisation import tsne_plot

import os
import pandas as pd
import zipfile
import make_rgb 
from astronomaly.dimensionality_reduction import pca
import gen_cutouts, settings, make_rgb
import numpy as np
import glob
import sys

#sys.path.append('/home/grespanm/github/KiDS_astronomaly/Feature_extraction/')
#sys.path.append('/home/grespanm/github/KiDS_astronomaly/Feature_extraction/Features/backbone/')


sys.path.append('/home/astrodust/mnt/github/KiDS_astronomaly/Feature_extraction/')
sys.path.append('/home/astrodust/mnt/github/KiDS_astronomaly/Feature_extraction/Features/backbone/')

from run_byol import BYOLTEGLIETest
import utils_byol

from datasetkids import KiDSDatasetloader

byol_test = BYOLTEGLIETest()


# Root directory for data
data_dir = settings.path_to_save_imgs
is_kids = True

df_list_obj= pd.read_parquet(os.path.join('/home/astrodust/mnt/github/data','big_dataframe.parquet'))
#df_list_obj= pd.read_parquet(os.path.join('/home/grespanm/github/data','big_dataframe.parquet'))
#df_list_obj['FOLDER'] = data_dir

## features info 
dim_reduction = 'pca'
feature_method = 'byol'
#model_choice =  'Resnet18' 

variance = 0.95
image_prep = 'sigmaclip_gray'


### GP info
AL_model =  'GP'
ei_tradeoff = 2

img_prep_list = []
force_rerun=False
vis= 'umap'
protege = 'kids'
batch = 200
score = 'acquisition'


#_batches_{batch}_score_{score}

print(img_prep_list) 
# Where output should be stored
output_dir = os.path.join(
    data_dir, 'DR4_astronomaly_output',
    f'DR4{"_".join(img_prep_list)}_dimred_{dim_reduction}_var_{variance}_prepbefore_{image_prep.replace("_", "")}_{AL_model}_tradeoff_{ei_tradeoff}', '')


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


############ creeate pca extracted features file to load

path_FE = '/home/astrodust/mnt/github/KiDS_astronomaly/Feature_extraction/pca_results'
name_features = f'combined_features_labels_0.95.npz'

felab = np.load(os.path.join(path_FE, name_features), allow_pickle=True)

pd.DataFrame(data= felab['features'], index =  felab['ids'] ).to_parquet(os.path.join(output_dir,'PCA_Decomposer_output.parquet'))



# These are transform functions that will be applied to images before feature
# extraction is performed. Functions are called in order.
image_transform_function = []
#for img_prep_tmp in img_prep_list:
#    image_transform_function = add_preprocessing(image_transform_function)

# You can apply a different set of transforms to the images that get displayed
# in the frontend. In this case, I want to see the original images before sigma
# clipping is applied.
display_transform_function = []

    

def run_pipeline():
    """
    Any script passed to the Astronomaly server must implement this function.
    run_pipeline must return a dictionary that contains the keys listed below.

    Parameters
    ----------

    Returns
    -------
    pipeline_dict : dictionary
        Dictionary containing all relevant data. Keys must include: 
        'dataset' - an astronomaly Dataset object
        'features' - pd.DataFrame containing the features
        'anomaly_scores' - pd.DataFrame with a column 'score' with the anomaly
        scores
        'visualisation' - pd.DataFrame with two columns for visualisation
        (e.g. TSNE or UMAP)
        'active_learning' - an object that inherits from BasePipeline and will
        run the human-in-the-loop learning when requested

    """

    # This creates the object that manages the data

    image_dataset = image_reader.ImageThumbnailsDataset(
        output_dir=output_dir, 
        transform_function=image_transform_function,
        display_transform_function=display_transform_function,
        display_image_size=424, fits_format=True, df_kids=df_list_obj)
        

    #dimensionality reduction
    if dim_reduction == 'pca':
        # Define file path
        output_file_path = os.path.join(output_dir, 'PCA_Decomposer_output.parquet')

        # Load the features from the parquet file
        features = pd.read_parquet(output_file_path)
        print(f"Loaded features from {output_file_path} with shape {features.shape}")


    print("Number of components:", features.shape[1])

    if True:
        #### Protege
        # Initial sort using PCA
        initial_steps = 10
        sorted_features = features.sort_values(features.columns[0])
        selected_inds = np.linspace(
            0, len(sorted_features)-1, initial_steps, dtype='int')
        selected_inds = sorted_features.index[selected_inds]
        anomalies = pd.DataFrame(
            [0]*len(sorted_features),
            index=sorted_features.index,
            columns=['score'])
        anomalies.loc[selected_inds] = 5
        anomalies = anomalies.sort_values('score', ascending=False)

    if False:
        sorted_features = features.sort_values(features.columns[0])
        df_hq_kids = pd.read_csv('/home/grespanm/github/data/kids_in_all_checked.csv')
        selected_inds = df_hq_kids['KIDS_ID_1']
        anomalies = pd.DataFrame(
            [0]*len(sorted_features),
            index=sorted_features.index,
            columns=['score'])
        anomalies.loc[selected_inds] = 5
        anomalies = anomalies.sort_values('score', ascending=False)

    if False:
        try:
            anomalies = pd.read_csv(os.path.join(output_dir, 'ml_scores.csv')).sort_values('score', 
                                                    ascending=False).set_index('Index')[['score']]
            anomalies.index.name = None
        except:
            anomalies = pd.read_csv(os.path.join(output_dir, 'ml_scores.csv')).sort_values('score', 
                                                    ascending=False).set_index('Unnamed: 0')[['score']]
            anomalies.index.name = None
        print(anomalies)

    if True:
        try:
            # This is used by the frontend to store labels as they are applied so
            # that labels are not forgotten between sessions of using Astronomaly
            if 'human_label' not in anomalies.columns:
                df = pd.read_csv(
                    os.path.join(output_dir, 'ml_scores.csv'), 
                    index_col=0,
                    dtype={'human_label': 'int'})
                df.index = df.index.astype('str')

                # This deals with the possibility of not all objects having labels
                #if len(anomalies) == len(df):
                #    anomalies = pd.concat(
                #        (anomalies, df['human_label']), axis=1, join='inner')
                    
                #### LABEL ALL #####
                inds = df.index[np.in1d(df.index, anomalies.index)]
                anomalies.loc[inds, "human_label"] = df["human_label"]
                print(f"{(anomalies.human_label!=-1).sum()} labels added")
                
        except FileNotFoundError:
            print('exception')
            pass

    # This is the active learning object that will be run on demand by the
    # frontend 
    #NB: the alpha here it used to weight the human labels 
    # in comparison to the algorithm one
    pipeline_active_learning = gaussian_process.GaussianProcess(
        features, output_dir=output_dir, force_rerun=False, ei_tradeoff=ei_tradeoff,
    )
    
    if vis=='tsne':
        pipeline_tsne = tsne_plot.TSNE_Plot(
            force_rerun=force_rerun,
            output_dir=output_dir,
            perplexity=5)
        vis_plot = pipeline_tsne.run(features.loc[anomalies.index])
    elif vis=='umap':
        # We use UMAP for visualisation which is run in the same way as other parts
        # of the pipeline.
        pipeline_umap = umap_plot.UMAP_Plot(
            force_rerun=force_rerun,
            output_dir=output_dir,
            shuffle=False)
        
        print(np.shape(features))
        vis_plot = pipeline_umap.run(features)


    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'active_learning': pipeline_active_learning,
            'visualisation': vis_plot}

