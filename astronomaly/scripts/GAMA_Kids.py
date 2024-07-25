import sys
#import faulthandler
#faulthandler.enable()
import warnings
warnings.filterwarnings("ignore")



sys.path.append('/home/grespanm/github/TEGLIE/teglie_scripts/')
sys.path.append('/home/grespanm/github/KiDS_astronomaly/')
import preprocessing
from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import shape_features, pretrained_cnn
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
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




def add_preprocessing(image_transform_function):

    if img_prep_tmp=='grey-avg':
        image_transform_function.append(image_preprocessing.grayscale_average)
    elif img_prep_tmp=='grey-lum':
        image_transform_function.append(image_preprocessing.grayscale_luminosity)
    elif img_prep_tmp=='grey-lightness':
        image_transform_function.append(image_preprocessing.grayscale_lightness)
    elif img_prep_tmp=='grey-desaturation':
        image_transform_function.append(image_preprocessing.grayscale_desaturation)
    elif img_prep_tmp=='rgb': 
        image_transform_function.append(make_rgb.make_rgb_one_image)
    elif img_prep_tmp=='rgb2d':
        image_transform_function.append(image_preprocessing.create_2d_rgb_image)
    elif img_prep_tmp=='rband':
        image_transform_function.append(image_preprocessing.image_get_first_band)
    elif img_prep_tmp=='sigmaclip':
        image_transform_function.append(image_preprocessing.image_transform_sigma_clipping)
    elif img_prep_tmp=='scaleclip':
         image_transform_function.append(preprocessing.scaling_clipping)
    elif img_prep_tmp=='scale':
        image_transform_function.append(image_preprocessing.image_transform_scale)

    return image_transform_function

# Root directory for data
data_dir = settings.path_to_save_imgs
is_kids=True
one_tile=False



if one_tile:
    table_tile = cutclass.getTableTile(tile_name='KIDS_40.6_-28.2')
    tab_to_use = gen_cutouts.apply_preprocessing(table_tile)[:50]
    df_list_obj = tab_to_use['ID', 'KIDS_TILE'].to_pandas().reset_index(drop=True)
    df_list_obj['FOLDER'] = settings.path_to_save_imgs 
    print(settings.path_to_save_imgs ) 
    df_list_obj.rename(columns={"ID": "KIDS_ID"}, inplace=True)

else:
    df_list_obj= pd.read_csv(os.path.join(data_dir,'GAMA',
                    'table_all_checked.csv')).drop_duplicates(subset='KIDS_ID').reset_index(drop=True)[:100]
    df_list_obj = df_list_obj
    df_list_obj['FOLDER'] = data_dir

dim_reduction = 'pca'
feature_method = 'cnn'
model_choice =  'zoobot' 
img_prep_list = ['image_transform_resize', 'scaleclip']
force_rerun=False
vis= 'umap'


print(img_prep_list) 
# Where output should be stored

output_dir = os.path.join(
    data_dir, 'GAMA_astronomaly_output', 
    f'GAMA_{"_".join(img_prep_list)}_dimred_{dim_reduction}_model_{feature_method}_weights_{model_choice}', '')
'''
if force_rerun==True and os.path.exists(glob.glob(os.path.join(output_dir,'*'))): 
    # loop through the list and delete each file
    for file_name in glob.glob(os.path.join(output_dir,'*')):
        file_path = os.path.join(output_dir, file_name)
        os.remove(file_path)
'''

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# These are transform functions that will be applied to images before feature
# extraction is performed. Functions are called in order.
image_transform_function = []
for img_prep_tmp in img_prep_list:
    image_transform_function = add_preprocessing(image_transform_function)

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
    if is_kids==True:
        image_dataset = image_reader.ImageThumbnailsDataset(
            output_dir=output_dir, 
            transform_function=image_transform_function,
            display_transform_function=display_transform_function,
            display_image_size=424, fits_format=True, df_kids=df_list_obj)
    else:
        image_dir = os.path.join(data_dir, 'ugri_images')
        image_dataset = image_reader.ImageThumbnailsDataset(
            directory=image_dir,
            output_dir=output_dir, 
            transform_function=image_transform_function,
            display_transform_function=display_transform_function,
            display_image_size=424, fits_format=True)


    ### feature selection
    
    if feature_method == 'ellipse':
        
        # Creates a pipeline object for feature extraction
        pipeline_ellipse =  shape_features.EllipseFitFeatures(
            percentiles=[90, 80, 70, 60, 50, 0],
            output_dir=output_dir, channel=0, force_rerun=force_rerun, 
            central_contour=False, upper_limit= 300)

        # Actually runs the feature extraction
        features = pipeline_ellipse.run_on_dataset(image_dataset)

    elif feature_method == 'cnn':
        #print('cnn')
        cnn =  pretrained_cnn.CNN_Features(model_choice,force_rerun=force_rerun)
        features = cnn.run_on_dataset(image_dataset)
    '''
    elif feature_method == 'te':
        te =  pretrained_cnn.TE_Features(model_choice, force_rerun=True)
        features = te.run_on_dataset(image_dataset)
    '''

    #dimensionality reduction
    if dim_reduction == 'pca':
        pipeline_pca = pca.PCA_Decomposer(force_rerun=force_rerun,
                                            output_dir=output_dir,
                                            threshold=0.95)
        features = pipeline_pca.run(features)
        print(np.shape(features))

    # Now we rescale the features using the same procedure of first creating
    # the pipeline object, then running it on the feature set
    #pipeline_scaler = scaling.FeatureScaler(force_rerun=force_rerun,
                                            #output_dir=output_dir)
    #features = pipeline_scaler.run(features)

    # The actual anomaly detection is called in the same way by creating an
    # Iforest pipeline object then running it
    pipeline_iforest = isolation_forest.IforestAlgorithm(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_iforest.run(features)

    # We convert the scores onto a range of 0-5
    pipeline_score_converter = human_loop_learning.ScoreConverter(
        force_rerun=False, output_dir=output_dir)
    anomalies = pipeline_score_converter.run(anomalies)

    try:
        # This is used by the frontend to store labels as they are applied so
        # that labels are not forgotten between sessions of using Astronomaly
        if 'human_label' not in anomalies.columns:
            df = pd.read_csv(
                os.path.join(output_dir, 'ml_scores.csv'), 
                index_col=0,
                dtype={'human_label': 'int'})
            df.index = df.index.astype('str')

            if len(anomalies) == len(df):
                anomalies = pd.concat(
                    (anomalies, df['human_label']), axis=1, join='inner')
    except FileNotFoundError:
        pass

    # This is the active learning object that will be run on demand by the
    # frontend 
    #NB: the alpha here it used to weight the human labels 
    # in comparison to the algorithm one
    pipeline_active_learning = human_loop_learning.NeighbourScore(
        alpha=0.2, output_dir=output_dir)
    
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
            output_dir=output_dir)
        vis_plot = pipeline_umap.run(features)

    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'active_learning': pipeline_active_learning,
            'visualisation': vis_plot}

