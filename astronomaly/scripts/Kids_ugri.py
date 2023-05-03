import sys
sys.path.append('/Users/mrgr/Documents/GitHub/FiLeK/')
#sys.path.append('/Users/mrgr/Documents/GitHub/KiDS_astronomaly/')
# An example with a subset of Galaxy Zoo data
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
import filek.make_rgb as make_rgb
from astronomaly.dimensionality_reduction import pca
import numpy as np
from filek import cutclass
import filek.utils as utils
import filek.gen_cutouts as gen_cutouts
import filek.settings as settings
 
'''
TODO:
 - something is wrong with the transformer as feature extractor, the shape is wrong
'''

# Root directory for data
data_dir = os.path.join(settings.path_to_save_imgs,'mock')
#os.path.join('/Users/mrgr/Documents/GitHub/KiDS_astronomaly/example_data/KiDS_cutouts')


is_kids=True
one_tile=False

if one_tile:
    table_tile = cutclass.getTableTile(tile_name='KIDS_40.6_-28.2')
    tab_to_use = gen_cutouts.apply_preprocessing(table_tile)[:50]
    df_list_obj = tab_to_use['ID', 'KIDS_TILE'].to_pandas().reset_index(drop=True)
    df_list_obj['FOLDER'] = settings.path_to_save_imgs  
    df_list_obj.rename(columns={"ID": "KIDS_ID"}, inplace=True)
else:
    df_list_obj= pd.read_csv(os.path.join(settings.path_to_save_imgs,'mock','df_tot.csv')).drop_duplicates(subset='KIDS_ID').reset_index(drop=True)[:-1]
    df_list_obj['FOLDER'] = data_dir

dim_reduction = 'pca'
feature_method = 'cnn'
model_choice =  'zoobot' #'resnet18'  
#img_prep= ['rband']
img_prep_list = ['rband', 'clipping']
force_rerun=True
vis= 'umap'

'''
img_prep_list = []
if len(img_prep)>0 and not isinstance(img_prep,list):
    img_prep_list.append(img_prep)
elif isinstance(img_prep,list):
    img_prep_list = img_prep 

if img_prep == '' or ('grey' not in img_prep and 'rband' not in img_prep):
    img_prep_list.append('rband')
'''    
print(img_prep_list) 
# Where output should be stored
output_dir = os.path.join(
    data_dir, 'astronomaly_output', 
    f'kids_mock_img_prep_{"_".join(img_prep_list)}_dim_red_{dim_reduction}_model_{feature_method}_weights_{model_choice}', '')

#sys.exit()
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_transform_function = []
# These are transform functions that will be applied to images before feature
# extraction is performed. Functions are called in order.
#for img_prep_tmp in img_prep:
for img_prep_tmp in img_prep_list:
    if img_prep_tmp=='grey':
        image_transform_function.append(image_preprocessing.image_transform_greyscale)
    elif img_prep_tmp=='rband':
        image_transform_function.append(image_preprocessing.image_get_first_band)
    elif img_prep_tmp=='clipping':
        image_transform_function.append(image_preprocessing.image_transform_sigma_clipping)
image_transform_function.append(image_preprocessing.image_transform_scale) #last one 

# You can apply a different set of transforms to the images that get displayed
# in the frontend. In this case, I want to see the original images before sigma
# clipping is applied.
display_transform_function = []
#NB: i modified the image_reader.py convert_array_to_image
    #image_preprocessing.image_get_first_band]
    #mage_preprocessing.image_transform_scale]
    #make_rgb.make_rgb_one_image ]
    

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
    pipeline_scaler = scaling.FeatureScaler(force_rerun=force_rerun,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)

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
