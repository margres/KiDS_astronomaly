# An example with a subset of Galaxy Zoo data
from astronomaly.data_management import image_reader
from astronomaly.preprocessing import image_preprocessing
from astronomaly.feature_extraction import shape_features
from astronomaly.postprocessing import scaling
from astronomaly.anomaly_detection import isolation_forest, human_loop_learning
from astronomaly.visualisation import umap_plot
#from astronomaly.visualisation import tsne_plot
import os
import pandas as pd
import zipfile
import sys
import numpy as np
from astronomaly.dimensionality_reduction import pca

sys.path.append('/home/grespanm/github/KiDS_astronomaly/Feature_extraction/')
sys.path.append('/home/grespanm/github/KiDS_astronomaly/Feature_extraction/Features/backbone/')

from run_byol import BYOLTEGLIETest
import utils_byol

from datasetkids import KiDSDatasetloader

byol_test = BYOLTEGLIETest()


# Root directory for data
#data_dir = os.path.join(os.getcwd(), 'example_data')

image_dir = byol_test.path_data

print('Image dir', image_dir)

# Where output should be stored
output_dir = os.path.join(byol_test.path_repository, 'astronomaly_output', 'kids_lenses')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# These are transform functions that will be applied to images before feature
# extraction is performed. Functions are called in order.
#image_transform_function = [
#    image_preprocessing.image_transform_sigma_clipping,
#    image_preprocessing.image_transform_scale]

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

    '''
    # This creates the object that manages the data
    image_dataset = image_reader.ImageThumbnailsDataset(
        directory=image_dir, output_dir=output_dir, 
        transform_function=image_transform_function,
        display_transform_function=display_transform_function,
        check_corrupt_data= True
    )

    # Creates a pipeline object for feature extraction
    pipeline_ellipse = shape_features.EllipseFitFeatures(
        percentiles=[90, 80, 70, 60, 50, 0],
        output_dir=output_dir, channel=2, force_rerun=False, 
        central_contour=False, upper_limit= 300
    )

    # Actually runs the feature extraction
    features = pipeline_ellipse.run_on_dataset(image_dataset)

    # Now we rescale the features using the same procedure of first creating
    # the pipeline object, then running it on the feature set
    pipeline_scaler = scaling.FeatureScaler(force_rerun=False,
                                            output_dir=output_dir)
    features = pipeline_scaler.run(features)
    
    '''
    loader = KiDSDatasetloader()
    image_dataset =  loader
    #file_path='features&labels_prepbefore_[50pxsigmaclipgray]_PCA_var_0.98_scaler_False.npz'

    #pd.DataFrame(byol_test.run_feature_extractor(variance_threshold=0.98, preprocessing_after_byol = [utils_byol.sigma_clipping_gray]   ))
    ft,lb = byol_test.run_feature_extractor(variance_threshold=0.98, preprocessing_after_byol = [utils_byol.sigma_clipping_gray]   )
    #get_features(os.path.join(byol_test, file_path))
    features = pd.DataFrame(data=ft)
    
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
    pipeline_active_learning = human_loop_learning.NeighbourScore(
        alpha=1, output_dir=output_dir)
    
    pipeline_pca = pca.PCA_Decomposer(force_rerun=False,
                                        output_dir=output_dir,
                                        threshold=1)
    features = pipeline_pca.run(features)

    # We use UMAP for visualisation which is run in the same way as other parts
    # of the pipeline.
    pipeline_umap = umap_plot.UMAP_Plot(
        force_rerun=False,
        output_dir=output_dir)
    vis_plot = pipeline_umap.run(features)

    # The run_pipeline function must return a dictionary with these keywords
    return {'dataset': image_dataset, 
            'features': features, 
            'anomaly_scores': anomalies,
            'active_learning': pipeline_active_learning,
            'visualisation': pipeline_umap }
