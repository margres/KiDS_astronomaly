import os
from nis import cat
from re import sub

import numpy as np
import pandas as pd
import umap
from astronomaly.anomaly_detection import (
    gaussian_process,
    human_loop_learning,
    isolation_forest,
    lof,
)
from astronomaly.data_management import image_reader
from astronomaly.dimensionality_reduction import pca
from astronomaly.feature_extraction import power_spectrum, shape_features
from astronomaly.postprocessing import scaling
from astronomaly.preprocessing import image_preprocessing
from astronomaly.utils import utils
from astronomaly.visualisation import umap_plot

# Select which resolution and subset to use
high_res = False
subset = False

# General directory
data_dir = "/home/verlon/Work_space/my_papers/detecting_diffuse_emission/"

# Images
image_dir = os.path.join(data_dir, "data", "")

# Catalogues
cat_dir = os.path.join(data_dir, "catalogues", "updated_28_June")


if high_res:
    image_dir = os.path.join(image_dir, "full_resolution")
    dir_ = os.listdir(image_dir)
    # list_of_files = os.listdir(os.path.join(image_dir, "full_resolution"))
    list_of_files = [f for f in dir_ if f[-5:] == ".fits"]

    if subset:
        catalogue_file = os.path.join(
            cat_dir, "full_catalogue_high_new_cut_subset_edge_sources_removed.csv"
        )
        output_dir = os.path.join(
            data_dir, "output", "astronomaly output", "high_res_subset", ""
        )
    else:
        catalogue_file = os.path.join(
            cat_dir, "full_catalogue_high_new_cut_edge_sources_removed.csv"
        )
        output_dir = os.path.join(
            data_dir, "output", "astronomaly output", "high_res", ""
        )


else:
    image_dir = os.path.join(image_dir, "convolved_low_resolution")
    dir_ = os.listdir(image_dir)
    # list_of_files = os.listdir(os.path.join(image_dir, "full_resolution"))
    list_of_files = [f for f in dir_ if f[-5:] == ".fits"]

    if subset:
        catalogue_file = os.path.join(
            cat_dir, "full_catalogue_convolved_new_cut_subset_edge_sources_removed.csv"
        )
        output_dir = os.path.join(
            data_dir, "output", "astronomaly output", "convolved_res_subset", ""
        )
    else:
        catalogue_file = os.path.join(
            cat_dir, "full_catalogue_convolved_new_cut_edge_sources_removed.csv"
        )
        output_dir = os.path.join(
            data_dir, "output", "astronomaly output", "convolved_res", ""
        )


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

catalogue = pd.read_csv(catalogue_file)


window_size = 128
image_transform_function = [
    # image_preprocessing.image_transform_inverse_sinh,
    image_preprocessing.image_transform_sigma_clipping,
    image_preprocessing.image_transform_remove_negatives,
    image_preprocessing.image_transform_scale,
]

display_transform_function = [
    # image_preprocessing.image_transform_inverse_sinh,
    # image_preprocessing.image_transform_sigma_clipping,
    image_preprocessing.image_transform_zscale,
    image_preprocessing.image_transform_scale,
]

# display_transform_function = image_transform_function

band_prefixes = []
bands_rgb = {}
plot_cmap = "hot"
feature_method = "representation"
dim_reduction = "pca"
force_rerun = True
subselect_label = False
subselected_list = False

def run_pipeline():

    image_dataset = image_reader.ImageDataset(
        directory=image_dir,
        list_of_files=list_of_files,
        window_size=window_size,
        output_dir=output_dir,
        plot_square=False,
        transform_function=image_transform_function,
        display_transform_function=display_transform_function,
        plot_cmap=plot_cmap,
        catalogue=catalogue,
        band_prefixes=band_prefixes,
        bands_rgb=bands_rgb,
        adaptive_sizing=True,
        min_window_size=0,
        display_image_size=256,
        display_interpolation="bicubic",
    )  # noqa
    
    if feature_method == "ellipse":
        pipeline_ellipse = shape_features.EllipseFitFeatures(
            percentiles=[90, 80, 70, 60, 50, 0],
            output_dir=output_dir,
            channel=0,
            force_rerun=force_rerun,
            upper_limit=200,
        )
        features_original = pipeline_ellipse.run_on_dataset(image_dataset)
        # features_original = pd.read_parquet(output_dir + 'EllipseFitFeatures_output.parquet')
        features_original = features_original.drop(
            columns=["Offset_90", "Aspect_90", "Theta_90"]
        )

    elif feature_method == "representation":
        if high_res:
            if subset:
                features_original = pd.read_parquet(
                    os.path.join(
                        data_dir,
                        "features",
                        "features_full_catalogue_high_new_cut_subset_edge_sources_removed.parquet",
                    )
                )
            else:
                features_original = pd.read_parquet(
                    os.path.join(
                        data_dir,
                        "features",
                        "features_full_catalogue_high_new_cut_edge_sources_removed.parquet",
                    )
                )

        else:
            if subset:
                features_original = pd.read_parquet(
                    os.path.join(
                        data_dir,
                        "features",
                        "features_full_catalogue_convolved_new_cut_subset_edge_sources_removed.parquet",
                    )
                )
            else:
                features_original = pd.read_parquet(
                    os.path.join(
                        data_dir,
                        "features",
                        "features_full_catalogue_convolved_new_cut_edge_sources_removed.parquet",
                    )
                )

    print("features", len(image_dataset.index), features_original.shape)
    features = features_original.copy()


######################################
    # # Used to select the features that match the 10 selected fits files only
    # select_names = list(catalogue["Unnamed: 0"])

    # features_original = features_original.query("index in @select_names").copy()
######################################

#### SCALING
    # # Now we rescale the features using the same procedure of first creating
    # # the pipeline object, then running it on the feature set
    # pipeline_scaler = scaling.FeatureScaler(force_rerun=True, output_dir=output_dir)
    # features = pipeline_scaler.run(features)
######################################

#### PCA
    if dim_reduction == "pca":
        pipeline_pca = pca.PCA_Decomposer(
            force_rerun=force_rerun, output_dir=output_dir, threshold=0.95
        )
        #   n_components=120)
        features = pipeline_pca.run(features_original)
        print("Number of components:", features.shape[1])
######################################


#### SCALING - Should this not be before PCA?
    # # Now we rescale the features using the same procedure of first creating
    # # the pipeline object, then running it on the feature set
    # pipeline_scaler = scaling.FeatureScaler(force_rerun=True, output_dir=output_dir)
    # features = pipeline_scaler.run(features)
######################################


#### iForest
    # # The actual anomaly detection is called in the same way by creating an
    # # Iforest pipeline object then running it
    # pipeline_iforest = isolation_forest.IforestAlgorithm(
    #     force_rerun=True, output_dir=output_dir
    # )
    # anomalies = pipeline_iforest.run(features)

    # # We convert the scores onto a range of 0-5
    # pipeline_score_converter = human_loop_learning.ScoreConverter(
    #     force_rerun=False, output_dir=output_dir
    # )
    # anomalies = pipeline_score_converter.run(anomalies)
######################################


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
    # pipeline_lof = lof.LOF_Algorithm(
    #     force_rerun=force_rerun, output_dir=output_dir)
    # anomalies = pipeline_lof.run(features)
    # # Sometimes LOF can fail quite dramatically
    # anomalies.loc[anomalies['score'] < -3] = -3
    anomalies = anomalies.sort_values('score', ascending=False)
######################################


    try:
        df = pd.read_csv(
            os.path.join(output_dir, "ml_scores.csv"),
            index_col=0,
            dtype={"human_label": "int"},
        )
        print("ml_scores file found")
        if subselect_label:
            df = df[df.human_label == selected_label]
        print(
            f"Out of {sum(df['human_label']!=-1)} labels, {sum(df['human_label']==5)} are interesting"
        )
        df.index = df.index.astype("str")
        # df.loc[df.index[200:], 'human_label'] = -1

        # if len(anomalies) == len(df):
        #     anomalies = pd.concat(
        #         (anomalies, df['human_label']), axis=1, join='inner')
        # This deals with the possibility of not all objects having labels

        #### LABEL ONLY THE FIRST 100 #####
        # ordered_inds = anomalies.sort_values('score', ascending=False).index
        # anomalies['human_label'] = -1
        # anomalies.loc[ordered_inds[:100], 'human_label'] = df.loc[ordered_inds[:100], 'human_label']

        #### LABEL ALL #####
        inds = df.index[np.in1d(df.index, anomalies.index)]
        anomalies.loc[inds, "human_label"] = df["human_label"]
        print(f"{(anomalies.human_label!=-1).sum()} labels added")

    except FileNotFoundError:
        pass

#### SUBSELECTED DATA
    if subselected_list:
        inds = image_dataset.index
        features = features.loc[inds]
        anomalies = anomalies.loc[inds]
        # Python doesn't sort intelligently and doesn't match linux
        nums = [int(i.split("_")[0].split("g")[-1]) for i in inds]
        sorted_inds = inds[np.argsort(nums)]
        anomalies = anomalies.loc[sorted_inds]
        anomalies.score = np.arange(len(anomalies))[::-1]
######################################


#### GP Active Learning
    pipeline_active_learning = gaussian_process.GaussianProcess(
        features, output_dir=output_dir, force_rerun=force_rerun, ei_tradeoff=3
    )
######################################


#### UMAP
    if features.shape[1] == 2:
        t_plot = features.loc[anomalies.index][:2000]
    else:
        pipeline_umap = umap_plot.UMAP_Plot(
            force_rerun=force_rerun,
            output_dir=output_dir,
            max_samples=2000,
            shuffle=True,
            n_neighbors=10,
            min_dist=0.0,
        )
        t_plot = pipeline_umap.run(features.loc[anomalies.index])
######################################


    return {
        "dataset": image_dataset,
        "features": features,
        "anomaly_scores": anomalies,
        "visualisation": t_plot,
        "active_learning": pipeline_active_learning,
    }