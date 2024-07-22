# Feature_extraction
A system for obtaining optical and radio morphology classes in astronomical image data in the form of a folder of images (Or classes in any image dataset). The representation learning relies on ImageNet weights as well fine-tunning using the self supervised learning algorithm BYOL. The representation can be extracted, clustered and visualized using sci-kit learn based clustering algorithms and manifold learning algorithms such as UMAP, PaCMAP and t-SNE.

This repo is still under construction but the project has an associated paper: https://arxiv.org/abs/2311.14157.

The Fine_tune.py allows the user to train a model that can reproduce the results in the paper by modfying the parameters "dataset" and "galaxyzooq_dir" to suit the computing enviroment. The notebook explore may be used to retrive the model trained and explore the feature space learned.

    #The code parameters:
    #augmentation probabilitiees
    g_p = 0.5
    v_p = 0.5
    h_p = 0.5
    g_r = 0.0
    r_r = 0.7
    r_c = 0.7
    valsplit = 0.1
    initial_weights = True
    continuation = False

    epochs = 50

    #Dataset arguments
    num_workers = 16
    batch_size = 32

    resize = 300
    
    dataset = "Galaxy_zoo"
    #training image floder for galaxy zoo
    galaxyzoo_dir = "/idia/projects/hippo/Koketso/galaxyzoo/galaxy_zoo"
    #galaxyzoo_dir = "/idia/projects/hippo/Koketso/galaxyzoo/galaxy_zoo_12"

    #Class validation image folder
    galaxyzooq_dir = "/idia/projects/hippo/Koketso/galaxyzoo/resized/galaxy_zoo_class_new"



    #Training arguments
    model_name = "Resnet18_Myweights"
    rep_layer = "avgpool"
    input_channel = 3
    
    patience = 5
    l_r = 1e-4
    best_loss = 5000000


