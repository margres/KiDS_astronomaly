#Functions
import hdbscan
import umap.umap_ as UMAP
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import IncrementalPCA
import time
import random




def reduce(dataset,n_components):
    start = time.time()
    pca = IncrementalPCA(n_components=n_components,
              copy=False,
              whiten=False,
              #svd_solver='arpack',   only works for the orginal pca
              batch_size = 50,
              #tol=0.0, 
              #iterated_power='auto',
             )
    pca = pca.fit(dataset)
    print(time.time() - start)
    return pca

def Similarity_index(reduced_data,labels):
    
    
    hd = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=False,
    gen_min_span_tree=True, leaf_size=40,
    metric='euclidean', min_cluster_size=100, min_samples=70)
    
    hd.fit(reduced_data)
    
    
    similarity = metrics.rand_score(hd.labels_,labels)
    
    return similarity

def  umap(dataset,scatter = True):
    start = time.time()
    reducer = UMAP.UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=True)
    
    u_embedded = reducer.fit_transform(dataset)
        #Save 2D represenatation

    pkl_filename = "uembedded_dataset.txt"
    #Load from file
    print("Saved : ", pkl_filename)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(u_embedded,file) 
    print("UMAP reduced in ",time.time() -start)

    if scatter == True:
        plt.style.use("seaborn")
        plt.scatter(u_embedded[:,0],u_embedded[:,1],s = 10,c = "black",alpha = 0.2)
        legend = [ "Distribution" ]
        
        plt.legend(legend, 
                   loc='lower right')
        plt.title("UMAP_Data")
        plt.savefig("UMAP_shade.png")
        plt.show()
    
    return u_embedded
def pickle_load(file_name):

    with open(file_name,'rb') as file:
        return pickle.load(file)


    return u_embedded

def HDBSCAN(reduced_data,min_cluster_size = 50,min_samples = 30):
    
    
    hd = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=False,
    gen_min_span_tree=True, leaf_size=40,
    metric='euclidean', min_cluster_size=min_cluster_size, min_samples=min_samples)
    
    hd.fit(reduced_data)
    
    
    return hd.labels_

# Shades en embedded 2D  dataset accoriding to clustering labels

def shade(embedded_dataset,predictions,numof_class = 2,name ="Embdding",save = False):
    colours = ['black','red','green','purple','blue','yellow','orange','cyan','magenta']*100
    #First we split the dataset according predicted classes
    classes = []


    #This section creates fake predictions for when you want to plat the data space plainly and illumintate points
    pseudo_predictions  = []
    if len(embedded_dataset)!= len(predictions):
        pseudo_predictions = [0]*len(embedded_dataset)
        for index in predictions:
            pseudo_predictions[index] =1
        predictions = pseudo_predictions



    for j in range(numof_class):
        #print("Isolating Class ")
        Class =[]
        for i in range(len(predictions)):
            if  predictions[i] ==j:
                Class.append(embedded_dataset[i])
        classes.append(Class)
    #Plotting the classses
        #Initialize plot
    plt.style.use("seaborn")
    plt.figure(figsize=(7,5))
    #plt.figure(facecolor="g")
    
    legend = []
    samples = []
    print("Plotting")
    for i in range(numof_class):
        classes[i] = np.array(classes[i])
        plt.scatter(classes[i][:,0],classes[i][:,1],s = 10,c = colours[i],alpha = 0.2)
        legend.append(str(i) +": " +colours[i] + " Cluster ")
        
    plt.legend(legend, 
               loc='lower right')
    plt.title(name)
    if save:
        plt.savefig(name+"shade.png")
    plt.show()
    
    return

# Returns an array showing how many elements are in clusters i.e [x elements in cluster 0, y elements in cluster 1, z elements in cluster 2]
def groups(array,lowest_class):
    groups = []
    counter = lowest_class
    while True:
        cluster = (array == counter).sum()
        if cluster != 0:
            groups.append(cluster)
            counter += 1
        else:
            break;
        
    return groups

# Returns n indices belonging to the group: group

def indices(labels,group, n):
    indices = []
    j = 0
    while j < n:
        i = random.randint(0,len(labels)-1)

        if i not in indices:
            if labels[i] == group:
                indices.append(i)
                j +=1

    return indices

# function below must return n of the most similar i
def similarity_search(input_index = int,data = list,number_of_neighbors = 10,return_indices = True,print_images = True):

    def distance(index1,index2):
        return np.linalg.norm(np.array(data[index1])-np.array(data[index2]))
 

    neighbors = []
    distances = []
    len_data = len(data)
    #Initialize random neighbors

    for j in range(number_of_neighbors):
        random_index = random.randint(0,len_data)
        distances.append(distance(input_index,random_index))
        neighbors.append(data[random_index])

    for i in range(len_data):

        distance_to_i = distance(input_index,i)
        furtherst_neighbor = max(distances) 

        if distance_to_i < furtherst_neighbor and i !=input_index:
            furthese_negihbor_index = distances.index(furtherst_neighbor)
            distances[furthese_negihbor_index]  = distance_to_i                                                         #if i is neare than the furthers neighbor of the input index, replace furthest neighbor with i
            neighbors[furthese_negihbor_index]   = i 


    return neighbors



def pca(data = list,n_components = 500, variance = 0.97):

    pca = IncrementalPCA(n_components =500)
    pca.fit(data)
    components  = pca.transform(data)
    var = pca.explained_variance_ratio_

    prefered_variance = 0;

    #Here I decide the number of components to keep, using the variance

    i = 0
    while prefered_variance < variance and i <n_components:
        prefered_variance += var[i]
        i +=1

    print("Variance to keep : ",prefered_variance," number of components : ",i )

    return components[:,0:i]













"""
I trained a CONV with 10 ekmbedded layers, achieved a loss of 180

Applied HDBSCAN AND UMAP on this

results comprable to HDBSCAN + PCA

truncation error means use forward slashes

KMeans and fit predict use high memory





"""


"""
Trained the CONV over more n_epochs
applied hdbscan + umap

struggled extracting the density peaks

"""

"""
I extracted features from a resnet50swav model

I need to automate the 

"""

"""
import umap.umap_
    then use umap.umap_.UMAP()



"""