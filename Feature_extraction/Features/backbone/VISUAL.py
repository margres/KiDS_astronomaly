#Functions
import hdbscan
import umap.umap_ as UMAP
import pacmap
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import TSNE
import time
import random
from mpl_toolkits.axes_grid1 import ImageGrid





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

def Similarity_index(labels,labels2):
    
    
    similarity = metrics.rand_score(labels2,labels)
    
    return similarity
## Visualisations

def tsine(dataset,scatter = True):
    x_embedded = TSNE(n_components = 2,
                      learning_rate = 'auto',
                     init = 'random',
                      verbose = 0,
                      )
    x_embedded = x_embedded.fit_transform(dataset)
    if scatter ==True:
        plt.style.use("seaborn")
        plt.scatter(x_embedded[:,0],x_embedded[:,1],s = 10,c = "black",alpha = 0.2)
        legend = [ "Distribution" ]
        plt.legend(legend, 
                   loc='lower right')
        plt.title("TSNE Distribution")
        plt.savefig("UMAP_shade.png")
        plt.show()

    #Save 2D represenatation

    pkl_filename = "embedded_dataset.txt"
    #Load from file
    """with open(pkl_filename, 'wb') as file:
        pickle.dump(x_embedded,file) 
    print(x_embedded)"""
    return x_embedded

def pmap(dataset,scatter = False, dim = 2):
    pc = pacmap.PaCMAP(n_components=dim,verbose = False,n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    p_embedded = pc.fit_transform(dataset)
    if scatter == True:
        plt.style.use("seaborn")
        plt.scatter(p_embedded[:,0],p_embedded[:,1],s = 10,c = "black",alpha = 0.2)
        legend = [ "Distribution" ]
        
        plt.legend(legend, 
                   loc='lower right')
        plt.title("PACMAP Distribution")
        plt.savefig("UMAP_shade.png")
        plt.show() 

    return p_embedded


def  umap(dataset,scatter = True,name = "UMAP Distribution", dim = 2, min_dist = 0.1, n_neighbors = 20,alpha = 0.2):
    start = time.time()
    reducer = UMAP.UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=min_dist, n_components=dim, n_epochs=None,
     n_neighbors= n_neighbors, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=43, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=4, unique=False, verbose=True)
    
    u_embedded = reducer.fit_transform(dataset)
        #Save 2D represenatation

    pkl_filename = "uembedded_dataset.txt"
    #Load from file
    print("Saved : ", pkl_filename)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(u_embedded,file) 
    print("UMAP reduced in ",time.time() -start)

    if scatter == True and dim ==2:
        plt.style.use("seaborn")
        plt.scatter(u_embedded[:,0],u_embedded[:,1],s = 10,c = "black",alpha = alpha)
        legend = [ "Distribution" ]
        
        plt.legend(legend, 
                   loc='lower right')
        plt.title(name)
        plt.axis('off')
        plt.savefig(name+".png")
        plt.show()
    
    return u_embedded
    
def pickle_load(file_name):

    with open(file_name,'rb') as file:
        return pickle.load(file)


    return u_embedded

def HDBSCAN(reduced_data,min_cluster_size = 50,min_samples = 30):
    
    
    hd = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    gen_min_span_tree=True, leaf_size=40,
    metric='euclidean', min_cluster_size=min_cluster_size, min_samples=min_samples)
    
    hd.fit(reduced_data)
    
    
    return hd.labels_

# Shades en embedded 2D  dataset accoriding to clustering labels

def shade(embedded_dataset,
          predictions,
          numof_class = 2,
          name ="Embedding",
          save = False,
          label = True,
          gz = False, 
          alpha = 0.2, 
          legendd =[],
          faint_class = -1, 
          limits = None, 
         hard_coloring = False):
    colours = ['black','blue','purple','yellow','red','green','orange','cyan','magenta']*100
    import matplotlib.cm as cm
    #colours = cm.rainbow(np.linspace(0, 1, numof_class))
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
            try:
            
                if  predictions[i] ==j:
                    Class.append(embedded_dataset[i])
            except:
                continue
        classes.append(Class)
    
    #Plotting the classses
        #Initialize plot
    plt.style.use("seaborn")
    plt.figure(figsize=(10,10))
    #plt.figure(facecolor="g")
    faint_alpha = alpha*2
    non_faint_alpha = alpha
    legend = []
    samples = []
    print("Plotting")
    for i in range(numof_class):
        try:
            classes[i] = np.array(classes[i])

            if i == faint_class:
                alpha = faint_alpha
            else:
                alpha = non_faint_alpha

            if i ==0:
                col = "black"
            else:
                if hard_coloring:
                    col = colours[i]
                else:
                    col = np.random.rand(1, 3)[0]


            plt.scatter(classes[i][:,0],classes[i][:,1],s = 2,c = col,alpha = alpha)
            legend.append(str(i) +": " +colours[i] + " Cluster ")
        except:
            numof_class = numof_class-1
            continue
    if label ==True:
        if gz == True:
            legend = legendd
        plt.legend(legend, loc=(1, 0),markerscale = 7, fontsize = 22)
    else:
        plt.legend([str(numof_class)+" Classes"],loc ='lower right',fontsize = 22)

        
    plt.title(name,fontsize = 22)
    if limits != None:
        plt.xlim((limits[0],limits[1]))
        plt.ylim((limits[2],limits[3]))
    plt.axis("off")
    if save:
        plt.savefig(name+"shade.png")
    plt.show()
    
    return

# Returns an array showing how many elements are in clusters i.e [x elements in cluster 0, y elements in cluster 1, z elements in cluster 2]
def groups(array,lowest_class = 0):
    groups = []
    counter = lowest_class
    while True:
        cluster = sum([int(a ==counter) for a in array])
        if counter != max(array)+1:
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
def indices_all(labels,group):
    indices = []
    #check whether a given index belong to a group
    #append the index
    #do this for all in labels
    indices = []
    for i in range(len(labels)):
        
        if labels[i] == group:
            indices.append(i)
    return indices
        
    

    return indices

# function below must return n of the most similar i
def similarity_search_old(input_index = int,data = list,number_of_neighbors = 10,return_indices = True,print_images = True):

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


def show_images(image_inputs = [],indices = [],n_by_n = True,sqrt = 8):

    
    images = []
    
    
    for index in indices:
        images.append(image_inputs[index])

        
    a = len(indices)
    b = 2*a
    
    row = 1
    if n_by_n:
        a,b = sqrt,sqrt
        row = b
        
        


    fig = plt.figure(figsize=(b,a))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(row,a),  # creates 2x2 grid of axes
                     axes_pad=0,  # pad between axes in inch.
                     )
    
    #show the images
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.set_xticks([]);ax.set_yticks([])
        plt.grid(False)
        ax.imshow((im).transpose((1,2,0)))#.astype('uint8'))

    plt.show()
    return




"""
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
        neighbors.append(random_index)
    max_before_loop = max(distances)
    max_after_loop = min(distances)

    while( max_before_loop != max_after_loop):

        for i in range(len_data):

            distance_to_i = distance(input_index,i)
            furtherst_neighbor = max(distances) 

            if distance_to_i < furtherst_neighbor and i not in neighbors:
                furthese_negihbor_index = distances.index(furtherst_neighbor)
                distances[furthese_negihbor_index]  = distance_to_i                                                         #if i is neare than the furthers neighbor of the input index, replace furthest neighbor with i
                neighbors[furthese_negihbor_index]   = i
        max_before_loop = max_after_loop
        max_after_loop = max(distances)
        a = np.sort(neighbors,axis = -1)
    

    return neighbors """
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
        neighbors.append(random_index)
    max_before_loop = max(distances)
    max_after_loop = min(distances)

    while( max_before_loop != max_after_loop):

        for i in range(len_data):

            distance_to_i = distance(input_index,i)
            furtherst_neighbor = max(distances) 

            if distance_to_i < furtherst_neighbor and i not in neighbors:
                furthese_negihbor_index = distances.index(furtherst_neighbor)
                distances[furthese_negihbor_index]  = distance_to_i      #if i is neare than the furthers neighbor of the input index, replace furthest neighbor with i
                neighbors[furthese_negihbor_index]   = i
        max_before_loop = max_after_loop
        max_after_loop = max(distances)
        a = np.sort(neighbors,axis = -1)    

    return np.sort(neighbors,axis = -1)   
def similarity_searchb(input_index = int,data = list,number_of_neighbors = 10,return_indices = True,print_images = True):

    def distance(index1,index2):
        return np.linalg.norm(np.array(data[index1])-np.array(data[index2]))
    neighbors = []
    distances = []
    len_data = len(data) 
    
    for j in range(number_of_neighbors):
        
        jthclosest_index = 0
        jthclosest_distance = distance(input_index,0) 
        for i in range(len_data):
            distance_to_input = distance(input_index,i)
            if jthclosest_distance >= distance_to_input:
 
                if i not in neighbors:
                    jthclosest_distance = distance_to_input
                    jthclosest_index = i

        neighbors.append(jthclosest_index)

    return neighbors                


def pca(data = list,n_components = 500, variance = 0.97,return_all = False):

    pca = IncrementalPCA(n_components =n_components)
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
    if return_all:
        return components
    else:
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
