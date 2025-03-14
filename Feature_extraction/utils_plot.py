import time
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
import re, os

def run_UMAP(features, n_neighbors=15, min_dist=0.1, 
            n_components=2,  densmap=False,
             spread=1.0, metric='euclidean'):
    # Convert features to numpy array if not already
    features = np.array(features)
    
    # Apply UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, 
                        densmap=False,
             spread=1.0, metric='euclidean', random_state=42)
    embedding = reducer.fit_transform(features)

    # Save UMAP embeddings
    umap_embeddings_path = 'umap_embeddings.tsv'
    np.savetxt(umap_embeddings_path, embedding, delimiter='\t')

    return embedding

def run_TSNE(features, n_components=2, perplexity=30, learning_rate=200, metric='euclidean'):
    # Convert features to numpy array if not already
    features = np.array(features)
    
    # Apply t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, metric=metric, random_state=42)
    embedding = tsne.fit_transform(features)
    return embedding



def plot_plotly(embedding, labels, title_plot, path_plt, show_plt=True):
    
    start_time = time.time()
    size_label_0=0.5
    size_other_labels=3

    # Create a DataFrame
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'label': labels
    })
    
    # Define a color palette
    colors_hex = ["#FFF5E1", "#9BEC00", "#C80036", "#4B70F5"]
    
    # Map colors to labels
    unique_labels = df['label'].unique()
    color_map = {label: color for label, color in zip(unique_labels, colors_hex)}
    
    # Apply alpha values
    df['alpha'] = df['label'].apply(lambda x: 0.4 if x == 0 else 0.8)
    df['size'] = df['label'].apply(lambda x: size_label_0 if x == 0 else size_other_labels)
    
    # Create Plotly figure
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color=df['label'],
        color_discrete_map=color_map,
        opacity=df['alpha'],
        size=df['size'],
        labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
        title=title_plot )
    
    # Update layout
    fig.update_layout(legend_title_text='Label')
    
    # Save the plot as a PNG file
    fig.write_html(path_plt+'.html')
    if show_plt:
        fig.show()

    end_time = time.time()
    print(f"Time taken to generate the plot: {end_time - start_time:.2f} seconds")



def plot_matplot(embedding, labels, title_plot, path_plt, show_plt=True):
    
    start_time = time.time()

    
    # Create a DataFrame
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'label': labels
    })
    
    # Define a color palette
    colors_hex = ["#FFF5E1", "#9BEC00", "#C80036", "#4B70F5"]
    
    # Map colors to labels
    unique_labels = df['label'].unique()
    colors = {label: color for label, color in zip(unique_labels, colors_hex)}
    
    # Plot using Matplotlib
    plt.figure(figsize=(10, 8))
    
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        if label == 0:
            plt.scatter(label_df['x'], label_df['y'], c=colors[label], label=f'Label {label}', alpha=0.4, s=0.5)
        else:
            plt.scatter(label_df['x'], label_df['y'], c=colors[label], label=f'Label {label}', alpha=0.8)
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.title(title_plot)
    plt.savefig(path_plt+'.png', dpi=200)
    if show_plt:
        plt.show()
    
    end_time = time.time()
    print(f"Time taken to generate the plot: {end_time - start_time:.2f} seconds")

def generate_plot_path(params='', notes='', path_plt_folder='', type_embedding=''):
    if 'n_neighbors' in params:
        type_embedding = 'UMAP'
        path_plt = f"{type_embedding}_nn_{params.get('n_neighbors', 15)}_minD_{params.get('min_dist', 0.1)}_dm_{params.get('densmap', False)}"
    else:
        type_embedding = 't-SNE'
        path_plt = f"{type_embedding}_perp_{params.get('perplexity', 30)}_lr_{params.get('learning_rate', 200)}"

    path_plt = os.path.join(path_plt_folder, 'png', re.sub(r'[= ,]', '_', path_plt + notes))
    return path_plt

def display_existing_plot(plot_path):
    img = plt.imread(plot_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    

def test_UMAP_params(features, labels, params_list, 
                    bool_plot_umap=True,
                    bool_plot_plotly=False,
                    type_embedding = 'UMAP',
                    notes ='',
                    path_plt_folder = '/users/grespanm/Plots/'):

    for params in params_list:
        if type_embedding=='UMAP':
            print('Running UMAP')
            embedding= run_UMAP(features,                       
                      n_neighbors=params.get('n_neighbors', 15), 
                      min_dist=params.get('min_dist', 0.1),  
                      densmap=params.get('densmap', False),
                      spread=params.get('spread', 1.0),
                      metric=params.get('metric', 'euclidean'))
            
            title_plot = f"{type_embedding} - n_neighbors={params.get('n_neighbors', 15)}, min_dist={params.get('min_dist', 0.1)}, densmap={params.get('densmap', False)}"
            
            #path_plt = f"{type_embedding}_n_neighbors{params.get('n_neighbors',15)}_mind_{params.get('min_dist', 0.1)}_d_{params.get('densmap', False)}"
            path_plt = generate_plot_path(params=params, notes=notes, path_plt_folder=path_plt_folder, type_embedding=type_embedding) 
        else:
            print('Running t-SNE')
            embedding= run_TSNE(features,                       
                      perplexity=params.get('perplexity', 30), 
                      learning_rate=params.get('learning_rate', 'auto'))

            title_plot = f"{type_embedding} - perplexity={params.get('perplexity', 30)}, learning_rate={params.get('learning_rate', 200)}"
        
            #path_plt = f"{type_embedding}_perplexity_{params.get('perplexity', 30)}_lr_{params.get('learning_rate', 200)}"
            path_plt = generate_plot_path(params=params, notes=notes, path_plt_folder=path_plt_folder, type_embedding=type_embedding) 

        title_plot = notes + title_plot
    
        
        if bool_plot_umap:
            #path_plt = os.path.join(path_plt_folder, 'png', re.sub(r'[= ,]', '_', path_plt) )
            plot_matplot(embedding, labels, title_plot, path_plt)
            
        if bool_plot_plotly:
            #path_plt = os.path.join(path_plt_folder, 'html',re.sub(r'[= ,]', '_', path_plt) )
            plot_plotly(embedding, labels, title_plot, path_plt)



from sklearn.neighbors import NearestNeighbors
from IPython.display import display, Image

def plot_matplot_nn(embedding, labels, title_plot, path_plt, show_plt=True):
    start_time = time.time()

    # Create a DataFrame
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'label': labels
    })
    
    # Define a color palette
    colors_hex = ["#FFF5E1", "#9BEC00", "#C80036", "#4B70F5"]
    
    # Map colors to labels
    unique_labels = df['label'].unique()
    colors = {label: color for label, color in zip(unique_labels, colors_hex)}
    
    # Plot using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(df['x'], df['y'], c=df['label'].map(colors), alpha=0.8)
    
    # Add color bar which maps labels to colors
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(title_plot)
    
    # Function to find and display 10 nearest neighbors when clicking on a point
    def onclick(event):
        if event.inaxes is not None:
            x, y = event.xdata, event.ydata
            click_point = np.array([[x, y]])
            
            # Calculate distances to all points
            distances, indices = nn.kneighbors(click_point, n_neighbors=11)
            
            # Get the labels of the nearest neighbors
            neighbor_labels = labels[indices.flatten()]
            
            # Display the clicked point and its 10 nearest neighbors
            display_points(click_point, neighbor_labels)
    
    # Function to display points and their labels (for interactive use)
    def display_points(points, neighbor_labels):
        # Display clicked point
        print(f"Clicked point: {points}")
        
        # Display 10 nearest neighbors and their labels
        for i, (point, label) in enumerate(zip(points, neighbor_labels)):
            if i == 0:
                print(f"Nearest neighbor {i+1}: {point} (clicked point) - Label: {label}")
            else:
                print(f"Nearest neighbor {i+1}: {point} - Label: {label}")
                # Assuming 'all_imgs.npz' contains images corresponding to labels, adjust this part as per your data structure
                display(Image(f'all_imgs.npz'))  # Replace with your image display code
                
    # Fit nearest neighbors for quick lookup
    nn = NearestNeighbors(n_neighbors=11).fit(embedding)
    
    # Connect click event to function
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.savefig(path_plt+'.png', dpi=200)
    if show_plt:
        plt.show()
    
    end_time = time.time()
    print(f"Time taken to generate the plot: {end_time - start_time:.2f} seconds")



