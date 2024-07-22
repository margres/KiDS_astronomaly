from IPython.display import display, clear_output
from matplotlib import image
import numpy as np
import PIL
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from astropy.stats import sigma_clipped_stats
import cv2
import pandas as pd

def image_to_array(directory):
    np_file_name = directory+".npy"
    
    
    array = []                                                #Likely bad idea due to memory requirements
    i = 0
    for sub_folder_path in glob.glob(directory + '/*'):
        
        for image in glob.glob(sub_folder_path + "/*.png"):
            pic = PIL.Image.open(image).convert('L')
        
            pic = np.array(pic)
            array.append(pic)             #indexing may slow down the computation
            clear_output()
            display(i)
            i +=1
    np.save(np_file_name,array)
    return np.array(array)
        
def folder_image_index_old(directory,index):
            
    array = []                                                #Likely bad idea due to memory requirements
    i = 0
    for sub_folder_path in glob.iglob(directory+ '/*'):
        
        for image in glob.glob(sub_folder_path + '/*'):
            if i == index:
                pic = PIL.Image.open(image)
                return pic #float(np.array(pic))
            i+=1
    return None

def single_image_to_array(directory):
    pic = PIL.Image.open(directory)
    pic = np.array(pic)
    return pic
    
    

def folder_image_index(folder_directory,index):
            
    array = []                                                #Likely bad idea due to memory requirements
    i = 0
    for sub_folder_path in glob.iglob(folder_directory+ '/*'):
        
        for image in glob.glob(sub_folder_path + '/*'):
            if i == index:
                pic = PIL.Image.open(image)
                return pic, image # Return the picture and the directory of the picture
            i+=1
    return None
def folder_image_name(directory,name,file_type = '.png'):
            
    array = []                                                #Likely bad idea due to memory requirements
    i = 0
    for sub_folder_path in glob.iglob(directory+ '/*'):
        
        for image in glob.glob(sub_folder_path + '/*'):
            if image.split('\\')[-1].split('/')[-1] == name+file_type:
                pic = PIL.Image.open(image)
                pic = pic.resize((400,400))
                return pic,image #float(np.array(pic))
            i+=1
    print("Not found")
    return None
"""
def folder_image_name(directory,name,file_type = ''):
    array = []                                                #Likely bad idea due to memory requirements
    i = 0
    for sub_folder_path in glob.iglob(directory+ '/*'):
        
        for image in glob.glob(sub_folder_path + '/*'):
            if image.split('/')[-1] == name+file_type:
                pic = PIL.Image.open(image)
                pic =pic.resize((400,400))
                return pic,image #float(np.array(pic))
            i+=1
    print("Not found")
    return None
"""

def get_npmeerkat():
    return np.load('C:/Users/Koketso/Documents/Scripts/data/meerkat2d.npy')

#folder_image_index('C:/Users/Koketso/Documents/Scripts/Deep Clustering/meerkat/meerkat',23)
def classes(directory):
    if directory =="hand_alphabet":
        imdir = '/idia/projects/hippo/Koketso/Train_Alphabet'
    elif directory =="dog_breed":
        imdir = '/idia/projects/hippo/Koketso/dog_breeds'
    elif directory =="hand_test":
        imdir = '/idia/projects/hippo/Koketso/Test_Alphabet'
    elif directory[0] =='/':
        imdir = directory
    else:
        print("hand_alphabet,hand_test or dog_breeds or the directory")

        return

    labels = []
    class_num = 0
    for Class in glob.glob(imdir +'/*'):
        for image in glob.glob(Class +'/*'):
            labels.append(class_num)
        class_num +=1
    return labels

#folder_image_index('C:/Users/Koketso/Documents/Scripts/Deep Clustering/meerkat/meerkat',23)
def names(directory):
    if directory =="hand_alphabet":
        imdir = '/idia/projects/hippo/Koketso/Train_Alphabet'
    elif directory =="dog_breed":
        imdir = '/idia/projects/hippo/Koketso/dog_breeds'
    elif directory =="hand_test":
        imdir = '/idia/projects/hippo/Koketso/Test_Alphabet'
    elif directory[0] =='/':
        imdir = directory
    else:
        print("hand_alphabet,hand_test or dog_breeds or the directory")

        return

    labels = []
    class_num = 0
    for Class in glob.glob(imdir +'/*'):
        for image in glob.glob(Class +'/*'):
            labels.append(image.split('/')[-1])
        class_num +=1
    return labels



def show_images(names = [],directory = "",n_by_n = False,sqrt = 4):
    
    #accesss the images in folder
    if directory == "meerkat":
        imdir = '/idia/projects/hippo/Koketso/meerkat'
    if directory == "imagenet":
        imdir = '/idia/projects/hippo/Koketso/dog_breeds'
    if directory == "galaxy_zoo":
        imdir = '/idia/projects/hippo/Koketso/galaxy_zoo_sub'
    if directory == "hand_alphabet":
        imdir = '/idia/projects/hippo/Koketso/Train_Alphabet'
    if directory[0]=='/':
       imdir = directory
    images = []
    for name in names:
        images.append(folder_image_name(imdir,name)[0])

        
    
    a = len(names)
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
        ax.imshow(im)

    plt.show()
    return

def image_transform_sigma_clipping(img, sigma=3, central=True):
    """
    Applies sigma clipping, fits contours and

    Parameters
    ----------
    img : np.ndarray
        Input image

    Returns
    -------
    np.ndarray

    """
    if len(img.shape) > 2:
        im = img[:, :, 0]
    else:
        im = img

    im = np.nan_to_num(im)  # OpenCV can't handle NaNs

    mean, median, std = sigma_clipped_stats(im, sigma=sigma)
    thresh = std + median
    img_bin = np.zeros(im.shape, dtype=np.uint8)

    img_bin[im <= thresh] = 0
    img_bin[im > thresh] = 1

    contours, hierarchy = cv2.findContours(img_bin, 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)

    x0 = img.shape[0] // 2
    y0 = img.shape[1] // 2

    for c in contours:
        if cv2.pointPolygonTest(c, (x0, y0), False) == 1:
            break

    contour_mask = np.zeros_like(img, dtype=np.uint8)
    if len(contours) == 0:
        # This happens if there's no data in the image so we just return zeros
        return contour_mask
    cv2.drawContours(contour_mask, [c], 0, (1, 1, 1), -1)

    new_img = np.zeros_like(img)
    new_img[contour_mask == 1] = img[contour_mask == 1]

    return new_img,contours


"""  w  = int(a/2)
    h = int(a/2)

    fig, axes = plt.subplots(w,h)
    j = 0
    for row in range(w):
        for column in range(h):
            ax = axes[row, column]
            ax.set_title(f"Image ({row}, {column})")
            ax.axis('off')
            ax.imshow(image[j])
            j +=1


    plt.show() """



def score(labels = "/idia/projects/hippo/Koketso/meerkat/labels.csv", image_name = ""):
    
    labels_file = pd.read_csv(labels)
    
    labels_file.set_index("Unnamed: 0",
            inplace = True)
    
    return labels_file.loc[image_name].user_score





def tag(labels = "/idia/projects/hippo/Koketso/meerkat/labels.csv", image_name = ""):
    
    labels_file = pd.read_csv(labels)
    
    labels_file.set_index("Unnamed: 0",
            inplace = True)
    
    return labels_file.loc[image_name].tag