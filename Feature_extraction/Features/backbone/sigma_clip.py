import sys
sys.path.insert(1, '/idia/projects/hippo/Koketso/galaxyzoo/Features/backbone')

import pandas as pd
import backbone.Imagefolder as imf
import importlib
from IPython.display import display,clear_output
import shutil
import numpy as np
from astropy.stats import sigma_clipped_stats
import cv2
import matplotlib.pyplot as plt
import PIL

non_sigma_clipped_imagefolder = "/idia/projects/hippo/Koketso/galaxyzoo/galaxy_zoo_biclass"
sigma_clipped_imagefolder  = "/idia/projects/hippo/Koketso/galaxyzoo/galaxy_zoo_biclass_clipped"

non_sigma_clipped_imagefolder_name = "galaxy_zoo_class"
sigma_clipped_imagefolder_name = "galaxy_zoo_biclass_clipped"
#Define the sigma_cliping function
"""
Loop over all the images => sigma_clip => save in a different folder
"""


Images_not_clipped = 0
Images_clipped = 0
num_of_images = 17333

for a in range(17333):#num_of_images):
    clear_output(wait = True)
    try:
        
    	#obtain the image by index
        
        image, Dir  = imf.folder_image_index(non_sigma_clipped_imagefolder,a)


        #The sigma_clip function takes arrays, so convert to arrays

        image = imf.single_image_to_array(Dir)

        #apply sigma_clipping
        sigma_clipped = imf.image_transform_sigma_clipping(image, sigma=4, central=True)

        #convert back to PIL image and save in the new folder
        pil_image = PIL.Image.fromarray(sigma_clipped)

        pil_image.save(Dir.replace(non_sigma_clipped_imagefolder_name,sigma_clipped_imagefolder_name),format = "png")
        
        Images_clipped +=1
    except:
    	Images_not_clipped +=1
   
    print("Images_not_clipped ",Images_not_clipped)
    print("Images_clipped ",Images_clipped)

