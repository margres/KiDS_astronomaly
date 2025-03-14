import cv2
import Imagefolder as imf
import numpy as np
import PIL
from IPython.display import display,clear_output




non_resized = "/idia/projects/hippo/Koketso/galaxyzoo/galaxy_zoo"



non_resized_name =""

new_folder_name = ""



#This function transforms a rectangle into the smallest square with the rectangle at center

def square(x,y,h,w):
    l = max([h,w])
    x_new = x
    y_new = y
    if h>w:
        x_new= x_new -abs(h-w)/2
    else:
        y_new = y_new -abs(h-w)/2
    
    return int(x_new),int(y_new),int(l)

#find a square slightly larger than the smallest square (one above), by a factor (scale)

def square_scale(x,y,h,w,scale = 2):
    l = max([h*scale,w*scale])
    x_new = x - abs(w-w*scale)/2
    y_new = y - abs(h-h*scale)/2
    h = h*scale
    w = w*scale
    if h>w:
        x_new= x_new -abs(h-w)/2
    else:
        y_new = y_new -abs(h-w)/2
    
    return int(x_new),int(y_new),int(l)





num = 0

passed = []

for a in range(0,170000):
    try:
        #obtain image from imagefolder directory
        pil_image,image_dir = imf.folder_image_index(non_resized,a)
        
        #check if already resized
        if pil_image.size[0] !=400:

            #convert to array

            image = imf.single_image_to_array(image_dir)

            #apply sigma cliping
            image_sig,contour = imf.image_transform_sigma_clipping(image, sigma=4, central=True)


            #obtaining contours
            gray = cv2.cvtColor(image_sig, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 30, 200)


            contours_clipped, hierarchy = cv2.findContours(edged, 
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


            #Stack all the contours, largest contour does not cointain objects of interest
            contours = np.vstack(contours_clipped)        

            c = np.vstack(contours_clipped)

            #Find the bounding rectangle around a galaxy

            x1,y1,w,h = cv2.boundingRect(c)

            #Transform rectangle into a sqaure
            x,y,l = square_scale(x1,y1,h,w)

            #Cropp image according to the square and resize
            new_image = cv2.resize(image[x:x+l,y:y+l],(400,400))

            final_pil_resized = PIL.Image.fromarray(new_image)

            final_pil_resized.save(image_dir.replace(non_resized_name,new_folder_name),format = "png")
            #Displays
            clear_output(wait = False)
            display(num," resized")
            num+=1
            #displays
            #plt.imshow(cv2.drawContours(image_sig, contours_clipped, -1, (0, 255, 0), 3))
        else:
            print("Already resized")
        
            
    except:
        pil_image.save(image_dir.replace(non_resized_name,new_folder_name),format = "png")
        display("saved", image_dir)
        pass
