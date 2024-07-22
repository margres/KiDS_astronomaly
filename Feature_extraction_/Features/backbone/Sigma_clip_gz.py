import cv2
import Imagefolder as imf
import numpy as np
import PIL
from IPython.display import display,clear_output



non_resized = "/idia/projects/hippo/Koketso/galaxyzoo/nonresized/galaxy_zoo_classRN"


non_resized_name = "galaxy_zoo_classRN/"

new_folder_name = "galaxy_zoo_classRN/"

num = 0


# get largest contours
for a in range(0,6000):
    if 1:
        image,image_dir = imf.folder_image_index(non_resized,a)

        alpha = imf.single_image_to_array(image_dir)
        # extract bgr image
        bgr = alpha[:,:,0:3]

        # extract alpha channel
        #alpha = alpha[:,:,3]
        """
        contours = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        """
        image_sig,contours = imf.image_transform_sigma_clipping(alpha, sigma=200, central=True)
        big_contour = max(contours, key=cv2.contourArea)


        #plt.imshow(cv2.drawContours(alpha, big_contour, -1, (0,255,0), 3))
        #clear_output(wait = True)

        # smooth contour
        peri = cv2.arcLength(big_contour, True)
        big_contour = cv2.approxPolyDP(big_contour, 0.0001 * peri, True)



        # draw white filled contour on black background
        contour_img = np.zeros_like(alpha)
        cv2.drawContours(contour_img, [big_contour], 0, (255,255,255), -1)
        #plt.show()


        # apply dilate to connect the white areas in the alpha channel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
        dilate = cv2.morphologyEx(contour_img, cv2.MORPH_DILATE, kernel)


        # make edge outline
        edge = cv2.Canny(dilate, 0, 200)

        # thicken edge
        edge = cv2.GaussianBlur(edge, (0,0), sigmaX=0.3, sigmaY=0.3)




        # make background
        result = np.full_like(alpha, (255,255,255))
        new_image = np.zeros_like(alpha)

        new_image[dilate>0] = alpha[dilate>0]
        #cv2.drawContours(new_image, big_contour, -1, (0,255,0), 3)
        
        # Save_image
        final_pil_resized = PIL.Image.fromarray(new_image)
        final_pil_resized.save(image_dir.replace(non_resized_name,new_folder_name),format = "png")
        
        #Displays
        clear_output(wait = True)
        display(num," resized")
        num+=1
        
 
    else:
        image.save(image_dir.replace(non_resized_name,new_folder_name),format = "png")
        display("saved", image_dir)
        pass
