
# import dependencies
import numpy as np
from color_correction import ColorCorrection
import cv2
import os
import glob
import tqdm

# define paths
image_path = 'test/images_color'
save_path = 'Corrected_images'
if not os.path.exists(save_path): # check if save path exists else create it
    os.makedirs(save_path)

# get all images in folder and subfolders
fmt = '.png' # format of images to correct
img_paths = glob.glob(image_path + '/**/*'+fmt, recursive=True)
n_img = len(img_paths)
print('Number of images found: ', n_img)

show = False # show figures from color correction
n_components = 15 # number of components for PLSR
degree = 3 # degree of polynomial
resize_f = 1 # reduce image size to speed up processing, may reduce image quality

i=0
for img_path in tqdm.tqdm(img_paths):
    # img_path = img_paths[i]
    # load image
    img = cv2.imread(img_path)
    img = img.astype(np.uint8)
    cc = ColorCorrection(img)
    img_corr, _, _ = cc.run_correction(resize_f=resize_f, 
                                 n_components=n_components, 
                                 degree=degree, 
                                 show=show)
    
    # save image in save_path with same file structure as image_path
    file_name = os.path.basename(img_path)
    file_path = os.path.dirname(img_path)
    file_path = file_path.replace(image_path, save_path)
    # check if path exists else create it
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    # print image type
    print('Image type: ', img_corr.dtype)
    cv2.imwrite(os.path.join(file_path, file_name), img_corr)
    
    # print progress bar in terminal tqdm with percentage
    print('Progress: ', round(i/n_img*100, 2), '%')
    i+=1
    
print('Done')