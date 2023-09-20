from ColorCorrectionML.ColorCorrectionML import *
import cv2
import time

img = cv2.imread('Images/img2.png')

cc = ColorCorrectionML(img, chart='Classic', illuminant='D50')

method = 'pls' # 'linear', 'lstsq', 'pls' 
# for linear regression, least square regression, and partial least square regression respectively
show = True

kwargs = {
    'method': method,
    'degree': 3, # degree of polynomial
    'interactions_only': False, # only interactions terms,
    'ncomp': 10, # number of components for PLS only
    'max_iter': 5000, # max iterations for PLS only
    'white_balance_mtd': 0 # 0: no white balance, 1: learningBasedWB, 2: simpleWB, 3: grayWorldWB,
    }

M, patch_size = cc.compute_correction(
    show=show,
    **kwargs
)
    

# resize img by 2
# img = cv2.resize(img, (0,0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

tic = time.time()
img_corr = cc.correct_img(img, show=True)
# img_corr = cc.Parallel_correct_img(img, chunks_=50000, show=True)
toc = time.time()
print('Time: ', toc-tic, 's')
