# ColorCorrectionML

ColorCorrectionML is a Python package for color correction of *sRGB* images using machine learning. It uses ML regression methods (linear, least sqaure, and partial least squares regression) to learn the color correction function from a training image with a color checker. The learned function is then applied to correct the color of a test image.

## Installation
```shell
pip install colorcorrectionML
```

## Usage
```python
from ColorCorrectionML import ColorCorrectionML
import cv2

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

img_corr = cc.correct_img(img, show=True)
# img_corr = cc.Parallel_correct_img(img, chunks_=50000, show=True)
```

## Example results
Scatter plot of the original image and the corrected image color values
![scatter plot](Images/scatter_plots.png)

Color correction results
![color correction results](Images/results1.png)



## Acknowledgemnts
I gratefully acknowledge [Devin Rippner](mailto:devin.rippner@usda.gov), [ORISE](https://orise.orau.gov/index.html), and the [USDA](https://www.usda.gov/) for their invaluable assistance and funding support in the development of this Repo. This project would not have been possible without their guidance and opportunities provided.


# NOTE
For our most recent, better color correction package, try the new [ColorCorrectionPackage](https://github.com/collinswakholi/ColorCorrectionPackage).


