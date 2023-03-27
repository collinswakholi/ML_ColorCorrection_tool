"""
Color correction class

@author: - Collins W.
Adopted from https://blog.francium.tech/using-machine-learning-for-color-calibration-with-a-color-checker-d9f0895eafdb
"""

# Importing the libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Reference colors in sRGB color space (D50 illuminant) for the xrite classic color checker 
# Values extracted from https://babelcolor.com/index_htm_files/RGB%20Coordinates%20of%20the%20Macbeth%20ColorChecker.pdf, 
# https://www.xrite.com/service-support/new_color_specifications_for_colorchecker_sg_and_classic_charts
# and https://xritephoto.com/documents/literature/en/ColorData-1p_EN.pdf

ref_babel_srgb_d50 = np.array([[115, 82, 68], # 1. Dark skin
                               [195, 149, 128], # 2. Light skin
                               [93, 123, 157], # 3. Blue sky
                               [91, 108, 65],  # 4. Foliage
                               [130, 129, 175], # 5. Blue flower
                               [99, 191, 171], # 6. Bluish green
                               [220, 123, 46], # 7. Orange
                               [72, 92, 168], # 8. Purplish blue 
                               [194, 84, 97], # 9. Moderate red 
                               [91, 59, 104], # 10. Purple
                               [161, 189, 62], # 11. Yellow green 
                               [229, 161, 40], # 12. Orange yellow
                               [42, 63, 147],  # 13. Blue 
                               [72, 149, 72], # 14. Green
                               [175, 50, 57], # 15. Red
                               [238, 200, 22], # 16. Yellow
                               [188, 84, 150], # 17. Magenta
                               [0, 137, 166],  # 18. Cyan
                               [245, 245, 240], # 19. White 9.5
                               [201, 202, 201], # 20. Neutral 8
                               [161, 162, 162], # 21. Neutral 6.5
                               [120, 121, 121], # 22. Neutral 5
                               [83, 85, 85], # 23. Neutral 3.5
                               [50, 50, 51] # 24. Black 2
])


ref_babel_srgb_d50_2 = np.array([[115, 82, 68],
                               [194, 150, 130],
                               [98, 122, 157],
                               [87, 108, 67],
                               [133, 128, 177],
                               [103, 189, 170],
                               [214, 126, 44],
                               [80, 91, 166],
                               [193, 90, 99],
                               [94, 60, 108],
                               [157, 188, 64],
                               [224, 163, 46],
                               [56, 61, 150],
                               [70, 148, 73],
                               [175, 54, 60],
                               [231, 199, 31],
                               [187, 86, 149],
                               [8, 133, 161],
                               [243, 243, 242],
                               [200, 200, 200],
                               [160, 160, 160],
                               [122, 122, 121],
                               [85, 85, 85],
                               [52, 52, 52]
])


# Class for color correction
class ColorCorrection:
    def __init__(self, img):
        self.img = img
        self.ref = ref_babel_srgb_d50_2
        self.type = None
        # check if image is open cv image or PIL image or Scipy image
        if type(self.img) == np.ndarray:
            self.type = 'cv2'
            self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            self.type = 'other'
            self.img_rgb = np.array(self.img)
            self.img = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR)
    
    
    def get_patch_sizes(self, img1, show=False):
        
        shp = img1.shape[:2]
        if show:
            img_fin = img1.copy()
        
        v = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)[:,:,2]
        v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
        ret, thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.01*shp[0]*shp[1]
        # draw rectangles on image
        Dims = []
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            
            ratio = w/h
            area = w*h
            if ratio<0.9 or ratio>1.20 or area<min_area: # get rid of rectangles that are too small or too tall/wide
                continue
            # Append dimensions to list
            Dims.append((w,h))
            if show:
                cv2.rectangle(img_fin,(x,y),(x+w,y+h),(0,255,0),2)

        if show:
            cv2.imshow('image', img_fin)
            cv2.waitKey(0)
        
        return np.round(np.mean(Dims, axis=0)).astype(int)                                                                 

   
    def extract_color_chart(self):
        detector = cv2.mcc.CCheckerDetector_create()
        detector.process(self.img, cv2.mcc.MCC24, 1)
        checkers = detector.getListColorChecker()
        print((detector.read))
        
        for checker in checkers:
            cdraw = cv2.mcc.CCheckerDraw_create(checker)
            img_draw = self.img.copy()
            cdraw.draw(img_draw)
            
            chartsRGB = checker.getChartsRGB()
            width, height = chartsRGB.shape[:2]
            roi = chartsRGB[0:width, 1]
            
            box_pts = checker.getBox()
            x1 = int(min(box_pts[:,0]))
            x2 = int(max(box_pts[:,0]))
            y1 = int(min(box_pts[:,1]))
            y2 = int(max(box_pts[:,1]))
            
            # crop image to bounding box
            img_roi = self.img[y1:y2, x1:x2]
            dims = self.get_patch_sizes(img_roi)
            
            # write image to file
            cv2.imwrite('chart.png', img_roi)
            
            # print(roi)
            rows = int(roi.shape[:1][0])
            src = chartsRGB[:,1].copy().reshape(int(rows/3), 1, 3)
            src = src.reshape(24, 3)
            
        return src, img_draw, dims
    
  
    def do_white_balance(self, s_lim=0.99):
        wb = cv2.xphoto.createGrayworldWB()
        # createLearningBasedWB() createSimpleWB() createGrayworldWB
        try:
            wb.setSaturationThreshold(s_lim)
        except:
            pass
        img = wb.balanceWhite(self.img_rgb)
        
        return img
    
    
    def run_correction(self,  resize_f = 1, n_components=15, degree=3, show=False):
        # linearize image
        src0, _,patch_dims = self.extract_color_chart()
        
        self.img_rgb = self.do_white_balance(0.995)
        self.img = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR)
        src, img_draw,_ = self.extract_color_chart()
        # img = self.img_rgb
            
        model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                            ('pls', PLSRegression(n_components=n_components, max_iter=5000))])
        
        model.fit(src, self.ref)
        print('Model score: ', model.score(src, self.ref))
        
        if resize_f != 1:
            img_size = self.img.shape[:2]
            # Resize image
            img_rgb = cv2.resize(self.img_rgb, (0,0), fx=resize_f, fy=resize_f, interpolation=cv2.INTER_CUBIC)
        else:
            img_rgb = self.img_rgb
        
        corr_img = img_rgb.copy()
        corr_img = corr_img.astype(np.uint32)
        
        for im in corr_img:
            im[:] = model.predict(im[:])
        
        # alternative to for loop using numpy vectorization
        #corr_img = model.predict(corr_img.reshape(-1, 3)).reshape(corr_img.shape)
            
        corr_img[corr_img>=255] = img_rgb[corr_img>=255]
        corr_img[corr_img<0] = img_rgb[corr_img<0]
        corr_img = corr_img.astype(np.uint8)
        
        # apply gamma to image
        # gamma = 1.4
        # corr_img = (np.power(corr_img.astype(np.uint8)/255, gamma)*255).astype(np.uint8)
        
        # preview original and color corrected image using matplotlib
        if show:
            src_pred = model.predict(src.astype(np.uint8))
            # scatter plot of predicted and actual values for before color correction and after color correction
            plt.scatter(self.ref, src0, c='r', s=40, label='Before Color Correction', edgecolors='k')
            plt.scatter(self.ref, src_pred, c='b', s=40, label='After Color Correction', edgecolors='k')
            
            # reshape to 1D array
            src1 = src0.reshape(-1)
            src2 = src_pred.reshape(-1)
            src_ref = self.ref.reshape(-1)
            
            # fit a line to the data points
            z = np.polyfit(src_ref, src1, 1)
            p = np.poly1d(z)
            plt.plot(src_ref, p(src_ref), 'r-', linewidth=0.5)
            
            z = np.polyfit(src_ref, src2, 1)
            p = np.poly1d(z)
            plt.plot(src_ref, p(src_ref), 'b-', linewidth=0.5)
            
            plt.legend(loc='upper left')
            plt.xlim([0, 255])
            plt.ylim([0, 255])
            plt.xlabel('Reference intensity')
            plt.ylabel('Predicted/Measured intensity')
            plt.show()
            
            
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(self.img_rgb)
            ax[0].set_title('Original Image')
            ax[1].imshow(corr_img)
            ax[1].set_title('Color Corrected Image')
            plt.show()
            
        if resize_f != 1:
            # Resize image to original size
            corr_img = cv2.resize(corr_img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
            
        # if input image is open cv image format, convert back to BGR
        if self.type == 'cv2':
            corr_img = cv2.cvtColor(corr_img, cv2.COLOR_RGB2BGR)
            img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
                           
        return corr_img, img_draw, patch_dims
    