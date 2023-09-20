import numpy as np
import cv2
import matplotlib.pyplot as plt
import colour as cl
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from ColorCorrectionML.reference_values import referenceColor
import datetime

import dask.array as da

class ColorCorrectionML:
    
    def __init__(self, img, chart='Classic', illuminant='D50'):
        self.img = img # image to be corrected 8-bit BGR from opencv
        # do gaussian blur on image
        # self.img = cv2.GaussianBlur(img, (5,5), 0)
        self.img_rgb = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.chart = chart
        self.illuminant = illuminant
        ref = referenceColor()
        self.reference = ref.getReference(self.chart, self.illuminant)
        self.model = None
        self.method = None
        self.degree = None
        self.max_iter = None
        self.interactions = None
        self.ncomp = None
        self.wb_lim = 0.95
        self.wb_type = 0
    
    @staticmethod
    def get_patch_sizes(img1):
        
        shp = img1.shape[:2]
        
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
    
        return np.round(np.mean(Dims, axis=0)).astype(int) 
    
    def extract_color_chart(self):
        detector = cv2.mcc.CCheckerDetector_create()
        detector.process(self.img, cv2.mcc.MCC24, 1)
        checkers = detector.getListColorChecker()
        # print((detector.read))
        
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
            
            rows = int(roi.shape[:1][0])
            src = chartsRGB[:,1].copy().reshape(int(rows/3), 1, 3)
            src = src.reshape(24, 3)
            
        return src, img_draw, dims
    
    @staticmethod
    def do_white_balance(img, type_=3, s_lim=0.95):
        
        if type_ == 1:
            wb = cv2.xphoto.createLearningBasedWB()
            img = wb.balanceWhite(img)
        elif type_ == 2:
            wb = cv2.xphoto.createSimpleWB()
            img = wb.balanceWhite(img)
    
        elif type_ == 3:
            wb = cv2.xphoto.createGrayworldWB()
            wb.setSaturationThreshold(s_lim)
            img = wb.balanceWhite(img)
        
        else:
            img = img
        
        return img
      
    def show_scatter(self, src, src_pred, save_plot=False):
        
        plt.scatter(self.reference, src, c='r', s=40, label='Before Color Correction', edgecolors='k')
        plt.scatter(self.reference, src_pred, c='b', s=40, label='After Color Correction', edgecolors='k')
        
        src_0 = src.reshape(-1)
        src_1 = src_pred.reshape(-1)
        src_ref = self.reference.reshape(-1)
        
        # fit a line to the data points
        z = np.polyfit(src_ref, src_0, 1)
        p = np.poly1d(z)
        plt.plot(src_ref,p(src_ref),"r--", linewidth=0.5, label='Before Color Correction fit')
        
        z = np.polyfit(src_ref, src_1, 1)
        p = np.poly1d(z)
        plt.plot(src_ref,p(src_ref),"b--", linewidth=0.5, label='After Color Correction fit')
        
        plt.legend(loc='upper left')
        plt.xlim([-10,265])
        plt.ylim([-10,265])
        plt.xlabel('Reference intensity')
        plt.ylabel('Predicted/Measured intensity')
        if save_plot:
            # get_string for current time
            t_str = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            plt.savefig('CS_' + t_str + '.png', dpi=600)
        plt.show()
          
    @staticmethod
    def DeltaE(rgb1, rgb2):
        lab1 = cl.XYZ_to_Lab(cl.sRGB_to_XYZ(rgb1/255))
        lab2 = cl.XYZ_to_Lab(cl.sRGB_to_XYZ(rgb2/255))
        deltaE = cl.difference.delta_E(lab1, lab2, method='CIE 2000') # methods: 'CIE 1976', 'CIE 1994', 'CIE 2000',
        # 'CMC', 'CAM02-UCS', 'CAM02-SCD', 'CAM02-LCD', 'CAM16-UCS', 'CAM16-SCD', 'CAM16-LCD', 'cie2000', 'cie1994', 
        # 'cie1976', 'ITP', 'DIN99'
        
        mean_deltaE = np.mean(deltaE)
        min_deltaE = np.min(deltaE)
        max_deltaE = np.max(deltaE)
        sd_deltaE = np.std(deltaE)
        
        # print('Mean DeltaE = ', mean_deltaE)
        # print('Min DeltaE = ', min_deltaE)
        # print('Max DeltaE = ', max_deltaE)
        # print('SD DeltaE = ', sd_deltaE)
        
        return min_deltaE, max_deltaE, mean_deltaE, sd_deltaE, deltaE
      
    @staticmethod
    def show_image(img, img_, save_img=False):
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        corr_img = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1,2, figsize=(20,10))
        ax[0].imshow(img_rgb)
        ax[0].set_title('Original Image')
        ax[1].imshow(corr_img)
        ax[1].set_title('Corrected Image')
        if save_img:
            # get_string for current time
            t_str = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
            plt.savefig('Img_' + t_str + '.png')
        plt.show()
           
    def compute_correction(self, show=False, **kwargs): 
        
        self.degree = kwargs.get('degree', 1)
        self.interactions = kwargs.get('interactions_only', False)
        self.max_iter = kwargs.get('max_iter', 500)
        self.method = kwargs.get('method', 'lstsq')
        self.ncomp = kwargs.get('ncomp', 2)
        self.wb_type = kwargs.get('white_balance_mtd', 0)
        
        # print('White balance method: ', self.wb_type)
        # print('Degree: ', self.degree)
        # print('Interactions: ', self.interactions)
        # print('Method: ', self.method)
        # print('Max iterations: ', self.max_iter)
        
        src0,_,patch_size = self.extract_color_chart()
        init_DeltaE = self.DeltaE(src0, self.reference)
        print('Initial mean DeltaE: ', init_DeltaE[2])
        
        # do white balance
        self.img_rgb = self.do_white_balance(self.img_rgb, type_=self.wb_type, s_lim=self.wb_lim)
        # self.show_image(self.img, cv2.cvtColor(self.img_rgb,cv2.COLOR_RGB2BGR))
        self.img = cv2.cvtColor(self.img_rgb,cv2.COLOR_RGB2BGR)
        
        
        src,_,_ = self.extract_color_chart()
        
        if self.degree > 1:
            src_ = PolynomialFeatures(degree=self.degree, interaction_only=self.interactions).fit_transform(src)
        else:
            src_ = src
        
        shp = src_.shape  
        if self.ncomp >= shp[1]:
            self.ncomp = shp[1]-1
        
        if self.method == 'linear': # linear regression
            self.model = LinearRegression()
            self.model.fit(src_, self.reference)
            
        elif self.method == 'lstsq': # least squares regression
            self.model = np.linalg.lstsq(src_, self.reference, rcond=None)[0]
            
        elif self.method == 'pls': # partial least squares regression
            print('Number of components: ', self.ncomp)
            self.model = PLSRegression(n_components=self.ncomp, max_iter=self.max_iter)
            self.model.fit(src_, self.reference)
        
        else:
            print('Method not recognized')
            return None
        
        
        if show:
            if self.method in ['pls', 'linear']:
                src_pred = self.model.predict(src_)
                print('Model fit score: ', self.model.score(src_, self.reference))
            elif self.method == 'lstsq':
                src_pred = np.matmul(src_, self.model).astype(np.uint8)
            
            final_DeltaE = self.DeltaE(src_pred, self.reference)
            print('Final mean DeltaE: ', final_DeltaE[2])
            self.show_scatter(src, src_pred, save_plot=True)
            
        return self.model, patch_size
    
    def correct_img(self, img, show=False):
        
        img_orig = img.copy()
        
        # #do gaussian blur on image
        # img = cv2.GaussianBlur(img, (5,5), 0)
        
        # do white balance on image
        img = self.do_white_balance(img, type_=self.wb_type, s_lim=self.wb_lim)
        
        sz = img.shape[:2]
        img0 = img.copy()
        
        img_ = img0.reshape(-1,3)
        if self.degree > 1:
            img_ = PolynomialFeatures(degree=self.degree, interaction_only=self.interactions).fit_transform(img_)
        
        img_ = img_.astype(np.float32)
        
        if self.method in ['pls', 'linear']:
            img_ = self.model.predict(img_)
        elif self.method == 'lstsq':
            img_ = np.matmul(img_, self.model)
        # img_[img_>255] = 255
        # img_[img_<0] = 0
        
        img_ = self.warp_extremes(img_, thresholds=[5,250])
        
        img_ = img_.reshape(sz[0],sz[1],3).astype(np.uint8)
        
        self.show_image(img_orig, img_, save_img=True) if show else None

        return img_
    
    def process_image_dask(self, img, chunks_=10000):
        img0 = img.copy()
        sz = img.shape[:2]
        img0 = img.reshape(-1,3)
        
        if self.degree > 1:
            img0 = PolynomialFeatures(degree=self.degree, interaction_only=self.interactions).fit_transform(img0)
        
        depth = img0.shape[1]
        
        img_ = da.from_array(img0, chunks=(chunks_,depth))
        
        if self.method in ['pls', 'linear']:
            img_da = da.map_blocks(self.model.predict, img_, dtype=np.float32)
        elif self.method == 'lstsq':
            img_da = da.map_blocks(np.matmul, img_, self.model, dtype=np.float32)
        
        img_ = img_da.compute()
        # img_[img_>255] = 255
        # img_[img_<0] = 0
        img_ = self.warp_extremes(img_, thresholds=[5,250])
        
        img_ = img_.reshape(sz[0],sz[1],3).astype(np.uint8)
        
        return img_
    
    def Parallel_correct_img(self, img, chunks_=10000, show=False):
        img_orig = img.copy()
        
        # #do gaussian blur on image
        # img = cv2.GaussianBlur(img, (5,5), 0)
        
        img = self.do_white_balance(img, type_=self.wb_type, s_lim=self.wb_lim)
            
        img_ = self.process_image_dask(img, chunks_=chunks_)
        
        self.show_image(img_orig, img_, save_img=True) if show else None

        return img_
    
    def warp_extremes(self, values, thresholds=[5,250]):
        """
        Warp values that are below thresholds[0] and above thresholds[1] to fit with in the range [0,255]
        """
        
        values = values.astype(np.float32)
        overshoot = values[values>thresholds[1]]
        undershoot = values[values<thresholds[0]]
        
        if len(overshoot)>0:
            diff = overshoot-thresholds[1]
            diff[diff>20]=20
            o_values = thresholds[1]+((255-thresholds[1])*(1-(1/(np.exp(0.5*diff)))))
            values[values>thresholds[1]] = np.round(o_values)
        if len(undershoot)>0:
            diff = thresholds[0]-undershoot
            diff[diff>20]=20
            u_values = thresholds[0]-(thresholds[0]*(1-(1/(np.exp(0.5*diff)))))
            values[values<thresholds[0]] = np.round(u_values)
            
        return values.astype(np.uint8)