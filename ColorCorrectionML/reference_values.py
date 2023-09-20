import numpy as np
# from colour.models import RGB_COLOURSPACES, XYZ_to_RGB, RGB_to_XYZ, Lab_to_XYZ, XYZ_to_Lab


# To Do:
# 1. Add other reference color values (D55, D65, D70, D75)
# 2. Add other color charts (ColorChecker24, ColorCheckerSG, ColorCheckerDC)
# 3. Add other color spaces (CIELab, CIELuv, CIELCH, CIELCHuv, CIECAM02, etc.)

class referenceColor:
    def __init__(self):
        self.srgb_d50 = np.array([[115, 82, 68], # 1. Dark skin
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
        
        self.srgb_d65 = []
        self.srgb_d55 = []
        self.srgb_d70 = []
        self.srgb_d75 = []
        
    
    def getReference(self, chart='Classic',ref_name='D50'):
        if chart == 'Classic' or chart == 'classic':
            Reference = eval('self.srgb_'+ref_name.lower())
            try:
                return Reference
            except AttributeError:
                raise ValueError('Unknown reference color space')
        else:
            raise ValueError('Chart not supported')