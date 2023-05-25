#!/usr/bin/env python

## dilate_erode.py

from Watershed import *

shed = Watershed(
#               data_image = "triangle1.jpg", 
               data_image = "triangle2.jpg", 
               binary_or_gray_or_color = "binary",
       )

shed.extract_data_pixels() 
print("Displaying the original image:")
shed.display_data_image()

dilated_image = shed.dilate(5, "square") 
shed.erode(dilated_image, 5, "square") 

##  To see the results with a circular structuring element
##  of radius 5 pixels, comment out the above two statements
##  and uncomment the following two statements:
#dilated_image = shed.dilate(5, "circular") 
#shed.erode(dilated_image, 5, "circular")        




