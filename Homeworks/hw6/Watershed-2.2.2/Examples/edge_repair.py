#!/usr/bin/env python

## edge_repair.py

'''
This script demonstrates how small gaps in the boundary of a
shape can be repaired by applying the two morphological
operators of 'dilate' and 'erode' in succession.
Ordinarily, you would want to use the same structuring
element for both operators.  However, when you use the same
structuring element, whether or not a gap in a boundary is
repaired through such a consecutive application of dilate
and erode operators depends much on the orientation of the
boundary in the vicinity of the gap and size of the gap.
So, whereas the gap is repaired for the image in
"broken_rectangle1.jpg", that is not the case for the image
in "broken_rectangle2.jpg" where the rectangular boundary is
not aligned with the horizontal and the vertical axes.
'''


from Watershed import *

shed = Watershed(
               data_image = "broken_rectangle1.jpg",
#               data_image = "broken_rectangle2.jpg",
               binary_or_gray_or_color = "binary",
       )

shed.extract_data_pixels() 
print("Displaying the original image:")
shed.display_data_image()

dilated_image = shed.dilate(5, "square")                                     
shed.erode(dilated_image, 5, "square")        

##  Comment out the above two statements and uncomment the
##  two statements below if you want to dilate and erode
##  with a circular structuring element of radius 5 pixels:
#dilated_image = shed.dilate(5, "circular")  
#shed.erode(dilated_image, 5, "circular")        

