#!/usr/bin/env python

##  distance_mapping.py

'''
This script first shows you a binary image containing several blobs.  You
are supposed to click inside one of the blobs to place a mark therein.
Subsequently, the script prints out in the terminal window in which you
execute the script a distance map with respect to the mark.

The distance map is shown only for the first mark you place inside a blob.

So there is no point to placing multiple marks in one or multiple blobs.

Unless you are using a wide terminal window, you might want to click inside
one of the smaller blobs in the binary image that shows up.

Also, as is mentioned at the top of every image that is displayed, do not
forget to close the image that is displayed in order for the script to
proceed to the next step.  Sometimes, you are asked to save the image
before you close it.

Pay close attention to the message at the top of each image that is
displayed.
'''



from Watershed import *

shed = Watershed(
               data_image = "artpic3.jpg",
               binary_or_gray_or_color = "binary",
       )

shed.extract_data_pixels() 
print("Displaying the original image:")
shed.display_data_image()
shed.connected_components("data")
shed.mark_blobs('distance_mapping')
shed.connected_components("marks")
print("\n\nStart dilating marked blob:\n\n")
shed.dilate_mark_in_its_blob(1)



