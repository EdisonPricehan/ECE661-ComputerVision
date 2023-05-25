#!/usr/bin/env python

## influence_zones.py

'''
This script demonstrates the calculation of the influence zones (IZ) in a
binary blob.

For a visually interesting demonstration, you must place at least two marks
inside a blob.  Each mark is dilated into its IZ and the boundaries between
the IZs constitute the geodesic skeleton of the binary blob.

Also, as is mentioned at the top of every image that is displayed, do not
forget to close the image that is displayed in order for the script to
proceed to the next step.  Sometimes, you are asked to save the image
before you close it.

Pay close attention to the message shown at the top of each image that is
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
shed.mark_blobs('influence_zones')
shed.connected_components("marks")
shed.compute_influence_zones_for_marks()





