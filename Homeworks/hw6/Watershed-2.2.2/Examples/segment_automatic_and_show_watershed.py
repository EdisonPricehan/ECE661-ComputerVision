#!/usr/bin/env python

##  segment_automatic_and_show_watershed.py

##  To see wathershed segmentation of an image that does not require any user
##  interaction, execute this script.
##  
##  As you will notice that, when no user interaction is involved, the Wathershed
##  algorithm over-segments the image.  For an example of the segmentation produced by
##  this script, for the following image
##  
##      orchid0001.jpg
##  
##  of an orchid, the script produced the segmentation shown in
##  
##      automatic_output_segmentation_for_orchid.jpg
##  

from Watershed import *

image_name = "orchid0001.jpg"

shed = Watershed(
               data_image = image_name,
               binary_or_gray_or_color = "color",
               size_for_calculations = 128,
               sigma = 1,
               gradient_threshold_as_fraction = 0.1,
               level_decimation_factor = 16,
               padding = 50,
       )
shed.extract_data_pixels() 
print("Displaying the original image:")
shed.display_data_image()
print("Calculating the gradient image")
shed.compute_gradient_image()
print("Computing Z level sets for the gradient image")
shed.compute_Z_level_sets_for_gradient_image()
print("Propagating influences:")
shed.propagate_influence_zones_from_bottom_to_top_of_Z_levels()
shed.display_watershed()
shed.display_watershed_in_color()

#   Extract up to 30 blob contours and, at the same time, only accept 
#   contours whose length is GREATER THAN 20 pixels:
contours = shed.extract_watershed_contours_with_random_sampling(30, 50)

#   You can also call the following method for extracting the watershed
#   contours
#contours = shed.extract_watershed_contours_separated()

#   Uncomment the following lines if you want to print out the contours:
#for contour in contours:
#    print "\n\ncontour: ", contour
#    print "contour length: ", len(contour)

shed.display_watershed_contours_in_color()
