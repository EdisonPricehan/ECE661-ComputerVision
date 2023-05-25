#!/usr/bin/env python

##  segment_automatic_and_use_region_growing_to_extract_blobs.py

##  This script uses region growing to extract blobs from the watershed contours,
##  which is unlike the script
##
##          segment_automatic_and_use_contours_to_extract_blobs.py
##
##  that uses the contours directly.
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
shed.extract_watershed_contours_with_random_sampling(35, 50)
shed.display_watershed_contours_in_color()

# Extract a maximum of 15 blobs and discard all blobs of size less than 80 pixels
shed.extract_segmented_blobs_using_region_growing(15,80)
shed.display_all_segmented_blobs()

