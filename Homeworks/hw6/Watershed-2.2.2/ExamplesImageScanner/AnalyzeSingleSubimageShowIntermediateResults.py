#!/usr/bin/env python

## AnalyzeSingleSubimageShowIntermediateResults.py

##   This script shows how you test your object detection logic on just one subimage
##   from the scanner dump directory.  As stated elsewhere, a scanner dump of
##   subimages is created by executing one of the two RunImageScanner script.

##   In comparison with the script AnalyzeSingleSubimage.py, the current script also
##   also shows you the intermediate results produced during the processing of the
##   subimage.

import ImageScanner

image_file = "DSC00781.JPG"

#subimage_file = "scanner_dump/subimage_22.jpg"
subimage_file = "scanner_dump/subimage_8.jpg"

scanner = ImageScanner.ImageScanner(
               data_image = image_file,
               subimage_filename = subimage_file,
               binary_or_gray_or_color = "color",
               scanner_dump_directory = "scanner_dump",
               scanning_window_width = 800,
               scanning_window_height = 800,
               color_filter = [(0,30),(0,255),(0,255)],
               min_brightness_level = 100,
               min_area_threshold = 1000,               
               max_area_threshold = 8000,
               max_number_of_blobs = 20,
               object_shape  = "circular",
       )


positions_of_detected_objects  =  \
              scanner.analyze_single_subimage_from_image_scan_and_show_intermediate_results()

print("\n\nObjects detected at pixel coordinates: %s" % str(positions_of_detected_objects))

