#!/usr/bin/env python

##  AnalyzeScannerDump.py

##  This script analyzes ALL of the subimages a scanner dump for the objects
##  that you would like to detect.
##
##  For each subimage, the script shows you the intermediate results that would
##  ordinarily consist of the output obtained by applying a color filter, the
##  binarized version of this images, the result of the connected components
##  algorithm applied to the binarized images, and an image of the final
##  blobs retained after simply shape heuristics are applied to each blob.



import ImageScanner

image_file = "DSC00781.JPG"

scanner = ImageScanner.ImageScanner(
               data_image = image_file,
               binary_or_gray_or_color = "color",
               scanner_dump_directory = "scanner_dump",
               scanning_window_width = 800,
               scanning_window_height = 800,
               color_filter = [(0,30),(0,255),(0,255)],
               min_brightness_level = 100,
#               min_area_threshold = 1000,               
               min_area_threshold = 2000,               
#               max_area_threshold = 8000,
               max_area_threshold = 16000,
               max_number_of_blobs = 20,
               object_shape  = "circular",
       )


positions_of_detected_objects = scanner.analyze_scanner_dump()

print("\n\nObjects detected at pixel coordinates: %s" % str(positions_of_detected_objects))

print("\nNumber of objects detectred: %d" % len(positions_of_detected_objects))

