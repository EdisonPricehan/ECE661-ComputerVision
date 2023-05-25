#!/usr/bin/env python

## RunImageScanner.py

##  This example script illustrates how to use the ImageScanner class in
##  the Watershed module in the non-interactive mode.

##  The non-interactive mode means that the original image will be scanned
##  one subimage at a time without the subimages being shown to the user.


import ImageScanner

image_file = "DSC00781.JPG"

scanner = ImageScanner.ImageScanner(
               data_image = image_file,
               binary_or_gray_or_color = "color",
               scanner_dump_directory = "scanner_dump",
               scanning_window_width = 800,
               scanning_window_height = 800
       )

##  NONINTERACTIVE:
scanner.scan_image_for_detections_noninteractive()


