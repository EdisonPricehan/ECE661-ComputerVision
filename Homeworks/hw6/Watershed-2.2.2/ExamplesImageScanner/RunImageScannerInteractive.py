#!/usr/bin/env python

## RunImageScannerInteractive.py

##  This example script illustrates how to use the ImageScanner class
##  in the Watershed module.

##  This script scans the image in an interactive mode.  What that means is
##  that each subimage is shown to the user before moving on to the next
##  subimage.  This makes it easier for the user to "understand" the image
##  at the level of very small objects in the original large image.

import pkg_resources
pkg_resources.require("Watershed>=2.2.0")
import ImageScanner

image_file = "DSC00781.JPG"

scanner = ImageScanner.ImageScanner(
               data_image = image_file,
               binary_or_gray_or_color = "color",
               scanner_dump_directory = "scanner_dump",
               scanning_window_width = 800,
               scanning_window_height = 800
       )

##  INTERACTIVE:
scanner.scan_image_for_detections_interactive()    

