#!/usr/bin/env python

##  hist_equalize_demo.py

##  The goal of this script is to demonstrate contrast enhancement with histogram
##  thresholding.

from Watershed import *

image_name = "fruitlets.jpg"

shed = Watershed(
               data_image = image_name,
               binary_or_gray_or_color = "color",
       )

shed.histogram_equalize()

