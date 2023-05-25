#!/usr/bin/env python

## LoG.py

'''
This script calculates the Laplacian-of-Gaussian (LoG) of a grayscale or
color image.  A color image is converted into a grayscale image before the
application of the LoG operator.  The LoG is calculated by taking a
difference of two Gaussian-smoothed images with two different values of
sigma.  The first Gaussian smoothed image is calculated with the sigma as
set in the constructor and the second with a sigma that 20% larger.
'''


from Watershed import *

shed = Watershed(
               data_image = "orchid0001.jpg",
               binary_or_gray_or_color = "color",
               size_for_calculations = 256,
               sigma = 1,
       )

shed.extract_data_pixels() 
print("Displaying the original image:")
shed.display_data_image()
shed.compute_LoG_image()

