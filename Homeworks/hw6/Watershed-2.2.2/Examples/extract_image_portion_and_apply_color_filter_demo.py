#!/usr/bin/env python

##  extract_image_portion_and_apply_color_filter_demo.py

##  The goal of this script is to demonstrate the following:
##
##     1) How to choose just a portion of the image for watershed segmentation; and
##
##     2) How to apply a color filter to the image.


##  SPECIFYING IMAGE PORTION FOR SEGMENTATION:
##
##    The module gives you two different methods for specifying a portion of the image
##    for watershed segmentation:
##
##       i) You can click at a point and then drag the mouse to define a rectangular
##          portion of the image;
##
##       2) You can specify any polygonal shaped area by clicking the mouse at the
##          vertices of the polygon you have in mind.
##
##    The first of these is provided by the method:
##
##            extract_image_region_interactively_by_dragging_mouse()
##
##    and the second by
##
##            extract_image_region_interactively_through_mouse_clicks()


##  APPLYING A COLOR FILTER TO THE IMAGE:
##
##    The module gives you two different methods for applying a color filter: You can
##    apply a filter to the HSV representation of the color, or its RGB
##    representation.  The methods for applying these color filters are:
##  
##             apply_color_filter_hsv()
##
##             apply_color_filter_rgb()
##  
##    For both these methods, you have two choices for specifying a filter: as a
##    triple of scalars or as a triple of pairs.  The filter must be specified either
##    as a triple of scalars, as in "(1,0.5,0.8)", or a triple of pairs, as in
##    "((0,35),(0,255),(0,255))".  When specified as a triple of scalars, each color
##    component is multiplied by the corresponding scalar.  And when specified as a
##    triple of pairs, each color component must be within the range dictated by the
##    corresponding pair. With HSV, you are more likely to use the second form of the
##    filter.  For example, the filter "((0,35),(0,255),(0,255))" works well if you
##    want to let through only the red and reddish pixels.  And, with RGB, you are
##    more likely to use the first form of the filter.  For example, if you wanted to
##    apply the watershed segmentation to just the R component of a color image, your
##    filter would be "(1,0,0)".

from Watershed import *

image_name = "fruitlets.jpg"

shed = Watershed(
               data_image = image_name,
               binary_or_gray_or_color = "color",
               size_for_calculations = 128,
               sigma = 1,
#               gradient_threshold_as_fraction = 0.05,
               gradient_threshold_as_fraction = 0.1,
               level_decimation_factor = 16,
               padding = 50,
#               color_filter = [1,0,0],
               color_filter = [(0,35),(0,255),(0,255)],
       )


#shed.extract_image_region_interactively_through_mouse_clicks()

shed.extract_image_region_interactively_by_dragging_mouse()

shed.apply_color_filter_hsv()

#shed.apply_color_filter_rgb()

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

shed.display_watershed_contours_in_color()
