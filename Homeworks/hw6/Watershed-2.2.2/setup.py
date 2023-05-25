#!/usr/bin/env python

### setup.py

from setuptools import setup, find_packages
import sys, os

setup(name='Watershed',
      version='2.2.2',
      author='Avinash Kak',
      author_email='kak@purdue.edu',
      maintainer='Avinash Kak',
      maintainer_email='kak@purdue.edu',
      url='https://engineering.purdue.edu/kak/distWatershed/Watershed-2.2.2.html',
      download_url='https://engineering.purdue.edu/kak/distWatershed/Watershed-2.2.2.tar.gz',
      description='An image segmentation algorithm based on the watershed paradigm',
      long_description='''

Consult the module API page at

      https://engineering.purdue.edu/kak/distWatershed/Watershed-2.2.2.html

for all information related to this module, including information related
to the latest changes to the code.  The page at the URL shown above lists
all of the module functionality you can invoke in your own code.  That page
also describes how you can directly access the segmented blobs in your own
code and how you can apply a color filter to an image before its segmentation.

With regard to the basic purpose of the module, it is a Python
implementation of the watershed algorithm for image segmentation.  This implementation
allows for both fully automatic and marker-assisted segmentation of an image.

Typical usage syntax:

::

        from Watershed import *
        shed = Watershed(
                   data_image = "orchid0001.jpg",
                   binary_or_gray_or_color = "color",
                   size_for_calculations = 128,
                   sigma = 1,
                   gradient_threshold_as_fraction = 0.1,
                   level_decimation_factor = 16,
                   padding = 20,
               )
        shed.extract_data_pixels()
        shed.display_data_image()
        shed.mark_image_regions_for_gradient_mods()                     #(A)
        shed.compute_gradient_image()
        shed.modify_gradients_with_marker_minima()                      #(B)
        shed.compute_Z_level_sets_for_gradient_image()
        shed.propagate_influence_zones_from_bottom_to_top_of_Z_levels()
        shed.display_watershed()
        shed.display_watershed_in_color()
        shed.extract_watershed_contours_seperated()
        shed.display_watershed_contours_in_color()

    The statements in lines (A) and (B) are needed only for marker-assisted
    segmentation with the module.  For a fully automated implemented of the
    BLM algorithm, you would need to delete those two statements.
          ''',

      license='Python Software Foundation License',
      keywords='image processing, image segmentation, computer vision',
      platforms='All platforms',
      classifiers=['Topic :: Scientific/Engineering :: Image Recognition', 'Programming Language :: Python :: 2.7', 'Programming Language :: Python :: 3.8'],
      packages=['Watershed','ImageScanner']
)
