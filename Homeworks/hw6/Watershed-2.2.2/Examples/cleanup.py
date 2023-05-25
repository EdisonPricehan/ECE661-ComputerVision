#!/usr/bin/env python

## cleanup.py

import os, glob

##  Ordinarily, the destructor of the Watershed module
##  should automatically cleanup the intermediate image
##  files created by some of the scripts in this directory.
##  However, should the module abort without executing its
##  destructor code, you can cleanup the directory yourself
##  by running this script.

for filename in glob.glob('_component_image_for_mark*'): os.remove(filename)
for filename in glob.glob('_gradient__*.jpg'): os.remove(filename)
for filename in glob.glob('_binarized_valley_created*'): os.remove(filename)
for filename in glob.glob('_region__*.jpg'): os.remove(filename)
for filename in glob.glob('_marker_modified_gradient*'): os.remove(filename)
for filename in glob.glob('_mark_for_*.jpg'): os.remove(filename)
for filename in glob.glob('_LoG__*.jpg'): os.remove(filename)






