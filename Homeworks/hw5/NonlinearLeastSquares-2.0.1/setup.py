#!/usr/bin/env python

### setup.py

#from distutils.core import setup

from setuptools import setup, find_packages
import sys, os

setup(name='NonlinearLeastSquares',
      version='2.0.1',
      author='Avinash Kak',
      author_email='kak@purdue.edu',
      maintainer='Avinash Kak',
      maintainer_email='kak@purdue.edu',
      url='https://engineering.purdue.edu/kak/distNonlinearLeastSquares/NonlinearLeastSquares-2.0.1.html',
      download_url='https://engineering.purdue.edu/kak/distNonlinearLeastSquares/NonlinearLeastSquares-2.0.1.tar.gz', 
      description='A Python module for solving optimization problems with nonlinear least-squares',
      long_description=''' 

Consult the module API page at 

      https://engineering.purdue.edu/kak/distNonlinearLeastSquares/NonlinearLeastSquares-2.0.1.html

for all information related to this module, including
information regarding the latest changes to the code. The
page at the URL shown above lists all of the module
functionality you can invoke in your own code.  

With regard to the basic purpose of this module, it provides
a domain agnostic implementation of nonlinear least-squares
algorithms (gradient-descent and Levenberg-Marquardt) for
fitting a model to observed data.  Typically, a model
involves several parameters and each observed data element
can be expressed as a function of those parameters plus
noise.  The goal of nonlinear least-squares is to estimate
the best values for the parameters given all of the observed
data.  In order to illustrate how to use the
NonlinearLeastSquares class, the module also comes with two
additional classes: **OptimizedSurfaceFit** and
**ProjectiveCamera.**  

The job of **OptimizedSurfaceFit** is to fit the best surface to noisy
height data over an XY-plane. The model in this case would
be an analytical expression for the height surface and the
goal of nonlinear least-squares would be to estimate the
best values for the parameters in the model.  

And the job of **ProjectiveCamera** is to demonstrate how
nonlinear least-squares can be used for estimating scene
structure from camera motion.  The underlying ideas is that
you take multiple images of a scene with a camera ---
something that you can simulate with **ProjectiveCamera**.
You feed the pixels thus recorded into the
NonlinearLeastSquares class to estimate the coordinates of
the scene structure points and, when using uncalibrated
cameras, to also estimate the extrinsic parameters of the
camera at each of its positions.

Starting with Version 2.0.0, the module includes code for
the sparse-bundle-adjustment variant of the
Levenberg-Marquardt algorithm.

Typical usage syntax for invoking the domain-agnostic
NonlinearLeastSquares through your own domain-specific class
such as OptimizedSurfaceFit or ProjectiveCamera is shown below:

::

        optimizer =  NonlinearLeastSquares(                                            
                         max_iterations = 200,
                         delta_for_jacobian = 0.000001,
                         delta_for_step_size = 0.0001,
                     )
    
        surface_fitter = OptimizedSurfaceFit(                                           
                             gen_data_synthetically = True,
                             datagen_functional = "7.8*(x - 0.5)**4 + 2.2*(y - 0.5)**2",
                             data_dimensions = (16,16), 
                             how_much_noise_for_synthetic_data = 0.3, 
                             model_functional = "a*(x-b)**4 + c*(y-d)**2",
                             initial_param_values = {'a':2.0, 'b':0.4, 'c':0.8, 'd':0.4},
                             display_needed = True,
                             debug = True,
                         )

        surface_fitter.set_constructor_options_for_optimizer(optimizer)  

        result = surface_fitter.calculate_best_fitting_surface('lm') 
        or 
        result = surface_fitter.calculate_best_fitting_surface('gd')  


                                       OR


        optimizer =  NonlinearLeastSquares.NonlinearLeastSquares(
                                             max_iterations = 400,
                                             delta_for_jacobian = 0.000001,
                                             delta_for_step_size = 0.0001,
                     )
        
        camera = ProjectiveCamera.ProjectiveCamera(
                             camera_type = 'projective',
                             alpha_x = 1000.0,
                             alpha_y = 1000.0,
                             x0 = 300.0,
                             y0 = 250.0,
                 )
        camera.initialize()

        world_points = camera.make_world_points_for_triangle()
        world_points_xformed = camera.apply_transformation_to_generic_world_points(world_points, ..... )

        ##  Now move the camera to different positions and orientations and then

        result = camera.get_scene_structure_from_camera_motion('lm')

                                       OR

        result = camera.get_scene_structure_from_camera_motion_with_bundle_adjustment()

          ''',

      license='Python Software Foundation License',
      keywords='gradient descent, nonlinear least-squares, optimization, bundle adjustment',
      platforms='All platforms',
      classifiers=['Topic :: Scientific/Engineering :: Information Analysis', 'Programming Language :: Python :: 3.8'],
      packages=['NonlinearLeastSquares','OptimizedSurfaceFit','ProjectiveCamera']
)
