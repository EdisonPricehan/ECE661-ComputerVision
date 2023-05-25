#!/usr/bin/env python
#-*- coding: latin-1 -*-

##  bundle_adjust_sfm_with_uncalibrated_cameras_translations_only.py

##    This script demonstrates how to use the sparse-bundle-adjustment capabilities
##    of the NonlinearLeastSquares module for solving problems that require estimating
##    both the scene structure and the camera parameters for the case when the data
##    is collected with uncalibrated cameras.

##    For any nonlinear least-squares method, you are required to supply starting value 
##    for the parameters you are estimating.  Therefore, it is interesting to study
##    at what point an algorithm starts getting trapped in a local minimum as you 
##    move the starting value farther and farther away from their true optimum values.
##    You can perform those kinds of studies with this script by changing the values
##    of the variables 'cam_pam_noise_factor' and 'structure_noise_factor'.

##    Note that this script should produce results identical to those produced by
##    the script 
##
##                sfm_with_uncalibrated_cameras_translations_only.py
##
##    but, of course, much faster because it calls on the bundle-adjustment variant
##    of the Levenberg-Marquardt algorithm.  The results from the two scripts would
##    be identical provided you use exactly the name number of world points, exactly
##    the same number of camera positions, etc., in both cases.

##   Calling syntax:
##
##        bundle_adjust_sfm_with_uncalibrated_cameras_translations_only.py



import NonlinearLeastSquares
import ProjectiveCamera
import numpy
import random
import sys

random.seed('abracadabra') 

#cam_pam_noise_factor = 1.0       ##  This creates an initial average error in the 6 
                                  ##  camera parameters for each camera that is large
                                  ##  enough to cause an average error of 60 pixels in
                                  ##  the projections for each of the measurements.  
                                  ##  Note that the six parameters for a camera are 
                                  ##  (w_x,w_y,w_z,t_x,t_y,t_z). The pixel displacement 
                                  ##  error of 60 pixels is brought down to 12 units by 
                                  ##  LM in a couple of iterations if you start with 
                                  ##  zero structure noise

cam_pam_noise_factor = 0.1               ##  creates an initial average error of 4.96 units
                                         ##  which is brought down to 0.26 units in a couple
                                         ##  of iterations.
#structure_noise_factor = 500            
structure_noise_factor = 0               ##  This controls the uncertainty in the initial
                                         ##  values supplied for the structure variables.
                                         ##  When set to 0, you can demonstrate how SBA
                                         ##  can be used for a simultaneous calibration of
                                         ##  of the camera in all its positions.


optimizer =  NonlinearLeastSquares.NonlinearLeastSquares(                  
                                     max_iterations = 400,
                                     delta_for_jacobian = 0.000001,
             )

#  This returns a camera whose optic axis is aligned with the world-Z axis and whose 
#  image plane is parallel to the world-XY plane. The parameters 'alpha_x' and 'alpha_y' 
#  are for the focal length in terms of the image sampling intervals along the x-axis 
#  and along the y-axis, respectively.   The parameters 'x0' and 'y0' are for the 
#  coordinates of the point in the camera image plane where the optic axis penetrates 
#  the image plane with respect to the origin in the image plane (which is usually a 
#  corner of the image):
camera = ProjectiveCamera.ProjectiveCamera( 
                     camera_type = 'projective',
                     alpha_x = 100.0,
                     alpha_y = 100.0,
                     x0 = 100.0,
                     y0 = 100.0,
         )
camera.initialize()
camera.print_camera_matrix()

##  To get around the problem of "nan" values for Rodrigues params when rotation is zero:
##  The argument to the 'rotate' function is in degrees
camera.rotate_previously_initialized_camera_around_world_X_axis(0.5)

world_points = camera.make_world_points_random(15)
print(world_points)
tracked_point_indexes_for_display = None
if len(world_points) > 6:
    tracked_point_indexes_for_display = sorted(random.sample(range(len(world_points)), 6))
    camera.set_tracked_point_indexes_for_display(tracked_point_indexes_for_display)
print("\n\ntracked_point_indexes_for_display: %s" % str(tracked_point_indexes_for_display))

#camera.display_world_points_double_triangles(world_points)
camera.set_num_world_points(len(world_points))

##  In the next statement, the first triple after 'world_points" is for the rotations
##  in degrees around the three world axes and the second triple is for the translations
##  along the three world axes. The large argument is to set the scale.
world_points_xformed = camera.apply_transformation_to_generic_world_points(world_points, (0,0,0), (0.0,0.0,50000.0), 1.0)
print("world_points_xformed: %s" % str(world_points_xformed))

##  Let us now move the camera around and collect the pixels:
number_of_camera_positions = 0
camera_params_ground_truth = []
#y_motion_delta = 500.0
y_motion_delta = 1000.0
all_pixels = []
for i in range(5):
    if i == 0:
        # The 2nd arg is the y_motion_delta which we set to zero for i=0
        camera.translate_a_previously_initialized_camera((0.0,0.0,0.0))
    else:
        camera.translate_a_previously_initialized_camera((0.0,y_motion_delta,0.0))
    camera.add_new_camera_to_list_of_cameras()
    camera_params_ground_truth.append(camera.get_current_camera_pose())
    pixels = camera.get_pixels_for_a_sequence_of_world_points(world_points_xformed)
    all_pixels.append(pixels)
    number_of_camera_positions += 1
print("\n\nall pixels with Y motions of the camera: %s" % str(all_pixels))
#x_motion_delta = 500.0
x_motion_delta = 1000.0
for i in range(5):
    camera.translate_a_previously_initialized_camera((x_motion_delta,0.0,0.0))
    camera.add_new_camera_to_list_of_cameras()
    camera_params_ground_truth.append(camera.get_current_camera_pose())
    pixels = camera.get_pixels_for_a_sequence_of_world_points(world_points_xformed)
    all_pixels.append(pixels)
    number_of_camera_positions += 1
print("\n\nall pixels with X and Y motions of the camera: %s" % str(all_pixels))

motion_history = camera._get_camera_motion_history()
print("\n\ncamera motion history: %s" % str(motion_history))

all_cameras = camera.get_all_cameras()
print("\n\nDisplaying all cameras:")
for item in all_cameras.items():
    print("\nFor camera %d" % item[0])
    print(item[1])

print("\n\nall pixels: %s" % str(all_pixels))
print("\ntotal number of camera positions: %d" % number_of_camera_positions)

camera.construct_X_vector_for_bundle_adjustment(all_pixels)

params_arranged_list = camera.construct_parameter_vec_for_uncalibrated_cameras_using_rodrigues_rotations()
print("\nAll parameters (camera + structure) stringified for one camera position: %s" % str(params_arranged_list))
print("\nNumber of all parameters (camera + structure) for estimation: %d" % len(params_arranged_list))
structure_params = params_arranged_list[6*len(all_cameras):]
print("\nStructure params: %s" % str(structure_params))

##  We will initialize the parameters by adding noise to the ground truth.  By varying
##  the amount of noise, we can study the power of the nonlinear-least-squares with
##  regard to the uncertainty in how the parameters are initialized.  But first we
##  need the ground truth:
ground_truth_dict = camera.set_all_parameters_to_ground_truth_for_sanity_check(world_points_xformed, camera_params_ground_truth)

##  Now construct the prediction vector:
camera.construct_Fvec_for_bundle_adjustment()

#  Get the structure ground truth:
structure_ground_truth = camera.construct_structure_ground_truth()
print("\n\nStructure ground truth: %s" % str(structure_ground_truth))

##  Now initialize the parameters:
initial_params_dict = {}
initial_params_list = []               # need this later for visualization 
initial_structure_params_dict = {}
initial_structure_params_list = []
for param in params_arranged_list:
    if param not in structure_params:
        if param.startswith('w_'):
            initial_params_dict[param] = ground_truth_dict[param] + cam_pam_noise_factor*random.uniform(-1.0,1.0)
        else:
            initial_params_dict[param] = ground_truth_dict[param] + 1000*cam_pam_noise_factor*random.uniform(-1.0,1.0)
    else:
        initial_params_dict[param] = ground_truth_dict[param] + structure_noise_factor*random.uniform(-1.0,1.0)
        initial_structure_params_list.append(initial_params_dict[param])
    initial_params_list.append(initial_params_dict[param])
camera.set_initial_values_for_structure([initial_structure_params_list[3*i:3*i+3] for i in range(len(initial_structure_params_list)//3)])
print("\n\nParameters and their initial values: %s" % str(initial_params_dict))
camera.set_params_list(params_arranged_list)
camera.set_initial_val_all_params_as_dict(initial_params_dict)
camera.set_initial_val_all_params(initial_params_list)
camera.set_constructor_options_for_optimizer_BA(optimizer)     
camera.display_structure()

result = camera.get_scene_structure_from_camera_motion_with_bundle_adjustment()   

######################### print out the calculated structure  ########################

print("\n\n\nRESULTS RETURNED BY bundle_adjust_sfm_with_calibrated_cameras_translations_only.py")

num_iterations_used = result['number_of_iterations']                     
error_norms_with_iterations = result['error_norms_with_iterations']      
final_param_values_list = result['parameter_values']                   
structure_param_values_list = final_param_values_list[-len(structure_ground_truth):]

print("\nError norms with iterations: %s" % str(error_norms_with_iterations))  
print("\nNumber of iterations used: %d" % num_iterations_used)                 
print("\nFinal values for the parameters:\n")                                  
for i in range(len(params_arranged_list)):
    print("%s  =>  %s     [ground truth: %s]   (initial value: %s) \n" % (params_arranged_list[i], final_param_values_list[i], ground_truth_dict[params_arranged_list[i]], initial_params_dict[params_arranged_list[i]]))  
