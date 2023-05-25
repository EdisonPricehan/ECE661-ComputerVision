#!/usr/bin/env python
#-*- coding: latin-1 -*-

##  sfm_with_calibrated_cameras_translations_only.py

##   This script demonstrates how to use the basic Levenberg-Marquardt algorithm
##   for solving problems that require estimating the scene structure from the
##   data collected by a calibrated camera in motion.  For the example shown here
##   the scene structure consists of randomly selected 15 points.

##   The camera is moved to a total of 10 different translational positions.
##   The camera motion is first along the positive world-X direction in steps of 
##   500 units and then along the positive world-Y direction, again in increments
##   of 500 units.

##   Calling syntax:
##
##             sfm_with_calibrated_cameras_translations_only.py


import NonlinearLeastSquares
import ProjectiveCamera
import numpy
import random
import sys

random.seed('abracadabra')      #  converges in 3 iterations, 15 random points,
                                #  10 camera views and structure noise factor of 4000
                                #  which translates into 40% noise

#random.seed('hello0000')        #  Converges in 2 iterations, 15 random points, 
                                #  10 camera views and structure noise factor of 4000
                                #  which translates into about 40% noise in the
                                #  initial values for the estimates.

##  The following variables controls about of noise in the initial values supplied
##  for the structure.  The scale of the structure is roughly 10,000 units.  So when
##  you set the following variable to 2000, you are adding roughly 20% noise to the
##  initial values for the structure variables.  And when the variable is set to 6000,
##  you are adding 60% noise.  Note that the structure_noise_factor is multiplied
##  by a random float between -1.0 and 1.0 and then added to the true value of a
##  structure variable.
#    structure_noise_factor = 2000
#    structure_noise_factor = 4000
structure_noise_factor = 6000


##  The following way to initialize the structure parameters will add
##  roughly 20% noise to the true values of the parameters.  If you set
##  this variable to false, the structure parameters will be initialized
##  randomly.  As you would expect, with random initializations will cause
##  the estimated parameter values to get stuck in a local minimum.
initialize_params_by_offsetting_true_values = True

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

camera.rotate_previously_initialized_camera_around_world_X_axis(0.5)

#world_points = camera.make_world_points_random(30)
world_points = camera.make_world_points_random(15)
print("\n\nworld points: %s" % str(world_points))
tracked_point_indexes_for_display = None
if len(world_points) > 6:
    tracked_point_indexes_for_display = sorted(random.sample(range(len(world_points)), 6))
    camera.set_tracked_point_indexes_for_display(tracked_point_indexes_for_display)
print("\n\ntracked_point_indexes_for_display: %s" % str(tracked_point_indexes_for_display))

camera.set_num_world_points(len(world_points))

##  In the next statement, the first triple after 'world_points" is for the rotations
##  in degrees around the three world axes and the second triple is for the translations
##  along the three world axes. The large argument is to set the scale.
world_points_xformed = camera.apply_transformation_to_generic_world_points(world_points, (0,0,0), (0.0,0.0,50000.0), 1.0)
print("world_points_xformed: %s" % str(world_points_xformed))

##  Move the camera to different positions first along the positive world Y-coordinate and 
##  then along the positive X-coordinate (for a total of 20 positions):
number_of_camera_positions = 0
Y_motion_delta = 0.0
all_pixels = []
for i in range(5):
#    if i==2: break
    camera.translate_a_previously_initialized_camera((0.0,Y_motion_delta,0.0))
    camera.add_new_camera_to_list_of_cameras()
    pixels = camera.get_pixels_for_a_sequence_of_world_points(world_points_xformed)
    print("\n\nPixels for Y-motion camera position %d: %s" % (i, str(pixels)))
    all_pixels.append(pixels)
    number_of_camera_positions += 1
    y_motion_delta =  1000.0
X_motion_delta = 1000.0
for i in range(5):
    camera.translate_a_previously_initialized_camera((X_motion_delta,0.0,0.0))
    camera.add_new_camera_to_list_of_cameras()
    pixels = camera.get_pixels_for_a_sequence_of_world_points(world_points_xformed)
    print("\n\nPixels for X-motion camera position %d: %s" % (i, str(pixels)))
    all_pixels.append(pixels)
    number_of_camera_positions += 1

#  Construct the Measurement Vector X for nonlinear least squares:
print("\n\nall pixels: %s" % str(all_pixels))
print("\ntotal number of camera positions: %d" % number_of_camera_positions)
camera.construct_X_vector(all_pixels)

#  Display the camera matrix for each camera position:
all_cameras = camera.get_all_cameras()
print("\n\nDisplaying all cameras: ")
for item in all_cameras.items():
    print("\nFor camera %d" % item[0])
    print(item[1])

#  Construct an ordered list of the SYMBOLIC NAMES to be used each of the coordinates for 
#  the scene points to be estimated.  This list looks like 
#  "['X_0', 'Y_0', 'Z_0', 'X_1', 'Y_1', 'Z_1', 'X_2' ......]"
params_arranged_list = camera.construct_parameter_vec_for_calibrated_cameras()
print("\nAll structure parameters: %s" % str(params_arranged_list))
print("\nNumber of all structure parameters for estimation: %d" % len(params_arranged_list))

camera_params_dict = {}
#  Note that, generically speaking, each 3x4 camera matrix has elements that symbolically 
#  look like 'p_11', 'p_12', etc.  We need to particularize these to each of the 20 camera
#  positions.  So think of the camera matrix elements as 'p_c_11', p_c_12', 'p_c_13', etc.
#  in which 'c' can be set to the integer index associated with each camera.  So for the
#  very first camera, camera matrix elements will look like 'p_0_11', 'p_0_12', 'p_0_13',
#  'p_0_21', and so on.  In the fourth line below, we first synthesize these symbolic names 
#  for the camera matrix elements and use the synthesized names as the keys in a dictionary.  
#  The value associated with each key is the actual numerical value of that element in the 
#  corresponding camera.
print("\n\nConstruct the symbols for the camera parameters for each camera: ")
for cam in all_cameras.items():
    for i in range(1,4):
        for j in range(1,5):
            camera_params_dict['p_' + str(cam[0]) + str('_') + str(i) + str(j)] = cam[1][i-1,j-1]

#  Construct the Predictor Vector Fvec for nonlinear least squares:
camera.construct_Fvec_for_calibrated_cameras(camera_params_dict)

##  The ground truth for comparison purposes:
ground_truth_dict = camera.set_structure_parameters_to_ground_truth(params_arranged_list, world_points_xformed)
#  Get the structure ground truth:
structure_ground_truth = camera.construct_structure_ground_truth()
print("\n\nStructure ground truth: %s" % str(structure_ground_truth))

print("\n\nInitialize all structure parameters: ")
initial_params_dict = {}
initial_params_list = []               #  need it later for displaying the results visually
if initialize_params_by_offsetting_true_values:
    for param in params_arranged_list:
        initial_params_dict[param] = ground_truth_dict[param] + structure_noise_factor*random.uniform(-1.0,1.0)
        initial_params_list.append(initial_params_dict[param])
else:
    for param in params_arranged_list:
        if param.startswith('X_') or param.startswith('Y_'):
            initial_params_dict[param] = float(random.randint(0,2000)*random.choice([-1,1]))
            initial_params_list.append(initial_params_dict[param])
        elif param.startswith('Z_'):
            initial_params_dict[param] = 5000.0 + float(random.randint(1,1000)*random.choice([-1,1]))
            initial_params_list.append(initial_params_dict[param])

camera.set_initial_values_for_structure([initial_params_list[3*i:3*i+3] for i in range(len(initial_params_list)//3)])

print("\n\nParameters and their initial values: %s" % str(initial_params_dict))
camera.set_params_list(params_arranged_list)
camera.set_initial_val_all_params_as_dict(initial_params_dict)
camera.set_initial_val_all_params(initial_params_list)
camera.set_constructor_options_for_optimizer(optimizer)     

#  Get the structure ground truth:
structure_ground_truth = camera.construct_structure_ground_truth()
print("\n\nStructure ground truth: %s" % str(structure_ground_truth))
camera.display_structure()

result = camera.get_scene_structure_from_camera_motion('lm')   

######################### display the calculated structure  ########################

print("\n\n\nRESULTS RETURNED BY sfm_with_calibrated_cameras_translations_only.py")

num_iterations_used = result['number_of_iterations']                     
error_norms_with_iterations = result['error_norms_with_iterations']      
final_param_values_vector = result['parameter_values']                   
final_param_values_list = final_param_values_vector.flatten().tolist()[0]

print("\nError norms with iterations: %s" % str(error_norms_with_iterations))  
print("\nNumber of iterations used: %d" % num_iterations_used)                 
print("\nFinal values for the parameters:\n")                                  
for i in range(len(params_arranged_list)):
    if params_arranged_list[i] in structure_ground_truth:
        print("%s  =>  %s     [ground truth: %s]   (initial value: %s) \n" % (params_arranged_list[i], final_param_values_list[i], structure_ground_truth[params_arranged_list[i]], initial_params_dict[params_arranged_list[i]]))  
    else:
        print("%s  =>  %s     \n" % (params_arranged_list[i], final_param_values_list[i]))  
