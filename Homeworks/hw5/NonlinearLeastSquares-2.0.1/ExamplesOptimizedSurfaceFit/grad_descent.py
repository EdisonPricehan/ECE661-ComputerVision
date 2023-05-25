#!/usr/bin/env python
#-*- coding: latin-1 -*-

##  grad_descent.py

##  This example calls on the NonlinearLeastSquares class to use the Gradient Descent
##  algorithm for estimating the parameters of a model surface that best fits the
##  synthetically generated noisy height data.
##
##  Note in line (D) the interaction between the domain-agnostic instance 'optimizer'
##  and the domain-specific instance 'surface_fitter'.  The domain-agnostic instance
##  'optimizer' has at its disposal two nonlinear least-squares algorithms:
##  Gradient-Descent and Levenberg-Marquardt.  However, before 'optimizer' can fire
##  up either of these algorithms, it needs the measurement data vector and the model
##  vector in a form that it understands.  What you see in line (D) is
##  'surface_fitter' exporting to 'optimizer' all of this information after it has
##  been formatted according to what is expected by the latter.

##  More specifically, as you can infer from the implementation of 
##
##            set_constructor_options_for_optimizer()
##
##  in the definition of the class OptimizeSurfaceFit, 'surface_fitter' conveys to
##  'optimizer' the following pieces of information in line (D):
##  
##      1) The measurement vector X.
##  
##      2) The initial values to be used for the parameters of the model function.
##
##      3) The Fvec vector, which is a vector of the predicted values, all in functional
##         form, for each of the data points in the measurement vector X.
##
##      4) The display function to be used for plotting the partial and the final
##         results if such results can be displayed in the form of a 2D or 3D graphic
##         with Python's matplotlib library.
##  
##  and some additional book-keeping information.
##
##  The code that you see in lines (E) through (N) is for invoking the desired nonlinear
##  least-squares algorithm, for receiving back the result, and for extracting from the
##  result the various items of information.
##
##  Call syntax:  grad_descent.py


import NonlinearLeastSquares
import OptimizedSurfaceFit

initial_param_values = {'a':2.0, 'b':0.4, 'c':0.8, 'd':0.4}                         #(A)

optimizer =  NonlinearLeastSquares.NonlinearLeastSquares(                           #(B)
                                     max_iterations = 400,
                                     delta_for_jacobian = 0.000001,
                                     delta_for_step_size = 0.0001,
             )

surface_fitter = OptimizedSurfaceFit.OptimizedSurfaceFit(                             #(C)
                                gen_data_synthetically = True,
                                datagen_functional = "7.8*(x - 0.5)**3 + 2.2*(y - 0.5)**2",
                                data_array_size = (16,16), 
                                how_much_noise_for_synthetic_data = 0.7,
                                model_functional = "a*(x-b)**3 + c*(y-d)**2",
                                initial_param_values = initial_param_values,
                                display_needed = True,
#                                display_size = (20,16),
#                                display_position = (500,300),
                                debug = True,
                 )

##  NOTE:  If the figures displayed on your screen are too small and/or at a location
##         that you do not like, you can fix the problem by uncommenting the constructor 
##         options 'display_size' and 'display_position' shown above.  In the
##         'display_size' tuple, the first element is the width and the second element 
##         the height of the display.   In the 'display_position' tuple, the first
##         element is the horizontal coordinate and the second element the vertical
##         coordinate of the upper left-hand corner of the display. The position 
##         coordinates are measured with respect to the upper left corner of your 
##         terminal screen. The horizontal coordinate is positive to the right and the 
##         vertical coordinate is positive pointing straight down.

surface_fitter.set_constructor_options_for_optimizer(optimizer)                     #(D)

result = surface_fitter.calculate_best_fitting_surface('gd')                        #(E)

num_iterations_used = result['number_of_iterations']                                #(F)

error_norms_with_iterations = result['error_norms_with_iterations']                 #(G)

final_param_values = result['parameter_values']                                     #(H)

final_param_values = final_param_values.flatten().tolist()[0]                       #(I)

print("\n\n\nRESULTS RETURNED BY grad_descent.py:")                                 #(J)

print("\nError norms with iterations: %s" % str(error_norms_with_iterations))       #(K)

print("\nNumber of iterations used: %d" % num_iterations_used)                      #(L)

print("\nFinal values for the parameters:\n")                                       #(M)
for i,param_val in enumerate(sorted(initial_param_values.items())):                 #(N)
    print("%s  =>  %s" % (param_val[0], final_param_values[i]))  

