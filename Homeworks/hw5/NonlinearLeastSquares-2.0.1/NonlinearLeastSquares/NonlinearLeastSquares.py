__version__ = '2.0.1'
__author__  = "Avinash Kak (kak@purdue.edu)"
__date__    = '2022-October-6'
__url__     = 'https://engineering.purdue.edu/kak/distNonlinearLeastSquares/NonlinearLeastSquares-2.0.1.html'
__copyright__ = "(C) 2022 Avinash Kak. Python Software Foundation."


__doc__ = '''

NonlinearLeastSquares.py

Version: ''' + __version__ + '''
   
Author: Avinash Kak (kak@purdue.edu)

Date: ''' + __date__ + '''


@title
CHANGE LOG:

  Version 2.0.1:

    Cleaned up the code to make it work with the current version of the
    matplotlib library that is used for displaying the results.  I have also
    removed the use of the "is" and "is not" comparison operators for numeric
    and string literals since such use has now been deprecated.  Also added
    additional comment blocks to the main functions in the module file.

  Version 2.0.0:

    This version includes sparse bundle adjustment (SBA) to speed up code
    execution when using nonlinear least-squares for solving
    structure-from-camera-motion problems with uncalibrated cameras.  SBA
    exploits the sparseness of the Jacobian encountered in such problems.
    This version also includes improvements to the Levenberg-Marquardt code
    itself. The most important improvement consists of how the "mu" factor
    (renamed "alambda" in this module) is initialized.  Version 2.0.0 also
    provides additional methods for the visualization of the results.

  Version 1.5.0:

    This version adds a new class, ProjectiveCamera, to the module for
    demonstrating how nonlinear least-squares can be used for estimating
    structure of a scene from the data recorded with a calibrated camera in
    motion.  For a simulated demonstration, you can create calibrated cameras
    from a generic instance of the ProjectiveCamera class that is then subject
    to various translational and rotational transformations.

  Version 1.1.1:

    This version includes constructor options to control the size and the
    position of the graphics for displaying the results of nonlinear
    least-squares optimization.

  Version 1.1:

    This version fixes a bug in the synthetic data generator function used for
    illustrating the functionality of the NonlinearLeastSquares class.
    Changes also made to the information that is printed out when the module
    is run in the debug mode.

  Version 1.0:

    In the form of a class named NonlinearLeastSquares, this module provides a
    domain agnostic implementation of nonlinear least-squares algorithms
    (gradient-descent and Levenberg-Marquardt) for fitting a model to observed
    data.  Typically, the model involves several parameters and each observed
    data element can be expressed as a function of those parameters plus
    noise.  The goal of nonlinear least-squares is to estimate the best values
    for the parameters given all of the observed data.  In order to illustrate
    how to use the NonlinearLeastSquares class, the module also comes with
    another class, OptimizeSurfaceFit, whose job is to fit the best surface
    (of a specified analytical form) to noisy height data over the
    XY-plane. The model in this case is the analytical description of the
    surface and the goal of nonlinear least-squares is to estimate the best
    values for the parameters in the analytical description.


@title
USAGE FOR OPTIMAL SURFACE FITTING:

    The module includes a domain specific class, OptimizedSurfaceFit, for
    demonstrating how nonlinear least-squares in the form of the
    Levenberg-Marquardt algorithm can be used for optimally fitting
    analytically specified surfaces to noisy data over a plane.
    
    For surface fitting, you need to create an instance of the
    NonlinearLeastSquares class as shown in line (A) below.  In addition,
    you need to create an instance of a domain-specific class such as
    OptimizedSurfaceFit as is shown at line (B) below.  Note that in the
    example shown in line (B), the domain-specific class has two duties:
    (1) to synthetically generate noisy height data; and (2) to call on
    NonlinearLeastSquares to fit an optimized model to the synthetically
    generated data.

        optimizer =  NonlinearLeastSquares(                              #(A)
                         max_iterations = 400,
                         delta_for_jacobian = 0.000001,
                         delta_for_step_size = 0.0001,        # only needed for GD
                     )

        surface_fitter = OptimizedSurfaceFit(                            #(B)                
                                gen_data_synthetically = True,
                                datagen_functional = "7.8*(x - 0.5)**3 + 2.2*(y - 0.5)**2",
                                data_array_size = (16,16),
                                how_much_noise_for_synthetic_data = 0.7,
                                model_functional = "a*(x-b)**3 + c*(y-d)**2",
                                initial_param_values = {'a':2.0, 'b':0.4, 'c':0.8, 'd':0.4},
                                display_needed = True,
                         )

        surface_fitter.set_constructor_options_for_optimizer(optimizer)  #(C)

        result = surface_fitter.calculate_best_fitting_surface('lm')     #(D)

        or 

        result = surface_fitter.calculate_best_fitting_surface('gd')     #(E)

    In line (C), it is the job of the OptimizedSurfaceFit instance as
    constructed in line (B) to export the domain-related information (after
    it is packaged under the hood into a domain-agnostic form) to the
    instance of NonlinearLeastSquares that was constructed in line (A).
    The information that the OptimizedSurfaceFit instance conveys to the
    NonlinearLeastSquares instance includes:

        1) The observed noisy data vector.  This vector is denoted X in the
           NonlinearLeastSquares class.  The same notation is used for the
           observed data vector in the usage-example class
           OptimizedSurfaceFit.

        2) The initial values to be used for the parameters of the model.

        3) A vector of the predicted values for each of the data elements
           in X, all in functional form involving the parameters of the
           model.  This vector is denoted Fvec in the NonlinearLeastSquares
           class.

        4) The display function to be used for plotting the partial and the
           final results if at all such results can be displayed in the
           form of a 2D or 3D graphic with Python's matplotlib library.

    Finally, the statement in line (D) lets you call on the
    Levenberg-Marquardt algorithm for estimating the model parameters. If
    you would rather use the more basic gradient-descent algorithm for that
    purpose, your call will look like what is shown in line (E).

    See the script 

                     leven_marq.py

    in the ExamplesOptimizedSurfaceFit subdirectory of the distribution for
    further information on what calls you need to sequence together for
    optimally estimating the parameters of an analytically specified
    surface that you want to fit to noisy data.


@title
USAGE FOR ESTIMATING THE SCENE STRUCTURE WITH A CALIBRATED CAMERA IN MOTION:

    You again need to create an instance of the NonlinearLeastSquares class
    as shown in line (F) below.  In addition, you need to create an
    instance of a class like ProjectiveCamera as shown at line (G) below.
    If you are writing your own code for camera modeling, it would be best
    if you subclass your code from the ProjectiveCamera class in the
    module.

        optimizer =  NonlinearLeastSquares.NonlinearLeastSquares(            #(F)
                                             max_iterations = 400,
                                             delta_for_jacobian = 0.000001,
                     )
        
        camera = ProjectiveCamera.ProjectiveCamera(                          #(G)
                             camera_type = 'projective',
                             alpha_x = 1000.0,
                             alpha_y = 1000.0,
                             x0 = 300.0,
                             y0 = 250.0,
                 )
        camera.initialize()

        world_points = camera.make_world_points_random(30)
        world_points_xformed = camera.apply_transformation_to_generic_world_points(world_points, ..... )

        ##
        ##  all_pixels = Move the camera to different positions and 
        ##               collect the pixels
        ##
        ##  Now construct the X vector and the corresponding prediction
        ##  vector (Fvec) for nonlinear least-squares:

        camera.construct_X_vector( all_pixels )                              #(H)
        camera.construct_parameter_vec_for_calibrated_cameras()              #(I)
        camera.construct_Fvec_for_calibrated_cameras(camera_params_dict)     #(J)
        result = camera.get_scene_structure_from_camera_motion('lm')         #(K)

    where the returned result will include the optimal estimations for the
    scene parameters.  Note that the parameter vector constructed in line
    (I) defines the hyperplane over which is defined the cost function.
    The goal of nonlinear least-squares is return that point on the
    parameter hyperplane where the cost function takes on the least value.
    Line (J) constructs the prediction vector in terms of the parameters
    used in line (I). The argument 'lm' in line (K) stands for
    "Levenberg-Marquardt".

    See the script 

              sfm_with_calibrated_cameras_translations_only.py 

    in the ExamplesStructureFromCameraMotion subdirectory of the
    distribution for further information on what calls you need to sequence
    together for estimating the structure of a scene that is being viewed
    with a moving camera.


@title
USING SPARSE BUNDLE ADJUSTMENT FOR ESTIMATING THE SCENE STRUCTURE WITH 
AN UNCALIBRATED CAMERA IN MOTION:

    You would again need to construct an instance of the
    NonlinearLeastSquares and an instance of the ProjectiveCamera class as
    shown in lines (F) and (G) above.  But now you would need to replace
    the code in lines (H) through (K) by:

        camera.construct_X_vector_for_bundle_adjustment( all_pixels )     #(L)
        camera.construct_parameter_vec_for_uncalibrated_cameras_with_rodrigues_rotations()
                                                                          #(M)
        camera.construct_Fvec_for_bundle_adjustment()                     #(O)
        result = camera.get_scene_structure_from_camera_motion_with_bundle_adjustment()
                                                                          #(P)

    The construction of the observation vector X in line (L) is specific to
    sparse bundle adjustment --- in the sense that you first want to list
    the pixels in all the cameras for the first world point, then you want
    to list the pixels in all the cameras for the second world point, and
    so on.  Ordering the observed data in this fashion is necessary to
    create the sort of block sparsity in the Jacobian that is exploited in
    SBA.

    Note that we assume that the intrinsic parameters of the camera are
    known already.  So the goal of the bundle adjustment is to estimate the
    six extrinsic parameters, three for translation of the camera and three
    (in the form of Rodrigues rotations) for the rotation.
        
    See the script 

       bundle_adjust_sfm_with_uncalibrated_cameras_translations_only.py

    in the ExamplesStructureFromCameraMotion subdirectory of the
    distribution for further information on what calls you need to sequence
    together for estimating both the camera parameters and the scene
    structure for a scene that is being viewed with a moving uncalibrated
    camera.

@title
INTRODUCTION:

    Nonlinear least-squares is a widely used set of algorithms for fitting
    a model to noisy data through the minimization of a cost function.  The
    observed data is represented as a vector (denoted X in this module) and
    how the model would predict each element of X by another vector that is
    denoted Fvec.  As you would expect, each element of Fvec is a function
    of the parameters of the model.  The goal of nonlinear least-squares is
    to find the best possible values for the parameters, best in the sense
    of minimizing a cost function that is almost always the square of the
    vector norm of the difference between X and Fvec.

    The simplest approach to solving a nonlinear least-squares problem is
    Gradient Descent (GD).  The mental imagery that best explains GD is to
    think of all of the model parameters as constituting a hyperplane and
    the cost function as a surface over the hyperplane.  Gradient descent
    consists of starting at some point in the hyperplane, looking straight
    up at the surface, and walking in the direction of the steepest descent
    on the surface. At each point on the hyperplane, you make the size of
    your next step proportional to the value of the gradient on the
    cost-function surface.  Assuming there are no local minima to trap your
    progress, GD will eventually take you to the point on the hyperplane
    that is directly below the global minimum for the cost function.

    Even when getting trapped in a local minimum is not an issue (because,
    let's say, you made a good choice for the starting point), the basic
    gradient-descent algorithm suffers from the shortcoming that the closer
    you get to the minimum, the smaller your steps will be, and, therefore,
    the slower your progress toward the destination.

    One way to get round this problem with gradient-descent is to use the
    Gauss-Newton formula to make a direct jump to the minimum --- assuming
    you are already sufficiently close to it.  However, should you
    inadvertently try to make a Gauss-Newton jump when still far from the
    minimum, your algorithm could become numerically unstable and crash.

    The algorithm that does a good job of combining the numerical
    robustness of gradient-descent with the speed of Gauss-Newton, while
    making sure that the latter is not invoked too early, is the
    Levenberg-Marquardt (LM) algorithm. Given a start point on the
    hyperplane spanned by the model parameters, LM starts by taking a GD
    step. In subsequent iterations, before committing itself to a step, LM
    checks whether or not that step can safely be taken with GN.  If not,
    it steers the path toward GD.  However, if the condition for taking the
    GN step safely is satisfied, it goes ahead with that.  In this manner,
    LM can get to the minimum in a small number of steps, often under 10,
    whereas for the same problem and the same start point, GD might take
    more than a hundred.
    
    The NonlinearLeastSquares class in this module provides implementations
    for both the basic Gradient Descent (GD) algorithm and the more
    efficient Levenberg-Marquardt algorithm for getting to the minimum of a
    cost function.  Starting with Version 2.0.0. this class also includes
    an SBA (Sparse Bundle Adjustment) variant of the Levenberg-Marquardt
    algorithm for optimal computation of both the structure and the camera
    parameters with the data collected by an uncalibrated camera in motion.

    From a programming standpoint, the most notable feature of
    NonlinearLeastSquares is that it is domain agnostic.  That is, you
    should be able to use NonlinearLeastSquares for solving any problem
    that requires a GD or an LM solution for finding the optimum values for
    a set of model parameters through the minimization of a cost function.

    The fact that NonlinearLeastSquares is generic implies that you have to
    write a class of your own for the specific domain in which you wish to
    use NonlinearLeastSquares.  The domain specific class that you create
    must export to NonlinearLeastSquares values for the following options
    in its constructor:

    -- The observed data vector.  This vector is denoted X in the
       NonlinearLeastSquares class. 

    -- The initial values to be used for the parameters of the model.

    -- A vector of the predicted values for each of the data elements in X,
       all in functional form involving the parameters of the model.  This
       vector is denoted Fvec in the NonlinearLeastSquares class.

    -- The display function to be used for plotting the partial and the
       final results if at all such results can be displayed in the form of
       a 2D or 3D graphic with Python's matplotlib library.

    And, if you wish for NonlinearLeastSquares to use your analytically
    specified partial derivatives in the Jacobian matrix that it needs for
    the next-step calculations, your domain-specific class must also export
    that matrix to NonlinearLeastSquares.  In the absence of a
    user-supplied Jacobian matrix, NonlinearLeastSquares estimates it
    numerically. [See the implementation code for the OptimizedSurfaceFit
    class supplied with this module --- that class is an example of a
    domain-specific class that uses NonlinearLeastSquares for carrying out
    the needed optimization --- for how your own domain-specific class can
    construct a Jacobian matrix in a functional form and supply it to
    NonlinearLeastSquares.]

    The NonlinearLeastSquares class provides several setter methods that
    your own domain-specific class can use to convey the above-mentioned
    information to NonlinearLeastSquares.

    To illustrate how you may wish to write your domain specific class,
    this module also comes with two additional classes named
    OptimizedSurfaceFit and ProjectiveCamera, the first for fitting
    surfaces to noisy data over an xy-plane and the second for estimating
    the structure of a 3D scene from the data recorded by a camera in
    motion.


@title
METHODS:

    Constructing an instance of NonlinearLeastSquares:

        optimizer =  NonlinearLeastSquares(   
                         X = None,
                         Fvec = None,
                         num_measurements = None,
                         num_parameters = None,
                         initial_params_dict = None, 
                         initial_param_values_file = None, 
                         jacobian_functionals_array = None,
                         delta_for_jacobian = 0.000001,
                         delta_for_step_size = 0.0001,
                         max_iterations = 200,
                     )

        In most usage scenarios, though, you are likely to call
        NonlinearLeastSquares directly with just the following constructor
        options and let your own domain specific class set the other
        options at run time.  That is, in your own code, you will first
        create an instance of NonlinearLeastSquares through the following
        call to its constructor:

        optimizer =  NonlinearLeastSquares(             
                         max_iterations = 400,
                         delta_for_jacobian = 0.000001,
                     )

        and subsequently have your own domain-specific class call the
        various setter methods of NonlinearLeastSquares for giving values
        to its other constructor options.  As to which setter methods your
        own class should call is presented in the rest of this section.

@title
CONSTRUCTOR OPTIONS:

        debug, debug2:

            When set to True, it prints out a lot of information at each
            iteration of the nonlinear least-squares algorithm invoked.

        display_function:

            If the problem you are trying to solve with nonlinear
            least-squares allows for the result of optimization to be
            visualized, use this constructor option to supply a reference
            to the function you would like to be used for such
            visualization.
    
        Fvec:
        Fvec_BA:
    
            These must be set to a numpy vector (actually a numpy matrix
            with just one column) whose elements are the "predictors" for
            the corresponding observed values in the X and the X_BA
            vectors, respectively.  (We use the symbol 'X' to denote the
            vector of measured data, as you will soon see. X_BA does the
            same for the bundle-adjustment variant of the basic
            Levenberg-Marquardt algorithm.) Each element of Fvec and
            Fvec_BA is, in general, a function of all of the model
            parameters.

        initial_params_dict:
    
            This is set to a dictionary whose keys are the model parameters
            and whose values the initial values for the model parameters.
            The initial values for the model parameters specify the point
            in the parameter hyperplane where you want to start the descent
            to the minimum.

        initial_param_values_file:

            If your problem involves a very large number of parameters, it
            may be more convenient to place all their initial values in a
            text file.  Each parameter and its initial value must be in a
            line all by itself.  See the file "initial_params_gd2.txt" in
            the Examples directory for how this file needs to be formatted.

        jacobian_functionals_array:

            If you wish to specify the partial derivatives of the
            functional elements in the Fvec vector with respect to the
            model parameters, you can supply those as a numpy chararray
            through this constructor option.

        num_measurements:

            This is simply the number of data values (meaning the 
            number of measurements) in X.
    
        num_parameters:
    
            This is the number of model parameters that you wish to calculate
            with nonlinear least-squares.

        X:
        X_BA:

            This variable must be set to a numpy vector (actually a numpy
            matrix with just one column) whose elements constitute the
            observed data.  The number of elements in X would equal the
            number of observed data elements that you are using for
            calculating the optimum values of the model parameters.
            Whereas the variable X stores the observations for any regular
            application of the Levenberg-Marquardt algorithm, the variable
            X_BA stores the observed data for the case when you want to use
            the bundle-adjustment variant of Levenberg-Marquardt.  Whereas
            the generic Levenberg-Marquardt places no constraints on how
            the observations are arranged in the vector X, that is not the
            case with the bundle-adjustment variant of the same algorithm.
            Both X and M_BA must be a numpy vector, meaning a numpy matrix
            with just one column.


@title
METHODS:

    (1)  grad_descent():

         This is the implementation of the basic gradient-descent
         algorithm.

    (2)  leven_marq():

         This is the implementation of the Levenberg-Marquardt algorithm
         mentioned in the Introduction section.

    (3)  set_debug():

         This allow your own domain-specific class to set the 'debug'
         attribute of an instance of NonlinearLeastSquares.

    (4)  set_display_function():

         Some problem domains allow the result of a nonlinear least-squares
         calculation to be displayed with 2D or 3D graphics.  For example,
         if the goal is to fit an analytical form (in the form of, say, a
         polynomial) to noisy height data over a flat plane and you use the
         nonlinear least-squares algorithm to calculate the best values for
         the parameters of the analytical form, you should be able to
         visualize the quality of your results by displaying both the
         original noisy data and the model you fit to the data.  When such a
         visualization of the results is possible, you can pass the
         definition of the your display function through this setter
         function.

    (5)  set_Fvec():
         set_Fvec_BA()

         Each of these methods expects for its main argument a numpy matrix
         with a single column whose each element must be a functional form
         that predicts the corresponding element in the measurement vector
         X or X_BA.  These functional forms will be functions of the model
         parameters.  The first of the two methods is for the regular
         implementation of the Levenberg-Marquardt algorithm and the second
         for the bundle-adjustment variant of the same.

    (6)  set_initial_params():

         This method expects for its main argument a dictionary of
         <key,value> pairs in which the keys are the model parameters and
         the the corresponding values the initial values to be given to
         those parameters.

    (7)  set_jacobian_functionals_array():

         This method expects for its argument an Nxp matrix of functionals
         for the partial derivatives needed for the Jacobian matrix.  N is
         the number of data elements in the X vector and p is the number of
         parameters in the model.  To elaborate, if you are using nonlinear
         least-squares to fit an optimal surface to noisy height values
         over the xy-plane, your X vector will be a single-column numpy
         matrix and each row of this vector would correspond to one height
         value at some (x,y) point. The corresponding row in the argument
         jacobian_functionals_array contains p functionals, with each
         functional being a partial derivative of the model functional
         (with its x and y set according to where the height was recorded)
         with respect to the parameter corresponding to the column index.

    (8)  set_num_measurements():

         This method sets the number of data elements in the X vector in
         the instance of the NonlinearLeastSquares on which the method is
         invoked.

    (9)  set_num_parameters():

         This method sets the number of model parameters that will be used
         in the nonlinear least-squares optimization.

    (10) set_X():
         set_X_BA():
                
         These two methods set the data vector, in the form of a numpy
         matrix consisting of only one column, in an instance of
         NonlinearLeastSquares.  The first of these for the regular
         implementation of the Levenberg-Marquardt algorithm and the second
         for the bundle-adjustment version of the same.  This is the data
         to which you which you want to fit a given model and you want
         NonlinearLeastSquares to estimate the best values for the
         parameters of the model.

@title
OPTIMIZED SURFACE FITTING:

    This section presents a class named OptimizedSurfaceFit to illustrate
    how you can use the functionality of NonlinearLeastSquares in your own
    code.  The goal of OptimizedSurfaceFit is to fit a model surface to
    noisy height data over the xy-plane in the xyz-coordinate frame, with
    the model surface being described by a polynomial function.  Here are
    some examples of such polynomials:

           "a*(x-b)**2 + c*(y-d)**2 + e"

           "a*x**2 + c*y**2"

           "a*x + b*y + c"
    
    where the value returned by the polynomial for given values of the
    coordinate pair (x,y) is the height above the xy-plane at that point.
    Given the sort of a model surface shown above, the problem becomes one
    of optimally estimating the value for the model parameters from the
    noisy observed data.  If vector X represents the measured data over a
    set of (x,y) points in the form of a vector of observations, we can now
    write for the cost function:

          d^2    =    || X  -  Fvec ||^2

    where Fvec is a vector of the predictions as dictated by the model at
    each of the (x,y) point.  That is, each element of the Fvec vector is a
    prediction for the corresponding element of the measurement vector X.
    The quantity d^2 is the square of the vector norm of the prediction
    error, meaning the difference between the observations in X and the
    predictions in Fvec.  Given X and Fvec vectors, We can call on
    NonlinearLeastSquares to help us find the best values for the
    parameters of the model surface.

    A typical call to OptimizedSurfaceFit's constructor looks like:

        surface_fitter = OptimizedSurfaceFit(                         
                                gen_data_synthetically = True,
                                datagen_functional = "7.8*(x - 0.5)**3 + 2.2*(y - 0.5)**2",
                                data_array_size = (16,16),
                                how_much_noise_for_synthetic_data = 0.7,
                                model_functional = "a*(x-b)**3 + c*(y-d)**2",
                                initial_param_values = {'a':2.0, 'b':0.4, 'c':0.8, 'd':0.4},
                                display_needed = True,
                                display_size = (12,8),                 
                                display_position = (500,300),
                                debug = True,
                         )

    or, if you wish to also supply the partial derivatives of the model
    functional that can be used by OptimizedSurfaceFit for specifying the
    the Jacobian matrix to NonlinearLeastSquares, like

        surface_fitter = OptimizedSurfaceFit(
                                gen_data_synthetically = True,
                                datagen_functional = "7.8*(x - 0.5)**3 + 2.2*(y - 0.5)**2",
                                data_array_size = (16,16), 
                                how_much_noise_for_synthetic_data = 0.5,
                                model_functional = "a*(x-b)**3 + c*(y-d)**2",
                                initial_param_values = {'a':2.0, 'b':0.4, 'c':0.8, 'd':0.4},
                                partials_for_jacobian = {'a':'(x-b)**2', 
                                                         'b':'-2*a*(x-b)', 
                                                         'c':'(y-d)**2', 
                                                         'd':'-2*c*(y-d)'},            
                                display_needed = True,
                                display_size = (12,8),                 
                                display_position = (500,300),
                         )
    
    With regard to the constructor option 'partials_for_jacobian', it is
    important to realize that what is passed to OptimizedSurfaceFit's
    constructor is not directly the Nxp Jacobian matrix (where N is the
    number of observations in X and p the number of parameters in the
    model).  Instead, it is a set of partial derivatives of the model
    functional with respect to the parameters of the functional.  However,
    OptimizedSurfaceFit knows how to translate into an Nxp numpy chararray
    of the functionals needed for the Jacobian.

    The constructor options for the OptimizedSurfaceFit class:

        data_array_size:                      

           The synthetic height that is generated by OptimizedSurfaceFit is
           over a unit square in the xy-plane. Both the x and the y
           coordinates of this square range over the interval 0.0 to 1.0.
           If you set this constructor option to, say, (16,16), the unit
           square will be sampled over a 16x16 grid.

        datagen_functional:           

            When gen_data_synthetically is set to True, you must supply an
            algebraic expression in the form of a string that
            OptimizedSurfaceFit can use to generate the height data.  Here
            is an example of what such a string looks like:

                       "7.8*(x - 0.5)**3 + 2.2*(y - 0.5)**2"

            You can call any function inside the string that the Python
            math library knows about.

        debug:

            This flag is passed on to NonlinearLeastSquares.  When set to
            True, it causes that class to display useful information during
            each iteration of the nonlinear least-squares algorithm.

        display_needed:

            Fitting optimal surfaces to height data lends itself well to 3D
            visualization.  So if you'd like to see the the surfaces that
            correspond to the optimal values for the model parameters, set
            this constructor option to True.

        display_position:

            It is set to a tuple of two integers, with the first integer
            specifying the horizontal coordinate and the second the
            vertical coordinate of the upper left corner of the display.
            These two coordinate values are with respect to the upper left
            corner of your terminal screen. Horizontal coordinates are
            positive to the right and vertical coordinates positive
            pointing down.  Setting this constructor parameter is optional.
            If not set, matplotlib will use its default values.

        display_size:

            It is set to a tuple of two integers, with the first integer
            specifying the width and the second the height of the display.
            Setting this constructor parameter is optional.  If not set,
            matplotlib will use its default values.

        gen_data_synthetically:

            If set to True, OptimizedSurfaceFit can generate the measurement
            height data for your synthetically according to the function
            you specify through the constructor option 'datagen_functional'.

        how_much_noise_for_synthetic_data:                    

           This option controls the amount of noise that is added to the
           height data generated according to the datagen_functional.  The
           best way to give a meaningful value to this construction option
           is to set to some fraction of the largest coefficient in the
           datagen_functional.

        initial_param_values:   

           Through this option, you can transmit to OptimizedSurfaceFit your
           initial guesses for the values of the parameters in the model
           functional.  If you cannot think of a guess, you try setting all
           the parameters to zero.  OptimizedSurfaceFit conveys your initial
           values for the parameters to the NonlinearLeastSquares class.
           You express the initial values in the form of a dictionary,
           whole keys are the name of the parameters and whose values the
           initial values to be given to those parameters.

        initial_param_values_file:

           If the number of parameters in the problem you are addressing is
           large, it may be more convenient to supply the initial values
           for the parameters through a text file. 

        measured_data_file:

           If the amount of measured data is large, it may be more convenient
           to feed it into the module through a text file.

        model_functional:                        

           This is the algebraic expression that we want to fit to the
           noisy height data.  OptimizedSurfaceFit will call on
           NonlinearLeastSquares to estimate the best values for the
           parameters of this algebraic expression.  For example, if the
           model_functional is "a*(x-b)**3 + c*(y-d)**2", the
           NonlinearLeastSquares will find the best possible values for the
           parameters a, b, c, and d --- best in the sense of minimizing
           the cost function described previously.

        model_functional_file:

           If the model functional is too long and/or too complex to be
           specified as an option directly in a call to the constructor,
           you can also place it in a text file through the
           model_functional_file option.

        optimizer:

           Through this constructor option, you can have an instance
           variable of the same name to hold a reference to an instance of
           NonlinearLeastSquares.

        partials_for_jacobian:

           Although the NonlinearLeastSquares class can numerically
           estimate the partial derivatives of the element of the Fvec
           vector with respect to the model parameters, with this option
           you can supply your own analytical forms for the partial
           derivatives that OptimizedSurfaceFit can convert into a Jacobian
           matrix before transmitting it to NonlinearLeastSquares.

    Here are the methods defined for OptimizedSurfaceFit:

        (1) construct_Fvec():

            This method constructs the Fvec vector that
            NonlinearLeastSquares needs for comparing with the measurement
            vector X. Each element of Fvec is a prediction for the
            corresponding element of X and this prediction is a functional
            form involving model parameters.

        (2) construct_jacobian_array_in_functional_form():

            This method is used only when the user supplies analytical
            forms for the partial derivatives of the model functional with
            respect to each of the model parameters.  (When the user does
            not supply such partial derivatives, NonlinearLeastSquares
            estimates the Jacobian through numerical approximations.)

        (3) display_function()

            The problem addressed by OptimizedSurfaceFit lends itself well
            to visualization of the quality of the results returned by
            NonlinearLeastSquares.  With the definition for this method
            that is provided, you can see how well the model parameters
            estimated by NonlinearLeastSquares fit the noisy height data.

        (4) gen_data():

            This method generates the height data over the xy-plane
            according to the analytical form that is supplied to it as its
            main argument.  We refer to this analytical form as the 'model
            functional'.

        (5) get_initial_params_from_file():

            If you want to use model functions that have a large number of
            parameters, it might be easier to place their values in a text
            file and have OptimizedSurfaceFit get it from the file by using
            this method.

        (6) get_measured_data_from_text_file():

            If you would like to supply the height data through a text file
            (rather than have the class generate it automatically for you),
            then this is the method to call for reading in the data from
            the file.  The method assumes that that individual data
            elements are separated by whitespace characters (space, tab,
            newline, etc.).  Since OptimizedSurfaceFit knows about the size
            of the array both along the x-coordinate and along the
            y-coordinate, it knows how to interpret the data in the text
            file.

        (7) get_model_functional_from_file():

            If the model functional is too long, you can get
            OptimizedSurfaceFit to read it from a text file by using this
            option.

        (8) set_constructor_options_for_optimizer()

            The responsibility of this method is to take all of the user
            supplied information and reconstitute it into a form that is
            needed by NonlinearLeastSquares taking into account the
            peculiarities of your domain.


@title
ESTIMATING SCENE STRUCTURE WHEN USING A CALIBRATED CAMERA IN MOTION:

    This section presents a class named ProjectiveCamera to illustrate how
    you can use the functionality of NonlinearLeastSquares in your own code
    for estimating the structure of a 3D scene from the data recorded by a
    calibrated camera in motion.

    To create a simulated structure-from-camera-motion demonstration with
    this module, you must first create an instance the ProjectiveCamera
    class.  A typical call to ProjectiveCamera's constructor looks like:

        camera = ProjectiveCamera.ProjectiveCamera(
                             camera_type = 'projective',
                             alpha_x = 1000.0,
                             alpha_y = 1000.0,
                             x0 = 300.0,
                             y0 = 250.0,
                 )
        camera.initialize()

    This returns a camera whose optic axis is aligned with the world-Z axis
    and whose image plane is parallel to the world-XY plane. The parameters
    'alpha_x' and 'alpha_y' are for the focal length of the camera in terms
    of the image sampling intervals along the x-axis and along the y-axis,
    respectively.  The parameters 'x0' and 'y0' are for the coordinates of
    the point in the camera image plane where the optic axis penetrates the
    image plane with respect to the origin in the image plane (which is
    usually a corner of the image).

        world_points = camera.make_world_points_for_triangle()
        world_points_xformed = camera.apply_transformation_to_generic_world_points( world_points, \
                                                                    (0,0,0), (0.0,0.0,5000.0), 1.0)

    which generates a triangle defined by its three vertices from a method
    defined for the ProjectiveCamera class and then moves the scene
    triangle along the optic axis of the camera (the world-Z axis) by 5000
    units.  After the transformation, the three vertices are at the
    coordinates (3000,3000,5000), (4000,3000,5000), and (4000,5000,5000).

    Subsequently, you must move the camera to different positions and
    orientations and use the camera matrix constructed by the
    ProjectiveCamera instance to project the world triangle into the camera
    images. You are going to need the following two methods defined for the
    ProjectiveCamera class for these camera motions:

        rotate_previously_initialized_camera_around_x_axis( theta_x_delta )

        translate_a_previously_initialized_camera( (0.0,y_motion_delta,0.0) )

    At each camera position/orientation achieved with the above two methods, you
    can record the pixels with the following call:

        pixels = camera.get_pixels_for_a_sequence_of_world_points( world_points_xformed )

    Subsequently, you must make the following call:

        construct_X_vector( all_pixels )        

    where 'all_pixels' is the set of all the pixel recorded in all the
    positions of the camera.

    You would also need to create a Prediction Vector, Fvec, for the
    observed data whose elements are predictor functionals in terms of the
    scene parameters that need to be estimated.  This is achieved with a 
    call like:

        construct_Fvec_for_calibrated_cameras( camera_params_dict )

    Now you are ready to call the following method:

        get_scene_structure_from_camera_motion('lm')

    which will invoke the Levenberg-Marquardt method on the
    NonlinearLeastSquares class to estimate the scene structure.

    The constructor options for the ProjectiveCamera class:

        camera_type:

           You can only set this constructor option to 'projective'
           in the current version of the module.  Eventually, I intend
           to include 'orthographic' as another possibility for this
           option.

        alpha_x:
        alpha_y:

           These options are for the focal length in terms of the image
           sampling intervals used along the image x-axis and along the
           image y-axis, respectively.

        x0:
        y0:

           These options are for the coordinates of the point in the camera
           image plane where the optic axis penetrates the image plane with
           respect to the origin in the image plane (which is usually a
           corner of the image).

        camera_rotation:

           Using the (roll,pitch,yaw) convention you can specify the
           rotation for the camera in the constructor itself.  However, for
           experimenting with structure-from-camera-motion experiments, it
           is easier to first construct a camera in its generic pose and to
           then call the rotate and translate methods on it in order to
           move to a different position and orientation.

        camera_translation:

           Using a triple to indicate displacements along the world-X,
           world-Y, and world-Z, you can specify a translation for the
           camera in the constructor itself.  However, for experimenting
           with structure-from-camera-motion experiments, it is easier to
           first construct a camera in its generic pose and to then call
           the rotate and translate methods on it in order to move to a
           different position and orientation.


    Here are the methods defined for ProjectiveCamera:

        (1) add_new_camera_to_list_of_cameras():

            You will find this utility method useful for enumerating all
            the different camera positions you will be using in a simulated
            structure-from-camera-motion experiment.

        (2) apply_transformation_to_generic_world_points()

            After you have constructed a scene object (typically just a
            simple shape like a triangle or a tetrahedron), you can call on
            this method to change its position and the pose in the world
            frame.  The method takes FOUR arguments: (1) The scene
            structure in the form of a list of homogeneous coordinates for
            the world points on the object.  (2) The first is a triple that
            specifies the rotation using the (roll,pitch,yaw)
            convention. (3) The second argument is a triple for the
            displacement along the world-X, world-Y, and world-Z
            coordinates. (4) The scale factor by which you want to expand
            or shrink the scene object.

        (3) construct_Fvec_for_calibrated_cameras(camera_params_dict)

            This method constructs the prediction vector Fvec vector that
            NonlinearLeastSquares needs for comparing with the measurement
            vector X. Each element of Fvec is a prediction for the
            corresponding element of X and this prediction is a functional
            form involving the structure parameters.
            
        (4) construct_parameter_vec_for_calibrated_cameras()

            This method constructs an ordered list of the SYMBOLIC NAMES to
            be used for each of the coordinates for the scene points that
            need to be estimated.  This list looks like 
            "['X_0', 'Y_0', 'Z_0', 'X_1', 'Y_1', 'Z_1', 'X_2' ......]"

        (5) construct_structure_ground_truth()

            This method packages the scene world points in a way that makes
            it convenient to output in your terminal window the estimated
            coordinates for the scene points, the ground-truth value for
            those coordinates, and the initial guesses supplied for them.

        (6) construct_X_vector(all_pixels)

            As mentioned in the Introduction, we use the notation X to
            represent a vector of all the observed data.  For a
            structure-from-camera-motion problem, the observed data
            consists of all the pixels in all of the camera positions.
            This method orders the x- and the y-coordinates of all the
            recorded pixels in the same fashion as the order given to the
            scene points in world-3D.  

        (7) display_structure()

            This method is used to display in the form of a Matplotlib
            figure the following three things simultaneously: the estimated
            scene structure, the actual world points used for the scene
            object, and the initial guesses supplied for those coordinates
            to the nonlinear least-squares algorithm.  The three parameters
            for this method are named 'structure_points_estimated',
            'world_points_xformed', and 'initial_values_supplied'.

        (8) get_all_cameras()

            This utility method is convenient for getting hold of all the
            cameras that supplied the data for solving the structure-from-
            camera-motion problem.  We consider an instance of
            ProjectiveCamera at each of its positions in world-3D as a
            distinct camera.  So if you move the camera to, say, 20
            different locations, you are in effect using 20 cameras.

        (9) get_pixels_for_a_sequence_of_world_points()

            For any given camera position, this method applies the
            corresponding camera matrix to each world point, which must be
            in homogeneous coordinates, in the sequence of world points
            supplied to the method as its argument.

        (10) get_scene_structure_from_camera_motion('lm')
             get_scene_structure_from_camera_motion_with_bundle_adjustment()

            You must call the first of these two methods for estimating the
            scene structure for the calibrated cameras case after you have
            collected all the pixel data from all the different positions
            of the camera.  Obviously, before you can call this method, you
            would need to construct the observation vector X from the pixel
            data the predictor vector Fvec from the parameters of the
            cameras at each of their positions.  And you call the second
            method for doing the same for the case of uncalibrated cameras
            if you want to use the bundle-adjustment version of the
            Levenberg-Marquardt algorithm.

        (11) initialize()
    
            This method packs the constructor options supplied to the
            ProjectiveCamera constructor in the form of the camera's
            intrinsic parameter matrix K.  If a translation and/or a
            rotation is specified for the camera through the constructor,
            those are also incorporated in the 3x4 camera matrix P put
            together by this method.
        
        (12) make_world_points_for_triangle()

            This method returns a scene object that consists of a triangle
            in world 3D. I have found a triangle defined by its three world
            points to be convenient for testing the basic logic of the
            algorithm for solving a structure-from-camera-motion problem.
            The triangle returned by this method can be subject to any
            orientation changing and position changing transformation.
        
        (13) make_world_points_from_tetrahedron_generic()

            Like the previous method, this method returns world points on a
            tetrahedron in world 3D that you can subsequently use for your
            simulated structure-from-moving-camera experiment.

        (14) print_camera_matrix()

            This utility method is convenient for displaying the 3x4 camera
            matrix for any or all of the positions of the camera.

        (15) rotate_previously_initialized_camera_around_world_X_axis()

            This method incrementally rotates the camera clockwise around
            the world-X axis by an angle 'theta' in degrees that is
            supplied to the method as its argument.

        (16) rotate_previously_initialized_camera_around_world_Y_axis()

            This method incrementally rotates the camera clockwise around
            the world-Y axis by an angle 'theta' in degrees that is
            supplied to the method as its argument.

        (17) set_constructor_options_for_optimizer( optimizer )
             set_constructor_options_for_optimizer_BA( optimizer )

            A ProjectiveCamera instance uses these method to pass on to
            NonlinearLeastSquares all the information needed by the latter
            (such as the observed data vector X or X_BA and the prediction
            vector Fvec or Fvec_BA) for constructing an optimum estimate of
            the scene structure.  The argument 'optimizer' that this method
            takes is an instance of NonlinearLeastSquares.

        (18) set_initial_values_for_params()

            Every nonlinear least-squares algorithm needs a starting guess
            for whatever it is that is being estimated.  In most cases, you
            would construct a random guess for the parameters and supply
            those values to this method in the form of a dictionary in
            which each key is the symbolic name of one of the parameters
            being estimated and the value a random guess for the parameter.

        (19) set_params_list( params_arranged_list )
            
            You will use this method to pass on to the instance of
            ProjectiveCamera an ordered list of the parameters you want
            estimated with nonlinear least-squares.

        (20) translate_a_previously_initialized_camera()

            This method incrementally displaces the camera by 'translation'
            that is supplied to it as its argument. The argument
            'translation' consists of a triple of real numbers that stand
            for a displacement along the world-X, along the world-Y, and
            along the world-Z.


@title
ESTIMATING SCENE STRUCTURE AND CAMERA PARAMETERS WHEN USING AN UNCALIBRATED CAMERA IN MOTION:
    
    The problem of scene construction becomes a lot more challenging when
    the camera in motion is uncalibrated.  When I say uncalibrated, I mean
    uncalibrated with respect to its extrinsic pararameters.  We assume in
    all cases in this document that the intrinsic parameters of the camera
    are known.

    What makes structure estimation more complicated in this case is that
    each new camera position adds 6 additional variables to the overall
    parameter space, 3 for translation and 3 for rotation as expressed by
    the Rodrigues parameters.  Let's say you have M camera positions and N
    structure points.  That would call for (6*M + 3*N) parameters to be
    estimated by the Levenberg-Marquardt algorithm because each camera has
    six external pose parameters associated with it and each structure
    point is defined by its three world coordinates.  Assuming that you can
    see all of the structure points in all the cameras, all of the pixels
    recorded in all of the camera position would constitute a 2*N*M
    dimensional observation vector X.  Therefore, your Jacobian will be of
    size [(2*N*M) x (6*M+3*N)].  For an example, say you have 20 camera
    positions and 100 structure points, your Jacobian will be of size
    4000x420.  Calculating a Jacobian of this size and multiplying it by
    its transpose could take many more than several minutes on a run of the
    mill machine.  And having to calculate the Jacobian multiple times in
    the iterative framework of nonlinear least-squares estimation could
    test your patience as you are waiting for the results.

    Fortunately, this otherwise long execution time can be significantly
    shortened if you take advantage of the sparsity of the Jacobian when
    using uncalibrated cameras.  As to why the Jacobian is sparse, consider
    the fact that each row of the Jacobian is a partial derivative of the
    prediction for one observation with respect to ALL the parameters.
    Obviously, when you take the partial derivative of the prediction
    function for a pixel in one specific camera with respect to the
    parameters of all other cameras, all those partial derivatives will be
    zero.  The implementations of the Levenberg-Marquardt algorithm that
    take advantage of the sparsity of the Jacobian are commonly referred to
    as "Sparse Bundle Adjustment" or just "Bundle Adjustment".

    This logic that this module uses for exploiting the sparsity of the
    Jacobian is based on the paper "SBA: A Software Package for Generic
    Sparse Bundle Adjustment" by Manolis Lourakis and Antonis Argyros that
    appeared in ACM Transactions on Mathematical Software, March 2009.
    
    In honor of these two authors, I have named all of the block Jacobian
    submatrices that take partial derivatives of the predictors with
    respect to the camera parameters as the Argyros matrices.  And I have
    named all of the block Jacobian matrices that do the same with respect
    to the structure variables as Lourakis matrices.

    In this module, the "bundle-adjustment" version of Levenberg-Marquardt
    algorithm can be found in the method

           bundle_adjust()

    of the NonlinearLeastSquares class.  As you will notice, the
    observation vector that I called X in the implementation of the method
    leven_marq() is now called X_BA in the implementation of the method
    bundle_adjust().  The reason has to do with the fact that the basic
    Levenberg-Marquardt algorithm that you see in leven_marq() does not care
    how the observed data is arranged in the vector X. Typically, in my
    personal computer vision code for multi-camera situations, I arrange
    the data by the camera.  That is, I group together all of the pixels
    recorded in one camera, followed by all of the pixels in the second
    camera, and so on.  That is how the observed data (and, therefore, also
    the prediction vector Fvec) is arranged in the method leven_marq().
    Unfortunately, that ordering of the data does not work for sparse
    bundle adjustment.  To maximally exploit sparse bundle adjustment, you
    must order the observation vector so that your first group together the
    pixels for what we may refer to as the first structure point, to be
    followed by the pixels in all the cameras for the second structure
    point, and so on.  Because this ordering is critical to the SBA
    algorithm described in the paper by Lourakis and Argyros, I have
    denoted the observation vector X_BA in the method bundle_adjust().  The
    order in which the predictor functionals are placed in the predictor
    vector Fvec must correspond to the order used in X_BA.  So the
    prediction vector in bundle_adjust() is named Fvec_BA.

    About how to invoke the bundle-adjustment variant of
    Levenberg-Marquardt in your own code, note the following methods of the
    ProjectiveCamera class that are specially meant for that purpose:

        (1)  construct_X_vector_for_bundle_adjustment()

             It is this method's job to arrange the observed data (meaning
             the pixels in the camera images) in the special order that is
             needed by the bundle-adjustment algorithm.

        (2)  construct_parameter_vec_for_uncalibrated_cameras_with_rodrigues_rotations()    

             When using uncalibrated cameras, the parameter vector must
             include three translational and three rotational parameters
             for each camera.  These are in addition to the three (X,Y,Z)
             parameters for each structure point being tracked in the
             cameras.  The job of this method is to synthesize this
             parameter vector.
             
        (3)  construct_Fvec_for_bundle_adjustment()

             The ordering used for the observed data (meaning the pixels in
             the cameras) in the X_BA vector must also be used in the
             prediction vector when you use the bundle-adjustment variant
             of the basic Levenberg-Marquardt algorithm.  This method
             creates the needed prediction vector.
       
        (4)  get_scene_structure_from_camera_motion_with_bundle_adjustment()

             It is this method of the ProjectiveCamera class that invokes
             the bundle_adjust() method of the NonlinearLeastSquares class.


@title
THE ExamplesOptimizedSurfaceFit DIRECTORY:

    See the 'ExamplesOptimizedSurfaceFit' directory in the distribution for
    examples of how you can use the NonlinearLeastSquares class for solving
    optimization problems.  These examples are based on the domain specific
    class OptimizedSurfaceFit that knows about fitting model surfaces to
    noisy height data over a flat plane.  You will see the following four
    scripts in this directory:

        leven_marq.py

        grad_descent.py    

        leven_marq_with_partial_derivatives.py

        grad_descent_with_partial_derivatives.py

    For the first two scripts, the NonlinearLeastSquares instance used will
    estimate the needed Jacobian matrix through appropriate numerical
    approximation formulas applied to the elements of the Fvec vector.  On
    the other hand, for the third and the fourth scripts, your own
    domain-specific class must construct the Jacobian matrix, in the form
    of an array of functions. In the case of the domain-specific class
    OptimizedSurfaceFit that comes with this module, this Jacobian matrix is
    constructed from the user-supplied partial derivatives for the model
    functional.

    In order to become familiar with the NonlinearLeastSquares class, you
    might wish to play with the four scripts listed above by:

    -- Trying different functional forms for the 'datagen_functional' for
       different shaped surfaces.

       When you change the algebraic form of 'datagen_functional' for the
       OptimizedSurfaceFit class, make sure that you also change the
       algebraic form supplied for 'model_functional'.  Note that nonlinear
       least-squares can only calculate the parameters of a model
       functional that best fit the noisy height data; it cannot conjure up
       a new mathematical form for the surface.  So the basic mathematical
       form of the 'model_functional' must be the same as that of the
       'datagen_functional'.

    -- Trying different degrees of noise.  

       As mentioned elsewhere, when you supply a numerical value for the
       constructor option 'how_much_noise_for_synthetic_data' for the
       OptimizedSurfaceFit class, the number you enter should be in
       proportion to the largest numerical coefficient in the 'datagen'
       functional.  Change this numerical value and see what happens to the
       quality of the final results.

    -- Try different values for the initial values of the model parameters.

       Since, depending on where the search for the optimum solution is
       started, all nonlinear least-squares methods can get trapped in a
       local minimum, see what happens when you change these initial
       values.
      
    -- Try different algebraic expressions for the 'model_functional'
       constructor option for the OptimizedSurfaceFit class.  But note that
       if you change the algebraic form of this functional, you must also
       change the algebraic form of the 'datagen_functional' option.

    -- Try running the example with and without the partial derivatives
       that are supplied through the 'partials_for_jacobian' option for the
       OptimizedSurfaceFit class.


@title
THE ExamplesStructureFromCameraMotion DIRECTORY:

    See the 'ExamplesStructureFromCameraMotion' directory in the
    distribution for the following three example scripts that show how you
    can use the NonlinearLeastSquares module for estimating the structure
    of a 3D scene from the images recorded by a moving camera:  

        sfm_with_calibrated_cameras_translations_only.py

        sfm_with_uncalibrated_cameras_translations_only.py

        bundle_adjust_sfm_with_uncalibrated_cameras_translations_only.py

    where the string "sfm" stands for "structure from motion".

    The first script listed above is for estimating the scene structure
    with a calibrated camera in motion.  As you play with this method, make
    sure you change the level of noise in the initial values supplied for
    the structure parameters to be estimated.  As you will see, the method
    works even when the initial values for the parameters are far from
    their true values.  Note that the ProjectiveCamera class makes it easy
    to specify calibrated cameras.  The constructor of the class first
    gives you a camera for which you can specify the internal and the
    external parameters through the constructor options. Subsequently, you
    can apply translational and rotational transformations to the camera to
    move it to different locations in world 3D.  Since the 3x4 camera
    matrices for all these positions of the camera are known, you end up
    with a set of fully calibrated cameras for experimenting with
    structure-from-motion simulations.

    The second and the third scripts listed above are for the case of
    uncalibrated cameras, with the former a straightforward application of
    the Levenberg-Marquardt algorithm and the latter a bundle-adjustment
    variant of the same.  Logically, both these methods must return
    identical answers.  (If you encounter a case when the two do not return
    the same answer, please send a bug report to me. I'd appreciate that
    very much.)

    Just to give you an idea of the speed-up you will get with
    bundle-adjustment, when I run the second script listed above on my
    laptop, it takes about 15 minutes for the number of structure points
    and the number of camera positions used in that script.  For exactly
    the same number of structure points and the camera positions, the third
    script takes only a couple of minutes.  You can only imagine the
    speed-up you will get with a C-based library for sparse bundle
    adjustment --- such as the "sba" library mentioned in the paper by
    Lourakis and Argyros that I cited earlier in this documentation page.


@title
CAVEAT

    Note the bundle-adjustment variant of the Levenberg-Marquardt algorithm
    that you see in the bundle-adjust() method of the NonlinearLeastSquares
    module is meant for just educational purposes.  Being pure Python, it
    cannot compete with highly optimized C-based code you will see in, say,
    the "sba" library mentioned in the article by Lourakis and Argyros that
    I have cited earlier in this documentation.  So if your needs for
    nonlinear least-squares for estimating the structure and the camera
    parameters are primarily of the production variety, go directly to
    those publicly available libraries.


@title
INSTALLATION:

    The NonlinearLeastSquares class was packaged using setuptools.  For
    installation, execute the following command-line in the source
    directory (this is the directory that contains the setup.py file after
    you have downloaded and uncompressed the package):
 
            sudo python setup.py install
    and/or
            sudo python3 setup.py install

    On Linux distributions, this will install the module file at a location
    that looks like

             /usr/local/lib/python3.6/dist-packages/

    If you do not have root access, you have the option of working directly
    off the directory in which you downloaded the software by simply
    placing the following statements at the top of your scripts that use
    the NonlinearLeastSquares class:

            import sys
            sys.path.append( "pathname_to_NonlinearLeastSquares_directory" )

    To uninstall the module, simply delete the source directory, locate
    where the NonlinearLeastSquares module was installed with "locate
    NonlinearLeastSquares" and delete those files.  As mentioned above, the
    full pathname to the installed version is likely to look like
    /usr/local/lib/python2.7/dist-packages/NonlinearLeastSquares*

    If you want to carry out a non-standard install of the
    NonlinearLeastSquares module, look up the on-line information on
    Disutils by pointing your browser to

              http://docs.python.org/dist/dist.html


@title
BUGS:

    Please notify the author if you encounter any bugs.  When sending
    email, please place the string 'NonlinearLeastSquares' in the subject
    line.


@title
ABOUT THE AUTHOR:

    Not too long ago, the author, Avinash Kak, finished a 17-year long
    "Objects Trilogy Project" with the publication of the last book "Designing
    with Objects" by John-Wiley. If interested, visit his web page at Purdue
    to find out what this project was all about. You might like "Designing
    with Objects" especially if you enjoyed reading Harry Potter as a kid (or
    even as an adult, for that matter).

    For all issues related to this module, contact the author at
    kak@purdue.edu

    If you send email, please place the string "NonlinearLeastSquares" in your
    subject line to get past the author's spam filter.


@title
COPYRIGHT:

    Python Software Foundation License

    Copyright 2022 Avinash Kak

@endofdocs
'''

import numpy
import numpy.linalg
import scipy
import math
import os,sys,glob,re
import itertools

numpy.set_printoptions(precision=3)


class NonlinearLeastSquares(object):
    def __init__(self, *args, **kwargs):
        'constructor'                       
        if args:
            raise Exception('''The NonlinearLeastSquares constructor can only be called with '''
                            '''the following keyword arguments: X, Fvec, num_measurements,  '''
                            '''num_parameters, initial_params_dict, jacobian_functionals_array, '''
                            '''initial_param_values_file, display_function''')
        allowed_keys = 'initial_param_values_file','initial_params_dict','measured_data','max_iterations','delta_for_jacobian','delta_for_step_size','jacobian_functionals_array','num_measurements','num_parameters','display_function','debug'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise Exception("Wrong key used in constructor call --- perhaps spelling error")
        X=Fvec=num_measurements=num_parameters=initial_param_values_file=initial_params_dict=measured_data_file=max_iterations=delta_for_jacobian=delta_for_step_size=jacobian_functionals_array=display_function=debug=None
        if 'initial_params_dict' in kwargs: initial_params_dict=kwargs.pop('initial_params_dict')
        if 'initial_param_values_file' in kwargs: initial_param_values_file=kwargs.pop('initial_param_values_file')
        if 'max_iterations' in kwargs: max_iterations=kwargs.pop('max_iterations')
        if 'delta_for_jacobian' in kwargs: delta_for_jacobian=kwargs.pop('delta_for_jacobian')
        if 'delta_for_step_size' in kwargs: delta_for_step_size=kwargs.pop('delta_for_step_size')
        if 'X' in kwargs: X=kwargs.pop('X')
        if 'Fvec' in kwargs: X=kwargs.pop('Fvec')
        if 'num_measurements' in kwargs: num_measurements=kwargs.pop('num_measurements')
        if 'num_parameters' in kwargs: num_parameters=kwargs.pop('num_parameters')
        if 'jacobian_functionals_array' in kwargs: jacobian_functionals_array=kwargs.pop('jacobian_functionals_array')
#        if 'initial_param_values_file' in kwargs: initial_param_values_file=kwargs.pop('initial_param_values_file')
        if 'debug' in kwargs: debug=kwargs.pop('debug')
        if initial_params_dict and initial_param_values_file:
            raise Exception("You must choose either the 'initial_param_values_file' or the 'initial_params_dict' option in the constructor, but not both")
        self.X = X
        self.Fvec = Fvec                 #  is a column vector --- meaning a numpy matrix with just one column
        self.X_BA = None
        self.Fvec_BA = None
        self.num_measurements = num_measurements
        self.num_parameters = num_parameters
        self.initial_params_dict = initial_params_dict
        self.jacobian_functionals_array = jacobian_functionals_array
        self.display_function = display_function
        if max_iterations:
            self.max_iterations = max_iterations
        else:
            raise Exception("The constructor must specify a value for max_iterations")        
        if delta_for_jacobian:
            self.delta_for_jacobian = delta_for_jacobian
        elif jacobian_functionals_array is None:        
            raise Exception("When not using 'jacobian_functionals_array', you must explicitly set 'delta_for_jacobian' in the constructor for NonlinearLeastSquares")
        self.delta_for_step_size = delta_for_step_size
        self.params_ordered_list = None
        self.params_arranged_list = None       # For scene reconstruction, we use arranged list and not ordered list
        self.problem = 'surface_fitting'       # set to "sfm" for solving "structure from camera motion" problems
        self.debug = debug if debug else False
        self.debug2 = False

    def set_problem(self, prob):
        '''
        If you are using this module to solve structure from camera motion (sfm) problems, use this method
        to set 'self.problem' to 'sfm_N' where 'N' is the number of world points you are tracking.   This 
        is needed because sfm needs the specialized display function defined for the ProjectiveCamera class.
        '''
        self.problem = prob

    def set_num_measurements(self, how_many_measurements):
        print("\nNumber of measurements: ", how_many_measurements)
        self.num_measurements = how_many_measurements

    def set_num_parameters(self, how_many_parameters):
        print("\nNumber of parameters: ", how_many_parameters)
        self.num_parameters = how_many_parameters

    def set_initial_params(self, initial_params_dict):
        self.initial_params_dict = initial_params_dict
        self.params_dict = initial_params_dict

    def set_params_ordered_list(self, params_list):
        self.params_ordered_list = sorted(params_list)

    def set_params_arranged_list(self, params_list):
        self.params_arranged_list = params_list

    def set_X(self, X):
        self.X = numpy.asmatrix(numpy.copy(X))

    def set_X_BA(self, X_BA):
        self.X_BA = numpy.asmatrix(numpy.copy(X_BA))

    def set_Fvec(self, Fvector):
        '''
        This method supplies the NonlinearLeastSquares class with the prediction vector
        whose each element is a functional form of the prediction in the observed data vector
        X.  Note that  Fvec is a column vector --- meaning a numpy matrix with just one column.  
        '''
        self.Fvec = Fvector  

    def set_Fvec_BA(self, Fvector_BA):
        '''
        You need to call this method for providing the NonlinearLeastSquares class with the
        prediction vector if you are going to be using the bundle-adjustment capabilities
        of the class.
        '''
        self.Fvec_BA = Fvector_BA

    def set_jacobian_functionals_array(self, jacobian_functionals_array):
        '''
        This method expects for its argument an Nxp matrix of functionals for the partial 
        derivatives needed for the Jacobian matrix.  N is the number of measurements in
        the X vector and p is the number of parameters in the model.  If you are using
        nonlinear least-squares to fit optimal surfaces to noisy measurements over the
        xy-plane, each element of the X vector would correspond to one such measurement at
        some (x,y) coordinates. And an element the argument jacobian_functionals_array chararray
        would correspond to the partial derivative of the model functional that already
        has incorporated the (x,y) coordinates corresponding to that row and that is 
        a partial derivative of the model with respect to the parameter corresponding to
        the column.
        '''
        self.jacobian_functionals_array = jacobian_functionals_array          # a chararray of size Nxp

    def set_display_function(self, display_function):
        self.display_function = display_function

    def set_debug(self, debug):
        self.debug = debug


    ###%%%
    #############################  Nonlnear Least-Squares with Basic Gradient Descent ############################

    def grad_descent(self):
        '''
        This is an implementation of the basic gradient descent algorithm for nonlinear least-squares as described 
        in my Lecture 13 notes at the lecture-notes website for Purdue University's ECE661: Computer Vision
        '''
        error_norm_with_iteration = []
        delta_for_jacobian = self.delta_for_jacobian if self.jacobian_functionals_array is None else None
        if self.delta_for_step_size is not None:
            delta_for_step_size = self.delta_for_step_size
        else:
            raise Exception("You must set the 'delta_for_step_size' option in the constructor for the gradient-descent algorithm")
        num_elements = len(self.Fvec)
        num_measurements = len(self.X)
        params_list = self.params_ordered_list if self.params_ordered_list is not None else self.params_arranged_list
        num_params  =  len(params_list)
        current_param_values = [self.params_dict[param] for param in params_list]
        current_param_values = numpy.matrix(current_param_values).T 
        current_fit_to_measurements = numpy.asmatrix(numpy.zeros_like(self.X))
        for i in range(num_measurements):
            current_fit_to_measurements[i,0] = \
                                   eval(self._eval_functional_element(self.Fvec[i,0], self.initial_params_dict))
        if self.debug:
            print("\ncurrent_fit_to_measurements:")
            print(str(current_fit_to_measurements))
        current_error = self.X - current_fit_to_measurements
        if self.debug:
            print("\ncurrent error:")
            print(str(current_error))
            print("current error shape: %s" % str(current_error.shape))
        current_error_norm = numpy.linalg.norm(self.X - current_fit_to_measurements)
        if current_error_norm < 1e-12:
            print("\nCurrent error norm: %.10f" % current_error_norm)
            print('''\nLooks like your initial choices for the parameters are perfect. '''
                  '''Perhaps there is nothing to be gained by invoking nonlinear least-squares '''
                  '''on your problem.''')
            sys.exit(1)
        error_norm_with_iteration.append(current_error_norm)
        if self.debug:
            print("current error norm: %s" % str(current_error_norm))
        new_param_values = new_fit_to_measurements = new_error_norm = None
        iteration_index = 0
        for iteration_index in range(self.max_iterations):
            if self.debug:
                print("\n\n ========================================  GD ITERATION: %d\n\n" % iteration_index)
            jacobian = numpy.asmatrix(numpy.zeros((num_measurements, num_params), dtype=float))
            if self.jacobian_functionals_array is not None:
                for i in range(num_measurements):
                    params_dict_local = {params_list[i] : current_param_values[i].tolist()[0][0] for i in range(num_params)}                
                    if self.debug is True and i == 0: 
                        print("\ncurrent values for parameters: %s" % str(sorted(params_dict_local.items())))
                    for j in range(num_params):
                        jacobian[i,j] = eval(self._eval_functional_element(self.jacobian_functionals_array[i,j], params_dict_local)) 
            else:
                for i in range(num_measurements):
                    params_dict_local = {params_list[i] : current_param_values[i].tolist()[0][0] for i in range(num_params)}
                    for j in range(num_params):
                        incremented_params_dict_local = {param : params_dict_local[param] for param in params_dict_local}
                        param = self.params_ordered_list[j] if self.params_ordered_list is not None else self.params_arranged_list[j]

                        evaled_element1 = self._eval_functional_element(self.Fvec[i][0], params_dict_local)
                        incremented_params_dict_local[param] = params_dict_local[param] + delta_for_jacobian
                        evaled_element2 = self._eval_functional_element(self.Fvec[i][0], incremented_params_dict_local)
                        jacobian[i,j] = (eval(evaled_element2) - eval(evaled_element1)) / delta_for_jacobian
                    params_dict_local = None
            if self.debug:
                print("jacobian:")
                print(str(jacobian))
#                print("jacobian shape: %s" % str(jacobian.shape))
            new_param_values = current_param_values + 2 * delta_for_step_size * (jacobian.T * current_error)
            if self.debug:
                print("\nnew parameter values:")
                print(str(new_param_values.T))
            new_params_dict = {params_list[i] : new_param_values[i].tolist()[0][0] for i in range(num_params)}
            if self.debug:
                print("new_params_dict: %s" % str(new_params_dict))
            new_fit_to_measurements = numpy.asmatrix(numpy.zeros_like(self.X))
            for i in range(num_measurements):
                new_fit_to_measurements[i,0] = eval(self._eval_functional_element(self.Fvec[i][0], new_params_dict))
            if self.debug:
                print("new_fit_to_measurements:")
                print(str(new_fit_to_measurements))
            new_error =  self.X - new_fit_to_measurements
            if self.debug:
                print("\nnew error:")
                print(str(new_error))
            new_error_norm = numpy.linalg.norm(self.X - new_fit_to_measurements)
            if self.debug:
                print("\nnew error norm: %s" % str(new_error_norm))
            if new_error_norm > error_norm_with_iteration[-1]:
                break
            error_norm_with_iteration.append(new_error_norm)
            if self.debug:
                print("\nerror norms with iterations: %s" % str(error_norm_with_iteration))
            if self.display_function is not None and iteration_index % int(self.max_iterations/5.0) == 0:
                self.display_function(new_fit_to_measurements, new_error_norm, iteration_index)
            current_param_values = new_param_values
        if self.debug:
            print("\nerror norms with iterations: %s" % str(error_norm_with_iteration))
            print("\niterations used: %d" % iteration_index)
            print("\n\nfinal values for the parameters: ") 
            print(str(new_param_values))
        if self.debug is True and iteration_index == self.max_iterations - 1:
            print("\n\nWARNING: max iterations reached without getting to the minimum")
        if self.display_function:
            self.display_function(new_fit_to_measurements, new_error_norm, iteration_index)
        result = {"error_norms_with_iterations" : error_norm_with_iteration,
                  "number_of_iterations" : iteration_index,
                  "parameter_values" : new_param_values}
        return result


    ###%%%
    #############  Nonlnear Least-Squares with the Levenberg-Marquardt Algorithm for Gradient Descent ############

    def leven_marq(self):
        '''
        This is an implementation of the Levenberg-Marquardt algorithm for nonlinear least-squares as described 
        in my Lecture 13 notes at the lecture-notes website for Purdue University's ECE661: Computer Vision

        This function is used the following scripts in the ExamplesOptimizedSurfaceFit directory of the distro:

                    leven_marq.py

                    leven_marq_with_partial_derivatives.py

        It is important to note that the above two scripts do NOT call leven_marq() function directly.  AS 
        stated in the main document page for the NonlinearLeastSquares module, the code in the file 
        NonlinearLeastSquares.py is written in a domain agnostic manner.  So you need a domain adaptation 
        class that knows how to package the arguments for calling leven_marq().  For the case of the two scripts
        listed above, that domain-specific class in the distro is OptimizedSurfaceFit.  It is a co-class of 
        the main module class NonlinearLeastSquares in the distribution.

        The function leven_marq() defined here is ALSO used in the following two scripts

            sfm_with_calibrated_cameras_translations_only.py
            sfm_with_uncalibrated_cameras_translations_only.py

        in the ExamplesStructureFromCameraMotion directory of the distribution.  Again, these example scripts
        do NOT directly call the leven_marq() function.  As shown in the two scripts, they must first construct
        an instance of the ProjectiveCamera class that knows how to bundle the arugments together for calling
        the leven_marq() function.
        '''
        if os.path.isdir("figs"):
            list(map(os.remove, glob.glob('figs/*.png')))
        else:
            os.mkdir("figs")
        error_norm_with_iteration = []
        error_norm_per_measurement_with_iteration = []
        alambda_with_iteration = []
        delta_for_jacobian = self.delta_for_jacobian if self.jacobian_functionals_array is None else None
        num_elements = len(self.Fvec)
        num_measurements = len(self.X)
        params_list = self.params_ordered_list if self.params_ordered_list is not None else self.params_arranged_list
        num_params  =  len(params_list)
        current_param_values = [self.params_dict[param] for param in params_list]
        current_param_values = numpy.matrix(current_param_values).T 
        current_fit_to_measurements = numpy.asmatrix(numpy.zeros_like(self.X))
        for i in range(num_measurements):
            current_fit_to_measurements[i,0] = eval(self._eval_functional_element(self.Fvec[i,0], self.initial_params_dict))
        current_error = self.X - current_fit_to_measurements
        print("\n\ncurrent error (before the iterations):")
        print(current_error.flatten().tolist()[0])
        current_error_norm = numpy.linalg.norm(self.X - current_fit_to_measurements)
        error_norm_with_iteration.append(current_error_norm)
        current_error_norm_per_measurement = current_error_norm / math.sqrt(num_measurements)
        print("\n\ncurrent error norm per measurement before iterations: %s" % str(current_error_norm_per_measurement))
        if current_error_norm_per_measurement < 1e-10:
            print("\n\nCurrent error norm: %.10f" % current_error_norm_per_measurement)
            print('''\n\nLooks like your initial choices for the parameters are perfect. Perhaps there is nothing'''
                  '''to be gained by invoking nonlinear least-squares on your problem.''')
            sys.exit(1)
        error_norm_per_measurement_with_iteration.append(current_error_norm_per_measurement)
        if self.display_function is not None and self.problem.startswith("sfm"):
            predicted_pixel_coordinates = current_fit_to_measurements.flatten().tolist()[0]
            predicted_pixels = [(predicted_pixel_coordinates[2*x], predicted_pixel_coordinates[2*x+1]) for x in range(len(predicted_pixel_coordinates) // 2)]
            self.display_function(predicted_pixels, None, current_error_norm_per_measurement)
        else:
            self.display_function(current_fit_to_measurements, current_error_norm_per_measurement, -1)
        new_param_values=new_fit_to_measurements=new_error_norm=new_error_norm_per_measurement=None
        iteration_index = 0
        self.alambda = None
        self.rho = None
        alambda = None
        rho = None
        #  An important feature of LM is that ONLY SOME OF THE ITERATIONS cause a reduction in 
        #  the error vector (which is the difference between the measured data and its predicted 
        #  values from the current knowledge of the parameters), the following list stores just
        #  those iteration index values that were productive in reducing this error.  This list is
        #  useful for deciding when to display the partial results.
        productive_iteration_index_values = []
        need_fresh_jacobian_flag = True
        A = g = None
        iterations_used = None
        best_estimated_structure = best_error_norm = best_error_norm_per_measurement=None
        for iteration_index in range(self.max_iterations):
            if need_fresh_jacobian_flag is True:
                print("\n\nCalculating a fresh jacobian\n\n")
                jacobian = numpy.asmatrix(numpy.zeros((num_measurements, num_params), dtype=float))
                for i in range(num_measurements):
                    params_dict_local = {params_list[i] : current_param_values[i].tolist()[0][0] for i in range(num_params)}
                if self.jacobian_functionals_array is not None:
                    '''
                    A functional form was supplied for the Jacobian.  Use it.
                    '''
                    for j in range(num_params):
                        jacobian[i,j] = \
                              eval(self._eval_functional_element(self.jacobian_functionals_array[i,j], params_dict_local)) 
                else:
                    '''
                    Estimate your own Jacobian
                    '''
                    for i in range(num_measurements):
                        for j in range(num_params):
                            incremented_params_dict_local = {param : params_dict_local[param] for param in params_dict_local}
                            param = self.params_ordered_list[j] if self.params_ordered_list is not None else self.params_arranged_list[j]
                            evaled_element1 = self._eval_functional_element(self.Fvec[i,0], params_dict_local)
                            incremented_params_dict_local[param] = incremented_params_dict_local[param] + delta_for_jacobian
                            evaled_element2 = self._eval_functional_element(self.Fvec[i,0], incremented_params_dict_local)
                            jacobian[i,j] = (eval(evaled_element2) - eval(evaled_element1)) / delta_for_jacobian
                print("\n\nFinished calculating the Jacobian")
                if self.debug:
                    print("\njacobian:")
                    print(jacobian)
                A = jacobian.T * jacobian
                g = jacobian.T * current_error
                if iteration_index == 0:
                    JtJ_max = max(A.diagonal().tolist()[0])
                    print("\n\nmax on the diagonal on J^T.J: %s" % str(JtJ_max))
                    self.alambda = JtJ_max / 1000
                    alambda = self.alambda                        
                    print("\n\nWe start with alambda = %f" % self.alambda)
            if self.debug:
                print("\ng vector for iteration_index: %d" % iteration_index)
                print(str(g.T))
            if abs(numpy.max(g)) < 0.0000001: 
                print("absolute value of the largest component of g below threshold --- quitting iterations")
                break
            B = self._pseudoinverse(A + alambda * numpy.asmatrix(numpy.identity(num_params)))
            new_delta_param = B * g
            new_param_values = current_param_values + new_delta_param
            if self.debug:
                print("\nnew parameter values:")
                print(str(new_param_values.T))
            new_params_dict = {params_list[i] : new_param_values[i].tolist()[0][0] for i in range(num_params)}
            if self.debug:
                print("\nnew_params_dict: %s" % str(sorted(new_params_dict.items())))
            new_fit_to_measurements = numpy.asmatrix(numpy.zeros_like(self.X))
            for i in range(num_measurements):
                new_fit_to_measurements[i,0] = eval(self._eval_functional_element(self.Fvec[i,0], new_params_dict))
            newly_projected_pixel_coordinates = new_fit_to_measurements.flatten().tolist()[0]
            newly_projected_pixels = [(newly_projected_pixel_coordinates[2*x], newly_projected_pixel_coordinates[2*x+1]) for x in range(len(newly_projected_pixel_coordinates) // 2)]
            if self.debug:
                print("\nnew_fit_to_measurements (shown as transpose):")
                print(str(new_fit_to_measurements.T))
            new_error =  self.X - new_fit_to_measurements
            if self.debug:
                print("\nnew error magnitudes at each measurement:")
                print(str(new_error.flatten().tolist()[0]))
            new_error_norm = numpy.linalg.norm(self.X - new_fit_to_measurements)
            new_error_norm_per_measurement = new_error_norm / math.sqrt(num_measurements)
            print("\nnew error norm per measurement: %s" % str(new_error_norm_per_measurement))

            rho = ( error_norm_with_iteration[-1] ** 2  - new_error_norm ** 2 ) / \
                                         (new_delta_param.T * g    +    alambda * new_delta_param.T * new_delta_param)
            if rho <= 0.0:
                need_fresh_jacobian_flag = False
                alambda = alambda * max( [1.0/3.0, 1.0 - (2.0 * rho - 1.0) ** 3] )     ## BIZARRE that a matrix is returned
                alambda = alambda.tolist()[0][0]
                if alambda > 1e12:
                    if self.debug:
                        print("\nIterations terminated because alambda exceeded limit") 
                    break
                if self.debug:
                    print("\nNO change in parameters for iteration_index: %d with alambda = %f" % (iteration_index, alambda))
                continue
            else:
                need_fresh_jacobian_flag = True
                if self.debug:
                    print("\n\n================================================Showing LM results for iteration: %d\n" 
                                                                        % (iteration_index+1))
                productive_iteration_index_values.append(iteration_index)
                alambda = self.alambda
                error_norm_with_iteration.append(new_error_norm)
                error_norm_per_measurement_with_iteration.append(new_error_norm_per_measurement) 
                print("\n\nerror norms per measurement for all iterations: %s" % str(error_norm_per_measurement_with_iteration))
                current_param_values = new_param_values
                best_param_values = new_param_values
                best_predicted_pixels  = newly_projected_pixels
                iterations_used = iteration_index + 1
                if self.display_function is not None:
                    if self.problem.startswith("sfm"):
                        num_structure_points = int(self.problem.split("_")[1])
                        estimated_structure = new_param_values[-3*num_structure_points:].flatten().tolist()[0]
                        estimated_structure = [estimated_structure[3*i:3*i+3] for i in range(num_structure_points)]
                        best_estimated_structure = estimated_structure
                        best_error_norm_per_measurement = new_error_norm_per_measurement
                        self.display_function(best_predicted_pixels, estimated_structure, new_error_norm_per_measurement, len(productive_iteration_index_values)-1)
                    else:
                        self.display_function(new_fit_to_measurements, new_error_norm_per_measurement, len(productive_iteration_index_values)-1)
        if self.debug:
#            print("\nerror norms for all iterations: %s" % str(error_norm_with_iteration))
            print("\nerror norms for all iterations: %s" % str(error_norm_per_measurement_with_iteration))
            print("\niterations used: %d" % (len(productive_iteration_index_values) - 1))
            print("\nproductive iteration index values: %s" % str(productive_iteration_index_values))
            print("\n\nfinal values for the parameters: ") 
            print(str(new_param_values.T))
        if self.debug is True and iteration_index == self.max_iterations - 1:
            print("\n\nWARNING: max iterations reached without getting to the minimum")
#        if self.display_function:
#            self.display_function(new_fit_to_measurements, new_error_norm, len(productive_iteration_index_values))
        if self.display_function is not None:
            if self.problem.startswith("sfm"):
                self.display_function(best_predicted_pixels, best_estimated_structure, best_error_norm_per_measurement, 
                                                               len(productive_iteration_index_values)-1)
            else:
                self.display_function(new_fit_to_measurements, new_error_norm_per_measurement, len(productive_iteration_index_values)-1)
        result = {"error_norms_with_iterations" : error_norm_per_measurement_with_iteration,
#                  "number_of_iterations" : len(productive_iteration_index_values),
                  "number_of_iterations" : iterations_used,
                  "parameter_values" : best_param_values}
        return result


    def leven_marq_v1_5(self):
        """
        This is the implementation of the leven_marq() function as it existed in Version 1.5.  On account of the
        fact that I made significant changes to this function in Version 2.0.0 of the module, I have retained the
        old version for the old-time users of my module.
        """
        if os.path.isdir("figs"):
            list(map(os.remove, glob.glob('figs/*.png')))
        else:
            os.mkdir("figs")
        error_norm_with_iteration = []
        delta_for_jacobian = self.delta_for_jacobian if self.jacobian_functionals_array is None else None
#        delta_for_step_size = self.delta_for_step_size if self.jacobian_functionals_array is None else None
        num_elements = len(self.Fvec)
        num_measurements = len(self.X)
        params_list = self.params_ordered_list if self.params_ordered_list is not None else self.params_arranged_list
        num_params  =  len(params_list)
        current_param_values = [self.params_dict[param] for param in params_list]
        current_param_values = numpy.matrix(current_param_values).T 
        current_fit_to_measurements = numpy.asmatrix(numpy.zeros_like(self.X))
        for i in range(num_measurements):
            current_fit_to_measurements[i,0] = \
                       eval(self._eval_functional_element(self.Fvec[i,0], self.initial_params_dict))
        if self.debug:
            print("\nleven_marq: current_fit_to_measurements (shown as transpose):")
            print(str(current_fit_to_measurements.T))
        current_error = self.X - current_fit_to_measurements
#        if self.debug:
        print("\ncurrent error (shown as transpose):")
        print(str(current_error.T))
        current_error_norm = numpy.linalg.norm(self.X - current_fit_to_measurements)

        if current_error_norm < 1e-12:
            print("\nCurrent error norm: %.10f" % current_error_norm)
            print('''\nLooks like your initial choices for the parameters are perfect. '''
                  '''Perhaps there is nothing to be gained by invoking nonlinear least-squares '''
                  '''on your problem.''')
            sys.exit(1)
        error_norm_with_iteration.append(current_error_norm)
        if self.debug:
            print("\ncurrent error norm: %s" % str(current_error_norm))
        if self.display_function is not None:
            self.display_function(current_fit_to_measurements, current_error_norm, -1)
        new_param_values = new_fit_to_measurements = new_error_norm = None
        iteration_index = 0
        alambda = 0.001
        #  If 10 CONSECUTIVE STEPS in the parameter hyperplane turn out to the wrong choices,
        #  we terminate the iterations.  If you want to change the number of consecutively 
        #  occurring stops, you have to make changes at three different places in this file, 
        #  including the statement shown below.
#        wrong_direction_flags = [0] * 10       
        wrong_direction_flags = [0] * 20
        #  An important feature of LM is that ONLY SOME OF THE ITERATIONS cause a reduction in 
        #  the error vector (which is the difference between the measured data and its predicted 
        #  values from the current knowledge of the parameters), the following list stores just
        #  those iteration index values that were productive in reducing this error.  This list is
        #  useful for deciding when to display the partial results.
        productive_iteration_index_values = [-1]
        for iteration_index in range(self.max_iterations):
            jacobian = numpy.asmatrix(numpy.zeros((num_measurements, num_params), dtype=float))
            if self.jacobian_functionals_array is not None:
                '''
                A functional form was supplied for the Jacobian.  Use it.
                '''
                for i in range(num_measurements):
                    params_dict_local = {params_list[i] : current_param_values[i].tolist()[0][0] for i in range(num_params)}
                    if self.debug is True and i == 0: 
                        print("\ncurrent values for parameters: %s" % str(sorted(params_dict_local.items())))
                    for j in range(num_params):
                        jacobian[i,j] = \
                          eval(self._eval_functional_element(self.jacobian_functionals_array[i,j], params_dict_local)) 
            else:
                '''
                Estimate your own Jacobian
                '''
                for i in range(num_measurements):
                    params_dict_local = {params_list[i] : current_param_values[i].tolist()[0][0] for i in range(num_params)}
                    if self.debug is True and i == 0: 
                        print("\ncurrent values for parameters: %s" % str(sorted(params_dict_local.items())))
                    for j in range(num_params):
                        incremented_params_dict_local = {param : params_dict_local[param] for param in params_dict_local}
                        param = self.params_ordered_list[j] if self.params_ordered_list is not None else self.params_arranged_list[j]
                        evaled_element1 = self._eval_functional_element(self.Fvec[i,0], params_dict_local)
                        incremented_params_dict_local[param] = params_dict_local[param] + delta_for_jacobian
                        evaled_element2 = self._eval_functional_element(self.Fvec[i,0], incremented_params_dict_local)
                        jacobian[i,j] = (eval(evaled_element2) - eval(evaled_element1)) / delta_for_jacobian
                    params_dict_local = None
            if self.debug:
                print("\njacobian:")
                print(str(jacobian))
#                print("\njacobian shape: %s" % str(jacobian.shape))
            A = jacobian.T * jacobian
            g = jacobian.T * current_error
            if self.debug:
                print("\ng vector for iteration_index: %d" % iteration_index)
                print(str(g.T))
            if abs(numpy.max(g)) < 0.0000001: 
                print("absolute value of the largest component of g below threshold --- quitting iterations")
                break
            B = numpy.linalg.inv(A + alambda * numpy.asmatrix(numpy.identity(num_params)))
            new_delta_param = alambda * g if iteration_index == 0 else B * g

            new_param_values = current_param_values + new_delta_param
            if self.debug:
                print("\nnew parameter values:")
                print(str(new_param_values.T))
            new_params_dict = {params_list[i] : new_param_values[i].tolist()[0][0] for i in range(num_params)}
            if self.debug:
                print("\nnew_params_dict: %s" % str(sorted(new_params_dict.items())))
            new_fit_to_measurements = numpy.asmatrix(numpy.zeros_like(self.X))
            for i in range(num_measurements):
                new_fit_to_measurements[i,0] = eval(self._eval_functional_element(self.Fvec[i,0], new_params_dict))
            if self.debug:
                print("\nnew_fit_to_measurements (shown as transpose):")
                print(str(new_fit_to_measurements.T))
            new_error =  self.X - new_fit_to_measurements
            if self.debug:
                print("\nnew error (shown as transpose):")
                print(str(new_error.T))
            new_error_norm = numpy.linalg.norm(self.X - new_fit_to_measurements)
            if self.debug:
                print("\nnew error norm: %s" % str(new_error_norm))
            if new_error_norm >= error_norm_with_iteration[-1]:
                alambda *= 10
                wrong_direction_flags.append(1)
#                wrong_direction_flags = wrong_direction_flags[-10:]   
                wrong_direction_flags = wrong_direction_flags[-20:]   
#                if alambda > 1e11:
                if alambda > 1e9:
                    if self.debug:
                        print("\nIterations terminated because alambda exceeded limit") 
                    break
                if all(x == 1 for x in wrong_direction_flags): 
                    if self.debug:
                        print("\n\nTERMINATING DESCENT BECAUSE reached a max of 20 consecutive bad steps")
                    break
                if self.debug:
                    print("\nNO change in parameters for iteration_index: %d with alambda = %f" % (iteration_index, alambda))
                continue
            else:
                if self.debug:
                    print("\n\n================================================ LM ITERATION: %d" 
                                                                        % len(productive_iteration_index_values))
                    print()
                productive_iteration_index_values.append(iteration_index)
                wrong_direction_flags.append(0)
#                wrong_direction_flags = wrong_direction_flags[-10:] 
                wrong_direction_flags = wrong_direction_flags[-20:] 
                alambda = 0.001
                error_norm_with_iteration.append(new_error_norm)
                if self.debug:
                    print("\nerror norms with iterations: %s" % str(error_norm_with_iteration))
                current_param_values = new_param_values
                if self.display_function is not None:
                    if len(productive_iteration_index_values) % 2 == 0:
                        self.display_function(new_fit_to_measurements, new_error_norm, len(productive_iteration_index_values)-1)
        if self.debug:
            print("\nerror norms with iterations: %s" % str(error_norm_with_iteration))
            print("\niterations used: %d" % (len(productive_iteration_index_values) - 1))
            print("\nproductive iteration index values: %s" % str(productive_iteration_index_values))
            print("\n\nfinal values for the parameters: ") 
            print(str(new_param_values.T))
        if self.debug is True and iteration_index == self.max_iterations - 1:
            print("\n\nWARNING: max iterations reached without getting to the minimum")
        if self.display_function:
            self.display_function(new_fit_to_measurements, new_error_norm, len(productive_iteration_index_values))
        result = {"error_norms_with_iterations" : error_norm_with_iteration,
                  "number_of_iterations" : len(productive_iteration_index_values) - 1,
                  "parameter_values" : new_param_values}
        return result


    ###%%%
    ################  Nonlnear Least-Squares for SfM (Structure from Motion) with Bundle Adjustment  #############

    def bundle_adjust(self, *args, **kwargs):
        """
        This is an implementation of the "bundle adjustment" version of the Levenberg-Marquardt algorithm for nonlinear
        least-squares.  Bundle adjustment takes advantage of the sparsity of the Jacobian that one sees in 
        applications such as estimating the scene structure with the images recorded with uncalibrated cameras.  
        The implementation shown here is based on the now celebrated paper "SBA: A Software Package for Generic
        Sparse Bundle Adjustment" by Manolis Lourakis and Antonis Argyros that appeared in ACM Transactions on 
        Mathematical Software, March 2009.

        This function is used in the following scripts 

            bundle_adjust_sfm_with_uncalibrated_cameras_translations_only.py 

        in the ExamplesStructureFromCameraMotion directory of the distribution.  

        Note that since bundle_adjust() is written in a generic manner, it is not called directly by the example
        scripts listed above.  As shown in the scripts, you need to first construct an instance of the class
        ProjectiveCamera that is a co-class of the main module class NonlinearLeastSquares in the distribiution.        
        """

        if args: raise Exception("The bundle_adjust can only be called with keyword arguments")
        allowed_keys = 'num_camera_params,num_structure_elements,num_cameras,num_params_per_camera,num_measurements_per_camera,initial_val_all_params'
        num_cameras=num_world_points=num_camera_params=num_structure_elements=num_cameras=num_cam_params_per_camera=num_measurements_per_camera=initial_val_all_params=None
        num_camera_params            =   kwargs.pop('num_camera_params')
        num_structure_elements       =   kwargs.pop('num_structure_elements')       ## they remain the same for all cams
        num_cameras                  =   kwargs.pop('num_cameras')
        num_cam_params_per_camera    =   kwargs.pop('num_cam_params_per_camera')
        num_measurements_per_camera  =   kwargs.pop('num_measurements_per_camera')
        initial_val_all_params       =   kwargs.pop('initial_val_all_params')
        error_norm_with_iteration = []
        error_norm_per_measurement_with_iteration = []
        delta_for_jacobian = self.delta_for_jacobian
        Fvec_as_list = self.Fvec_BA[:,0].tolist()
        num_Fvec_elements = len(Fvec_as_list)
        num_world_points       = num_measurements_per_camera // 2
        params_list = self.params_arranged_list
        num_params  =  len(params_list)
        num_measurements = len(self.X_BA)
        params_for_camera_dict        = {i : None for i in range(num_cameras)}
        initial_param_vals_for_cam    = {i : None for i in range(num_cameras)}
        current_param_vals_for_cam    = {i : None for i in range(num_cameras)}
        for c in range(num_cameras):
            params_for_camera_dict[c]        = params_list[c*num_cam_params_per_camera : (c+1)*num_cam_params_per_camera]
            initial_param_vals_for_cam[c]    = initial_val_all_params[c*num_cam_params_per_camera : (c+1)*num_cam_params_per_camera]
        structure_params  =   params_list[num_cameras * num_cam_params_per_camera : ]
        initial_param_vals_for_structure = initial_val_all_params[ num_cameras * num_cam_params_per_camera : ]
        # We now place all the initial values for the params in the current_param list for iterative processing
        current_param_values = initial_val_all_params
        current_param_values_vec = numpy.matrix(current_param_values).T
        current_fit_to_measurements = numpy.asmatrix(numpy.zeros_like(self.X_BA))
        for data_index in range(num_measurements_per_camera * num_cameras):
            current_fit_to_measurements[data_index,0] = eval(self._eval_functional_element(self.Fvec_BA[data_index,0], self.initial_params_dict))
        if self.debug:
            print("\nbundle_adjust: current_fit_to_measurements (shown as transpose):")
            print(str(current_fit_to_measurements.T))
        current_error = self.X_BA - current_fit_to_measurements
        print("\ncurrent error (this is before the iterations):")
        print(current_error.flatten().tolist()[0])
        current_error_norm = numpy.linalg.norm(self.X_BA - current_fit_to_measurements)
        error_norm_with_iteration.append(current_error_norm)
        current_error_norm_per_measurement = current_error_norm / math.sqrt(num_measurements)
        print("\n\ncurrent error norm per measurement before iterations: %s" % str(current_error_norm_per_measurement))
        if self.display_function is not None and self.problem.startswith("sfm"):
            predicted_pixel_coordinates = current_fit_to_measurements.flatten().tolist()[0]
            predicted_pixels = [(predicted_pixel_coordinates[2*x], predicted_pixel_coordinates[2*x+1]) for x in range(len(predicted_pixel_coordinates) // 2)]
            self.display_function(predicted_pixels, None, current_error_norm_per_measurement)
        if current_error_norm_per_measurement < 1e-9:
            print("\nCurrent error norm: %.10f" % current_error_norm)
            print('''\nLooks like your initial choices for the parameters are perfect. '''
                  '''Perhaps there is nothing to be gained by invoking nonlinear least-squares '''
                  '''on your problem.''')
            sys.exit(1)
        error_norm_per_measurement_with_iteration.append(current_error_norm_per_measurement)
        # Next we need to calculate what L&A refer to as \epsilon_ij, which is the error associated with
        # the i-th point in the j_th camera.  Note that the first two elements of "current_error" is for the 
        # first world point in the first camera. I believe that this 2-element vector would be \epsilon_11.
        # The next two elements of "current_error" are for the first element in the second camera. These 
        # would be represented by \epsilon_12, and so on.
        epsilon_array    =   [[None for _ in range(num_cameras)] for _ in range(num_world_points)]
        current_error_as_list = current_error.flatten().tolist()[0]
        epsilons_arranged_by_points = [current_error_as_list[pt*2*num_cameras : (pt+1)*2*num_cameras] for pt in range(num_world_points)]
        if self.debug2:
            print("\n\nepsilons_arranged_by_points: %s" % str(epsilons_arranged_by_points))
        for point_index in range(num_world_points):
            for cam_index in range(num_cameras):
                epsilon_array[point_index][cam_index] = numpy.matrix([epsilons_arranged_by_points[point_index][2*cam_index],
                                                                epsilons_arranged_by_points[point_index][2*cam_index+1]]).T
        if self.debug2:
            print("\n\nepsilon_ij array of vectors:")
            for point_index in range(num_world_points):
                for cam_index in range(num_cameras):
                    print(epsilon_array[point_index][cam_index])

        new_param_values = new_fit_to_measurements = new_error_norm = None
        iteration_index = 0
        ##  We now define for each camera two matrices that are denoted A and B in the paper by Lourakis and Argyros.  
        ##  There is an A matrix for each point and each camera, as is the case with the B matrices also.  We refer 
        ##  to the array of all A matrices as the Argyros array.  And we refer to the array of all B matrices as the
        ##  Lourakis array:
        Argyros_array    =   [[None for _ in range(num_cameras)] for _ in range(num_world_points)]
        Lourakis_array   =   [[None for _ in range(num_cameras)] for _ in range(num_world_points)]
        productive_iteration_index_values = []
        best_estimated_structure = best_error_norm = None
        need_fresh_jacobian_flag = True
        need_sanity_check = False
        self.alambda = None
        self.rho = None
        alambda = None
        rho = None
#        self.debug2 = True
        self.debug2 = False
        iterations_used = None
        for iteration_index in range(self.max_iterations): 
            if need_fresh_jacobian_flag is True:
                if need_sanity_check is True:
                    print("\n\n|||||||||||||||||||||||| entering sanity checking code ||||||||||||||||||||||||||||||||||||")
                    if num_cameras > 6 or num_world_points > 9:
                        sys.exit("It is best to run sanity check on cases involving less than six cameras and less than nine points")
                    jacobian = numpy.asmatrix(numpy.zeros((num_measurements, num_params), dtype=float))
                    params_dict_local = {params_list[i] : current_param_values[i] for i in range(num_params)}
                    for i in range(num_measurements):
                        for j in range(num_params):
                            param = self.params_arranged_list[j]
                            evaled_element1 = self._eval_functional_element(self.Fvec_BA[i,0], params_dict_local)
                            incremented_params_dict_local = {param : params_dict_local[param] for param in params_dict_local}
                            incremented_params_dict_local[param] = incremented_params_dict_local[param] + delta_for_jacobian
                            evaled_element2 = self._eval_functional_element(self.Fvec_BA[i,0], incremented_params_dict_local)
                            jacobian[i,j] = (eval(evaled_element2) - eval(evaled_element1)) / delta_for_jacobian
                    print("\n\nJacobian:")
                    print(jacobian)
                    print("\nsize of the jacobian: %s" % str(jacobian.shape))
                    sanity_A = jacobian.T * jacobian
                    sanity_g = jacobian.T * current_error

                    sanity_JtJ_max = max(sanity_A.diagonal().tolist()[0])
                    print("\n\nmax on the diagonal on J^T.J: %s" % str(sanity_JtJ_max))
                    print("|||||||||||||||||||||||| exiting sanity checking code ||||||||||||||||||||||||||||||||||||\n\n\n")

                print("\n\n---------------------------Running SBA in iteration: %d---------------\n" % iteration_index)
                params_dict_local = {params_list[i] : current_param_values[i] for i in range(num_params)}
                for point_index in range(num_world_points):            
                    for cam_index in range(num_cameras):
                        params_for_cam =  params_for_camera_dict[cam_index]
                        A_matrix_for_cam_and_point = numpy.asmatrix(numpy.zeros(shape=(2, len(params_for_cam))))
                        B_matrix_for_cam_and_point = numpy.asmatrix(numpy.zeros(shape=(2, 3)))  # 2 for (x,y), 3 for (X,Y,Z)
                        x_cord_prediction = self.Fvec_BA[2*num_cameras*point_index + 2*cam_index,0]
                        y_cord_prediction = self.Fvec_BA[2*num_cameras*point_index + 2*cam_index + 1,0]
                        for param_index,param in enumerate(params_for_cam):
                            evaled_x_cord_predi = eval(self._eval_functional_element(x_cord_prediction, params_dict_local))
                            evaled_y_cord_predi = eval(self._eval_functional_element(y_cord_prediction, params_dict_local))
                            incremented_params_dict_local = {param : params_dict_local[param] for param in params_dict_local}
                            incremented_params_dict_local[param] += delta_for_jacobian
                            incremented_evaled_x_cord_predi = eval(self._eval_functional_element(x_cord_prediction,
                                                                                incremented_params_dict_local))
                            incremented_evaled_y_cord_predi = eval(self._eval_functional_element(y_cord_prediction,
                                                                                incremented_params_dict_local))
                            A_matrix_for_cam_and_point[0,param_index] = (incremented_evaled_x_cord_predi -  
                                                       evaled_x_cord_predi) / delta_for_jacobian
                            A_matrix_for_cam_and_point[1,param_index] = (incremented_evaled_y_cord_predi - 
                                                       evaled_y_cord_predi) / delta_for_jacobian
                        Argyros_array[point_index][cam_index] = A_matrix_for_cam_and_point
                for cam_index in range(num_cameras):
                    for point_index in range(num_world_points):            
                        B_matrix_for_cam_and_point = numpy.asmatrix(numpy.zeros(shape=(2, 3)))  # 2 for (x,y), 3 for (X,Y,Z)
                        x_cord_prediction = self.Fvec_BA[2*num_cameras*point_index + 2*cam_index,0]
                        y_cord_prediction = self.Fvec_BA[2*num_cameras*point_index + 2*cam_index + 1,0]
                        evaled_x_cord_predi = eval(self._eval_functional_element(x_cord_prediction, params_dict_local))
                        evaled_y_cord_predi = eval(self._eval_functional_element(y_cord_prediction, params_dict_local))
                        for param_index,param in enumerate(structure_params[point_index*3:point_index*3+3]):
                            incremented_params_dict_local = {param : params_dict_local[param] for param in params_dict_local}
                            incremented_params_dict_local[param] += delta_for_jacobian
                            incremented_evaled_x_cord_predi = eval(self._eval_functional_element(x_cord_prediction,
                                                                                incremented_params_dict_local))
                            incremented_evaled_y_cord_predi = eval(self._eval_functional_element(y_cord_prediction,
                                                                                incremented_params_dict_local))
                            B_matrix_for_cam_and_point[0,param_index] = (incremented_evaled_x_cord_predi -  
                                                                          evaled_x_cord_predi) / delta_for_jacobian
                            B_matrix_for_cam_and_point[1,param_index] = (incremented_evaled_y_cord_predi - 
                                                                          evaled_y_cord_predi) / delta_for_jacobian
                        Lourakis_array[point_index][cam_index] = B_matrix_for_cam_and_point
                if self.debug2:
                    print("\n\nShowing all A matrices (the Argyros array of matrices):")
                    for point_index in range(num_world_points):
                        for cam_index in range(num_cameras):
                            print(Argyros_array[point_index][cam_index])     
                    print("\n\nShowing all B matrices (the Lourakis array of matrices):")
                    for point_index in range(num_world_points):
                        for cam_index in range(num_cameras):
                            print(Lourakis_array[point_index][cam_index])     
                ## We now estimate the Jacobian from the A and the B matrices computed.  We need to do so
                ## in order to initialize the value of mu in the LM algorithm:
                BAjacobian = numpy.asmatrix(numpy.zeros((num_measurements, num_params), dtype=float))
                row_band_size = 2*num_cameras
                for i in range(num_measurements):
                    for j in range(num_camera_params):
                        row_band_index = i // (2*num_cameras)
                        within_rb_index   = i % (2*num_cameras)
                        row_index_for_matrix = within_rb_index // 2                    
                        ii = i // 2
                        jj = j // 6
                        if row_index_for_matrix == jj:
                            m = i%2
                            n = j%6
                            BAjacobian[i,j] = Argyros_array[row_band_index][jj][m,n]  
                    for j in range(3):
                        row_band_index = i // (2*num_cameras)
                        within_rb_index   = i % (2*num_cameras)
                        jj = num_cameras*6 + row_band_index * 3
                        m = i%2
                        n = j%3
                        BAjacobian[i,jj+j] = Lourakis_array[row_band_index][within_rb_index//2][m,n]  
                print("\n\nBAjacobian:")
                print(BAjacobian)
                print("\nsize of the BAjacobian: %s" % str(BAjacobian.shape))

                if need_sanity_check is True:
                    print("\n\n|||||||||||||||||||||||| entering sanity checking code ||||||||||||||||||||||||||||||||||||")
                    try:
                        assert numpy.array_equal(jacobian, BAjacobian), \
                           "the sanity check based on exact equality failed --- will try approximate for equality"
                    except:
                        for row in range(num_measurements):
                            for col in range(num_params):
                                if abs(jacobian[row,col] - BAjacobian[row,col]) > 1e-9:
                                    sys.exit("SANITY check failed even in the approximate sense for row=%d  col=%d" %(row,col))
                    print("\n\n|||||||||||||||||||||||| exiting sanity checking code ||||||||||||||||||||||||||||||||||||")
                if iteration_index == 0:
                    BA_JtJ =  BAjacobian.T * BAjacobian
                    BA_diag_max = max(BA_JtJ.diagonal().tolist()[0])
                    print("\n\nmax on the diagonal on J^T.J: %s" % str(BA_diag_max))
                    self.alambda = BA_diag_max / 1000
                    alambda = self.alambda                        
                    print("\n\nWe start with alambda = %f" % self.alambda)
                #  This will serve the same purpose as the g vector for the LM algo
                g_BA  =  BAjacobian.T * current_error
                ##  Now we create the U and V arrays:
                U_array = [numpy.asmatrix(numpy.zeros(shape=(6,6))) for _ in range(num_cameras)]  
                V_array = [numpy.asmatrix(numpy.zeros(shape=(3,3))) for _ in range(num_world_points)] 
                for cam_index in range(num_cameras):
                    for point_index in range(num_world_points):
                        U_array[cam_index] += Argyros_array[point_index][cam_index].T * Argyros_array[point_index][cam_index]
                for point_index in range(num_world_points):
                    for cam_index in range(num_cameras):
                        V_array[point_index] += Lourakis_array[point_index][cam_index].T * Lourakis_array[point_index][cam_index]
                if self.debug2:
                    print("\n\nUarray:") 
                    print(U_array)
                    print("\n\nVarray:")
                    print(V_array) 
                W_array    =   [[None for _ in range(num_cameras)] for _ in range(num_world_points)]
                for cam_index in range(num_cameras):
                    for point_index in range(num_world_points):
                        W_array[point_index][cam_index]  =  Argyros_array[point_index][cam_index].T * \
                                                                           Lourakis_array[point_index][cam_index]
                if self.debug2:
                    print("\n\nShowing all W_array:")
                    for point_index in range(num_world_points):
                        for cam_index in range(num_cameras):
                            print(W_array[point_index][cam_index])
                ##  Now we need to compute \epsilon_a_j for the j-th camera
                error_cam_param      = [numpy.asmatrix(numpy.zeros(shape=(6,1))) for _ in range(num_cameras)]
                ##  and \epsilon_b_i for the i-th point
                error_struct_param   = [numpy.asmatrix(numpy.zeros(shape=(3,1))) for _ in range(num_world_points)]
                for cam_index in range(num_cameras):
                    for point_index in range(num_world_points):            
                        error_cam_param[cam_index] += Argyros_array[point_index][cam_index].T * \
                                                                             epsilon_array[point_index][cam_index]
                for point_index in range(num_world_points):
                    for cam_index in range(num_cameras):
                        error_struct_param[point_index]  +=  Lourakis_array[point_index][cam_index].T * \
                                                                             epsilon_array[point_index][cam_index]
                if self.debug2:
                    print("\n\nDisplaying error_cam_param:")
                    print(error_cam_param)
                    print("\n\nDisplaying error_struct_param:")
                    print(error_struct_param)

            ##  Now we need to augment each element of the square U_array and each element of the square V_array
            ##  by adding \mu to the diagonal:  (\mu in the paper is the same as alambda here)
            Ustar_array = [U_array[j].copy() for j in range(num_cameras)]  # if you have 6 cam params per cam
            Vstar_array = [V_array[i].copy() for i in range(num_world_points)] # for the 3 coordinates of a world p
            for cam_index in range(num_cameras):
                for i in range(6):
                    Ustar_array[cam_index][i,i] +=  alambda   
            for point_index in range(num_world_points):       
                for i in range(3):                ##   arrays
                    Vstar_array[point_index][i,i] += alambda
            if self.debug2:
                print("\n\n\nDisplaying Ustar array:")
                print(Ustar_array)
                print("\n\n\nDisplaying Vstar array:")
                print(Vstar_array)
            ##  Now let us calculate the Y array:
            Y_array    =   [[None for _ in range(num_cameras)] for _ in range(num_world_points)]
            for cam_index in range(num_cameras):
                for point_index in range(num_world_points):    
                    Y_array[point_index][cam_index]  =  W_array[point_index][cam_index] * self._pseudoinverse(Vstar_array[point_index])
            if self.debug2:
                print("\n\nDisplay the Y array of matrices:")
                print(Y_array)
            error_cam  =  [numpy.asmatrix(numpy.zeros(shape=(6,1))) for _ in range(num_cameras)]
            for cam_index in range(num_cameras):
                tempsum = numpy.asmatrix(numpy.zeros(shape=(6,1)))
                for point_index in range(num_world_points):    
                    tempsum += (Y_array[point_index][cam_index] * error_struct_param[point_index])
                error_cam[cam_index]  =   error_cam_param[cam_index]  -  tempsum
            S_array    =   [[None for _ in range(num_cameras)] for _ in range(num_cameras)]            
            for cam_index1 in range(num_cameras):
                for cam_index2 in range(num_cameras):        
                    tempsum2 = numpy.asmatrix(numpy.zeros(shape=(6,6)))
                    for point_index in range(num_world_points):                                     
                        tempsum2 += Y_array[point_index][cam_index1] * W_array[point_index][cam_index2].T
                    if cam_index1 == cam_index2:
                        S_array[cam_index1][cam_index2] = Ustar_array[cam_index1] - tempsum2
                    else:
                        S_array[cam_index1][cam_index2] = - tempsum2
            if self.debug2:
                print("\n\nThe S matrix:")
                print(S_array)
            # At this point S is a mxm matrix whose every element itself is a 6x6 matrix where 6 is for the
            # six camera parameters for each camera position.  m is the total number of camera positions.
            S = numpy.asmatrix(numpy.zeros(shape=(6*num_cameras, 6*num_cameras)))
            for i in range(num_cameras):
                for j in range(num_cameras):
                    for m in range(6):
                        for n in range(6):
                            S[i*6+m, j*6+n] = S_array[i][j][m,n] 
            if self.debug2:
                print("\n\nThe S matrix:")
                print(S)
            #  We now define a long vector \Delta_cam that is a column-wise concatenation of the all the
            #  camera specific \delta_a in the error_cam array:
            error_cam_concatenated = numpy.asmatrix(numpy.zeros(shape=(6*num_cameras,1)))
            for cam_index in range(num_cameras):
                error_cam_concatenated[cam_index*6:(cam_index+1)*6, 0]  =  error_cam[cam_index]
            if self.debug2:
                print("\n\nerror_cam_concatenated:")
                print(error_cam_concatenated)
            # Suppose \delta_a represents the next step size for the camera params for each camera.  It is a column 
            # vec with 6 elements. When we concatenate it for all m cameras, we get a 6*m element long \Delta_cam
            # that has the steps to take for all the camera parameters for all m cameras:
            Delta_cam  =  self._pseudoinverse(S) * error_cam_concatenated
            if self.debug2:
                print("\n\nThe calculated deltas for the 6 parameters for all camera positions:")
                print(Delta_cam)
            # Now break Delta_cam into camera specific portions because you are going to need them later:
            Delta_cam_array = [Delta_cam[cam_index*6:(cam_index+1)*6, 0] for cam_index in range(num_cameras)]
            if self.debug2:
                print("\n\nDelta_cam_array to show the individual camera components in Delta_cam")
                print(Delta_cam_array)
            # Next we need to calculate Delta_b for all the structure points. Delta_b is a column-wise concatenation
            # of world-point specific delta_b_i that we calculate in the following loop:
            Delta_b = numpy.asmatrix(numpy.zeros(shape=(3*num_world_points, 1))) 
            for point_index in range(num_world_points):
                tempsum = numpy.asmatrix(numpy.zeros(shape=(3,1)))                 
                for cam_index in range(num_cameras):
                    tempsum +=  W_array[point_index][cam_index].T *  Delta_cam_array[cam_index]
                Delta_b[point_index*3:(point_index+1)*3, 0] = self._pseudoinverse(Vstar_array[point_index]) * (error_struct_param[point_index] - tempsum)

            if self.debug2:
                print("\n\nDelta_b column vector:")
                print(Delta_b)
            Delta_all = numpy.asmatrix(numpy.zeros(shape=(6*num_cameras + 3*num_world_points, 1)))
            Delta_all[:6*num_cameras,0] = Delta_cam
            Delta_all[6*num_cameras:,0] = Delta_b
            Delta_all_as_list = Delta_all.flatten().tolist()[0]
            if self.debug2:
                print("\n\nDelta_all_as_list:                    %s" % str(Delta_all_as_list))
            new_delta_param = Delta_all
            if need_sanity_check is True:
                left_side_eqn_9  = sanity_A + alambda *  numpy.asmatrix(numpy.identity(num_params))
                sanity_B = self._pseudoinverse(sanity_A + alambda * numpy.asmatrix(numpy.identity(num_params)))
                sanity_new_delta_param = sanity_B * sanity_g
                print("\n\nThe delta in params as produced by LM: %s" % str(sanity_new_delta_param.flatten().tolist()[0]))
            new_param_values = list(map(lambda x,y:x+y, current_param_values, Delta_all_as_list))
            if self.debug2:
                print("\n\nnew parameter values:")
                print(new_param_values)
            new_params_dict = {params_list[i] : new_param_values[i] for i in range(num_params)}
            if self.debug2:
                print("\nnew_params_dict: %s" % str(sorted(new_params_dict.items())))
            new_fit_to_measurements = numpy.asmatrix(numpy.zeros_like(self.X_BA))
            for i in range(num_measurements):
                new_fit_to_measurements[i,0] = eval(self._eval_functional_element(self.Fvec_BA[i,0], new_params_dict))
            if self.debug2:
                print("\nnew_fit_to_measurements (shown as transpose):")
                print(str(new_fit_to_measurements.T))
            new_error =  self.X_BA - new_fit_to_measurements
            if self.debug2:
                print("\n\nnew_error:")
                print(new_error.T)
            epsilon_array    =   [[None for _ in range(num_cameras)] for _ in range(num_world_points)]
            current_error_as_list = current_error.flatten().tolist()[0]
            epsilons_arranged_by_points = [current_error_as_list[pt*2*num_cameras : (pt+1)*2*num_cameras] for pt in range(num_world_points)]
            for point_index in range(num_world_points):
                for cam_index in range(num_cameras):
                    epsilon_array[point_index][cam_index] = numpy.matrix([epsilons_arranged_by_points[point_index][2*cam_index],
                                                                epsilons_arranged_by_points[point_index][2*cam_index+1]]).T
            print("\nnew error at iteration %d:" %  iteration_index)
            print(new_error.flatten().tolist()[0])
            new_error_norm = numpy.linalg.norm(new_error)
            new_error_norm_per_measurement = new_error_norm / math.sqrt(len(self.X_BA))            
            print("\nnew error norm per measurement: %s" % str(new_error_norm_per_measurement))
            newly_projected_pixel_coordinates = new_fit_to_measurements.flatten().tolist()[0]
            newly_projected_pixels = [(newly_projected_pixel_coordinates[2*x], newly_projected_pixel_coordinates[2*x+1]) for x in range(len(newly_projected_pixel_coordinates) // 2)]
            rho = ( error_norm_with_iteration[-1] ** 2  - new_error_norm ** 2 ) / \
                                         (new_delta_param.T * g_BA    +    alambda * new_delta_param.T * new_delta_param)
            if rho <= 0.0:
                need_fresh_jacobian_flag = False
                alambda = alambda * max( [1.0/3.0, 1.0 - (2.0 * rho - 1.0) ** 3] )     ## BIZARRE that a matrix is returned
                alambda = alambda.tolist()[0][0]
                if alambda > 1e11:
                    print("\nIterations terminated because alambda exceeded limit") 
                    break
                print("\n\nThe current GN direction did not work out.  Will try a new direction.")
                if self.debug:
                    print("\nNO change in parameters for iteration_index: %d with alambda = %f" % (iteration_index, alambda))
                Ustar_array = Vstar_array = Delta_cam = Delta_b = Delta_all = None
                continue
            else:
                need_fresh_jacobian_flag = True
                print("\n\n====================  Showing Results for SBA ITERATION: %d  ===========================" 
                                                                % (iteration_index + 1))
                productive_iteration_index_values.append(iteration_index)
                alambda = self.alambda
                error_norm_per_measurement_with_iteration.append(new_error_norm_per_measurement)
                error_norm_with_iteration.append(new_error_norm)
                print("\n\nerror norms per measurement for all iterations: %s" % str(error_norm_per_measurement_with_iteration))
                current_param_values = new_param_values
                best_param_values = new_param_values
                best_predicted_pixels  = newly_projected_pixels
                iterations_used = iteration_index + 1
                if self.display_function is not None:
                    if self.problem.startswith("sfm"):
                        num_structure_points = int(self.problem.split("_")[1])
                        estimated_structure = current_param_values[-3*num_structure_points:]
                        estimated_structure = [estimated_structure[3*i:3*i+3] for i in range(num_structure_points)]
                        best_estimated_structure = estimated_structure
                        best_error_norm_per_measurement = new_error_norm_per_measurement
                        self.display_function(best_predicted_pixels, estimated_structure, new_error_norm_per_measurement, len(productive_iteration_index_values)-1)
        if self.debug:
            print("\nerror norms with iterations: %s" % str(error_norm_per_measurement_with_iteration))
#            print("\niterations used: %d" % (len(productive_iteration_index_values) - 1))
            print("\niterations used: %d" % iterations_used)
            print("\nproductive iteration index values: %s" % str(productive_iteration_index_values))
            print("\n\nfinal values for the parameters: ") 
            print(str(new_param_values))
        if self.debug is True and iteration_index == self.max_iterations - 1:
            print("\n\nWARNING: max iterations reached without getting to the minimum")
        if self.display_function is not None:
            if self.problem.startswith("sfm"):
                self.display_function(best_predicted_pixels, estimated_structure, new_error_norm_per_measurement, len(productive_iteration_index_values)-1)
            else:
                if len(productive_iteration_index_values) % 2 == 0:
                    self.display_function(new_fit_to_measurements, new_error_norm, len(productive_iteration_index_values)-1)
        result = {"error_norms_with_iterations" : error_norm_per_measurement_with_iteration,
#                  "number_of_iterations" : len(productive_iteration_index_values) - 1,
                  "number_of_iterations" : iterations_used,
                  "parameter_values" : new_param_values}
        return result



    ###%%%
    ##################################  Private Methods of NonlinearLeastSquares  ################################

    def _get_initial_params_from_file(self, filename):
        if not filename.endswith('.txt'): 
            sys.exit("Aborted. _get_initial_params_from_file() is only for CSV files")
        initial_params_dict = {}
#        initial_params_list = [line for line in [line.strip() for line in open(filename,"rU")] if line is not '']
        initial_params_list = [line for line in [line.strip() for line in open(filename,"rU")] if line != '']
        for record in initial_params_list:
            initial_params_dict[record[:record.find('=')].rstrip()] = float(record[record.find('=')+1:].lstrip())
        self.params_dict = initial_params_dict
        self.params_ordered_list = sorted(self.params_dict) if self.params_ordered_list is not None else self.params_arranged_list
        return initial_params_dict

    def _get_measured_data_from_text_file(self, filename):
        if not filename.endswith('.txt'): 
            sys.exit("Aborted. _get_measured_data_from_text_file() is only for txt files")
        all_data = list(map(float, open(filename).read().split()))
        if self.debug:
            print("_get_measured_data_from_text_file: all_data")
            print(str(all_data))
        X = numpy.matrix(all_data).T
        xnorm = numpy.linalg.norm(X)
        if self.debug:
            print("_get_measured_data_from_text_file:  norm of X: %s" % str(xnorm))  

    def _eval_functional_element(self, Fvec_element, params_dict):
        '''
        Evaluates one element of the prediction vector Fvec by substituting for its parameters
        (both camera and structure) the values as supplied by the dictionary params_dict
        This is the evaluation function used for the basic LM algorithm.
        '''
        augmented_element = Fvec_element
        for param in params_dict:
            regex = r'\b' + param + r'\b'         
            if isinstance(augmented_element, (bytes)):
                if re.search(regex, augmented_element.decode('utf-8')):
                    augmented_element = re.sub(regex, str(params_dict[param]), augmented_element.decode('utf-8'))
            else:
                if re.search(regex, augmented_element):
                    augmented_element = re.sub(regex, str(params_dict[param]), augmented_element)
        return augmented_element

    def _eval_functional_element2(self, Fvec_element, param_list, param_val_list):
        '''
        Although this method does basically the same thing as the previous method, this one 
        is meant for the sparse-bundle-adjustment implementation of LM.  Now the second argument, 
        param_val_list, is a list for the current values for the camera parameters --- BUT ONLY FOR 
        ONE CAMERA --- and for the structure parameters.  Note that this method was written assuming 
        that the second argument is a list as opposed to a dict.
        '''
        augmented_element = Fvec_element
        for i,param in enumerate(param_list):
            regex = r'\b' + param + r'\b'         
            if isinstance(augmented_element, (bytes)):
                if re.search(regex, augmented_element.decode('utf-8')):
                    augmented_element = re.sub(regex, str(param_val_list[i]), augmented_element.decode('utf-8'))
            else:
                if re.search(regex, augmented_element):
                    augmented_element = re.sub(regex, str(param_val_list[i]), augmented_element)
        return augmented_element


    def _pseudoinverse(self, A):
        return (A.T * A).I * A.T

