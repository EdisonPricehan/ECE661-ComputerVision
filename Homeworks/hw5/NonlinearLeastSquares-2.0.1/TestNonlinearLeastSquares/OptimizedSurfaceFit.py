#from NonlinearLeastSquares import NonlinearLeastSquares

##  OptimizedSurfaceFit.py
##  Main Module Version: 2.0.1

##  OptimizedSurfaceFit is a part of the Avi Kak's NonlinearLeastSquares Python module
##  for demonstrating how the module can be used for optimized fitting of a surface 
##  (whose analytical form is known) to noisy data over a plane.

##  The goal of OptimizedSurfaceFit is to fit a model surface to noisy height data
##  over the xy-plane in the xyz-coordinate frame, with the model surface being
##  described by a polynomial function.  Here are some examples of such polynomials:
##
##           "a*(x-b)**2 + c*(y-d)**2 + e"
##
##           "a*x**2 + c*y**2"
##
##           "a*x + b*y + c"
##
##  where the value returned by the polynomial for given values of the coordinate
##  pair (x,y) is the height above the xy-plane at that point.  Given the sort of a
##  model surface shown above, the problem becomes one of optimally estimating the
##  value for the model parameters from the noisy observed data.


import numpy
import numpy.linalg
import os,sys,glob
import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

numpy.set_printoptions(precision=3)

class OptimizedSurfaceFit(object):

    def __init__(self, *args, **kwargs):
        'constructor'                       
        if args:
            raise Exception('''The OptimizedSurfaceFit constructor can only be called with '''
                            '''the following keyword arguments: initial_param_values_file, '''
                            '''initial_param_values, model_functional_file, measured_data_file, '''  
                            '''model_functional, data_array_size, gen_data_synthetically, '''
                            '''partials_for_jacobian, datagen_functional, optimizer, '''
                            '''how_much_noise_for_synthetic_data,display_size,display_position''')
        allowed_keys = 'initial_param_values_file','initial_param_values','model_functional_file','measured_data_file','model_functional','data_array_size','gen_data_synthetically','partials_for_jacobian','datagen_functional','how_much_noise_for_synthetic_data','optimizer','display_needed','debug','display_size','display_position'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise Exception("Wrong key used in constructor call --- perhaps spelling error")
        initial_param_values=initial_param_values_file=model_functional=model_functional_file=measured_data_file=gen_data_synthetically=data_array_size=partials_for_jacobian=datagen_functional=how_much_noise_for_synthetic_data=optimizer=display_needed=display_size=display_position=debug=None
        if 'initial_param_values' in kwargs: initial_param_values=kwargs.pop('initial_param_values')
        if 'initial_param_values_file' in kwargs: initial_param_values_file=kwargs.pop('initial_param_values_file')
        if 'model_functional' in kwargs: model_functional=kwargs.pop('model_functional')
        if 'model_functional_file' in kwargs: model_functional_file=kwargs.pop('model_functional_file')
        if 'measured_data_file' in kwargs: measured_data_file=kwargs.pop('measured_data_file')
        if 'gen_data_synthetically' in kwargs: gen_data_synthetically=kwargs.pop('gen_data_synthetically')
        if 'data_array_size' in kwargs: data_array_size=kwargs.pop('data_array_size')
        if 'partials_for_jacobian' in kwargs: partials_for_jacobian=kwargs.pop('partials_for_jacobian')
        if 'datagen_functional' in kwargs: datagen_functional=kwargs.pop('datagen_functional')
        if 'optimizer' in kwargs: optimizer=kwargs.pop('optimizer') 
        if 'how_much_noise_for_synthetic_data' in kwargs: how_much_noise_for_synthetic_data=kwargs.pop('how_much_noise_for_synthetic_data')
        if 'display_needed' in kwargs: display_needed=kwargs.pop('display_needed')
        if 'display_size' in kwargs: display_size=kwargs.pop('display_size')
        if 'display_position' in kwargs: display_position=kwargs.pop('display_position')
        if 'debug' in kwargs: debug=kwargs.pop('debug') 
        if gen_data_synthetically and measured_data_file:
            raise Exception("You must choose either 'measured_data_file' or 'gen_data_synthetically' option in the constructor")
        if model_functional and model_functional_file:
            raise Exception("You must choose either 'model_functional_file' or 'model_functional' option in the constructor")
        if initial_param_values and initial_param_values_file:
            raise Exception("You must choose either 'initial_param_values_file' or 'initial_param_values' option in the constructor, but not both")
        if how_much_noise_for_synthetic_data is not None and datagen_functional is None:
            raise Exception("Using the constructor option 'how_much_noise_for_synthetic_data' when 'datagen_functional' is pointless") 
        self.debug = debug if debug is not None else False
        self.optimizer = optimizer
        self.display_needed = display_needed if display_needed is not None else False
        self.display_size = display_size
        self.display_position = display_position
        self.num_measurements = None
        if data_array_size:
            self.data_array_size = data_array_size
        else:
            raise Exception("The constructor must specify the intrinsic dimensionality of the data")
        if how_much_noise_for_synthetic_data:
            self.how_much_noise_for_synthetic_data = how_much_noise_for_synthetic_data
        else:
            self.how_much_noise_for_synthetic_data = 0.0
        if measured_data_file is True and datagen_functional is None:
            self.X = self._get_measured_data_from_text_file(measured_data_file)
        elif datagen_functional:
            self.X = self.gen_data(datagen_functional)
        else:
            raise Exception("You have neither provided a measured data file nor supplied a datagen functional")
        if initial_param_values:
            self.initial_params_dict = self.params_dict = initial_param_values
        else:
            self.initial_params_dict = self.params_dict = self._get_initial_params_from_file(initial_param_values_file) 
        self.params_ordered_list = sorted(self.params_dict)  
        self.num_parameters = len(self.params_ordered_list)
        if model_functional_file:
            self.model_functional = self.get_model_functional_from_file(model_functional_file)
        else:
            self.model_functional = model_functional
        self.partials_for_jacobian = partials_for_jacobian

    def set_constructor_options_for_optimizer(self, algo):
        '''
        This method conveys the following information from an instance of OptimizedSurfaceFit to an 
        instance of NonlinearLeastSquares:

            1)  The measurement vector X.
            2)  The initial values to be used for the parameters of the model function.
            3)  The functionals for the partial derivatives to be used in the Jacobian matrix, 
                assuming that the option 'partials_for_jacobian' was set in the instance of 
                OptimizedSurfaceFit that was created.
            4)  The Fvec vector, which is a vector of the predicted values, all in functional 
                form, for each of the data elements in the measurement vector X.
            5)  The display function to be used for plotting the partial and the final results if
                such results can be displayed in the form of a 2D or 3D graphic with Python's 
                matplotlib library.

        and some additional book-keeping information.
        '''
        self.optimizer = algo
        algo.set_X(self.X)
        algo.set_initial_params(self.initial_params_dict)
        if self.partials_for_jacobian:
            algo.set_jacobian_functionals_array(self.construct_jacobian_array_in_functional_form())
        algo.set_Fvec(self.construct_Fvec())
        if self.display_needed:
            algo.set_display_function(self.display_function)
        algo.set_params_ordered_list(self.initial_params_dict)
        algo.set_num_measurements(self.num_measurements)
        algo.set_num_parameters(self.num_parameters)
        algo.set_debug(self.debug)

    def construct_jacobian_array_in_functional_form(self):
        '''
        The method returns a Jacobian matrix in its functional form (with respect to
        the parameters).  The size of the matrix is N x p  where N is the total number
        of measurements and p is the total number of parameters.
        '''
        spatial_res_x = 1.0 / self.data_array_size[0]
        spatial_res_y = 1.0 / self.data_array_size[1]
        if self.debug:
            print("\n partial derivative functionals: %s" % str(sorted(self.partials_for_jacobian.items())))
        jacob_array = numpy.chararray((self.num_measurements, self.num_parameters), itemsize=100)
        ij = itertools.product(range(self.data_array_size[0]), range(self.data_array_size[1]))
        for i in range(numpy.prod(self.data_array_size)):
            x_index = i // self.data_array_size[0]          
            y_index = i - x_index * self.data_array_size[0]
            for p in range(self.num_parameters):
                param = self.params_ordered_list[p]
                jacob_functional_for_element = self.partials_for_jacobian[param]
                jacob_functional_for_element = str.replace(jacob_functional_for_element, 'x', str(x_index * spatial_res_x))
                jacob_functional_for_element = str.replace(jacob_functional_for_element, 'y', str(y_index * spatial_res_y))
                jacob_array[i,p] = jacob_functional_for_element
        return jacob_array
   
    def calculate_best_fitting_surface(self, algo):
        if algo == 'lm':
            result_dict = self.optimizer.leven_marq()
        elif algo == 'gd':
            result_dict = self.optimizer.grad_descent()
        return result_dict

    def get_initial_params_from_file(self, filename):
        if not filename.endswith('.txt'): 
            sys.exit("Aborted. _get_initial_params_from_file() is only for CSV files")
        initial_params_dict = {}
#        initial_params_list = [line for line in [line.strip() for line in open(filename,"rU")] if line is not '']
        initial_params_list = [line for line in [line.strip() for line in open(filename,"rU")] if line != '']
        for record in initial_params_list:
            initial_params_dict[record[:record.find('=')].rstrip()] = float(record[record.find('=')+1:].lstrip())
        self.params_dict = initial_params_dict
        self.params_ordered_list = sorted(self.params_dict)             #need ordered list for jacobian calculations
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

    def gen_data(self, datagen_functional):
        d1,d2 = self.data_array_size 
        X1,X2 = None,None
        delta_x = 1.0 / d1
        delta_y = 1.0 / d2
        datagen_functional = str.replace(datagen_functional, 'x', 'i * ' + str(delta_x))
        datagen_functional = str.replace(datagen_functional, 'y', 'j * ' + str(delta_y))
        X1 = numpy.fromfunction(lambda i,j: eval(datagen_functional), (d1,d2), dtype=float)
        if self.how_much_noise_for_synthetic_data > 0.0:
            X2 = self.how_much_noise_for_synthetic_data *  (numpy.random.random_sample((d1,d2)) - 0.5)
        X = X1 + X2 if X2 is not None else X1
        X = numpy.asmatrix(X)
        X = X.reshape((numpy.prod(self.data_array_size),1))
        if self.debug:
            print("\ngen_data: X as a vector (shown as the transpose of the data vector): ") 
            print(str(X.T))
        X_as_list = X.reshape((1,numpy.prod(self.data_array_size))).tolist()[0]
        self.num_measurements = len(X_as_list)
        FILE = open("newdata.txt", 'w')
        list(map(FILE.write, list(map(lambda x: "%.6f  " % x, X_as_list))))
        return X

    def get_model_functional_from_file(self, filename):
        if not filename.endswith('.txt'): 
            sys.exit("Aborted. _get_model_functional() is only for txt files")
        model_functional = open(filename).read().strip()
        return model_functional

    def construct_Fvec(self):
        ''' 
        This is a vector of the same size as the number of measurements.  Each element of the
        vector is the model functional instantiated with the (x,y) coordinates that correspond
        to that measurement.  Recall, the noisy data is measured over a d0xd1 grid in the
        (x,y)-plane.  The grid is assumed to sample a unit square in the first quadrant of the
        plane.  That is, we assume that both the x and the y coordinates range over the
        interval (0,1).
        '''
        spatial_res_x = 1.0 / self.data_array_size[0]
        spatial_res_y = 1.0 / self.data_array_size[1]
        self.spatial_res_x = spatial_res_x
        self.spatial_res_y = spatial_res_y
        F = numpy.chararray((self.data_array_size[0], self.data_array_size[1]), itemsize=100)
        ij = itertools.product(range(self.data_array_size[0]), range(self.data_array_size[1]))
        for _, (i,j) in enumerate(ij):
            functional_for_element = self.model_functional
            functional_for_element = str.replace(functional_for_element, 'x', str(i * spatial_res_x))
            functional_for_element = str.replace(functional_for_element, 'y', str(j * spatial_res_y))
            F[i,j] = functional_for_element
        self.Fvec = F.reshape((numpy.prod(self.data_array_size),1))
        if self.debug: 
            print("\nValue of Fvec:")
            print(str(self.Fvec))
        return self.Fvec

    def display_function(self, new_fit_to_measurements, new_error_norm, iteration_index):
        new_fit_to_measurements_as_arr = numpy.asmatrix(new_fit_to_measurements.reshape((self.data_array_size[0],self.data_array_size[1])))
        xx,yy = numpy.meshgrid(numpy.linspace(0, 1, self.data_array_size[0]),  numpy.linspace(0, 1, self.data_array_size[1]))
        zz = [[new_fit_to_measurements_as_arr[i,j] for j in range(self.data_array_size[0])] 
                                                                      for i in range(self.data_array_size[1])]
        # Generate points for the scatter plot of the actually measured data points
        xxx = xx.reshape((1,self.data_array_size[0]*self.data_array_size[1])).tolist()[0]
        yyy = yy.reshape((1,self.data_array_size[0]*self.data_array_size[1])).tolist()[0]
        zzz = self.X.reshape((1,self.data_array_size[0]*self.data_array_size[1])).tolist()[0]
        zmax = numpy.max(zz) if numpy.max(zz) >= numpy.max(zzz) else numpy.max(zzz)
        zmin = numpy.min(zz) if numpy.min(zz) <= numpy.min(zzz) else numpy.min(zzz)
        if self.display_size is not None: 
            w,h = self.display_size
            fig = plt.figure(figsize=(w,h))
        else:
            fig = plt.figure()
        new_error_norm_str = "%.4f" % new_error_norm
        fig.suptitle("iteration: " + str(iteration_index) + "     error: " + 
                                                     new_error_norm_str, fontsize=14, fontweight='bold')
        ##   ax = fig.gca(projection='3d')                
        ax = fig.add_subplot(111, projection='3d')  ## Version 2.0.1 fix for the above statement
        ## Version 1.5.0 fix:  xx, yy, and zz needed to be turned into numpy arrays
#        surf = ax.plot_surface(xx, yy, zz, color="yellow")
        surf = ax.plot_surface(numpy.array(xx), numpy.array(yy), numpy.array(zz), color="yellow")
        ax.set_zlim(zmin, zmax)
        ax.scatter(xxx,yyy,zzz, c='r', marker='o')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.savefig("figs/figure_" + str(iteration_index))
        if self.display_position is not None: 
            try:
                manager = plt.get_current_fig_manager()
                pos = self.display_position
                manager.window.wm_geometry('+' + str(pos[0]) + '+' + str(pos[1]))
            except: pass
        plt.show()
