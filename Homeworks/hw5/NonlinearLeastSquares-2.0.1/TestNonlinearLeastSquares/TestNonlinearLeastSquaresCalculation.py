import NonlinearLeastSquares
import OptimizedSurfaceFit
import unittest

class TestNonlinearLeastSquaresCalculation(unittest.TestCase):

    def setUp(self):
        print("Testing nonlinear least-squares")
        self.optimizer = NonlinearLeastSquares.NonlinearLeastSquares(max_iterations = 10,
                                            delta_for_jacobian = 0.000001,
                                            delta_for_step_size = 0.00001)
        self.surface_fitter = OptimizedSurfaceFit.OptimizedSurfaceFit(
                            gen_data_synthetically = True,
                            datagen_functional = "7.8*x + 2.2*y + 1",
                            data_array_size = (8,8), 
                            how_much_noise_for_synthetic_data = 0.5,
                            model_functional = "a*x + c*y + e",
                            initial_param_values = {'a':0.0, 'c':0.0, 'e':0.0},
                         )

        self.surface_fitter.set_constructor_options_for_optimizer(self.optimizer)

    def test_nonlinear_least_squares_calculation(self):
        result = self.surface_fitter.calculate_best_fitting_surface('gd')
        final_param_values = result['parameter_values']
        final_param_values = final_param_values.flatten().tolist()[0]
        self.assertTrue(len(final_param_values) == 3)

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestNonlinearLeastSquaresCalculation, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

