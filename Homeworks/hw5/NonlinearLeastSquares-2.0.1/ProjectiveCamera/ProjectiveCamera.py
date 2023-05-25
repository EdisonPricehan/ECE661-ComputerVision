#!/usr/bin/env python

##  ProjectiveCamera.py

##  Main Module Version: 2.0.1

##  This class is a part of Avi Kak's Python module named NonlinearLeastSquares.  The purpose of this 
##  class is to demonstrate how Nonlinear Least Squares can be used to estimate the scene structure 
##  from the motion of the camera.  That is, you move the camera to different positions (and, if 
##  desired, different orientations) and record the pixels at each position.  It is relatively easy
##  to reconstruct the scene from the pixels thus recorded --- especially if you know the camera
##  parameters.  This class simulates the camera and then transforms the pixel data recorded and 
##  the projection functionals for each camera position into a form that can be used by the main 
##  NonlinearLeastSquares class for estimating the scene structure.

import numpy                                                                    
import numpy.linalg
import scipy
import sys,os,glob
import math
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore")

#numpy.set_printoptions(precision=3,suppress=True)

def euclidean(point1, point2):
    '''
    inhomogeneous 3D coordinates expected
    '''
    distance = math.sqrt(reduce(lambda x,y: x+y, map(lambda x: x**2, [(point1[i] - point2[i]) for i in range(len(point1))])))
    return distance

class ProjectiveCamera(object):
    '''    
    The parameters 'alpha_x' and 'alpha_y' are for the focal length in terms of the image 
    sampling intervals along the x-axis and along the y-axis, respectively.   The parameters 
    'x0' and 'y0' are for the coordinates of the point in the camera image plane where the 
    optic axis penetrates the image plane with respect to the origin in the image plane 
    (which is usually a corner of the image):
    '''
    def __init__(self, *args, **kwargs):
        if args:
            raise ValueError(
                    '''Camera constructor can only be called with keyword arguments for the follwoing 
                       keywords: camera_type,alpha_x,alpha_y,x0,y0,camera_rotation,
                       camera_translation, partials_for_jacobian, display_needed''')
        allowed_keys = 'camera_type','alpha_x','alpha_y','x0','y0','camera_rotation','camera_translation','partials_for_jacobian','display_needed'
        keywords_used = kwargs.keys()
        for keyword in keywords_used:
            if keyword not in allowed_keys:
                raise Exception("Wrong key used in constructor call --- perhaps spelling error")
        camera_type=alpha_x=alpha_y=x0=y0=camera_rotation=camera_translation=partials_for_jacobian=display_needed=None
        if 'camera_type' in kwargs:                camera_type = kwargs.pop('camera_type')
        if 'alpha_x' in kwargs:                    alpha_x = kwargs.pop('alpha_x')
        if 'alpha_y' in kwargs:                    alpha_y = kwargs.pop('alpha_y')
        if 'x0' in kwargs:                         x0 = kwargs.pop('x0')
        if 'y0' in kwargs:                         y0 = kwargs.pop('y0')
        if 'camera_rotation' in kwargs:            camera_rotation = kwargs.pop('camera_rotation')            
        if 'camera_translation' in kwargs:         camera_translation = kwargs.pop('camera_translation')
        if kwargs: 
            raise ValueError("You have used unrecognized keyword for supplying constructor arguments") 
        if camera_type is not None and camera_type in ('scaled_orthographic', 'projective'): 
            self.camera_type = camera_type
        else:
            raise ValueError("""You must specify the camera type and it must be either """
                             """'scaled_orthographic' or 'projective'""") 
        if alpha_x is not None:
            self.alpha_x = alpha_x
        else:
            self.alpha_x = 1.0
        if alpha_y is not None:
            self.alpha_y = alpha_y
        else:
            self.alpha_y = 1.0
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = 0.0
        if y0 is not None:
            self.y0 = y0
        else:
            self.y0 = 0.0
        if camera_rotation is not None:
            self.camera_rotation = camera_rotation
        else:
            self.camera_rotation = numpy.asmatrix(numpy.diag([1.0, 1.0, 1.0]))
        if camera_translation is not None:
            self.camera_translation = camera_translation
        else:
            self.camera_translation = numpy.matrix([0.0, 0.0, 0.0])
        self.partials_for_jacobian = partials_for_jacobian
        self.display_needed = display_needed if display_needed is not None else False
        self.num_measurements = None
        self.list_of_cameras = {}   # A 'camera' means a "camera position'. The keys of the dict enumerate the positions
        self.motion_history =  []
        self.structure_ground_truth = None  
        self.initial_values_for_structure = None
        self.all_params_list = None
        self.initial_val_all_params = None
        self.initial_param_vals_dict = None
        self.num_structure_points = 0
        self.num_world_points  = 0
        self.num_cameras = 0
        self.tracked_point_indexes_for_display = None
        self.debug = True

    def initialize(self):
        # Camera intrinsic param matrix:
        self.K = numpy.matrix([[self.alpha_x, 0, self.x0], [0, self.alpha_y, self.y0], [0, 0, 1.0]])
        self.K_inverse = self.K.I
        print("\nThe K matrix:\n")
        print(self.K)
        print("\nThe inverse of the K matrix:\n")
        print(self.K_inverse)
        # Camera matrix:
        self.P = self.K * numpy.append(self.camera_rotation, self.camera_translation.T, 1)
        print("Camera matrix P produced by the initializer:")
        print(self.P)
        #  We represent the pose by a list of 6 items, with the first three standing for the Rodrigues rotation 
        #  and the last three representing the translations along the world-X, world-Y, and world-Z.
        self.P_initial = self.P.copy()
        self.pose = ['nan','nan','nan', 0.0, 0.0, 0.0]

    def set_num_world_points(self, num):
        self.num_structure_points = num

    def set_num_cameras(self, num):
        self.num_cameras = num

    def print_camera_matrix(self):
        print("\n\nCamera matrix:")
        print(self.P)

    def get_pixel_for_world_point(self, world_point):
        '''
        World point expected in homogeneous coordinates as a 4-item list
        '''
        world_point_as_one_row_matrix = numpy.matrix(world_point)
        xvec = self.P * world_point_as_one_row_matrix.T
        xvec = xvec.T
        x_aslist = xvec[0,:].tolist()[0]
        x,y = int(x_aslist[0]/x_aslist[2]), int(x_aslist[1]/x_aslist[2])
        print("\nworld point: %s" % str(world_point))
        print("pixels: %s" % str((x,y)))
        return x,y

    def get_pixels_for_a_sequence_of_world_points(self, world_points):
        '''
        Each item in the list of world_points needs to be a 4-item list for the
        homegeneous coords of the point in question
        '''
        image_pixels = []
        for point in world_points:
            pixel = self.get_pixel_for_world_point(point)
            image_pixels.append(pixel)
        return image_pixels

    def rotate_previously_initialized_camera_around_world_X_axis(self, theta):
        '''
        We use the (roll,pitch,yaw) convention for describing the camera rotation, with the 
        rotation around the world X-axis as the roll, around the Y-axis as the pitch, and
        around the Z-axis as the yaw. The rotation angle theta needs to be in degrees.     
        '''
        cos_theta =  scipy.cos( theta * scipy.pi / 180 )                                        
        sin_theta =  scipy.sin( theta * scipy.pi / 180 )         
        rot_X = numpy.matrix([[1.0,0.0,0.0],[0.0,cos_theta,-sin_theta],[0.0,sin_theta,cos_theta]])
        left_3by3 =  self.P[0:3,0:3]
        last_col   = self.P[:,3]
        new_left_3by3 = left_3by3 * rot_X
        self.P = numpy.append(new_left_3by3, last_col, 1)
        self.motion_history.append(['rotate_X', theta])

    def rotate_previously_initialized_camera_around_world_Y_axis(self, theta):
        '''
        We use the (roll,pitch,yaw) convention for describing the camera rotation, with the 
        rotation around the world X-axis as the roll, around the Y-axis as the pitch, and
        around the Z-axis as the yaw. The rotation angle theta needs to be in degrees.     
        '''
        cos_theta =  scipy.cos( theta * scipy.pi / 180 )                                        
        sin_theta =  scipy.sin( theta * scipy.pi / 180 )         
        rot_Y = numpy.matrix([[cos_theta,0.0,sin_theta],[0.0,1.0,0.0],[-sin_theta,0.0,cos_theta]])
#        rot_Y = numpy.matrix([[cos_theta,0.0,-sin_theta],[0.0,1.0,0.0],[sin_theta,0.0,cos_theta]])
        left_3by3 =  self.P[0:3,0:3]
        last_col   = self.P[:,3]
        new_left_3by3 =  left_3by3 * rot_Y
        self.P = numpy.append(new_left_3by3, last_col, 1)
        self.motion_history.append(['rotate_Y', theta])

    def translate_a_previously_initialized_camera(self, translation):
        '''
        The parameter 'translation' is a 3-element list, with the first element indicating the
        translation along X, the second along Y, and the third along Z.
        '''
        left_3by3 = self.P[0:3,0:3]
        last_column   =  self.P[:,3]
        new_last_column = last_column + self.K * numpy.asmatrix(translation).T
        self.P = numpy.append(left_3by3, new_last_column, 1)
        self.motion_history.append(['translate', translation])

    def add_new_camera_to_list_of_cameras(self):
        how_many_already = len(self.list_of_cameras)
        self.list_of_cameras[how_many_already] = self.P
        self.num_cameras += 1

    def get_all_cameras(self):
        return self.list_of_cameras

    def get_current_camera_pose(self):
        P = self.P_initial.copy()
        left_3by3 = self.P[0:3,0:3]        
        last_column   =  self.P[:,3]
        camera_translation = self.K.I * last_column
        t_x,t_y,t_z = camera_translation[:,0].flatten().tolist()[0]
        R = self.K.I * left_3by3                            # this is the camera rotation matrix R
        w_x,w_y,w_z  = self.contruct_rodrigues_rotation([R[0,0],R[0,1],R[0,2],R[1,0],R[1,1],R[1,2],R[2,0],R[2,1],R[2,2]])
        new_pose_R_and_t = None
        if len(self.motion_history) == 0:
            return w_x,w_y,w_z,t_x,t_y,t_z
        else:
            for pose_change in self.motion_history:
                left_3by3 = P[0:3,0:3]
                R = self.K.I * left_3by3                            # this is the camera rotation matrix R
                last_column  =  P[:,3]
                camera_translation = (self.K.I * last_column).flatten().tolist()[0]
                if pose_change[0] == 'translate':
                    new_translate = numpy.matrix(list(map(lambda x,y:x+y, camera_translation, pose_change[1])))
                    t_current = new_translate
                    R_current = self.K.I * left_3by3
                    new_pose_R_and_t = numpy.append(R_current, new_translate.T, 1)
                    new_pose = numpy.append(new_pose_R_and_t, [[0,0,0,1]], 0)
                    P = numpy.append(left_3by3,  self.K * new_translate.T, 1)
                elif pose_change[0] == 'rotate_X':
                    if pose_change[1] > 1e-6:
                        R_current = self.K.I * left_3by3
                        t_current = self.K.I * last_column
                        cam_pose_matrix = numpy.append(R_current, t_current, 1)
                        cam_pose_matrix = numpy.append(cam_pose_matrix, [[0,0,0,1]], 0)
                        new_pose = self._rotate_cam_frame_around_X_axis(cam_pose_matrix, pose_change[1])   # new_pose is 4x4
                        new_pose_R_and_t = new_pose[:3,]
                        P  =  self.K * new_pose_R_and_t
                elif pose_change[0] == 'rotate_Y':
                    if pose_change[1] > 1e-6:
                        R_current = self.K.I * left_3by3
                        t_current = self.K.I * last_column
                        cam_pose_matrix = numpy.append(R_current, t_current, 1)
                        cam_pose_matrix = numpy.append(cam_pose_matrix, [[0,0,0,1]], 0)
                        new_pose = self._rotate_cam_frame_around_Y_axis(cam_pose_matrix, pose_change[1])
                        new_pose_R_and_t = new_pose[:3,]
                        P  =  self.K * new_pose_R_and_t
                elif pose_change[0] == 'rotate_Z':
                    if pose_change[1] > 1e-6:
                        R_current = self.K.I * left_3by3
                        t_current = self.K.I * last_column
                        cam_pose_matrix = numpy.append(R_current, t_current, 1)
                        new_pose = self._rotate_cam_frame_around_Z_axis(cam_pose_matrix, pose_change[1])
                        new_pose_R_and_t = new_pose[:3,]
                        P  =  self.K * new_pose_R_and_t
            if new_pose_R_and_t is None:
                return w_x,w_y,w_z,t_x,t_y,t_z       
            else:
                R = new_pose_R_and_t[:,0:3]
                t = new_pose_R_and_t[:,3]
                t_x,t_y,t_z = t.flatten().tolist()[0]
                w_x,w_y,w_z  = self.contruct_rodrigues_rotation([R[0,0],R[0,1],R[0,2],R[1,0],R[1,1],R[1,2],R[2,0],R[2,1],R[2,2]])
                if self.debug:
                    print("\n\nget_current_cam_pose: Printing the 6-tuple for the cam position: %s" % str([w_x,w_y,w_z,t_x,t_y,t_z]))
                return w_x,w_y,w_z,t_x,t_y,t_z

    def set_structure_parameters_to_ground_truth(self, structure_params, world_point_coords):
        gt_dict = {structure_params[i] : 0.0 for i in range(len(structure_params))}
        ground_truth_for_structure = []
        for point_index in range(len(world_point_coords)):
            gt_dict['X_' + str(point_index)] = world_point_coords[point_index][0]
            gt_dict['Y_' + str(point_index)] = world_point_coords[point_index][1]
            gt_dict['Z_' + str(point_index)] = world_point_coords[point_index][2]
#            ground_truth_for_structure += world_point_coords[point_index]
        self.ground_truth_for_structure = ground_truth_for_structure
        return gt_dict

    def set_all_parameters_to_ground_truth_for_sanity_check(self, world_point_coords, camera_gt):
        '''
        This method is useful for sanity checks in an implementation of sparse bundle adjustment.  When you 
        set the camera parameters to their ground truth values, the optimizer should not wander off from 
        those values in its search for the optimum.  Ideally, this would be case if you set ALL the parameters 
        to their ground-truth values.
        '''
#        all_params = self.p.flatten().tolist()[0]
        all_params = self.all_params_list
        number_of_cams = len(self.list_of_cameras)
        all_camera_params = all_params[:6*number_of_cams]
        structure_params =  all_params[6*number_of_cams:]
        gt_dict = {all_params[i] : 0.0 for i in range(len(all_params))}
        ground_truth_for_structure = []
        for cam in range(number_of_cams):
            gt_dict['w_' + str(cam) + str('_x')] = camera_gt[cam][0]
            gt_dict['w_' + str(cam) + str('_y')] = camera_gt[cam][1]
            gt_dict['w_' + str(cam) + str('_z')] = camera_gt[cam][2]
            gt_dict['t_' + str(cam) + str('_x')] = camera_gt[cam][3]
            gt_dict['t_' + str(cam) + str('_y')] = camera_gt[cam][4]    
            gt_dict['t_' + str(cam) + str('_z')] = camera_gt[cam][5]    
        for point_index in range(len(world_point_coords)):
            gt_dict['X_' + str(point_index)] = world_point_coords[point_index][0]
            gt_dict['Y_' + str(point_index)] = world_point_coords[point_index][1]
            gt_dict['Z_' + str(point_index)] = world_point_coords[point_index][2]
            ground_truth_for_structure += world_point_coords[point_index]
        self.ground_truth_for_structure = ground_truth_for_structure
        return gt_dict

    def set_initial_values_for_structure(self, values_list):
        '''
        This method makes it more convenient to pass the initial values to the display function 
        that is invoked by the NonlinearLeastSquares class.
        '''
        self.initial_values_for_structure = values_list

    def set_3D_generic_transform_for_3D_scene_objects(self):
        rot3D = numpy.matrix([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])    
        trans3D = numpy.matrix([0.0,0.0,0.0])
        transform = numpy.append(rot3D, trans3D.T, 1)
        self.scene_transform_3D = numpy.append(transform, [[0,0,0,1]], 0)

    def pixels_on_a_line_between_two_image_points(self, line, N):
        '''
        The parameter `line' is a tuple of two points (point1, point2) where the
        points point1 and point2 are expected to be in homogeneous coordinates.  That
        is, each is a triple.  Returns N points between the two given points.
        '''
        pixels = []
        point1,point2 = line
        x_span = point2[0] - point1[0]
        y_span = point2[1] - point1[1]     
        w_span = point2[2] - point1[2]     
        del_x = x_span / (N + 1)
        del_y = y_span / (N + 1)
        del_w = w_span / (N + 1)
        for i in range(1,N+1):
            pixels.append( [point1[0] + i*del_x, point1[1] + i*del_y, point1[2] + i*del_w] ) 
        return pixels

    def points_on_a_line_between_two_world_points(self, line, N, world_points):
        '''
        The parameter `line' is a tuple of two world points (point1, point2) in HOMOGENEOUS
        COORDINATES.  That is, each point is a 4-vector.  Returns N points between the two 
        given points, INCLUDING THE END POINTS.
        '''
        point1,point2 = line
        x_span = point2[0] - point1[0]
        y_span = point2[1] - point1[1]     
        z_span = point2[2] - point1[2]     
        w_span = point2[3] - point1[3]     
        del_x = x_span / N
        del_y = y_span / N
        del_z = z_span / N
        for i in range(N+1):
            new_point = [point1[0] + i*del_x, point1[1] + i*del_y, point1[2] + i*del_z, 1]
            if not self.is_point_in_a_list_of_points(new_point, world_points):
                world_points.append( new_point )
        return world_points

    def is_point_in_a_list_of_points(self, point, list_of_points):
        x,y,z,w = point
        xx,yy,zz = x/w,y/w,z/w
        for pt in list_of_points:
            xxx,yyy,zzz,www = pt
            xxxx,yyyy,zzzz = xxx/www,yyy/www,zzz/www 
            if (abs(xx-xxxx) < 0.5) & (abs(yy-yyyy) < 0.5) & (abs(zz-zzzz) < 0.5):
                return True
            else:
                continue
        return False

    def make_world_points_from_tetrahedron_generic(self, points_per_line):
        '''
        The 'points_per_line' is the number of points you want on each edge of the tetra
        '''
        xz_point1 = (100,0,0,1)
        xz_point2 = (200,0,200,1)
        xz_point3 = (300,0,0,1)       
        xz_point4 = (200,0,-200,1)       
        y_apex_1  = (200,200,0,1)
        lines = []
        # base of the tetrahedron:
        line1 = (xz_point1, xz_point2)
        line2 = (xz_point2, xz_point3)
        line3 = (xz_point3, xz_point4)
        line4 = (xz_point4, xz_point1)
        lines += [line1,line2,line3,line4]
        # lines from the apex to the base corners:
        line5 = (xz_point1, y_apex_1)
        line6 = (xz_point2, y_apex_1)
        line7 = (xz_point3, y_apex_1)
        line8 = (xz_point4, y_apex_1)
        lines += [line5,line6,line7,line8]
        world_points = []
        for line in lines:
            world_points = self.points_on_a_line_between_two_world_points(line,points_per_line - 1, world_points)
        print("\nworld points: %s" % str(world_points))
        print("\nnumber of world points: %d" % len(world_points))
        self.num_world_points = len(world_points)
        self.world_points = world_points
        return world_points

    def make_world_points_for_triangle(self):
        xy_point1 = (3000.0,3000.0,0.0,1.0)
        xy_point2 = (4000.0,3000.0,0.0,1.0)
        xy_point3 = (4000.0,5000.0,0.0,1.0)       
        world_points = [xy_point1, xy_point2, xy_point3]
        print("\nworld points: %s" % str(world_points))
        print("\nnumber of world points: %d" % len(world_points))
        self.num_world_points = len(world_points)
        self.world_points = world_points
        return world_points

    def make_world_points_for_double_triangle(self):
        # triangle 1
        point1 = (3000.0,3000.0,0.0,1.0)
        point2 = (4000.0,3000.0,0.0,1.0)
        point3 = (4000.0,5000.0,0.0,1.0)       
        #triangle 2
        point4 = (3000.0,3000.0,500.0,1.0)
        point5 = (4000.0,3000.0,500.0,1.0)
        point6 = (4000.0,5000.0,500.0,1.0)       
        world_points = [point1,point2,point3,point4,point5,point6]
#        print("\nworld points: %s" % str(world_points))
        print("\nnumber of world points: %d" % len(world_points))
        self.num_world_points = len(world_points)
        self.world_points = world_points
        return world_points

    def make_world_points_random(self, how_many):
#        assert how_many % 3 == 0, "For random generation, make the number of world point a multiple of 3"
        world_points = []
        for _ in range(how_many):
#            world_points.append( (random.randint(0,1000), random.randint(0,1000), random.randint(0,1000), 1) )
            world_points.append( (random.randint(0,10000), random.randint(0,10000), random.randint(0,10000), 1) )
        print("\nnumber of world points: %d" % len(world_points))
        self.num_world_points = len(world_points)
        self.world_points = world_points
        return world_points

    def apply_transformation_to_generic_world_points(self, points, rotation, translation, scale):
        '''
        The parameter 'rotation' is a triple of rotation angles in degrees around the world X-axis, the 
        world Y-axis and the world Z-axis, respectively.  The parameter 'translation' is a triple of real 
        numbers that are the translations of the object along the world-X, the translation along world-Y, 
        and the translation along the world-Z.  IMPORTANT:  When an object undergoes these rotations, 
        it is important to realize that they are NOT with respect to a local coordinate frame centered
        on the object.  On the other hand, they are with respect to the world frame. Say that is an
        object point on the positive side of the world-Z axis.  When you rotate this object through 
        90 degrees with respect to the world-Y axis, that point will move to a location on the positive
        world-X axis.
        '''
        (rotx,roty,rotz) = rotation
        (tx,ty,tz) = translation
        (scx, scy, scz) = (scale, scale, scale)
        self.set_3D_generic_transform_for_3D_scene_objects()
        self._rotate_3D_scene_around_world_X_axis(rotx)
        self._rotate_3D_scene_around_world_Y_axis(roty)
        self._rotate_3D_scene_around_world_Z_axis(rotz)
        self._translate_3D_scene(translation)
        print("\n3D transform for the object:\n")
        print(self.scene_transform_3D)
        self._scale_3D_scene(scale)
        world_points_xformed_homogeneous = []
        for point in points:
            world_points_xformed_homogeneous.append(self.scene_transform_3D * numpy.matrix(point).T)
        world_points_xformed = []
        for pt in world_points_xformed_homogeneous:
            world_points_xformed.append( pt[:,0].flatten().tolist()[0] )
        print("\nworld points transformed: %s" % str(world_points_xformed))
        self.world_points_xformed = world_points_xformed
        return world_points_xformed

    def display_structure_and_pixels(self, predicted_pixels=None, scene_structure=None, reprojection_error=None, iteration_index=None):
        try:
            measured_pixel_coords =  self.X.flatten().tolist()[0]
        except:
            measured_pixel_coords =  self.X_BA.flatten().tolist()[0]
        measured_pixels = [(measured_pixel_coords[2*x], measured_pixel_coords[2*x+1]) for x in range(len(measured_pixel_coords) // 2)]
        if self.tracked_point_indexes_for_display is not None:
            measured_pixels  =  [measured_pixels[x] for x in self.tracked_point_indexes_for_display]
            predicted_pixels  = [predicted_pixels[x] for x in self.tracked_point_indexes_for_display]
        fig = plt.figure(1)
        ax = fig.add_subplot(121)
        subplt1 = plt.subplot(121)
        plt.title("measured pixels in red, predicted pixels in blue",  fontsize=14, fontweight='bold')
        xm_coords = [pixel[0] for pixel in measured_pixels]
        ym_coords = [pixel[1] for pixel in measured_pixels]
        subplt1.plot(xm_coords, ym_coords, 'ro')
        xp_coords = [pixel[0] for pixel in predicted_pixels]
        yp_coords = [pixel[1] for pixel in predicted_pixels]
        subplt1.plot(xp_coords, yp_coords, 'bx')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
#        plt.subplots_adjust(hspace = .5)                     # h stands for height space between the two subplots
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        ax2 = plt.subplot(122, projection='3d')
        if self.tracked_point_indexes_for_display is None:
            ground_truth=self.structure_ground_truth if self.structure_ground_truth else None
            initial_values=self.initial_values_for_structure if self.initial_values_for_structure else None
        else:
            ground_truth= [self.structure_ground_truth[x] for x in self.tracked_point_indexes_for_display]
            initial_values = [self.initial_values_for_structure[x] for x in self.tracked_point_indexes_for_display]
        if reprojection_error is not None:
            error_str = "%.4f" % reprojection_error
            ax2.set_title("iteration: " + str(iteration_index) + "     reproj. error: " + error_str, fontsize=14, fontweight='bold')
        if iteration_index is not None:
            ax2.set_title("iteration: " + str(iteration_index), fontsize=14, fontweight='bold')
        else:
            ax2.set_title("The Ground Truth and Initial Guess", fontsize=14, fontweight='bold')
        XI=YI=ZI=XG=YG=ZG=XM=YM=ZM=None
        for triangle_index in range(len(ground_truth)//3):
            if initial_values is not None:
                XI = [point[0] for point in initial_values[3*triangle_index:3*triangle_index+3]]
                YI = [point[1] for point in initial_values[3*triangle_index:3*triangle_index+3]]
                ZI = [point[2] for point in initial_values[3*triangle_index:3*triangle_index+3]]
                XI.append(XI[0])
                YI.append(YI[0])
                ZI.append(ZI[0])
            if ground_truth is not None:
                XG = [point[0] for point in ground_truth[3*triangle_index:3*triangle_index+3]] 
                YG = [point[1] for point in ground_truth[3*triangle_index:3*triangle_index+3]]
                ZG = [point[2] for point in ground_truth[3*triangle_index:3*triangle_index+3]]
                XG.append(XG[0])
                YG.append(YG[0])
                ZG.append(ZG[0])
            if scene_structure is not None:
                XM = [point[0] for point in scene_structure[3*triangle_index:3*triangle_index+3]]
                YM = [point[1] for point in scene_structure[3*triangle_index:3*triangle_index+3]]
                ZM = [point[2] for point in scene_structure[3*triangle_index:3*triangle_index+3]]
                XM.append(XM[0])
                YM.append(YM[0])
                ZM.append(ZM[0])
            if XI is not None:
                ax2.plot(XI,YI,ZI, 'xm-', label='initial guess')
            if XG is not None:
                ax2.plot(XG,YG,ZG, 'xr-', label='ground truth')
            if XM is not None:
                ax2.plot(XM,YM,ZM, 'xb-', label='scene structure')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        plt.show()



    def display_structure(self, scene_structure=None, reprojection_error=None, iteration_index=None):
        '''
        Each of the three parameters is a list of coordinate triples in World 3D.  The first parameter
        is for the coordinate triples for the estimated scene structure, the second list for the ground 
        truth, and the third for the initial guesses supplied.  The third parameter stands for the point 
        in parameter hyperplane for starting the downhill path to the optimum solution for the scene
        structure.
        '''
        if self.tracked_point_indexes_for_display is None:
            ground_truth=self.structure_ground_truth if self.structure_ground_truth else None
            initial_values=self.initial_values_for_structure if self.initial_values_for_structure else None
        else:
            ground_truth= [self.structure_ground_truth[x] for x in self.tracked_point_indexes_for_display]
            initial_values = [self.initial_values_for_structure[x] for x in self.tracked_point_indexes_for_display]

        fig = plt.figure()
        if reprojection_error is not None:
            error_str = "%.4f" % reprojection_error
            fig.suptitle("iteration: " + str(iteration_index) + "     reproj. error: " + error_str, fontsize=14, fontweight='bold')
        else:
            if iteration_index is not None:
                fig.suptitle("iteration: " + str(iteration_index), fontsize=14, fontweight='bold')
            else:
                fig.suptitle("The Ground Truth and Initial Guess", fontsize=14, fontweight='bold')
        ax = fig.gca(projection='3d')
        XI=YI=ZI=XG=YG=ZG=XM=YM=ZM=None
        for triangle_index in range(len(ground_truth)//3):
            if initial_values is not None:
                XI = [point[0] for point in initial_values[3*triangle_index:3*triangle_index+3]]
                YI = [point[1] for point in initial_values[3*triangle_index:3*triangle_index+3]]
                ZI = [point[2] for point in initial_values[3*triangle_index:3*triangle_index+3]]
                XI.append(XI[0])
                YI.append(YI[0])
                ZI.append(ZI[0])
            if ground_truth is not None:
                XG = [point[0] for point in ground_truth[3*triangle_index:3*triangle_index+3]] 
                YG = [point[1] for point in ground_truth[3*triangle_index:3*triangle_index+3]]
                ZG = [point[2] for point in ground_truth[3*triangle_index:3*triangle_index+3]]
                XG.append(XG[0])
                YG.append(YG[0])
                ZG.append(ZG[0])
            if scene_structure is not None:
                XM = [point[0] for point in scene_structure[3*triangle_index:3*triangle_index+3]]
                YM = [point[1] for point in scene_structure[3*triangle_index:3*triangle_index+3]]
                ZM = [point[2] for point in scene_structure[3*triangle_index:3*triangle_index+3]]
                XM.append(XM[0])
                YM.append(YM[0])
                ZM.append(ZM[0])
            if XI is not None:
                ax.plot(XI,YI,ZI, 'xm-', label='initial guess')
            if XG is not None:
                ax.plot(XG,YG,ZG, 'xr-', label='ground truth')
            if XM is not None:
                ax.plot(XM,YM,ZM, 'xb-', label='scene structure')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def display_world_points_double_triangles(self, scene_structure):
        '''
        Each of the three parameters is a list of coordinate triples in World 3D.  The first parameter
        is for the coordinate triples for the estimated scene structure, the second list for the ground 
        truth, and the third for the initial guesses supplied.  The third parameter stands for the point 
        in parameter hyperplane for starting the downhill path to the optimum solution for the scene
        structure.
        '''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for triangle_index in range(len(scene_structure)//3):
            XM = [point[0] for point in scene_structure[3*triangle_index:3*triangle_index+3]]
            YM = [point[1] for point in scene_structure[3*triangle_index:3*triangle_index+3]]
            ZM = [point[2] for point in scene_structure[3*triangle_index:3*triangle_index+3]]
            XM.append(XM[0])
            YM.append(YM[0])
            ZM.append(ZM[0])
            ax.plot(XM,YM,ZM, 'xr-', label='world points')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def display_structure_double_triangles_final_results(self, estimated_structure):
        '''
        Each of the three parameters is a list of coordinate triples in World 3D.  The first parameter
        is for the coordinate triples for the estimated scene structure, the second list for the ground 
        truth, and the third for the initial guesses supplied.  The third parameter stands for the point 
        in parameter hyperplane for starting the downhill path to the optimum solution for the scene
        structure.
        '''
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for triangle_index in range(len(estimated_structure)//3):
            XM = [point[0] for point in estimated_structure[3*triangle_index:3*triangle_index+3]]
            YM = [point[1] for point in estimated_structure[3*triangle_index:3*triangle_index+3]]
            ZM = [point[2] for point in estimated_structure[3*triangle_index:3*triangle_index+3]]
            XM.append(XM[0])
            YM.append(YM[0])
            ZM.append(ZM[0])
            ax.plot(XM,YM,ZM, 'xb-', label='estimated structure')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()


    #####################   package the objects needed by the NonlinearLeastSquares class   ####################
 
    def construct_X_vector(self, pixel_coordinates):
        '''
        The function parameter 'pixel_coordinates' is a list of lists (LoL), with each list in LoL being 
        the set of pixels recorded from one position and orientation of the camera.
        '''
        #  Since 'pixel_coordinates' is a LIST OF LISTS, with each list being for one camera pos, 
        #  we can say:
#        self.display_structure_and_pixels(pixel_coordinates[0], pixel_coordinates[1])
#        sys.exit("delib in construct X")
        self.num_cameras = len(pixel_coordinates)   
        print("\nnumber of camera positions: %d" % self.num_cameras)        
        X = []
        for pixels_for_one_camera_pos in pixel_coordinates:      
            for pixel in pixels_for_one_camera_pos:
                X += pixel
        self.X_size = len(X)
        self.num_measurements = len(X)
        self.X = numpy.matrix(X).T


    def construct_X_vector_for_bundle_adjustment(self, pixel_coordinates):
        '''
        The function parameter 'pixel_coordinates' is a list of lists (LoL), with each list in LoL being 
        the set of pixels recorded from one position and orientation of the camera.

        For sparse bundle adjustment logic, the elements of the measurement vector X need to be arranged 
        differently from what is accomplished by the previous method.  You need to arrange the pixels in 
        the world-point order as opposed to the camera order.  To explan, let's say you have m cameras and 
        n world points. Your first 2m elements in the X vector will be for THE FIRST WORLD POINT in ALL 
        of the m cameras.  Of these 2m elements, the first two elements will be for the x- and the 
        y-coordinates of the first world point in the first camera.  The next pair of two elements will be 
        for the SAME world point but in the second camera, and so on.
        '''
        #  Since 'pixel_coordinates' is a LIST OF LISTS, with each list being for one camera pos, 
        #  we can say:
        print("\nnumber of camera positions: %d" % self.num_cameras)        
        X = []
        for point_index in range(self.num_world_points):
            for cam_index in range(self.num_cameras):
                X += pixel_coordinates[cam_index][point_index]
        self.X_size = len(X)
        self.num_measurements = len(X)
        self.X_BA = numpy.matrix(X).T

    def construct_structure_ground_truth(self):
        structure_ground_truth_dict = {}
        structure_ground_truth = []
        for (i,point) in enumerate(self.world_points_xformed):
            X,Y,Z,W = point
            var1, var2, var3 = 'X_' + str(i), 'Y_' + str(i), 'Z_' + str(i)
            structure_ground_truth_dict[var1] = float(X) / W
            structure_ground_truth_dict[var2] = float(Y) / W
            structure_ground_truth_dict[var3] = float(Z) / W
            structure_ground_truth.append([structure_ground_truth_dict[var1], structure_ground_truth_dict[var2], structure_ground_truth_dict[var3]])
        self.structure_ground_truth = structure_ground_truth
        return structure_ground_truth_dict

    def construct_Fvec(self):
        ''' 
        This method is for demonstrating the most general case of estimating the camera matrix directly
        for each position of the camera WITHOUT placing any R-induced constraints on the elements of the
        matrix --- which is NEVER a good thing to do but nonethless useful for educational purposes.

        This method constructs the Prediction Vector for the observed data in "self.X". This is a vector 
        of the same size as the number of measurements in "self.X". The elements of the Prediction Vector 
        are functionals involving the parameters to be estimated.

        Think of p_c as your 3x4 camera projection matrix for the camera indexed 'c' and p_c_ij its 
        ij-th element.
        '''
        functional_x =  '(p_c_11*X_ + p_c_12*Y_ + p_c_13*Z_ + p_c_14) / (p_c_31*X_ + p_c_32*Y_ + p_c_33*Z_ + p_c_34)' 
        functional_y =  '(p_c_21*X_ + p_c_22*Y_ + p_c_23*Z_ + p_c_24) / (p_c_31*X_ + p_c_32*Y_ + p_c_33*Z_ + p_c_34)' 
        Fvec = []
        for i in range(self.num_cameras):
            functional_x_for_cam = str.replace( functional_x, 'c', str(i) )
            functional_y_for_cam = str.replace( functional_y, 'c', str(i) )
            for j in range(self.num_world_points):
                tempx = str.replace( functional_x_for_cam, 'X_', 'X_' + str(j))
                tempx = str.replace( tempx, 'Y_', 'Y_' + str(j))
                tempx = str.replace( tempx, 'Z_', 'Z_' + str(j))
                Fvec.append( tempx )
                tempy = str.replace( functional_y_for_cam, 'X_', 'X_' + str(j))
                tempy = str.replace( tempy, 'Y_', 'Y_' + str(j))
                tempy = str.replace( tempy, 'Z_', 'Z_' + str(j))
                Fvec.append( tempy )
        self.Fvec = numpy.matrix(Fvec).T
        print("\n\nprinting Fvec:")
        print(self.Fvec)
        return self.Fvec

    def construct_Fvec_for_calibrated_cameras(self, camera_params_dict):
        ''' 
        This method constructs the Prediction Vector for the observed data in "self.X". This is a vector 
        of the same size as the number of measurements in "self.X". The elements of the Prediction Vector 
        are functional involving the parameters to be estimated.
        '''
        functional_x =  '(p_c_11*X_ + p_c_12*Y_ + p_c_13*Z_ + p_c_14) / (p_c_31*X_ + p_c_32*Y_ + p_c_33*Z_ + p_c_34)' 
        functional_y =  '(p_c_21*X_ + p_c_22*Y_ + p_c_23*Z_ + p_c_24) / (p_c_31*X_ + p_c_32*Y_ + p_c_33*Z_ + p_c_34)' 
        Fvec = []
        for i in range(self.num_cameras):
            functional_x_for_cam = str.replace( functional_x, 'c', str(i) )
            functional_y_for_cam = str.replace( functional_y, 'c', str(i) )
            for j in range(self.num_world_points):
                tempx = str.replace( functional_x_for_cam, 'X_', 'X_' + str(j))
                tempx = str.replace( tempx, 'Y_', 'Y_' + str(j))
                tempx = str.replace( tempx, 'Z_', 'Z_' + str(j))
                Fvec.append( tempx )
                tempy = str.replace( functional_y_for_cam, 'X_', 'X_' + str(j))
                tempy = str.replace( tempy, 'Y_', 'Y_' + str(j))
                tempy = str.replace( tempy, 'Z_', 'Z_' + str(j))
                Fvec.append( tempy )
        for i in range(len(Fvec)):
            for param in camera_params_dict:
                Fvec[i] = str.replace( Fvec[i], param, str(camera_params_dict[param]) )
        self.Fvec = numpy.matrix(Fvec).T
        print("\n\nprinting Fvec:")
        print(self.Fvec)
        return self.Fvec

    def contruct_rodrigues_rotation(self, R):
        ''' 
        The parameter R is a **list** of elements of the rotation matrix R in the form 
        [r_11, r_12, r_13, r_21, r_22, r_23, r_31, r_32, r_33].  This method returns a 3-element
        Rodrigues vector in the form of a **list** [w_x, w_y, w_z]  
        '''
        R_trace = R[0] + R[4] +	R[8]
        phi = scipy.arccos( (R_trace - 1) / 2.0 )
        if phi < 1e-6:
            return 'nan','nan','nan'
        phi_over_2sinphi = phi / (2.0 * scipy.sin(phi))
        w_x = phi_over_2sinphi *  (R[7] - R[5])
        w_y = phi_over_2sinphi *  (R[2] - R[6])	
        w_z = phi_over_2sinphi *  (R[3] - R[1])
        return w_x,w_y,w_z
    

    def construct_R_from_rodrigues_rotation(self, w):
        '''
        The parameter w is a **list** of three elements that is the Rodrigues vector 
        representation of the rotation matrix R.  It returns R in the form of a **list**
        [r_11, r_12, r_13, r_21, r_22, r_23, r_31, r_32, r_33] of the elements of R.
        '''
        if w == ('nan','nan','nan'):
            return (1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
        w_vec = numpy.matrix(w).T            # construct a one-column matrix for w
        w_vec_crossprod_matrix  = numpy.asmatrix([ [  0.0,  -w[2],   w[1]  ],
                                                   [  w[2],  0.0,   -w[0]  ],
                                                   [ -w[1],  w[0],   0.0   ] ] )
        phi = numpy.linalg.norm(w_vec)
        sinphi_over_phi = scipy.sin(phi) / phi
        one_minus_cosphi_over_phi_squared = (1.0 - scipy.cos(phi)) / (phi ** 2)
        R = numpy.asmatrix(numpy.diag([1.0,1.0,1.0]))    +  sinphi_over_phi * w_vec_crossprod_matrix   \
                                +   one_minus_cosphi_over_phi_squared * w_vec_crossprod_matrix ** 2
        R = R.flatten().tolist()[0]
        return R
    
    def construct_Fvec_for_uncalibrated_cameras_with_known_intrinsic_params_no_rodrigues(self):
        ''' 
        We assume that the intrinsic parameter matrix K is known, but the extrinsic parameters for
        each of the camara positions is NOT known.  However, we place no R-induced constraints when
        calculating the extrinsic parameters --- WHICH IS NEVER A SAFE THING TO DO but nonetheless
        useful for educational demonstrations.  "No R-induced" stands for "no Rodrigues".

        This method constructs the Prediction Vector for the observed data in "self.X". This is a vector 
        of the same size as the number of measurements in "self.X". The elements of the Prediction Vector 
        are functionals involving the parameters to be estimated.

        Since we know the intrinsic parameters of the camera, we know (p_x,p_y).  These are the coordinates
        of the center of the image frame where the optic axis penetrates origin in the camera image plane.
        Notice how we replace p_x by the intrinsic element K[0,2] and p_y by K[1,2] when we construct
        functional_x_for_cam and functional_y_for_cam which are the x- and the y-coordinates of the
        predicted position of the pixel for a given world point.

        As can be seen from the constructions for functional_x_for_cam and functional_y_for_cam, 
        the meaning of f_x is the same as K[0,0] and that of f_y the same as K[1,1] where K is the
        3x3 intrinsic matrix for for the camera.

        r_c is the 3x3 rotation matrix and t_c the 3-element translational vector for the c-th camera.
        '''
        functional_x =  '(  f_x   * (r_c_11*X_ + r_c_12*Y_ + r_c_13*Z_ + t_c_x)  +             \
                            alpha * (r_c_21*X_ + r_c_22*Y_ + r_c_23*Z_ + t_c_y)  +             \
                            p_x   * (r_c_31*X_ + r_c_32*Y_ + r_c_33*Z_ + t_c_z)   )  /         \
                                    (r_c_31*X_ + r_c_32*Y_ + r_c_33*Z_ + t_c_z)'
        functional_y =  '(  f_y   * (r_c_21*X_ + r_c_22*Y_ + r_c_23*Z_ + t_c_y)  +             \
                            p_y   * (r_c_31*X_ + r_c_32*Y_ + r_c_33*Z_ + t_c_z)   )  /         \
                                    (r_c_31*X_ + r_c_32*Y_ + r_c_33*Z_ + t_c_z)'

        ##  Our goal now is to construct a prediction for each of the observed pixels in all of
        ##  the camera positions. The functional_x shown above is a generic form of this prediction
        ##  for the horizontal coordinate of a pixel and functional_y for the vertical coordinate
        ##  in the the c-th camera in terms of the rotation matrix r_c and the translation vector
        ##  t_c for the camera.
        Fvec = []
        for i in range(self.num_cameras):
            functional_x_for_cam = str.replace( functional_x, 'c', str(i) )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'f_x',   str(self.K[0,0]) )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'alpha', str(self.K[0,1]) )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'p_x',   str(self.K[0,2]) )
            functional_y_for_cam = str.replace( functional_y, 'c', str(i) )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'f_y',   str(self.K[1,1]) )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'p_y',   str(self.K[1,2]) )

            for j in range(self.num_world_points):
                tempx = str.replace( functional_x_for_cam, 'X_', 'X_' + str(j))
                tempx = str.replace( tempx, 'Y_', 'Y_' + str(j))
                tempx = str.replace( tempx, 'Z_', 'Z_' + str(j))
                Fvec.append( tempx )
                tempy = str.replace( functional_y_for_cam, 'X_', 'X_' + str(j))
                tempy = str.replace( tempy, 'Y_', 'Y_' + str(j))
                tempy = str.replace( tempy, 'Z_', 'Z_' + str(j))
                Fvec.append( tempy )
        self.Fvec = numpy.matrix(Fvec).T
        print("\n\nprinting Fvec:")
        print(self.Fvec)
        return self.Fvec

    def construct_Fvec_for_uncalibrated_cameras_with_known_intrinsic_params_and_with_rodrigues_rotations(self):
        ''' 
        This method constructs the Prediction Vector for the observed data in "self.X". This is a vector 
        of the same size as the number of measurements in "self.X". The elements of the Prediction Vector 
        are functional involving the parameters to be estimated.

        Recall that the elements of self.X alternate between the x- and the y-coordinates for all the pixels
        observed in all of the camera positions.  So if you are tracking N world points in M camera images,
        the size of self.X is 2NM.

        Since we know the intrinsic parameters of the camera, we know (p_x,p_y).  These are the coordinates
        of the center of the image frame where the optic axis penetrates with the image origin in the 
        camera image plane.  As mentioned earlier, the meaning of p_x is the same as that for K[0,2] and
        of p_y the same as for K[1,2] in the 3x3 intrinsic parameter matrix K.

        You need to place each element that eventually is replaced by a numerical value by a parenthesized form.
        If you don't, you end up with things like '+-0.45634**2' which confuses the math processor.

        In what follows, w_c is the 3-element Rodrigues vector that represents for the 3x3 rotation matrix
        r_c for the c-th camera.  The three elements of w_c are denoted (w_c_x, w_c_y, w_c_z).  We start with
        how to get the nine elements of the matrix r_c from the 3 elements of the vector w_c.

        '''
#       r_11  =  wx**2             +      (1-wx**2)*cphi                          (shown without normalization)
        r_c_11  =  '((w_c_x)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)  +  (1 - ((w_c_x)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))' 
#       r_12  =  wx*wy*(1-cphi)    -      wz*sphi
        r_c_12  =  '((w_c_x)*(w_c_y)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  -   ((w_c_z) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)) * scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_13  =  wx*wz*(1-cphi)    +      wy*sphi
        r_c_13  =  '((w_c_x)*(w_c_z)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  +   ((w_c_y) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_21  =  wx*wy*(1-cphi)    +      wz*sphi
        r_c_21  =  '((w_c_x)*(w_c_y)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  +   ((w_c_z) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_22  =  wy**2             +      (1-wy**2)*cphi
        r_c_22  =  '((w_c_y)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)  +  (1 - ((w_c_y)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))' 
#       r_23  =  wy*wz*(1-cphi)    -      wx*sphi
        r_c_23  =  '((w_c_y)*(w_c_z)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  -   ((w_c_x) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))' 
#       r_31  =  wx*wz*(1-cphi)    -      wy*sphi
        r_c_31  =  '((w_c_x)*(w_c_z)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  -   ((w_c_y) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_32  =  wy*wz*(1-cphi)    +      wx*sphi
        r_c_32  =  '((w_c_y)*(w_c_z)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  +   ((w_c_x) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_33  =  wz**2             +      (1-wz**2)*cphi
        r_c_33  =  '((w_c_z)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)  +   (1 - ((w_c_z)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))' 

#        functional_x =  '(  f_x * (r_11*X_ + r_12*Y_ + r_13*Z_ + t_x)  +  alpha * (r_21*X_ + r_22*Y_ + r_23*Z_ + t_y)  +   p_x * (r_31*X_ + r_32*Y_ + r_33*Z_ + t_z)   )  /  (r_31*X_ + r_32*Y_ + r_33*Z_ + t_z)'
        functional_x =  '(  f_x * ((r_11)*X_ + (r_12)*Y_ + (r_13)*Z_ + t_x)  +  alpha * ((r_21)*X_ + (r_22)*Y_ + (r_23)*Z_ + t_y)  +   p_x * ((r_31)*X_ + (r_32)*Y_ + (r_33)*Z_ + t_z)   )  /  ((r_31)*X_ + (r_32)*Y_ + (r_33)*Z_ + t_z)'
#        functional_y =  '(  f_y  * (r_21*X_ + r_22*Y_ + r_23*Z_ + t_y)  +  p_y  * (r_31*X_ + r_32*Y_ + r_33*Z_ + t_z)   )  / (r_31*X_ + r_32*Y_ + r_33*Z_ + t_z)'
        functional_y =  '(  f_y  * ((r_21)*X_ + (r_22)*Y_ + (r_23)*Z_ + t_y)  +  p_y  * ((r_31)*X_ + (r_32)*Y_ + (r_33)*Z_ + t_z)   )  / ((r_31)*X_ + (r_32)*Y_ + (r_33)*Z_ + t_z)'

        Fvec = []
        for i in range(self.num_cameras):
            functional_x_for_cam = str.replace( functional_x,         'r_11', r_c_11 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_12', r_c_12 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_13', r_c_13 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_21', r_c_21 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_22', r_c_22 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_23', r_c_23 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_31', r_c_31 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_32', r_c_32 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_33', r_c_33 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 't_x',  't_c_x')
            functional_x_for_cam = str.replace( functional_x_for_cam, 't_y',  't_c_y')
            functional_x_for_cam = str.replace( functional_x_for_cam, 't_z',  't_c_z')
            functional_x_for_cam = str.replace( functional_x_for_cam, '_c_', '_' + str(i) + '_' )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'f_x',   str(self.K[0,0]) )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'alpha', str(self.K[0,1]) )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'p_x',   str(self.K[0,2]) )

            functional_y_for_cam = str.replace( functional_y,         'r_21', r_c_21 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_22', r_c_22 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_23', r_c_23 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_31', r_c_31 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_32', r_c_32 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_33', r_c_33 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 't_y',  't_c_y')
            functional_y_for_cam = str.replace( functional_y_for_cam, 't_z',  't_c_z')
            functional_y_for_cam = str.replace( functional_y_for_cam, '_c_', '_' + str(i) + '_' )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'f_y',   str(self.K[1,1]) )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'p_y',   str(self.K[1,2]) )

            for j in range(self.num_world_points):
                tempx = str.replace( functional_x_for_cam, 'X_', 'X_' + str(j))
                tempx = str.replace( tempx, 'Y_', 'Y_' + str(j))
                tempx = str.replace( tempx, 'Z_', 'Z_' + str(j))
                Fvec.append( tempx )
                tempy = str.replace( functional_y_for_cam, 'X_', 'X_' + str(j))
                tempy = str.replace( tempy, 'Y_', 'Y_' + str(j))
                tempy = str.replace( tempy, 'Z_', 'Z_' + str(j))
                Fvec.append( tempy )

        self.Fvec = numpy.matrix(Fvec).T
        return self.Fvec


    def construct_Fvec_for_bundle_adjustment(self):
        ''' 
        This method constructs the Prediction Vector for the observed data in "self.X". This is a vector 
        of the same size as the number of measurements in "self.X". The elements of the Prediction Vector 
        are functional involving the parameters to be estimated.

        Since we know the intrinsic parameters of the camera, we know (p_x,p_y).  These are the coordinates
        of the center of the image frame where the optic axis penetrates with the image origin in the 
        camera image plane.

        You need to place each element that eventually is replaced by a numerical value by a parenthesized form.
        If you don't, you end up with things like '+-0.45634**2' which confuses the math processor.
        '''
#       r_11  =  wx**2             +      (1-wx**2)*cphi                          (shown without normalization)
        r_c_11  =  '((w_c_x)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)  +  (1 - ((w_c_x)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))' 
#       r_12  =  wx*wy*(1-cphi)    -      wz*sphi
        r_c_12  =  '((w_c_x)*(w_c_y)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  -   ((w_c_z) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)) * scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_13  =  wx*wz*(1-cphi)    +      wy*sphi
        r_c_13  =  '((w_c_x)*(w_c_z)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  +   ((w_c_y) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_21  =  wx*wy*(1-cphi)    +      wz*sphi
        r_c_21  =  '((w_c_x)*(w_c_y)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  +   ((w_c_z) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_22  =  wy**2             +      (1-wy**2)*cphi
        r_c_22  =  '((w_c_y)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)  +  (1 - ((w_c_y)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))' 
#       r_23  =  wy*wz*(1-cphi)    -      wx*sphi
        r_c_23  =  '((w_c_y)*(w_c_z)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  -   ((w_c_x) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))' 
#       r_31  =  wx*wz*(1-cphi)    -      wy*sphi
        r_c_31  =  '((w_c_x)*(w_c_z)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  -   ((w_c_y) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_32  =  wy*wz*(1-cphi)    +      wx*sphi
        r_c_32  =  '((w_c_y)*(w_c_z)/((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*(1 - scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)))  +   ((w_c_x) / math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.sin(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))'
#       r_33  =  wz**2             +      (1-wz**2)*cphi
        r_c_33  =  '((w_c_z)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2)  +   (1 - ((w_c_z)**2) / ((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))*scipy.cos(math.sqrt((w_c_x)**2+(w_c_y)**2+(w_c_z)**2))' 

#        functional_x =  '(  f_x * (r_11*X_ + r_12*Y_ + r_13*Z_ + t_x)  +  alpha * (r_21*X_ + r_22*Y_ + r_23*Z_ + t_y)  +   p_x * (r_31*X_ + r_32*Y_ + r_33*Z_ + t_z)   )  /  (r_31*X_ + r_32*Y_ + r_33*Z_ + t_z)'
        functional_x =  '(  f_x * ((r_11)*X_ + (r_12)*Y_ + (r_13)*Z_ + t_x)  +  alpha * ((r_21)*X_ + (r_22)*Y_ + (r_23)*Z_ + t_y)  +   p_x * ((r_31)*X_ + (r_32)*Y_ + (r_33)*Z_ + t_z)   )  /  ((r_31)*X_ + (r_32)*Y_ + (r_33)*Z_ + t_z)'
#        functional_y =  '(  f_y  * (r_21*X_ + r_22*Y_ + r_23*Z_ + t_y)  +  p_y  * (r_31*X_ + r_32*Y_ + r_33*Z_ + t_z)   )  / (r_31*X_ + r_32*Y_ + r_33*Z_ + t_z)'
        functional_y =  '(  f_y  * ((r_21)*X_ + (r_22)*Y_ + (r_23)*Z_ + t_y)  +  p_y  * ((r_31)*X_ + (r_32)*Y_ + (r_33)*Z_ + t_z)   )  / ((r_31)*X_ + (r_32)*Y_ + (r_33)*Z_ + t_z)'

        Fvec = []
        for i in range(self.num_cameras):
            functional_x_for_cam = str.replace( functional_x,         'r_11', r_c_11 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_12', r_c_12 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_13', r_c_13 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_21', r_c_21 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_22', r_c_22 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_23', r_c_23 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_31', r_c_31 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_32', r_c_32 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'r_33', r_c_33 )
            functional_x_for_cam = str.replace( functional_x_for_cam, 't_x',  't_c_x')
            functional_x_for_cam = str.replace( functional_x_for_cam, 't_y',  't_c_y')
            functional_x_for_cam = str.replace( functional_x_for_cam, 't_z',  't_c_z')
            functional_x_for_cam = str.replace( functional_x_for_cam, '_c_', '_' + str(i) + '_' )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'f_x',   str(self.K[0,0]) )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'alpha', str(self.K[0,1]) )
            functional_x_for_cam = str.replace( functional_x_for_cam, 'p_x',   str(self.K[0,2]) )

            functional_y_for_cam = str.replace( functional_y,         'r_21', r_c_21 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_22', r_c_22 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_23', r_c_23 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_31', r_c_31 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_32', r_c_32 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'r_33', r_c_33 )
            functional_y_for_cam = str.replace( functional_y_for_cam, 't_y',  't_c_y')
            functional_y_for_cam = str.replace( functional_y_for_cam, 't_z',  't_c_z')
            functional_y_for_cam = str.replace( functional_y_for_cam, '_c_', '_' + str(i) + '_' )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'f_y',   str(self.K[1,1]) )
            functional_y_for_cam = str.replace( functional_y_for_cam, 'p_y',   str(self.K[1,2]) )

            for j in range(self.num_world_points):
                tempx = str.replace( functional_x_for_cam, 'X_', 'X_' + str(j))
                tempx = str.replace( tempx, 'Y_', 'Y_' + str(j))
                tempx = str.replace( tempx, 'Z_', 'Z_' + str(j))
                Fvec.append( tempx )
                tempy = str.replace( functional_y_for_cam, 'X_', 'X_' + str(j))
                tempy = str.replace( tempy, 'Y_', 'Y_' + str(j))
                tempy = str.replace( tempy, 'Z_', 'Z_' + str(j))
                Fvec.append( tempy )

        mapped_indexes = self._index_mapping_for_BA(self.num_cameras, self.num_world_points)
        Fvec_reorganized = [Fvec[i] for i in mapped_indexes]
        self.Fvec_BA = numpy.matrix(Fvec_reorganized).T

    def testFvecForBA(self):
        exp   =  self.Fvec_BA[-1]
        print("\n\ndisplaying the last element of Fvec_BA: %s" % exp)
        param_vals_dict = self.initial_param_vals_dict
        print("\n\nparam_vals_dict: %s" % str(param_vals_dict))

    def construct_parameter_vec(self):
        '''
        This method constructs the parameter vector for estimating directly the camera matrix for each position
        of the camera.  HOWEVER NOTE THAT IS NEVER NEVER A SAFE THING TO dO but nonetheless useful in an
        educational setting for explaining its consequences.
        '''
        params_camera_template = 'p_c_11,p_c_12,p_c_13,p_c_14,p_c_21,p_c_22,p_c_23,p_c_24,p_c_31,p_c_32,p_c_33,p_c_34'
        for i in range(self.num_cameras):
            if i == 0:
                camera_params_str =  str.replace(params_camera_template, 'c', str(i))
            else:
                camera_params_str += ',' + str.replace(params_camera_template, 'c', str(i))
        camera_params_list = camera_params_str.split(',')
        structure_params_template = 'X_k,Y_k,Z_k'
        for j in range(self.num_world_points):
            if j == 0:
                structure_params_str = str.replace(structure_params_template, 'k', str(j))
            else:
                structure_params_str += ',' + str.replace(structure_params_template, 'k', str(j))
        structure_params_list = structure_params_str.split(',')
        all_params_list = camera_params_list + structure_params_list
        self.initial_params_list = all_params_list
        if self.debug:
            print("\ncamera params list: %s" % str(all_params_list))
        self.p  =  numpy.matrix(all_params_list).T
        return all_params_list

    def construct_parameter_vec_for_calibrated_cameras(self):
        '''
         Call this function only for the case when you estimating the structure using calibrated params
        '''
        structure_params_template = 'X_k,Y_k,Z_k'
        for j in range(self.num_world_points):
            if j == 0:
                structure_params_str = str.replace(structure_params_template, 'k', str(j))
            else:
                structure_params_str += ',' + str.replace(structure_params_template, 'k', str(j))
        structure_params_list = structure_params_str.split(',')
        self.initial_params_list = structure_params_list
        print("\nstructure params list: %s" % str(structure_params_list))
        return structure_params_list

    def construct_parameter_vec_for_uncalibrated_cameras_with_known_intrinsic_params(self):
        '''
        This method constructs the parameter vector for uncalibrated cameras but with known intrinsic 
        parameters.  But note that this method is for the case when you attempt to estimate the elements
        of the rotation matrix directly at each position of the camera --- WHICH IS NEVER A SAFE THING
        TO DO but nonetheless useful in an educational setting for demonstrating the consequences thereof.
        '''
        params_camera_template = 'r_c_11,r_c_12,r_c_13,t_c_x,r_c_21,r_c_22,r_c_23,t_c_y,r_c_31,r_c_32,r_c_33,t_c_z'
        for i in range(self.num_cameras):
            if i == 0:
                camera_params_str =  str.replace(params_camera_template, 'c', str(i))
            else:
                camera_params_str += ',' + str.replace(params_camera_template, 'c', str(i))
        camera_params_list = camera_params_str.split(',')
        structure_params_template = 'X_k,Y_k,Z_k'
        for j in range(self.num_world_points):
            if j == 0:
                structure_params_str = str.replace(structure_params_template, 'k', str(j))
            else:
                structure_params_str += ',' + str.replace(structure_params_template, 'k', str(j))
        structure_params_list = structure_params_str.split(',')
        all_params_list = camera_params_list + structure_params_list
        self.initial_params_list = all_params_list
        print("\ncamera params list: %s" % str(all_params_list))
        return all_params_list

    def construct_parameter_vec_for_uncalibrated_cameras_using_rodrigues_rotations(self):
        '''
         Call this function only for the case when you are estimating the structure using uncalibrated cameras
        '''
        params_camera_template = 'w_c_x,w_c_y,w_c_z,t_c_x,t_c_y,t_c_z'
        self.num_cam_params_per_camera = 6
        num_camera_params = 0
        for i in range(self.num_cameras):
            if i == 0:
                camera_params_str =  str.replace(params_camera_template, 'c', str(i))
            else:
                camera_params_str += ',' + str.replace(params_camera_template, 'c', str(i))
            num_camera_params += 6
        self.num_camera_params = num_camera_params
        camera_params_list = camera_params_str.split(',')
        structure_params_template = 'X_k,Y_k,Z_k'
        num_structure_elements = 0
        for j in range(self.num_world_points):
            if j == 0:
                structure_params_str = str.replace(structure_params_template, 'k', str(j))
            else:
                structure_params_str += ',' + str.replace(structure_params_template, 'k', str(j))
            num_structure_elements += 3
        self.num_structure_elements = num_structure_elements
        structure_params_list = structure_params_str.split(',')
        self.structure_params_list = structure_params_list
        all_params_list = camera_params_list + structure_params_list
        self.initial_params_list = all_params_list
        print("\nALL params list: %s" % str(all_params_list))
        self.all_params_list = all_params_list
        return all_params_list

    def get_structure_params_list(self):
        return self.structure_params_list

    def set_params_list(self, params_list_arranged):
        self.params_list_arranged = params_list_arranged

    def set_initial_val_all_params_as_dict(self, initial_params_dict):
        self.initial_param_vals_dict = initial_params_dict

    def set_initial_val_all_params(self, initial_val_all_params):
        self.initial_val_all_params = initial_val_all_params

    def set_tracked_point_indexes_for_display(self, tracked_point_indexes):
        self.tracked_point_indexes_for_display = tracked_point_indexes

    def set_constructor_options_for_optimizer(self, algo):
        '''
        This method conveys the following information from an instance of ProjectiveCamera to an 
        instance of NonlinearLeastSquares:
            1)  The measurement vector X.
            2)  The initial values to be used for the parameters of the scene structure.
            3)  The Fvec vector, which is a vector of the predicted values, all in functional 
                form, for each of the data elements in the measurement vector X.
            4)  The display function to be used for plotting the partial and the final results if
                such results can be displayed in the form of a 2D or 3D graphic with Python's 
                matplotlib library.
            5)  and some additional book-keeping information.
        '''
        self.optimizer = algo
        algo.set_X(self.X)
        algo.set_params_arranged_list(self.params_list_arranged)
#        algo.set_initial_params(self.initial_params_dict)
        algo.set_initial_params(self.initial_param_vals_dict)
        if self.partials_for_jacobian:
            algo.set_jacobian_functionals_array(self.construct_jacobian_array_in_functional_form())
        algo.set_Fvec(self.Fvec)
        if self.display_needed:
            algo.set_display_function(self.display_function)
        algo.set_num_measurements(self.num_measurements)
        algo.set_num_parameters(len(self.params_list_arranged))
#        algo.set_display_function(self.display_structure)
        algo.set_display_function(self.display_structure_and_pixels)
        algo.set_problem("sfm_" + str(self.num_structure_points))
        algo.set_debug(self.debug)

    def set_constructor_options_for_optimizer_BA(self, algo):
        '''
        This method conveys the following information from an instance of ProjectiveCamera to an 
        instance of NonlinearLeastSquares:
            1)  The measurement vector X_BA.
            2)  The initial values to be used for the parameters of the scene structure.
            3)  The Fvec vector, which is a vector of the predicted values, all in functional 
                form, for each of the data elements in the measurement vector X_BA.
            4)  The display function to be used for plotting the partial and the final results if
                such results can be displayed in the form of a 2D or 3D graphic with Python's 
                matplotlib library.
            5)  and some additional book-keeping information.
        '''
        self.optimizer = algo
        algo.set_X_BA(self.X_BA)
        algo.set_params_arranged_list(self.params_list_arranged)
        algo.set_initial_params(self.initial_param_vals_dict)
        if self.partials_for_jacobian:
            algo.set_jacobian_functionals_array(self.construct_jacobian_array_in_functional_form())
        algo.set_Fvec_BA(self.Fvec_BA)
        if self.display_needed:
            algo.set_display_function(self.display_function)
        algo.set_num_measurements(self.num_measurements)
        algo.set_num_parameters(len(self.params_list_arranged))
        algo.set_display_function(self.display_structure_and_pixels)
        algo.set_problem("sfm_" + str(self.num_structure_points))
        algo.set_debug(self.debug)

    def get_scene_structure_from_camera_motion(self, algo):
        if algo == 'lm':
            result_dict = self.optimizer.leven_marq()
        elif algo == 'gd':
            result_dict = self.optimizer.grad_descent()
        return result_dict

    def get_scene_structure_from_camera_motion_with_bundle_adjustment(self):
        result_dict = self.optimizer.bundle_adjust( 
                                  num_camera_params           =  self.num_camera_params,
                                  num_structure_elements      =  self.num_structure_elements,
                                  num_cameras                 =  len(self.list_of_cameras),
                                  num_cam_params_per_camera   =  self.num_cam_params_per_camera,
                                  num_measurements_per_camera =  self.num_measurements // len(self.list_of_cameras),
                                  initial_val_all_params = self.initial_val_all_params,
                      )
        return result_dict

    ##################    Private Methods of the Projective Camera Class     ##################

    def _rotate_3D_scene_around_world_X_axis(self, theta):
        '''
        This rotation through theta is around the world-X axis with respect to the world origin. 
        Think of an object point on the world-Z axis.  If you rotate that object through, say, 
        90 degrees with this method, the object point in question will move to somewhere on the
        world-X axis.  The rotation angle theta must be in degrees
        '''
        cos_theta =  scipy.cos( theta * scipy.pi / 180 )                                        
        sin_theta =  scipy.sin( theta * scipy.pi / 180 )         
        rot_X = numpy.matrix([[1.0,0.0,0.0],[0.0,cos_theta,-sin_theta],[0.0,sin_theta,cos_theta]])
        rot_X = numpy.append(rot_X, numpy.matrix([[0,0,0]]).T, 1)
        rot_X = numpy.append(rot_X, [[0,0,0,1]], 0)
        self.scene_transform_3D = self.scene_transform_3D * rot_X


    def _rotate_3D_scene_around_world_Y_axis(self, theta):
        '''
        This rotation through theta is around the world-Y axis with respect to the world origin. 
        The rotation angle theta must be in degrees
        '''
        cos_theta =  scipy.cos( theta * scipy.pi / 180 )                                        
        sin_theta =  scipy.sin( theta * scipy.pi / 180 )         
        rot_Y = numpy.matrix([[cos_theta,0.0,sin_theta],[0.0, 1.0, 0.0],[-sin_theta,0.0,cos_theta]])
        rot_Y = numpy.append(rot_Y, numpy.matrix([[0,0,0]]).T, 1)
        rot_Y = numpy.append(rot_Y, [[0,0,0,1]], 0)
        self.scene_transform_3D = rot_Y * self.scene_transform_3D

    def _rotate_3D_scene_around_world_Z_axis(self, theta):
        '''
        This rotation through theta is around the world-Z axis with respect to the world origin. 
        The rotation angle theta must be in degrees
        '''
        cos_theta =  scipy.cos( theta * scipy.pi / 180 )                                        
        sin_theta =  scipy.sin( theta * scipy.pi / 180 )         
        rot_Z = numpy.matrix([[cos_theta,-sin_theta,0.0],[sin_theta,cos_theta,0.0],[0.0,0.0,1.0]])
        rot_Z = numpy.append(rot_Z, numpy.matrix([[0,0,0]]).T, 1)
        rot_Z = numpy.append(rot_Z, [[0,0,0,1]], 0)
        self.scene_transform_3D = rot_Z * self.scene_transform_3D

    def _translate_3D_scene(self, translation):
        '''
        If you also need to rotate the object, you are likely to want to rotate the
        object at the origin before applying the translation transform: The argument
        `translation' must be a list of 3 numbers, indicating the translation vector
        '''
        rot3D = numpy.matrix([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])    
        trans3D = numpy.matrix(translation)
        transform = numpy.append(rot3D, trans3D.T, 1)
        transform = numpy.append(transform, [[0,0,0,1]], 0)
        self.scene_transform_3D = transform * self.scene_transform_3D

    def _scale_3D_scene(self, scale):
        if scale == 1.0: return
        left_upper_3by3 = self.scene_transform_3D[0:3,0:3]
        scale_diag = numpy.diag([scale,scale,scale])
        scale_as_matrix = numpy.asmatrix(scale_diag)
        new_left_upper_3by3  =  scale_as_matrix * left_upper_3by3
        new_upper_3by4  =  numpy.append(new_left_upper_3by3, self.scene_transform_3D[0:3,3], 1)
        new_scaled_xform =  numpy.append(new_upper_3by4, [[0,0,0,1]], 0)
        self.scene_transform_3D  = new_scaled_xform

    def _rotate_cam_frame_around_X_axis(self, cam_pose_matrix, theta):
        '''
        This rotation through theta is around the world-X axis with respect to the world origin. 
        Think of an object point on the world-Z axis.  If you rotate that object through, say, 
        90 degrees with this method, the object point in question will move to somewhere on the
        world-X axis.  The rotation angle theta must be in degrees
        '''
        cos_theta =  scipy.cos( theta * scipy.pi / 180 )                                        
        sin_theta =  scipy.sin( theta * scipy.pi / 180 )         
        rot_X = numpy.matrix([[1.0,0.0,0.0],[0.0,cos_theta,-sin_theta],[0.0,sin_theta,cos_theta]])
        rot_X = numpy.append(rot_X, numpy.matrix([[0,0,0]]).T, 1)
        rot_X = numpy.append(rot_X, [[0,0,0,1]], 0)
        return cam_pose_matrix * rot_X

    def _rotate_cam_frame_around_Y_axis(self, cam_pose_matrix, theta):
        '''
        This rotation through theta is around the world-Y axis with respect to the world origin. 
        The rotation angle theta must be in degrees
        '''
        cos_theta =  scipy.cos( theta * scipy.pi / 180 )                                        
        sin_theta =  scipy.sin( theta * scipy.pi / 180 )         
        rot_Y = numpy.matrix([[cos_theta,0.0,sin_theta],[0.0,1.0,0.0],[-sin_theta,0.0,cos_theta]])
        rot_Y = numpy.append(rot_Y, numpy.matrix([[0,0,0]]).T, 1)
        rot_Y = numpy.append(rot_Y, [[0,0,0,1]], 0)
        return cam_pose_matrix * rot_Y

    def _rotate_cam_frame_around_Z_axis(self, cam_pose_matrix, theta):
        '''
        This rotation through theta is around the world-Z axis with respect to the world origin. 
        The rotation angle theta must be in degrees
        '''
        cos_theta =  scipy.cos( theta * scipy.pi / 180 )                                        
        sin_theta =  scipy.sin( theta * scipy.pi / 180 )         
        rot_Z = numpy.matrix([[cos_theta,-sin_theta,0.0],[sin_theta,cos_theta,0.0],[0.0,0.0,1.0]])
        rot_Z = numpy.append(rot_Z, numpy.matrix([[0,0,0]]).T, 1)
        rot_Z = numpy.append(rot_Z, [[0,0,0,1]], 0)
        return cam_pose_matrix * rot_Z

    def _get_camera_motion_history(self):
        return self.motion_history

    def _index_mapping_for_BA(self, num_cameras, num_points):
        R = range(num_cameras * num_points * 2)
        R_cams = [R[lower*num_points*2 : (lower+1)*num_points*2] for lower in range(num_cameras)]
        R_points = [[] for _ in range(num_points)]
        for i in range(num_points):
            for j,sublist in enumerate(R_cams):
                R_points[i].append(sublist[2*i])
                R_points[i].append(sublist[2*i+1])
        flattened =  [item for sublist in R_points for item in sublist]
        return flattened
