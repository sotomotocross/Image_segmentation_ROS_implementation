from __future__ import print_function
from keras_segmentation.predict import model_from_checkpoint_path
from keras_segmentation.predict import predict
from tf.transformations import euler_from_quaternion
from math import *
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import  Imu
import tensorflow as tf
import math
import numpy.matlib
import numpy as np
import cv2
import rospy
from img_seg_cnn.msg import PredData, PolyCalcCustom, PolyCalcCustomTF

import roslib
roslib.load_manifest('img_seg_cnn')

red = (153, 0, 18)

class FeatureCalculator:
    
    """
    Feature calculator class for image processing.

    This class calculates features from images, transforms them, and publishes the results.

    Attributes:
        (attributes go here)
    """
    
    max_index = None  # Declare max_index as a class variable
    
    def __init__(self):
        """
        Initialize the FeatureCalculator.

        Creates subscribers, publishers, and initializes necessary variables.
        """
        
        # Create subscribers
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.updateImu)
        self.pos_sub = rospy.Subscriber("/mavros/global_position/local", Odometry, self.OdomCb)
        self.vel_uav = rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped, self.VelCallback)

        # Create publishers
        self.pub_pred_data = rospy.Publisher("/pred_data", PredData, queue_size=1000)
        self.pub_polycalc_custom = rospy.Publisher("/polycalc_custom", PolyCalcCustom, queue_size=1000)
        self.pub_polycalc_custom_tf = rospy.Publisher("/polycalc_custom_tf", PolyCalcCustomTF, queue_size=1000)
        
        # Generate instances of classes
        self.bridge = CvBridge()

        # uav state variables
        self.uav_vel_body = np.array([0.0, 0.0, 0.0, 0.0])
        self.vel_uav = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ZED stereo camera translation and rotation variables
        self.transCam = [0.0, 0.0, -0.14]
        self.rotCam = [0.0, 1.57, 0.0]
        self.phi_cam = self.rotCam[0]
        self.theta_cam = self.rotCam[1]
        self.psi_cam = self.rotCam[2]

        # ZED stereocamera intrinsic parameters
        self.cu = 360.5
        self.cv = 240.5
        self.ax = 252.07
        self.ay = 252.07        
    
    def updateImu(self, msg):
        """
        Update IMU data callback function.

        Args:
            msg (sensor_msgs.msg.Imu): IMU data message.
        """
        self.phi_imu = msg.orientation.x
        self.theta_imu = msg.orientation.y
        self.psi_imu = msg.orientation.z
        self.w_imu = msg.orientation.w
        self.phi_imu, self.theta_imu, self.psi_imu = euler_from_quaternion(
            [self.phi_imu, self.theta_imu, self.psi_imu, self.w_imu])

    # Callback function updating the Odometry measurements (rostopic /mavros/global_position/local)
    def OdomCb(self, msg):
        """
        Update Odometry data callback function.

        Args:
            msg (nav_msgs.msg.Odometry): Odometry data message.
        """
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        # print("Message position: ", msg.pose.pose.position)

    # Callback function updating the Velocity measurements (rostopic /mavros/local_position/velocity_body)
    def VelCallback(self, msg):
        """
        Update Velocity data callback function.

        Args:
            msg (geometry_msgs.msg.TwistStamped): Velocity data message.
        """

        self.vel_uav[0] = msg.twist.linear.x
        self.vel_uav[1] = msg.twist.linear.y
        self.vel_uav[2] = msg.twist.linear.z
        self.vel_uav[3] = msg.twist.angular.x
        self.vel_uav[4] = msg.twist.angular.y
        self.vel_uav[5] = msg.twist.angular.z
        # print("Message uav velocity: ", self.vel_uav)        

    
    # Function calling the feature transformation from the image plane on a virtual image plane
    def featuresTransformation(self, mp, phi, theta):
        
            """
            Transform features from image plane to a virtual image plane.

            Args:
                mp (numpy.array): Input feature vector.
                phi (float): Euler angle phi.
                theta (float): Euler angle theta.

            Returns:
                numpy.array: Transformed feature vector.
            """

            Rphi = np.array([[1.0, 0.0, 0.0], [0.0, cos(phi), -sin(phi)],
                            [0.0, sin(phi), cos(phi)]]).reshape(3, 3)
            Rtheta = np.array([[cos(theta), 0.0, sin(theta)], [
                            0.0, 1.0, 0.0], [-sin(theta), 0.0, cos(theta)]]).reshape(3, 3)
            Rft = np.dot(Rphi, Rtheta)

            N = len(mp)
            M = int(N+(N/2))
            cartesian_ft = np.zeros(M)

            for i, j in zip(range(0, M-2, 3), range(0, N-1, 2)):
                cartesian_ft[i] = mp[j]

            for i, j in zip(range(1, M-1, 3), range(1, N, 2)):
                cartesian_ft[i] = mp[j]

            for i in range(2, M, 3):
                cartesian_ft[i] = self.z

            cartesian_tf_ft = np.zeros(M)

            for i in range(0, M, 3):
                cartesian_tf_ft[i:i+3] = np.dot(Rft, cartesian_ft[i:i+3])

            return cartesian_tf_ft

    def feature_normalization(self, feature_pixel_virtual, hull_points_virtual, cu, cv, ax, ay):
        """
        Normalize features in the virtual image plane.

        Args:
            feature_pixel_virtual (numpy.array): Virtual image plane feature vector.
            hull_points_virtual (numpy.array): Virtual image plane hull points.
            cu (float): Image center x-coordinate.
            cv (float): Image center y-coordinate.
            ax (float): Horizontal focal length.
            ay (float): Vertical focal length.

        Returns:
            tuple: Normalized feature vector and hull points.
        """
        N = len(feature_pixel_virtual)
        norm_feat = np.zeros(N)

        for i in range(0,N,3):
                    norm_feat[i] =  self.z*((feature_pixel_virtual[i]-cu)/ax)
                    norm_feat[i+1] =  self.z*((feature_pixel_virtual[i+1]-cv)/ay)
                    norm_feat[i+2] =  feature_pixel_virtual[i+2]     

        K = len(hull_points_virtual)
        norm_hull = np.zeros((int(K),3))
        for i in range(0,K,1):
                norm_hull[i,0] =  self.z*((hull_points_virtual[i,0]-cu)/ax)
                norm_hull[i,1] =  self.z*((hull_points_virtual[i,1]-cv)/ay)
                norm_hull[i,2] =  hull_points_virtual[i,2]

        return norm_feat, norm_hull
    
    def cartesian_from_pixel(self, ft_pixel, cu, cv, ax, ay):
            """
            Convert pixel coordinates to Cartesian coordinates.

            Args:
                ft_pixel (numpy.array): Pixel coordinates.
                cu (float): Image center x-coordinate.
                cv (float): Image center y-coordinate.
                ax (float): Horizontal focal length.
                ay (float): Vertical focal length.

            Returns:
                numpy.array: Cartesian coordinates.
            """
            
            N = len(ft_pixel)
            
            tf_feat = np.zeros(N)
            
            for i in range(0,N-1,2):
                tf_feat[i] = self.z*((ft_pixel[i]-cu)/ax)
            
            for i in range(1,N,2):
                tf_feat[i] = self.z*((ft_pixel[i]-cv)/ay)
            
            
            return tf_feat
    
    def pixels_from_cartesian(self, mp_cartesian, cu, cv, ax, ay):        

            """
            Convert Cartesian coordinates to pixel coordinates.

            Args:
                mp_cartesian (numpy.array): Cartesian coordinates.
                cu (float): Image center x-coordinate.
                cv (float): Image center y-coordinate.
                ax (float): Horizontal focal length.
                ay (float): Vertical focal length.

            Returns:
                numpy.array: Pixel coordinates.
            """
            M = len(mp_cartesian)
            # print("length of cartesian transformed feature vector: ", M)
            virtual_plane_pixel = np.zeros(M)
            
            for i in range(0,M,3):
                # print("index for cartesian to pixel transformation: ", i)
                virtual_plane_pixel[i] = (mp_cartesian[i]/mp_cartesian[i+2])*ax + cu
                virtual_plane_pixel[i+1] = (mp_cartesian[i+1]/mp_cartesian[i+2])*ay + cv
                virtual_plane_pixel[i+2] = mp_cartesian[i+2]
                
            # print("virtual plane features in pixel: ", virtual_plane_pixel)        
            return virtual_plane_pixel
       
    
    def calculate_polygon_features(self, contours_blk, max_index):
        
        """
        Calculate polygon features from contours.

        Args:
            contours_blk (list): List of contours.
            max_index (int): Index of the largest contour.

        Returns:
            tuple: Polygon features and information.
        """
        # Existing code for polygon features calculation
        # ------------------------------------------------------------------------------------------------
        # Polygon approximation - Features, Angle and Area calculation
        # ------------------------------------------------------------------------------------------------
        if len(contours_blk) > 0 and cv2.contourArea(contours_blk[max_index]) > 200:
            # Extract relevant information from the contour
            blackbox = cv2.minAreaRect(contours_blk[max_index])
            (x_min, y_min), (w_min, h_min), angle = blackbox
            box = cv2.boxPoints(blackbox)
            box = np.int0(box)
            alpha = angle
            
            sigma = cv2.contourArea(contours_blk[0])
            sigma_square = math.sqrt(sigma)
            sigma_square_log = np.log(sigma_square)
                
                
            epsilon = 0.01 * cv2.arcLength(contours_blk[max_index], True)
            # get approx polygons
            approx = cv2.approxPolyDP(contours_blk[max_index], epsilon, True)
                
            # hull is convex shape as a polygon
            hull = cv2.convexHull(contours_blk[max_index])
                
            hull_points = np.zeros((len(hull),2)) # Pre-allocate matrix
            hull_points = np.array(hull_points, dtype=np.int_)
            stacked_feature_vector = hull_points.flatten('A') 
            for i in range(0,len(hull_points)):
                hull_points[i,:] = [hull[i][0][0], hull[i][0][1]]                
            
            c = np.argsort(hull_points[:,1])
            d = c[0]
            f = c[1]
            feature_vector = hull_points.flatten()
            Ical = (1/len(hull_points))*np.matlib.repmat(np.eye(2),1,len(hull_points))                         
            barycenter_features = np.dot(Ical, feature_vector)
            x = (hull_points[d,0]-self.cu)+(hull_points[f,0]-self.cu)-2*(barycenter_features[0]-self.cu)
            y = (hull_points[d,1]-self.cv) + (hull_points[f,1]-self.cv) - 2*(barycenter_features[1]-self.cv)
            tangent = -x/(-y)
            # tangent = y/x
            angle_radian = math.atan2(-x,-y)
            angle_deg = math.degrees(angle_radian)
                
            determinants = np.zeros(len(hull_points))
            last_det = np.linalg.det([[hull_points[0,0], hull_points[len(hull_points)-1,0]],[hull_points[0,1], hull_points[len(hull_points)-1,1]]])
            determinants[len(hull_points)-1] = last_det
                
            for i in range(0,len(hull_points)-1,1):
                determinants[i] = np.linalg.det([[hull_points[i,0], hull_points[i+1,0]],[hull_points[i,1], hull_points[i+1,1]]])
                    
                
            custom_sigma = 0.5*sum(determinants)
            custom_sigma_square =  math.sqrt(custom_sigma)
            custom_sigma_square_log = np.log(custom_sigma_square)     
    
        return alpha, sigma, sigma_square, sigma_square_log, hull_points, tangent, angle_radian, angle_deg, custom_sigma, custom_sigma_square, custom_sigma_square_log, feature_vector, box, barycenter_features, stacked_feature_vector, Ical

    def transform_features(self, feature_vector, Ical, hull_points):
        """
        Transform features from image plane to a normalized plane.

        Args:
            feature_vector (numpy.array): Input feature vector.
            Ical (numpy.array): Identity matrix for normalization.
            hull_points (numpy.array): Hull points.

        Returns:
            tuple: Transformed features and information.
        """
        
        feature_cartesian = self.cartesian_from_pixel(feature_vector, self.cu, self.cv, self.ax, self.ay)
        feature_cartesian_virtual = self.featuresTransformation(feature_cartesian, self.phi_imu, self.theta_imu)
        feature_pixel_virtual = self.pixels_from_cartesian(feature_cartesian_virtual, self.cu, self.cv, self.ax, self.ay)         
            
        K = len(hull_points)
        hull_points_transformed = np.zeros((int(K),3))
            
        m = 0
            
        for i in range(0,K,1):
                 hull_points_transformed[i,0] =  feature_pixel_virtual[m]
                 hull_points_transformed[i,1] =  feature_pixel_virtual[m+1]
                 hull_points_transformed[i,2] =  feature_pixel_virtual[m+2]
                 m+=3                        

        normalized_feature_vector, normalized_hull_points = self.feature_normalization(feature_pixel_virtual, hull_points_transformed, self.cu, self.cv, self.ax, self.ay)
            
        c_transformed = np.argsort(normalized_hull_points[:,1])
        d_transformed = c_transformed[0]
        f_transformed = c_transformed[1]   
            
        L = len(feature_vector)
        transformed_features_only_image_plane = np.zeros(L)     
            
        for i,j in zip(range(0,L-1,2),range(0,K,1)):
                transformed_features_only_image_plane[i] = normalized_hull_points[j,0]
        for i,j in zip(range(1,L,2),range(0,K,1)):
                transformed_features_only_image_plane[i] = normalized_hull_points[j,1]
                
            
        transformed_barycenter_features = np.dot(Ical, transformed_features_only_image_plane)            
            
        transformed_x = (normalized_hull_points[d_transformed,0]-self.cu)+(normalized_hull_points[f_transformed,0]-self.cu)-2*(transformed_barycenter_features[0]-self.cu)
        transformed_y = (normalized_hull_points[d_transformed,1]-self.cv) + (normalized_hull_points[f_transformed,1]-self.cv) - 2*(transformed_barycenter_features[1]-self.cv)
        transformed_tangent = -transformed_x/(-transformed_y)
        transformed_angle_radian = math.atan2(transformed_y,transformed_x)
        transformed_angle_radian = math.atan2(-transformed_x,-transformed_y)
        transformed_angle_deg = math.degrees(transformed_angle_radian)
                        
        transformed_determinants = np.zeros(len(normalized_hull_points))
        last_transformed_det = np.linalg.det([[normalized_hull_points[0,0], normalized_hull_points[len(normalized_hull_points)-1,0]],[normalized_hull_points[0,1], normalized_hull_points[len(normalized_hull_points)-1,1]]])
        transformed_determinants[len(normalized_hull_points)-1] = last_transformed_det
            
        for i in range(0,len(normalized_hull_points)-1,1):
            transformed_determinants[i] = np.linalg.det([[normalized_hull_points[i,0], normalized_hull_points[i+1,0]],[normalized_hull_points[i,1], normalized_hull_points[i+1,1]]])
            
            
        transformed_sigma = 0.5*sum(transformed_determinants)
        transformed_sigma_square =  math.sqrt(transformed_sigma)
        transformed_sigma_square_log = np.log(transformed_sigma_square)
        
        
        return feature_cartesian_virtual, feature_pixel_virtual, normalized_hull_points, d_transformed, f_transformed, transformed_features_only_image_plane, transformed_barycenter_features, transformed_tangent, transformed_angle_radian, transformed_angle_deg, transformed_sigma, transformed_sigma_square, transformed_sigma_square_log

    def extract_moments(self, contours_blk):
        """
        Extract moments from contours.

        Args:
            contours_blk (list): List of contours.

        Returns:
            tuple: Extracted moments and information.
        """
        M = cv2.moments(contours_blk[0])
            
        cX = int(M["m10"] / M["m00"])
        cX_float = M["m10"] / M["m00"]
            
        cY = int(M["m01"] / M["m00"])
        cY_float = M["m01"] / M["m00"]       
            
        moments = np.zeros(24)
        moments[0] = M["m00"] 
        moments[1] = M["m10"] 
        moments[2] = M["m01"] 
        moments[3] = M["m20"] 
        moments[4] = M["m11"] 
        moments[5] = M["m02"] 
        moments[6] = M["m30"] 
        moments[7] = M["m21"]
        moments[8] = M["m12"]
        moments[9] = M["m03"] 
        moments[10] = M["mu20"] 
        moments[11] = M["mu11"] 
        moments[12] = M["mu02"] 
        moments[13] = M["mu30"] 
        moments[14] = M["mu21"] 
        moments[15] = M["mu12"] 
        moments[16] = M["mu03"] 
        moments[17] = M["nu20"] 
        moments[18] = M["nu11"] 
        moments[19] = M["nu02"] 
        moments[20] = M["nu30"] 
        moments[21] = M["nu21"] 
        moments[22] = M["nu12"] 
        moments[23] = M["nu03"]         
        
        return moments, cX, cX_float, cY, cY_float

    def calculate_features(self, contours_blk, max_index):
        """
        Calculate various features from contours.

        Args:
            contours_blk (list): List of contours.
            max_index (int): Index of the largest contour.

        Returns:
            tuple: Extracted features and information.
        """
        alpha, sigma, sigma_square, sigma_square_log, hull_points, tangent, angle_radian, angle_deg, custom_sigma, custom_sigma_square, custom_sigma_square_log, feature_vector, box, barycenter_features, stacked_feature_vector, Ical = self.calculate_polygon_features(
            contours_blk, max_index
        )
        
        # Extract moments
        moments, cX, cX_float, cY, cY_float  = self.extract_moments(contours_blk)

        # Combine all the relevant features into a single dictionary
        features_pred_data = {
                'box_1': box[0][:],
                'box_2': box[1][:],
                'box_3': box[2][:],
                'box_4': box[3][:],
                'cX': cX,
                'cY': cY,
                'alpha': alpha,
                'sigma': sigma,
                'sigma_square': sigma_square,
                'sigma_square_log': sigma_square_log
            }

        # Feature transformation
        feature_cartesian_virtual, feature_pixel_virtual, normalized_hull_points, d_transformed, f_transformed, transformed_features_only_image_plane, transformed_barycenter_features, transformed_tangent, transformed_angle_radian, transformed_angle_deg, transformed_sigma, transformed_sigma_square, transformed_sigma_square_log = self.transform_features(
            feature_vector, Ical, hull_points)

        features_polycalc_custom = {
            'd': d_transformed,
            'f': f_transformed,
            'custom_sigma': custom_sigma,
            'custom_sigma_square': custom_sigma_square,
            'custom_sigma_square_log': custom_sigma_square_log,
            'tangent': tangent,
            'angle_radian': angle_radian,
            'angle_deg': angle_deg
        }

        features_polycalc_custom_tf = {
            'd_transformed': d_transformed,
            'f_transformed': f_transformed,
            'transformed_sigma': transformed_sigma,
            'transformed_sigma_square': transformed_sigma_square,
            'transformed_sigma_square_log': transformed_sigma_square_log,
            'transformed_tangent': transformed_tangent,
            'transformed_angle_radian': transformed_angle_radian,
            'transformed_angle_deg': transformed_angle_deg,
            'moments': moments
        }

        return features_pred_data, stacked_feature_vector, barycenter_features, features_polycalc_custom, transformed_features_only_image_plane, transformed_barycenter_features, features_polycalc_custom_tf

    def publish_extracted_features(self, features_pred_data, stacked_feature_vector, barycenter_features, features_polycalc_custom, transformed_features_only_image_plane, transformed_barycenter_features, features_polycalc_custom_tf):
        """
        Publish the extracted features.

        Args:
            features_pred_data (dict): Dictionary of predicted data features.
            stacked_feature_vector (numpy.array): Stacked feature vector.
            barycenter_features (numpy.array): Barycenter feature vector.
            features_polycalc_custom (dict): Dictionary of custom polygon calculation features.
            transformed_features_only_image_plane (numpy.array): Transformed feature vector in the image plane.
            transformed_barycenter_features (numpy.array): Transformed barycenter feature vector.
            features_polycalc_custom_tf (dict): Dictionary of transformed polygon calculation features.
        """
        try:
            # Create instances of the message types
            pred_data_msg = PredData()
            polycalc_custom_msg = PolyCalcCustom()
            polycalc_custom_tf_msg = PolyCalcCustomTF()

            # ------------------------------------------------------------------------------------------------
            # box features when using box = cv2.boxPoints(blackbox)
            # ------------------------------------------------------------------------------------------------
            pred_data_msg.box_1 = features_pred_data['box_1']
            pred_data_msg.box_2 = features_pred_data['box_2']
            pred_data_msg.box_3 = features_pred_data['box_3']
            pred_data_msg.box_4 = features_pred_data['box_4']

            pred_data_msg.cX = features_pred_data['cX']
            pred_data_msg.cY = features_pred_data['cY']
            pred_data_msg.alpha = features_pred_data['alpha']
            pred_data_msg.sigma = features_pred_data['sigma']
            pred_data_msg.sigma_square = features_pred_data['sigma_square']
            pred_data_msg.sigma_square_log = features_pred_data['sigma_square_log']

            self.pub_pred_data.publish(pred_data_msg)
            # ------------------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------------------
            # polygon feature calculation WITHOUT virtual image transformation
            # ------------------------------------------------------------------------------------------------
            polycalc_custom_msg.features = stacked_feature_vector
            polycalc_custom_msg.barycenter_features = barycenter_features
            polycalc_custom_msg.d = features_polycalc_custom['d']
            polycalc_custom_msg.f = features_polycalc_custom['f']

            polycalc_custom_msg.custom_sigma = features_polycalc_custom['custom_sigma']
            polycalc_custom_msg.custom_sigma_square = features_polycalc_custom['custom_sigma_square']
            polycalc_custom_msg.custom_sigma_square_log = features_polycalc_custom['custom_sigma_square_log']

            polycalc_custom_msg.tangent = features_polycalc_custom['tangent']
            polycalc_custom_msg.angle_radian = features_polycalc_custom['angle_radian']
            polycalc_custom_msg.angle_deg = features_polycalc_custom['angle_deg']

            self.pub_polycalc_custom.publish(polycalc_custom_msg)
            # ------------------------------------------------------------------------------------------------

            # ------------------------------------------------------------------------------------------------
            # polygon feature calculation WITH virtual image transformation
            # ------------------------------------------------------------------------------------------------
            polycalc_custom_tf_msg.transformed_features = transformed_features_only_image_plane
            polycalc_custom_tf_msg.transformed_barycenter_features = transformed_barycenter_features
            polycalc_custom_tf_msg.d_transformed = features_polycalc_custom_tf['d_transformed']
            polycalc_custom_tf_msg.f_transformed = features_polycalc_custom_tf['f_transformed']

            polycalc_custom_tf_msg.transformed_sigma = features_polycalc_custom_tf['transformed_sigma']
            polycalc_custom_tf_msg.transformed_sigma_square = features_polycalc_custom_tf['transformed_sigma_square']
            polycalc_custom_tf_msg.transformed_sigma_square_log = features_polycalc_custom_tf['transformed_sigma_square_log']

            polycalc_custom_tf_msg.transformed_tangent = features_polycalc_custom_tf['transformed_tangent']
            polycalc_custom_tf_msg.transformed_angle_radian = features_polycalc_custom_tf['transformed_angle_radian']
            polycalc_custom_tf_msg.transformed_angle_deg = features_polycalc_custom_tf['transformed_angle_deg']
            polycalc_custom_tf_msg.moments = features_polycalc_custom_tf['moments']

            self.pub_polycalc_custom_tf.publish(polycalc_custom_tf_msg)
            # ------------------------------------------------------------------------------------------------

        except Exception as e:
            print(f"Error publishing extracted features: {e}")
