#!/usr/bin/env python
from __future__ import print_function
from img_seg_cnn.msg import PREDdata, POLYcalc_custom, POLYcalc_custom_tf
from keras_segmentation.predict import model_from_checkpoint_path
from keras_segmentation.predict import predict
from tf.transformations import euler_from_quaternion
from math import *
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3Stamped, TwistStamped
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String, Empty, Int16, Float32, Bool, UInt16MultiArray, UInt32MultiArray, UInt64MultiArray
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import tensorflow as tf
import math
import numpy.matlib
import numpy as np
import cv2
import rospy
import sys

import roslib
roslib.load_manifest('img_seg_cnn')

red = (153, 0, 18)
dim = (720, 480)

class image_converter:

  def __init__(self):
    self.image_pub_first_image = rospy.Publisher(
        "/image_raww", Image, queue_size=10)
    self.image_pub_second_image = rospy.Publisher(
        "/image_raww_comb", Image, queue_size=10)
    self.pub_pred_data = rospy.Publisher(
        "/pred_data", PREDdata, queue_size=1000)
    self.pub_polycalc_custom = rospy.Publisher(
        "/polycalc_custom", POLYcalc_custom, queue_size=1000)
    self.pub_polycalc_custom_tf = rospy.Publisher(
        "/polycalc_custom_tf", POLYcalc_custom_tf, queue_size=1000)
    self.pred_data = PREDdata()
    self.polycalc_custom = POLYcalc_custom()
    self.polycalc_custom_tf = POLYcalc_custom_tf()
    self.bridge = CvBridge()
    self.ros_image_pub = rospy.Publisher(
        "image_bounding_box", Image, queue_size=10)

    # Create subscribers
    self.image_sub = rospy.Subscriber(
        "/iris_demo/ZED_stereocamera/camera/left/image_raw", Image, self.callback)
    self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.updateImu)
    self.pos_sub = rospy.Subscriber(
        "/mavros/global_position/local", Odometry, self.OdomCb)
    self.vel_uav = rospy.Subscriber(
        "/mavros/local_position/velocity_body", TwistStamped, self.VelCallback)

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
        self.phi_imu = msg.orientation.x
        self.theta_imu = msg.orientation.y
        self.psi_imu = msg.orientation.z
        self.w_imu = msg.orientation.w
        self.phi_imu, self.theta_imu, self.psi_imu = euler_from_quaternion(
            [self.phi_imu, self.theta_imu, self.psi_imu, self.w_imu])

  # Callback function updating the Odometry measurements (rostopic /mavros/global_position/local)

  def OdomCb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        # print("Message position: ", msg.pose.pose.position)

  # Callback function updating the Velocity measurements (rostopic /mavros/local_position/velocity_body)

  def VelCallback(self, msg):

        self.vel_uav[0] = msg.twist.linear.x
        self.vel_uav[1] = msg.twist.linear.y
        self.vel_uav[2] = msg.twist.linear.z
        self.vel_uav[3] = msg.twist.angular.x
        self.vel_uav[4] = msg.twist.angular.y
        self.vel_uav[5] = msg.twist.angular.z
        # print("Message uav velocity: ", self.vel_uav)

  # Function calling the feature transformation from the image plane on a virtual image plane

  def featuresTransformation(self, mp, phi, theta):

        Rphi = np.array([[1.0, 0.0, 0.0], [0.0, cos(phi), -sin(phi)],
                        [0.0, sin(phi), cos(phi)]]).reshape(3, 3)
        # print("Rphi: ", Rphi)
        Rtheta = np.array([[cos(theta), 0.0, sin(theta)], [
                          0.0, 1.0, 0.0], [-sin(theta), 0.0, cos(theta)]]).reshape(3, 3)
        # print("Rtheta: ", Rtheta)
        Rft = np.dot(Rphi, Rtheta)
        # print("Rft: ", Rft)
        # print("Shape of Rft: ", Rft.shape)

        N = len(mp)
        # print("Length of features: ", N)
        M = int(N+(N/2))
        # print("new length: ", M)
        cartesian_ft = np.zeros(M)

        for i, j in zip(range(0, M-2, 3), range(0, N-1, 2)):
              cartesian_ft[i] = mp[j]

        for i, j in zip(range(1, M-1, 3), range(1, N, 2)):
              cartesian_ft[i] = mp[j]

        for i in range(2, M, 3):
              cartesian_ft[i] = self.z

        # print("cartesian: ", cartesian_ft)
        # print("Shape of cartesian: ", cartesian_ft.shape)

        cartesian_tf_ft = np.zeros(M)
        # print("mpv_dot: ", mpv_dot)
        # print("Shape of mpv_dot: ", mpv_dot.shape)

        for i in range(0, M, 3):
              # print("Index of mpv_dot: ", i)
              cartesian_tf_ft[i:i+3] = np.dot(Rft, cartesian_ft[i:i+3])
              # print("Index of mpv_dot[i:i+3]: ", cartesian_tf_ft[i:i+3])

        return cartesian_tf_ft

  def feature_normalization(self, feature_pixel_virtual, hull_points_virtual, cu, cv, ax, ay):
      N = len(feature_pixel_virtual)
      norm_feat = np.zeros(N)
      # print("feature_pixel_virtual: ", feature_pixel_virtual)
      for i in range(0,N,3):
                 norm_feat[i] =  self.z*((feature_pixel_virtual[i]-cu)/ax)
            #      print("feature_pixel_virtual[i]", feature_pixel_virtual[i+1])
            #      print("feature_pixel_virtual[i]-cv", feature_pixel_virtual[i+1]-cv)
            #      print("((feature_pixel_virtual[i]-cv)/ay)", ((feature_pixel_virtual[i+1]-cv)/ay))
            #      print("self.z*((feature_pixel_virtual[i]-cv)/ay)", self.z*((feature_pixel_virtual[i+1]-cv)/ay))
                 norm_feat[i+1] =  self.z*((feature_pixel_virtual[i+1]-cv)/ay)
                 norm_feat[i+2] =  feature_pixel_virtual[i+2]     
      # print("normalized feature vector: ", norm_feat)

      K = len(hull_points_virtual)
      norm_hull = np.zeros((int(K),3))
      # print("hull_points_virtual: ", hull_points_virtual)
      for i in range(0,K,1):
            norm_hull[i,0] =  self.z*((hull_points_virtual[i,0]-cu)/ax)
            # print("hull_points_virtual[i,1]", hull_points_virtual[i,1])
            # print("hull_points_virtual[i,1]-cv", hull_points_virtual[i,1]-cv)
            # print("((hull_points_virtual[i,1]-cv)/ay)", ((hull_points_virtual[i,1]-cv)/ay))
            # print("self.z*((hull_points_virtual[i,1]-cv)/ay)", self.z*((hull_points_virtual[i,1]-cv)/ay))
            norm_hull[i,1] =  self.z*((hull_points_virtual[i,1]-cv)/ay)
            norm_hull[i,2] =  hull_points_virtual[i,2]
      # print("norm_hull: ", norm_hull)

      return norm_feat, norm_hull
  
  def cartesian_from_pixel(self, ft_pixel, cu, cv, ax, ay):
        
        N = len(ft_pixel)
        # print("Length of features: ", N)
        tf_feat = np.zeros(N)
        # print("Length of tf_feat: ", tf_feat)   
        
        # print("ft_pixel: ", ft_pixel)    
        
        for i in range(0,N-1,2):
              # print("even index: ", i)
              # print("x component ft_pixel[i]: ", ft_pixel[i])
              tf_feat[i] = self.z*((ft_pixel[i]-cu)/ax)
        
        for i in range(1,N,2):
              # print("odd index: ", i)
              # print("y component ft_pixel[i]: ", ft_pixel[i])
              tf_feat[i] = self.z*((ft_pixel[i]-cv)/ay)
        
        # print("transformed feature vector: ", tf_feat)
        # print("self.z: ", self.z)
        
        return tf_feat
  
  def pixels_from_cartesian(self, mp_cartesian, cu, cv, ax, ay):        
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

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape

    def create_blank(width, height, rgb_color=(0, 0, 0)):
      # Create black blank image
      image = np.zeros((height, width, 3), np.uint8)
      # Since OpenCV uses BGR, convert the color first
      color = tuple(reversed(rgb_color))
      # Fill image with color
      image[:] = color
      return image

    with graph.as_default():
    	pr, seg_img = predict( 
	              	model=mdl, 
	              	inp=cv_image 
                      )
    segimg = seg_img.astype(np.uint8)

    copy_image = np.copy(cv_image)
    copy_mask = np.copy(segimg)
    ww = copy_mask.shape[1]
    hh = copy_mask.shape[0]
    red_mask = create_blank(ww, hh, rgb_color=red)
    copy_mask=cv2.bitwise_and(copy_mask,red_mask)
    combo_image=cv2.addWeighted(copy_image, 1, copy_mask,1 ,1)

    mask = cv2.inRange(segimg, (130, 130, 130), (255, 255, 255))
    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    contours_blk, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     _, contours_blk, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("contours: ", contours_blk)
    areas = [cv2.contourArea(c) for c in contours_blk]
    max_index = np.argmax(areas)
    # cnt = countours_blk[max_index]
    # contours_blk.sort(key=cv2.minAreaRect)
    if len(contours_blk) > 0 and cv2.contourArea(contours_blk[max_index]) > 200:
          # Box creation for the detected coastline
            blackbox = cv2.minAreaRect(contours_blk[max_index])
            (x_min, y_min), (w_min, h_min), angle = blackbox            
            box = cv2.boxPoints(blackbox)
            box = np.int0(box)
            # print("box: ", box)  
            # print("angle = ", angle)         

            M = cv2.moments(contours_blk[0])
            # print("moments: ", M)
            # print("how many moments? = ", len(M))
            cX = int(M["m10"] / M["m00"])
            cX_float = M["m10"] / M["m00"]
            # print("cX = ", cX)
            # print("cX_float = ", cX_float)
            cY = int(M["m01"] / M["m00"])
            cY_float = M["m01"] / M["m00"]
            # print("moments area = ", M["m00"])
            # print("cY = ", cY)
            # print("cY_float = ", cY_float)
            # moments:  {'m00': 16104.0, 
            #            'm10': 5996134.333333333, 
            #            'm01': 3605177.0, 
            #            'm20': 2951430083.0, 
            #            'm11': 1620242707.5833333, 
            #            'm02': 915574641.6666666, 
            #            'm30': 1614344515429.1, 
            #            'm21': 862419613826.5, 
            #            'm12': 464282693972.36664, 
            #            'm03': 253538506205.5, 
            #            'mu20': 718840233.0634146, 
            #            'mu11': 277897601.5572734, 
            #            'mu02': 108489370.8439517, 
            #            'mu30': -19888995277.239746, 
            #            'mu21': -5256153190.191376, 
            #            'mu12': -1045710561.0260849, 
            #            'mu03': -4427782.458190918, 
            #            'nu20': 2.7718189052864903, 
            #            'nu11': 1.0715619275337225, 
            #            'nu02': 0.4183306321717444, 
            #            'nu30': -0.6043359641035431, 
            #            'nu21': -0.15971055155839226, 
            #            'nu12': -0.031774380317448656, 
            #            'nu03': -0.0001345401385747088}
            # print("m00 = ", M["m00"] )
            # print("m10 = ", M["m10"] )
            # print("m01 = ", M["m01"] )
            # print("m20 = ", M["m20"] )
            # print("m11 = ", M["m11"] )
            # print("m02 = ", M["m02"] )
            # print("m30 = ", M["m30"] )
            # print("m21 = ", M["m21"] )
            # print("m12 = ", M["m12"] )
            # print("m03 = ", M["m03"] )
            # print("mu20 = ", M["mu20"] )
            # print("mu11 = ", M["mu11"] )
            # print("mu02 = ", M["mu02"] )
            # print("mu30 = ", M["mu30"] )
            # print("mu21 = ", M["mu21"] )
            # print("mu12 = ", M["mu12"] )
            # print("mu03 = ", M["mu03"] )
            # print("nu20 = ", M["nu20"] )
            # print("nu11 = ", M["nu11"] )
            # print("nu02 = ", M["nu02"] )
            # print("nu30 = ", M["nu30"] )
            # print("nu21 = ", M["nu21"] )
            # print("nu12 = ", M["nu12"] )
            # print("nu03 = ", M["nu03"] )
            
            
            
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
            
      
            
            # Sorting of the orientation of the detected coastline
            if angle < -45:
                angle = 90 + angle
            if w_min < h_min and angle > 0:
                angle = (90 - angle) * -1
            if w_min > h_min and angle < 0:
                angle = 90 + angle

            # ------------------------------------------------------------------------------------------------
            # Polygon approximation - Features, Angle and Area calculation
            # ------------------------------------------------------------------------------------------------
            alpha = angle
            # print("angle of the contour: ", alpha)
            sigma = cv2.contourArea(contours_blk[0])
            sigma_square = math.sqrt(sigma)
            sigma_square_log = np.log(sigma_square)
            # print("opencv_sigma:", sigma)
            # print("opencv_sigma_square:", sigma_square)
            # print("opencv_sigma_square_log:", sigma_square_log)
            
            # print("initial approximation: ", contours_blk[max_index])
            # print(len(contours_blk[max_index]), "objects were found in this image.")            
            
            # edge = cv2.Canny(contours_blk[max_index], 10, 250)
            # print("edge approximation: ", edge)
            
            epsilon = 0.01 * cv2.arcLength(contours_blk[max_index], True)
            # get approx polygons
            approx = cv2.approxPolyDP(contours_blk[max_index], epsilon, True)
            # print("polygon approximation: ", approx)
            # print(len(approx), "polygon vertices were found in this image.")
            # hull is convex shape as a polygon
            hull = cv2.convexHull(contours_blk[max_index])
            # print("hull approximation: ", hull)
            hull_points = np.zeros((len(hull),2)) # Pre-allocate matrix
            # print("\n")
            # print("hull points length: ", len(hull_points))
            for i in range(0,len(hull_points)):
              hull_points[i,:] = [hull[i][0][0], hull[i][0][1]]    
            # print("hull_points: ", hull_points)
            # a=np.argmin(hull_points[:,1])
            c = np.argsort(hull_points[:,1])
            d = c[0]
            f = c[1]
            # print("1st new index:", d)
            # print("2nd new index:", f)
            # print("1st np.argmin(hull_points): ", c)
            # arr = np.delete(hull_points, a, axis=0)
            # print("arr: ", arr)
            # b=np.argmin(arr[:,1])
            # print("2nd np.argmin(hull_points): ", b)
            feature_vector = hull_points.flatten()
            # print("feauture vector: ", feature_vector)
            Ical = (1/len(hull_points))*np.matlib.repmat(np.eye(2),1,len(hull_points))                         
            # print("Ical: ", Ical)
            bayrcenter_features = np.dot(Ical, feature_vector)
            # print("barycenter features: ", bayrcenter_features)
            # x = (hull_points[0,0]-self.cu)+(hull_points[len(hull_points)-1,0]-self.cu)-2*(bayrcenter_features[0]-self.cu)
            x = (hull_points[d,0]-self.cu)+(hull_points[f,0]-self.cu)-2*(bayrcenter_features[0]-self.cu)
            # y = (hull_points[0,1]-self.cv) + (hull_points[len(hull_points)-1,1]-self.cv) - 2*(bayrcenter_features[1]-self.cv)
            y = (hull_points[d,1]-self.cv) + (hull_points[f,1]-self.cv) - 2*(bayrcenter_features[1]-self.cv)
            tangent = -x/(-y)
            # tangent = y/x
            angle_radian = math.atan2(-x,-y)
            # angle_radian = math.atan2(y,x)
            angle_deg = math.degrees(angle_radian)
            # print("tangent:", tangent)
            # print("angle_radian:", angle_radian)
            # print("angle_deg:", angle_deg)
            
            determinants = np.zeros(len(hull_points))
            # print("length of vertices: ", len(hull_points))
            last_det = np.linalg.det([[hull_points[0,0], hull_points[len(hull_points)-1,0]],[hull_points[0,1], hull_points[len(hull_points)-1,1]]])
            determinants[len(hull_points)-1] = last_det
            # print("determinants vector: ", determinants)
            
            for i in range(0,len(hull_points)-1,1):
                  # print("determinant index: ", i)
                  determinants[i] = np.linalg.det([[hull_points[i,0], hull_points[i+1,0]],[hull_points[i,1], hull_points[i+1,1]]])
                  # print("determinants[i]: ", determinants[i])
                  
            # print("determinants vector: ", determinants)
            
            custom_sigma = 0.5*sum(determinants)
            custom_sigma_square =  math.sqrt(custom_sigma)
            custom_sigma_square_log = np.log(custom_sigma_square)
            # print("custom_sigma:", custom_sigma)
            # print("custom_sigma_square:", custom_sigma_square)
            # print("custom_sigma_square_log:", custom_sigma_square_log)       
            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
            
            
            # ------------------------------------------------------------------------------------------------
            # Polygon features transformation - Features, Angle and Area re-calculation
            # ------------------------------------------------------------------------------------------------
            
            feature_cartesian = self.cartesian_from_pixel(feature_vector, self.cu, self.cv, self.ax, self.ay)
            # print("features in cartesian from pixel: ", feature_cartesian)
            feature_cartesian_virtual = self.featuresTransformation(feature_cartesian, self.phi_imu, self.theta_imu)
            # print("feature_cartesian_virtual: ", feature_cartesian_virtual)
            feature_pixel_virtual = self.pixels_from_cartesian(feature_cartesian_virtual, self.cu, self.cv, self.ax, self.ay)
            # print("feature_pixel_virtual: ", feature_pixel_virtual)
            # print("\n")           
            
            K = len(hull_points)
            # print("K: " , K)
            hull_points_transformed = np.zeros((int(K),3))
            # print("hull_points_transformed: ", hull_points_transformed)
            
            m = 0
            
            for i in range(0,K,1):
                 hull_points_transformed[i,0] =  feature_pixel_virtual[m]
                 hull_points_transformed[i,1] =  feature_pixel_virtual[m+1]
                 hull_points_transformed[i,2] =  feature_pixel_virtual[m+2]
                 m+=3                        
            # print("after hull_points_transformed: ", hull_points_transformed)

            normalized_feature_vector, normalized_hull_points = self.feature_normalization(feature_pixel_virtual, hull_points_transformed, self.cu, self.cv, self.ax, self.ay)
            # print("normalized feature vector: ", normalized_feature_vector)
            # print("normalized_hull_points: ", normalized_hull_points)
            # print("self.z: ", self.z)
            
            c_transformed = np.argsort(normalized_hull_points[:,1])
            d_transformed = c_transformed[0]
            f_transformed = c_transformed[1]
            # print("1st new index:", d_transformed)
            # print("2nd new index:", f_transformed)
            # print("1st np.argsort(hull_points_transformed[:,1]): ", c)       
            
            L = len(feature_vector)
            transformed_features_only_image_plane = np.zeros(L)     
            
            for i,j in zip(range(0,L-1,2),range(0,K,1)):
                transformed_features_only_image_plane[i] = normalized_hull_points[j,0]
            for i,j in zip(range(1,L,2),range(0,K,1)):
                transformed_features_only_image_plane[i] = normalized_hull_points[j,1]
                
            # print("transformed_features_only_image_plane: ", transformed_features_only_image_plane)
            
            transformed_barycenter_features = np.dot(Ical, transformed_features_only_image_plane)
            # print("transformed barycenter features: ", transformed_barycenter_features)
            
            # print("normalized_hull_points: ", normalized_hull_points)
            # print("normalized_feature_vector: ", normalized_feature_vector)
            
            transformed_x = (normalized_hull_points[d_transformed,0]-self.cu)+(normalized_hull_points[f_transformed,0]-self.cu)-2*(transformed_barycenter_features[0]-self.cu)
            transformed_y = (normalized_hull_points[d_transformed,1]-self.cv) + (normalized_hull_points[f_transformed,1]-self.cv) - 2*(transformed_barycenter_features[1]-self.cv)
            # transformed_tangent = transformed_y/transformed_x
            # print("transformed_y/transformed_x: ", transformed_y/transformed_x)
            transformed_tangent = -transformed_x/(-transformed_y)
            # print("transformed_y/transformed_x: ", transformed_y/transformed_x)
            transformed_angle_radian = math.atan2(transformed_y,transformed_x)
            transformed_angle_radian = math.atan2(-transformed_x,-transformed_y)
            transformed_angle_deg = math.degrees(transformed_angle_radian)
            # print("transformed_tangent:", transformed_tangent)
            # print("transformed_angle_radian:", transformed_angle_radian)
            # print("transformed_angle_deg:", transformed_angle_deg)
                        
            transformed_determinants = np.zeros(len(normalized_hull_points))
            # print("length of vertices: ", len(normalized_hull_points))
            last_transformed_det = np.linalg.det([[normalized_hull_points[0,0], normalized_hull_points[len(normalized_hull_points)-1,0]],[normalized_hull_points[0,1], normalized_hull_points[len(normalized_hull_points)-1,1]]])
            transformed_determinants[len(normalized_hull_points)-1] = last_transformed_det
            # print("transformed_determinants vector: ", transformed_determinants)
            
            for i in range(0,len(normalized_hull_points)-1,1):
                  # print("determinant index: ", i)
                  transformed_determinants[i] = np.linalg.det([[normalized_hull_points[i,0], normalized_hull_points[i+1,0]],[normalized_hull_points[i,1], normalized_hull_points[i+1,1]]])
                  # print("transformed_determinants[i]: ", transformed_determinants[i])
                  
            # print("transformed_determinants vector: ", transformed_determinants)
            
            transformed_sigma = 0.5*sum(transformed_determinants)
            transformed_sigma_square =  math.sqrt(transformed_sigma)
            transformed_sigma_square_log = np.log(transformed_sigma_square)
            # print("transformed_sigma:", transformed_sigma)
            # print("transformed_sigma_square:", transformed_sigma_square)
            # print("transformed_sigma_square_log:", transformed_sigma_square_log)

            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
                        
            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
            
            # print("\n")
            
            # ------------------------------------------------------------------------------------------------
            # box features when using box = cv2.boxPoints(blackbox)
            # ------------------------------------------------------------------------------------------------
            # pred_data = PREDdata()
            self.pred_data.box_1 = box[0][:]
            # print("box_1 = ", self.pred_data.box_1)
            self.pred_data.box_2 = box[1][:]
            # print("box_2 = ", self.pred_data.box_2)
            self.pred_data.box_3 = box[2][:]
            # print("box_3 = ", self.pred_data.box_3)
            self.pred_data.box_4 = box[3][:]
            # print("box_4 = ", self.pred_data.box_4)
            
            self.pred_data.cX = cX
            self.pred_data.cY = cY
            self.pred_data.alpha = alpha
            self.pred_data.sigma = sigma
            self.pred_data.sigma_square = sigma_square
            self.pred_data.sigma_square_log = sigma_square_log
            
            
            # print("self.pred_data.cX: ", self.pred_data.cX)
            # print("self.pred_data.cY: ", self.pred_data.cY)
            # print("self.pred_data.alpha: ", self.pred_data.alpha)
            # print("self.pred_data.sigma: ", self.pred_data.sigma)
            # print("self.pred_data.sigma_square: ", self.pred_data.sigma_square)
            # print("self.pred_data.sigma_square_log: ", self.pred_data.sigma_square_log)
            
            # print("Normalized centroid x: ", (self.pred_data.cX-self.cu)/self.ax)
            # print("Normalized centroid y: ", (self.pred_data.cY-self.cv)/self.ay)
            
            self.pub_pred_data.publish(self.pred_data)
            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
            
            
            
            # ------------------------------------------------------------------------------------------------
            # polygon feature calculation WITHOUT virtual image transformation
            # ------------------------------------------------------------------------------------------------
            hull_points = np.array(hull_points, dtype=np.int_)
            stacked_feature_vector = hull_points.flatten('A')            
            # print("stacked_feature_vector: ", stacked_feature_vector)
            # print("feature_vector: ", feature_vector)
            self.polycalc_custom.features = stacked_feature_vector
            # self.polycalc_custom.features = feature_vector
            self.polycalc_custom.barycenter_features = bayrcenter_features
            self.polycalc_custom.d = d 
            self.polycalc_custom.f = f
            
            self.polycalc_custom.custom_sigma = custom_sigma
            self.polycalc_custom.custom_sigma_square = custom_sigma_square
            self.polycalc_custom.custom_sigma_square_log = custom_sigma_square_log
            
            self.polycalc_custom.tangent = tangent
            self.polycalc_custom.angle_radian = angle_radian
            self.polycalc_custom.angle_deg = angle_deg
            
            # print("self.polycalc_custom.features: ", self.polycalc_custom.features)
            # print("self.polycalc_custom.barycenter_features ", self.polycalc_custom.barycenter_features)
            # print("self.polycalc_custom.d: ", self.polycalc_custom.d)
            # print("self.polycalc_custom.f: ", self.polycalc_custom.f)
            
            # print("self.polycalc_custom.custom_sigma: ", self.polycalc_custom.custom_sigma)
            # print("self.polycalc_custom.custom_sigma_square: ", self.polycalc_custom.custom_sigma_square)
            # print("self.polycalc_custom.custom_sigma_square_log: ", self.polycalc_custom.custom_sigma_square_log)
            
            # print("self.polycalc_custom.tangent: ", self.polycalc_custom.tangent)
            # print("self.polycalc_custom.angle_radian: ", self.polycalc_custom.angle_radian)
            # print("self.polycalc_custom.angle_deg: ", self.polycalc_custom.angle_deg)
            
            self.pub_polycalc_custom.publish(self.polycalc_custom)
            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
            
            
            # ------------------------------------------------------------------------------------------------
            # polygon feature calculation WITH virtual image transformation
            # ------------------------------------------------------------------------------------------------
            # transformed_features_only_image_plane = np.array(transformed_features_only_image_plane, dtype=np.int_)
            # print("transformed_features_only_image_plane: ", transformed_features_only_image_plane)
            # print("transformed_features_only_image_plane.shape: ", transformed_features_only_image_plane.shape)
            # print("transformed_features_only_image_plane.dtype: ", transformed_features_only_image_plane.dtype)
            # a = np.array(transformed_features_only_image_plane, dtype=np.uint32)
            # print("a: ", a)
            # print("a.shape: ", a.shape)
            # print("a.dtype: ", a.dtype)
            self.polycalc_custom_tf.transformed_features = transformed_features_only_image_plane
            # self.polycalc_custom_tf.transformed_features = a
            self.polycalc_custom_tf.transformed_barycenter_features = transformed_barycenter_features
            self.polycalc_custom_tf.d_transformed = d_transformed
            self.polycalc_custom_tf.f_transformed = f_transformed
            
            self.polycalc_custom_tf.transformed_sigma = transformed_sigma
            self.polycalc_custom_tf.transformed_sigma_square = transformed_sigma_square
            self.polycalc_custom_tf.transformed_sigma_square_log = transformed_sigma_square_log
            
            self.polycalc_custom_tf.transformed_tangent = transformed_tangent
            self.polycalc_custom_tf.transformed_angle_radian = transformed_angle_radian
            self.polycalc_custom_tf.transformed_angle_deg = transformed_angle_deg
            self.polycalc_custom_tf.moments = moments
            
            print("self.polycalc_custom_tf.transformed_features: ", self.polycalc_custom_tf.transformed_features)
            print("self.polycalc_custom_tf.transformed_barycenter_features: ", self.polycalc_custom_tf.transformed_barycenter_features)
            print("self.polycalc_custom_tf.d_transformed ", self.polycalc_custom_tf.d_transformed)
            print("self.polycalc_custom_tf.f_transformed: ", self.polycalc_custom_tf.f_transformed)
            
            print("self.polycalc_custom_tf.transformed_sigma: ", self.polycalc_custom_tf.transformed_sigma)
            print("self.polycalc_custom_tf.transformed_sigma_square: ", self.polycalc_custom_tf.transformed_sigma_square)
            print("self.polycalc_custom_tf.transformed_sigma_square_log: ", self.polycalc_custom_tf.transformed_sigma_square_log)
            
            print("self.polycalc_custom_tf.transformed_tangent: ", self.polycalc_custom_tf.transformed_tangent)
            print("self.polycalc_custom_tf.transformed_angle_radian: ", self.polycalc_custom_tf.transformed_angle_radian)
            print("self.polycalc_custom_tf.transformed_angle_deg: ", self.polycalc_custom_tf.transformed_angle_deg)
            print("self.polycalc_custom_tf.moments = ", self.polycalc_custom_tf.moments)
            
            self.pub_polycalc_custom_tf.publish(self.polycalc_custom_tf)
            # ------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------
            
            # print("\n")
            
            # cv2.imshow("Combined prediction", combo_image)
            # cv2.imshow("Prediction image window", segimg) 
            # cv2.imshow("Bounded contour image", ros_image) 
            cv2.waitKey(3)

    try:
      open_cv_image = self.bridge.cv2_to_imgmsg(segimg, "bgr8")
      open_cv_image.header.stamp = data.header.stamp
      self.image_pub_first_image.publish(open_cv_image)
      
      combo_open_cv_image = self.bridge.cv2_to_imgmsg(combo_image, "bgr8")
      combo_open_cv_image.header.stamp = data.header.stamp
      self.image_pub_second_image.publish(combo_open_cv_image)
      
      # cv2.drawContours(segimg, [box], 0, (0, 0, 255), 1)
      # cv2.drawContours(segimg, [approx], 0, (0, 0, 255), 1)
      cv2.drawContours(segimg, [hull], 0, (0, 0, 255), 1)
      cv2.line(segimg, (int(x_min), 54), (int(x_min), 74), (255, 0, 0), 1)
      # cv2.drawContours(combo_image, [box], 0, (0, 0, 255), 1)
      # cv2.line(combo_image, (int(x_min), 54), (int(x_min), 74), (255, 0, 0), 1)
      
      cv_image = cv2.resize(segimg, (720, 480))
      ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
      ros_image.header.stamp = data.header.stamp
      self.ros_image_pub.publish(ros_image)
    except CvBridgeError as e:
      print(e)



def main(args):
      
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
      tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
      tf.config.experimental.set_virtual_device_configuration( gpus[0],
                                                              [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU \n")
    except RuntimeError as e:
      # Visible devices must be set before GPUs have been initialized
      print(e)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config = config)
  gpu = tf.test.gpu_device_name()

  # Check available GPU devices.
  print("The following GPU devices are available: %s" % tf.test.gpu_device_name())
  
  global mdl
  # mdl = model_from_checkpoint_path("src/img_seg_cnn/model_checkpoints/mobilenet_unet/mobilenet_unet_224")
  mdl = model_from_checkpoint_path("src/img_seg_cnn/model_checkpoints/mobilenet_segnet/mobilenet_segnet224")
  global graph
  graph = tf.get_default_graph()

  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)