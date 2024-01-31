from __future__ import print_function
from keras_segmentation.predict import model_from_checkpoint_path
from keras_segmentation.predict import predict
from math import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import tensorflow as tf
import numpy as np
import cv2
import rospy
from feature_calculator import FeatureCalculator

import roslib
roslib.load_manifest('img_seg_cnn')

red = (153, 0, 18)

class DetectionProcessor:
    """
    Image detection processor class.

    This class is responsible for processing images using a segmentation model
    and extracting features using a FeatureCalculator.

    Attributes:
        mdl: Segmentation model for image processing.
        graph: TensorFlow graph for the model.
        feature_calculator: Instance of FeatureCalculator for feature extraction.
    """
    
    def __init__(self, mdl, graph):
        """
        Initialize the DetectionProcessor.

        Args:
            mdl (str): Path to the segmentation model defined in detection_node.py.
            graph: TensorFlow graph for the model (default is None).
        """
        self.mdl = mdl
        self.graph = graph
        
        self.init_ros()
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Generate instances of classes
        self.feature_calculator = FeatureCalculator()
        
    def init_ros(self):
        """
        Initialize ROS-related components, including subscribers, publishers, and variables.
        """
        
        # Create subscribers
        self.image_sub = rospy.Subscriber("/iris_demo/ZED_stereocamera/camera/left/image_raw", Image, self.callback)
        
        # Create publishers
        self.image_pub_first_image = rospy.Publisher("/image_raww", Image, queue_size=10)
        self.image_pub_second_image = rospy.Publisher("/image_raww_comb", Image, queue_size=10)
        self.ros_image_pub = rospy.Publisher("image_bounding_box", Image, queue_size=10)             

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

    def callback(self, data):
        """
        ROS callback function for processing incoming image data.

        Args:
            data (sensor_msgs.msg.Image): Incoming image data.
        """
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
    
        # Perform image processing and detection
        processed_image = self.process_image(cv_image)


    def process_image(self, cv_image):
        """
        Process the incoming image using the segmentation model and extract features.

        Args:
            cv_image (numpy.array): Input image in OpenCV format.
        
        Returns:
            tuple: Extracted features, including predictions, feature vectors, and transformed features.
        """

        
        
        def create_blank(width, height, rgb_color=(0, 0, 0)):
            # Create black blank image
            image = np.zeros((height, width, 3), np.uint8)
            # Since OpenCV uses BGR, convert the color first
            color = tuple(reversed(rgb_color))
            # Fill image with color
            image[:] = color
            return image

        with self.graph.as_default():
            pr, seg_img = predict( 
                model=self.mdl, 
                inp=cv_image 
            )
        segimg = seg_img.astype(np.uint8)

        copy_image = np.copy(cv_image)
        copy_mask = np.copy(segimg)
        ww = copy_mask.shape[1]
        hh = copy_mask.shape[0]
        red_mask = create_blank(ww, hh, rgb_color=red)
        copy_mask = cv2.bitwise_and(copy_mask, red_mask)
        combo_image = cv2.addWeighted(copy_image, 1, copy_mask, 1, 1)

        mask = cv2.inRange(segimg, (130, 130, 130), (255, 255, 255))
        kernel = np.ones((1, 1), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=3)
        mask = cv2.dilate(mask, kernel, iterations=3)
        
        contours_blk, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours_blk]
        max_index = np.argmax(areas)
        if len(contours_blk) > 0 and cv2.contourArea(contours_blk[max_index]) > 200:
              # Box creation for the detected coastline
            blackbox = cv2.minAreaRect(contours_blk[max_index])
            (x_min, y_min), (w_min, h_min), angle = blackbox
            hull = cv2.convexHull(contours_blk[max_index])
        
        # Publish the processed images
        try:
            open_cv_image = self.bridge.cv2_to_imgmsg(segimg, "bgr8")
            open_cv_image.header.stamp = rospy.Time.now()
            self.image_pub_first_image.publish(open_cv_image)

            combo_open_cv_image = self.bridge.cv2_to_imgmsg(combo_image, "bgr8")
            combo_open_cv_image.header.stamp = rospy.Time.now()
            self.image_pub_second_image.publish(combo_open_cv_image)

            cv2.drawContours(segimg, [hull], 0, (0, 0, 255), 1)
            cv2.line(segimg, (int(x_min), 54), (int(x_min), 74), (255, 0, 0), 1)

            cv_image_resized = cv2.resize(segimg, (720, 480))
            ros_image = self.bridge.cv2_to_imgmsg(cv_image_resized, "bgr8")
            ros_image.header.stamp = rospy.Time.now()
            self.ros_image_pub.publish(ros_image)
        except CvBridgeError as e:
            print(e)
        
        # Display the processed images for debugging/visualization
        cv2.imshow("Combined prediction", combo_image)
        cv2.imshow("Prediction image window", segimg) 
        cv2.waitKey(3)
        
        # Extract features using the FeatureCalculator class
        extracted_features = self.feature_calculator.calculate_features(contours_blk, max_index)

        # Unpack the extracted features dictionary
        (features_pred_data, stacked_feature_vector, barycenter_features, features_polycalc_custom, transformed_features_only_image_plane, transformed_barycenter_features, features_polycalc_custom_tf) = extracted_features

        # Publish the extracted features
        self.feature_calculator.publish_extracted_features(features_pred_data, stacked_feature_vector, barycenter_features, features_polycalc_custom, transformed_features_only_image_plane, transformed_barycenter_features, features_polycalc_custom_tf)


        return extracted_features
