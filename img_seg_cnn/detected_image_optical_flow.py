#!/usr/bin/env python
import rospy
import sys
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from img_seg_cnn.msg import Optical_flow_custom


def process_image(msg):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    # print("img shape: ", img.shape)
    
    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally 
    # expensive
    prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("prev_gray shape: ", prev_gray.shape)
    
    # Creates an image filled with zero
    # intensities with the same dimensions 
    # as the frame
    mask = np.zeros_like(img)
    
    # Sets image saturation to maximum
    mask[..., 1] = 255
    
    # cv2.imshow("image",img)
        
    img = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    # Converts each frame to grayscale - we previously 
    # only converted the first frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                       None,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
    
    # print("flow: ", flow)
    print("mean of flow: ", np.mean(flow))
    
    final_flow = flow[flow != 0]
    
    # print("flow: ", final_flow)
    print("mean of final_flow: ", np.mean(final_flow))
          
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
      
    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
      
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
      
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
      
    # Opens a new window and displays the output frame
    # cv2.imshow("dense optical flow", rgb)
      
    # Updates previous frame
    prev_gray = gray
    
    msg_optical_flow = Optical_flow_custom()
    msg_optical_flow.mean_flow = np.mean(flow)
    msg_optical_flow.mean_final_flow = np.mean(final_flow)
    pub_optical_flow.publish(msg_optical_flow)
        
    cv2.waitKey(1)



if __name__ == '__main__':
    while not rospy.is_shutdown():
        rospy.init_node('image_sub')
        rospy.loginfo('image_sub node started')
        print("Subscribe me one more time")
        rospy.Subscriber("/image_raww", Image, process_image)
        pub_optical_flow = rospy.Publisher(
        "/optical_flow_output", Optical_flow_custom, queue_size=1000)
        # rospy.Rate(1000)
        rospy.spin()
