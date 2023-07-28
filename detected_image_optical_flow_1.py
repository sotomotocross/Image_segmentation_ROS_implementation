#!/usr/bin/env python
import rospy
import sys
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class optical_flow:

    def __init__(self):
        self.image_sub = rospy.Subscriber("/image_raww", Image, self.process_image)

    def process_image(self, msg):
        bridge = CvBridge()
        self.img = bridge.imgmsg_to_cv2(msg, "bgr8")
        print("img: ", self.img)
        print("img shape: ", self.img.shape)
        cv2.imshow("image",self.img)
        cv2.waitKey(1)


# if __name__ == '__main__':
#     while not rospy.is_shutdown():
#         rospy.init_node('image_sub')
#         rospy.loginfo('image_sub node started')
#         # print("Subscribe me one more time")
#         # rospy.Subscriber("/image_raww_comb", Image, process_image)
#         image_sub = rospy.Subscriber("/image_raww", Image, process_image)
#         # rospy.Subscriber("/iris_demo/ZED_stereocamera/camera/left/image_raw", Image, process_image)

#         print("image_sub shape: ", test_img)
#         print("img shape: ", test_img)

#         # Converts frame to grayscale because we
#         # only need the luminance channel for
#         # detecting edges - less computationally
#         # expensive
#         # prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
#         # # print("prev_gray shape: ", prev_gray.shape)

#         # # Creates an image filled with zero
#         # # intensities with the same dimensions
#         # # as the frame
#         # mask = np.zeros_like(img)

#         # # Sets image saturation to maximum
#         # mask[..., 1] = 255

#         # while(first.any()):

#         #     cv2.imshow("image",img)
#         #     bridge = CvBridge()
#         #     img = bridge.imgmsg_to_cv2(msg, "bgr8")

#         #     # Converts each frame to grayscale - we previously
#         #     # only converted the first frame to grayscale
#         #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         #     # Calculates dense optical flow by Farneback method
#         #     flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
#         #                                     None,
#         #                                     0.5, 3, 15, 3, 5, 1.2, 0)

#         #     # Computes the magnitude and angle of the 2D vectors
#         #     magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

#         #     # Sets image hue according to the optical flow
#         #     # direction
#         #     mask[..., 0] = angle * 180 / np.pi / 2

#         #     # Sets image value according to the optical flow
#         #     # magnitude (normalized)
#         #     mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

#         #     # Converts HSV to RGB (BGR) color representation
#         #     rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

#         #     # Opens a new window and displays the output frame
#         #     cv2.imshow("dense optical flow", rgb)

#         #     # Updates previous frame
#         #     prev_gray = gray

#         rospy.spin()


def main(args):

  of = optical_flow()
  rospy.init_node('image_sub for optical flow', anonymous=True)
  try:
    rospy.Rate(1)
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
