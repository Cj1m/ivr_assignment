#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import math
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters


class target_finder:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing2', anonymous=True)
        # Subscribe to camera 1 output processed by image1.py
        self.image_sub1 = message_filters.Subscriber("/image_topic1", Image)
        # Subscribe to camera 2 output processed by image2.py
        self.image_sub2 = message_filters.Subscriber("/image_topic2", Image)

        # Synchronize subscriptions into one callback
        ts = message_filters.TimeSynchronizer([self.image_sub1, self.image_sub2], 1)
        ts.registerCallback(self.callback)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # Get the time
        self.time_trajectory = rospy.get_time()


    # Receive data from camera 1 and camera 2
    def callback(self, data1, data2):
        # Receive image 1

        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.waitKey(3)
        cv2.imshow("image", self.cv_image1)

        complete_target1 = self.threshold_targets(self.cv_image1)
        complete_target2 = self.threshold_targets(self.cv_image2)

        #cv2.imshow("threshold", complete_target1)
        target_centers1 = self.find_centers(complete_target1)
        target_centers2 = self.find_centers(complete_target2)

        base_frame_xzcoords = self.get_base_frame_position(self.cv_image1)
        base_frame_yzcoords = self.get_base_frame_position(self.cv_image2)

        base_frame_center = [base_frame_xzcoords[0], base_frame_yzcoords[0], base_frame_yzcoords[1]]

    def find_centers(self, complete_target):
        cnts, hierarchy = cv2.findContours(complete_target, cv2.RETR_CCOMP,
                                cv2.CHAIN_APPROX_TC89_L1)

        centers = []
        for c in cnts:
           M = cv2.moments(c)
           cX = int(M["m10"] / M["m00"])
           cY = int(M["m01"] / M["m00"])
           centers.append((cX, cY))

        return centers

    def threshold_targets(self, image):
        # Threshold for both box and target on light background
        threshold_objects1 = cv2.inRange(image, (70, 100, 130), (120, 150, 160))
        # Threshold for target on dark background
        threshold_target2 = cv2.inRange(image, (27, 55, 70), (40, 70, 95))

        # Threshold box for when its behind robots
        threshold_box = cv2.inRange(image, (0, 7, 12), (0.5, 45, 65))

        complete_target = threshold_objects1 + threshold_target2 + threshold_box

        kernel = np.ones((2, 2), np.uint8)
        complete_target = cv2.erode(complete_target, kernel, iterations=1)
        complete_target = cv2.dilate(complete_target, kernel, iterations=3)

        return complete_target


    def get_base_frame_position(self, image):
        thresholded_image1 = cv2.inRange(image, (102, 102, 102), (140, 140, 140))

        kernel = np.ones((2, 2), np.uint8)
        thresholded_image = cv2.erode(thresholded_image1, kernel, iterations=1)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(thresholded_image, kernel, iterations=3)

        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)

        cnt = contours[0]
        M = cv2.moments(cnt)
        center_1 = int(M['m10'] / M['m00'])
        center_2 = int(M['m01'] / M['m00'])
        return [center_1, center_2]


# call the class
def main(args):
    v = target_finder()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


