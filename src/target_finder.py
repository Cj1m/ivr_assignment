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
        cv2.imshow("image", self.cv_image2)
        thresholded_image1 = cv2.inRange(self.cv_image2, (70, 100, 130), (120, 150, 160))
        thresholded_image2 = cv2.inRange(self.cv_image2, (27, 55, 70), (40, 70, 95))
        cv2.imshow("threshold", thresholded_image1)



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


