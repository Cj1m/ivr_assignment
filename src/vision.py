#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters


class vision:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing2', anonymous=True)
    #Subscribe to camera 1 output processed by image1.py
    self.image_sub1 = message_filters.Subscriber("/image_topic1",Image)
    #Subscribe to camera 2 output processed by image2.py
    self.image_sub2 = message_filters.Subscriber("/image_topic2",Image)

    # Synchronize subscriptions into one callback
    ts = message_filters.TimeSynchronizer([self.image_sub1, self.image_sub2], 1)
    ts.registerCallback(self.callback)

    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    #Get the time
    self.time_trajectory = rospy.get_time()
    #Joint 1 co-ordinates
    self.joint1_pos = {"x": 0, "y" : 0, "z": 0}
    # Joint 23 co-ordinates
    self.joint23_pos = {"x": 0, "y": 0, "z": 0}
    # Joint 4 co-ordinates
    self.joint4_pos = {"x": 0, "y": 0, "z": 0}
    # Joint EE co-ordinates
    self.jointEE_pos = {"x": 0, "y": 0, "z": 0}

  # Receive data from camera 1 and camera 2
  def callback(self, data1, data2):
    # Receive image 1

    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")
    except CvBridgeError as e:
      print(e)

    #cv2.imshow("ftp", self.cv_image1)
    cv2.waitKey(1)
    #Get each joints co-ordinates from camera 1
    image1_joint1 = self.get_joint_coordinates(self.cv_image1, (0,10,10), (5, 255, 255))
    image1_joint23 = self.get_joint_coordinates(self.cv_image1, (10,0,0), (255, 5, 5))
    image1_joint4 = self.get_joint_coordinates(self.cv_image1, (0,10,0), (5, 255, 5))
    image1_jointEE = self.get_joint_coordinates(self.cv_image1, (0,0,10), (5, 5, 255))

    #Get each joints co-ordinates from camera 2
    image2_joint1 = self.get_joint_coordinates(self.cv_image2, (0, 10, 10), (5, 255, 255))
    image2_joint23 = self.get_joint_coordinates(self.cv_image2, (10, 0, 0), (255, 5, 5))
    image2_joint4 = self.get_joint_coordinates(self.cv_image2, (0, 10, 0), (5, 255, 5))
    image2_jointEE = self.get_joint_coordinates(self.cv_image2, (0, 0, 10), (5, 5, 255))

    # Compare co-ordinates obtained from camera 1 and camera 2
    # Update each joints global variable position
    self.set_coordinates(image1_joint1, image2_joint1, self.joint1_pos)
    self.set_coordinates(image1_joint23, image2_joint23, self.joint23_pos)
    self.set_coordinates(image1_joint4, image2_joint4, self.joint4_pos)
    self.set_coordinates(image1_jointEE, image2_jointEE, self.jointEE_pos)

    print(self.jointEE_pos)

  def set_coordinates(self, image1_joint, image2_joint, joint_pos):
    joint_pos["x"] = image2_joint[0]
    joint_pos["y"] = image1_joint[0]
    # If one camera 1 is obscured then use the other camera
    if image1_joint[1] == 0:
      joint_pos["z"] = image2_joint[1]
    # If neither camera is obscured then set it to the mean of the two
    # If both joints are obscured then leave it unchanged
    elif image2_joint[1] != 0:
      joint_pos["z"] = (image2_joint[1] + image1_joint[1]) / 2

  def get_joint_coordinates(self, image, lower_thresh, upper_thresh):

    thresholded_image = cv2.inRange(image, lower_thresh, upper_thresh)
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
  v = vision()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)


