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
    # Colour threshold dictionary
    self.upper_threshold = {'yellow':  (5, 255, 255), 'blue' : (255, 5, 5), 'green' : (5, 255, 5), 'red': (5, 5, 255) }
    #Colour lower threshold disctionary
    self.lower_threshold = {'yellow': (0,10,10), 'blue' : (10, 0, 0), 'green' :  (0, 10, 0), 'red': (0, 0, 10) }


  # Receive data from camera 1 and camera 2
  def callback(self, data1, data2):
    # Receive image 1

    try:
      self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")
    except CvBridgeError as e:
      print(e)

    cv2.imshow("image 1", self.cv_image1)
    cv2.imshow("image 2", self.cv_image2)
    cv2.waitKey(1)
    #Get each joints co-ordinates from camera 1
    image1_joint1 = self.get_joint_coordinates(self.cv_image1, "yellow")
    image1_joint23 = self.get_joint_coordinates(self.cv_image1, "blue" )
    image1_joint4 = self.get_joint_coordinates(self.cv_image1, "green" )
    image1_jointEE = self.get_joint_coordinates(self.cv_image1, "red")

    #Get each joints co-ordinates from camera 2
    image2_joint1 = self.get_joint_coordinates(self.cv_image2, "yellow")
    image2_joint23 = self.get_joint_coordinates(self.cv_image2,  "blue" )
    image2_joint4 = self.get_joint_coordinates(self.cv_image2, "green" )
    image2_jointEE = self.get_joint_coordinates(self.cv_image2,  "red")

    # Compare co-ordinates obtained from camera 1 and camera 2
    # Update each joints global variable position
    self.set_coordinates(image1_joint1, image2_joint1, self.joint1_pos)
    self.set_coordinates(image1_joint23, image2_joint23, self.joint23_pos)
    self.set_coordinates(image1_joint4, image2_joint4, self.joint4_pos)
    self.set_coordinates(image1_jointEE, image2_jointEE, self.jointEE_pos)

    print("Joint 2 position: " + str(self.joint23_pos))
    print("Joint 4 position: " + str(self.joint4_pos))

    link1_angle = self.get_angle_between_points(self.joint1_pos, self.joint23_pos)
    link2_angle = self.get_angle_between_points(self.joint23_pos, self.joint4_pos)

    # Get length of each link
    link1_dist_pixels  = self.distance_between_joints(self.joint1_pos, self.joint23_pos)
    link3_dist_pixels = self.distance_between_joints(self.joint23_pos, self.joint4_pos)
    link4_dist_pixels = self.distance_between_joints(self.joint4_pos, self.jointEE_pos)
    pixel_in_metres = 2/link1_dist_pixels

    #Get rotation matrix
    v1 = self.get_vector_between_joints(self.joint4_pos, self.joint23_pos)
    d = self.get_vector_between_joints(self.joint23_pos, self.joint1_pos)
    #print("v1:" + str(v1))
    #print("d:" + str(d))
    print(self.rotation_matrix_X(self.joint4_pos, v1, d))
    print(self.rotation_matrix_Y(self.joint4_pos, v1, d))
    print(self.rotation_matrix_Z(self.joint4_pos, v1, d))

    print(link4_dist_pixels*pixel_in_metres)

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

  def get_joint_coordinates(self, image, colour):

    thresholded_image = cv2.inRange(image, self.lower_threshold[colour], self.upper_threshold[colour])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(thresholded_image, kernel, iterations=3)
    
    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # If current joint is obscured by another joint use the co-ordinates of that joint
    if len(contours) == 0:
      colours = np.array(["yellow", "blue", "green", "red"])
      next_colour_position = np.char.find(colours, colour)[0] + 1
      if next_colour_position > 3:
        return self.get_joint_coordinates(image, colours[0])
      else:
         return self.get_joint_coordinates(image, colours[next_colour_position])
    else:
      cnt = contours[0]
      M = cv2.moments(cnt)
      center_1 = int(M['m10'] / M['m00'])
      center_2 = int(M['m01'] / M['m00'])
      return [center_1, center_2]

  # Returns the angle between two points in a 2D plane about a horizontal plane
  def get_angle_between_points(self, point1, point2):

    xz_angle = ((math.atan2(point2['z'] - point1['z'], point2['x'] - point1['x'])) + math.pi/2) % math.pi
    yz_angle = ((math.atan2(point2['z'] - point1['z'], point2['y'] - point1['y'])) + math.pi/2) % math.pi

    return [xz_angle, yz_angle]

  def distance_between_joints(self, point1, point2):
    return math.sqrt(math.pow(point2['x'] - point1['x'], 2) + math.pow(point2['y'] - point1['y'], 2)
                     + math.pow(point2['z'] - point1['z'], 2))

  def get_vector_between_joints(self, point1, point2):
    return [point2['x'] - point1['x'], point2['y'] - point1['y'], point2['z'] - point1['z']]

  def rotation_matrix_X(self, point2, v, d):
    [a,b,c] = d
    [x1, y1, z1] = v
    cos_alpha = (z1 * (point2['z'] - c)) / (math.pow(z1, 2) + math.pow(y1, 2))
    sin_alpha = (point2['y'] - b - y1*cos_alpha) / z1
    return [[1, 0, 0], [0, cos_alpha, sin_alpha], [0, -sin_alpha, cos_alpha]]

  def rotation_matrix_Y(self, point2, v, d):
    [a, b, c] = d
    [x1, y1, z1] = v
    cos_beta = (z1*(point2['z'] - c) + x1*point2['x'] - x1*a) / (math.pow(x1, 2) + math.pow(z1, 2))
    sin_beta = -((point2['x'] - a - x1*cos_beta) / z1)
    return [[cos_beta, 0, -sin_beta], [0, 1, 0], [sin_beta, 0, cos_beta]]

  def rotation_matrix_Z(self, point2, v, d):
    [a, b, c] = d
    [x1, y1, z1] = v

    cos_gamma = (y1*point2['y'] - y1*b + x1*point2['x'] - x1*a) / (math.pow(x1, 2) + math.pow(y1, 2))
    sin_gamma = (point2['x'] - a - x1*cos_gamma) / y1
    return [[cos_gamma, sin_gamma, 0], [-sin_gamma, cos_gamma, 0], [0, 0, 1]]

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


