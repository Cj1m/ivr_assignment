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
from sympy import *
from mpmath import *
from sympy.abc import x,y,z,w
import message_filters


class vision:

  # Defines publisher and subscriber
  def __init__(self):
    # initialize the node named image_processing
    rospy.init_node('image_processing2', anonymous=True)
    #Create new topic for publishing rotation matrices to
    self.rot_pub = rospy.Publisher("rot_pub", Float64MultiArray, queue_size = 1)
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

    # cv2.imshow("image 1", self.cv_image1)
    # cv2.imshow("image 2", self.cv_image2)
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

    link1_vec = {"x": self.joint23_pos['x'] - self.joint1_pos['x'],
                 "y": self.joint23_pos['y'] - self.joint1_pos['y'],
                 "z": self.joint23_pos['z'] - self.joint1_pos['z']}
    link2_vec = {"x": self.joint4_pos['x'] - self.joint23_pos['x'],
                 "y": self.joint4_pos['y'] - self.joint23_pos['y'],
                 "z": self.joint4_pos['z'] - self.joint23_pos['z']}
    link3_vec = {"x": self.jointEE_pos['x'] - self.joint4_pos['x'],
                 "y": self.jointEE_pos['y'] - self.joint4_pos['y'],
                 "z": self.jointEE_pos['z'] - self.joint4_pos['z']}

    # link1_angle = self.get_angle_between_points(self.joint1_pos, self.joint23_pos)
    # link2_angle = self.get_angle_between_points(self.joint23_pos, self.joint4_pos)
    # link3_angle = self.get_angle_between_points(self.joint4_pos, self.jointEE_pos)

    link1_angle = [0,0]
    link2_angle = self.get_angle_between_points(link1_vec, link2_vec)
    link3_angle = self.get_angle_between_points(link2_vec, link3_vec)

    print (link1_angle)

    # Get length of each link
    link1_dist_pixels = self.distance_between_joints(self.joint1_pos, self.joint23_pos)
    link3_dist_pixels = self.distance_between_joints(self.joint23_pos, self.joint4_pos)
    link4_dist_pixels = self.distance_between_joints(self.joint4_pos, self.jointEE_pos)
    pixel_in_metres = 2/link1_dist_pixels

    #print("Actual EE: " + str(self.jointEE_pos))
   # print("Actual 23: " + str(self.joint23_pos))
   # print("Actual 4: " + str(self.joint4_pos))

    # print("Estimated: " + str(self.get_EE_with_forward_kinematics([link1_angle, link2_angle, link3_angle],
    #                                           [link1_dist_pixels, link3_dist_pixels, link4_dist_pixels],
    #                                           [self.joint1_pos['x'], self.joint1_pos['y'], self.joint1_pos['z']]
    #                                           )))

    print(link1_angle)
    print(link2_angle)
    print(link3_angle)

    link1_angle = [0,0]
    link2_angle = [0,0]
    link3_angle = [0,0]

    """
    w = link1Angle
    x= link2_angle[1]
    y= link2_angle[0]
    z=link3_angle[1]
    """

    x, y, z, w = symbols('x y z w')

    rot_2 = [[1, 0, 0],
                     [0, math.cos(link2_angle[1]), -math.sin(link2_angle[1])],
                     [0, math.sin(link2_angle[1]), math.cos(link2_angle[1])]]
    rot_3 = [[math.cos(link2_angle[0]), 0, math.sin(link2_angle[0])],
                    [0, 1, 0],
                    [-math.sin(link2_angle[0]), 0, math.cos(link2_angle[0])]]
    #print(np.matmul(rot_2, rot_3))
    a12 = np.matrix([[1, 0, 0, 0],
                     [0, math.cos(x), -math.sin(x), 0],
                     [0, math.sin(x), math.cos(x), link1_dist_pixels*pixel_in_metres],
                     [0, 0, 0, 1]])
    a23 = np.matrix([[math.cos(y), 0, math.sin(y), 0],
                    [0, 1, 0, 0],
                    [-math.sin(y), 0, math.cos(y), 0],
                    [0, 0, 0, 1]])

    a34 = np.matrix([[1, 0, 0, 0],
                    [0, math.cos(z), -math.sin(z), 0],
                    [0, math.sin(z), math.cos(z), link3_dist_pixels*pixel_in_metres],
                    [0, 0, 0, 1]])
    a12 = Matrix([[1, 0, 0, 0],
                  [0, math.cos(x), -math.sin(x), 0],
                  [0, math.sin(x), math.cos(x), 2],
                  [0, 0, 0, 1]])
    a23 = Matrix([[math.cos(y), 0, math.sin(y), 0],
                  [0, 1, 0, 0],
                  [-math.sin(y), 0, math.cos(y), 0],
                  [0, 0, 0, 1]])

    a34 = Matrix([[1, 0, 0, 0],
                  [0, math.cos(z), -math.sin(z), 0],
                  [0, math.sin(z), math.cos(z), 3],
                  [0, 0, 0, 1]])
    p4 = [0, 0, 2, 1]
    link1Angle = self.find_Z_angle(self.jointEE_pos, a12, a23, a34, p4)

    a01 = Matrix([[math.cos(w), -math.sin(w), 0, 0],
                  [math.sin(w), math.cos(w), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    p4 = Matrix([0, 0, link4_dist_pixels*pixel_in_metres, 1])
    link1Angle = self.find_Z_angle(self.jointEE_pos, a12, a23, a34, p4)

    a01 = np.matrix([[math.cos(w), -math.sin(w), 0, 0],
                     [math.sin(w), math.cos(w), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    x = np.matmul(np.matmul(np.matmul(np.matmul(a01, a12), a23), a34), p4)
    print (x)
    print ([link1Angle, link2_angle[1], link2_angle[0], link3_angle[1]])
    print (self.get_jacobian([link1Angle, link2_angle[1], link2_angle[0], link3_angle[1]], x))
    # print("Link 1 angle: " + str(self.find_Z_angle(self.jointEE_pos, a12, a23, a34, p4)))
    # print("Link 2 angle: " + str(link2_angle[1]))
    # print("Link 3 angle: " + str(link2_angle[0]))
    # print("Link 4 angle: " + str(link3_angle[1]))


    # #Get rotation matrix
    # v1 = self.get_vector_between_joints(self.joint23_pos, self.joint4_pos)
    # d = self.get_vector_between_joints(self.joint1_pos, self.joint23_pos)
    #
    # joint2_rot_matrix = self.rotation_matrix_X(v1, d);

    # Publish the results
    # try:
    #   self.rot_pub.publish(np.array(joint2_rot_matrix).flatten(), "bgr8")
    # except CvBridgeError as e:
    #   print(e)

    # print (type(self.rotation_matrix_X(v1, d)))
    # print ("Base Vector: " + str(-d))
    # print ("Vector T: " + str(-v1))
    # print("Rotation matrix X: " + str(self.rotation_matrix_X(v1, d)))
    # print("Rotation matrix Y: " + str(self.rotation_matrix_Y(v1, d)))
    # print("Rotation matrix Z: " + str(self.rotation_matrix_Z(v1, d)))
    #
    # print(link4_dist_pixels*pixel_in_metres)

  """Algorithm works when accurate angle measurements for joints 2,3,4 are used"""
  def find_Z_angle(self, EE, a12, a23, a34, p4):
    closest_angle = 400000000000000000000000000
    closest_dist = 10000000000000000000000000

    #Find end-effector position relative to joint 1
    EE = {"x": EE['x'] - self.joint1_pos['x'], "y": EE['y'] - self.joint1_pos['y'], "z": self.joint1_pos['z'] - EE['z']}
    EEVector = [EE['x'], EE['y'], EE['z']]

    for cur_z_degrees in range(-180, 180):
      cur_z = cur_z_degrees * (math.pi / 180)
      a01 = np.matrix([[math.cos(cur_z), -math.sin(cur_z), 0, 0],
                       [math.sin(cur_z), math.cos(cur_z), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
      estimated_EE = np.matmul(np.matmul(np.matmul(np.matmul(a01, a12), a23), a34), p4)
      estimated_EE = ([estimated_EE[0, 0], estimated_EE[0, 1], estimated_EE[0, 2]])

      if (np.linalg.norm(np.subtract(np.array(estimated_EE), np.array(EEVector))) < closest_dist):

        closest_angle = cur_z_degrees
        closest_dist = np.linalg.norm(np.subtract(np.array(estimated_EE), np.array(EEVector)))

    return (closest_angle * (math.pi / 180))

  def get_jacobian(self, joint_angles, x):
    x, y, z, w = symbols('x y z w')
    t1_val, t2_val, t3_val, t4_val = joint_angles
    return (Matrix(x[:3,3]).jacobian([x, y, z, w])).subs([(x, t1_val), (y, t2_val), (z, t3_val), (w, t4_val)])

  def get_EE_with_forward_kinematics(self, link_angles, link_distances, joint1_pos):
    print(link_angles)
    kinematics_joint23_pos = self.get_next_point_pos(joint1_pos, link_distances[0], link_angles[0])
    print ("Estimated Joint23 Pos; " + str(kinematics_joint23_pos))
    kinematics_joint4_pos = self.get_next_point_pos(kinematics_joint23_pos, link_distances[1], link_angles[1])
    print ("Estimated Joint4 Pos; " + str(kinematics_joint4_pos))
    kinematics_jointEE_pos = self.get_next_point_pos(kinematics_joint4_pos, link_distances[2], link_angles[2])

    return kinematics_jointEE_pos

  def get_next_point_pos(self, curPos, linkLength, angle):
      #print (linkLength)
      x = math.sin(angle[0]) * math.cos(angle[1]) * linkLength
      y = math.sin(angle[1]) * linkLength
      z = math.cos(angle[0]) * math.cos(angle[1]) * linkLength

      return [curPos[0] + x, curPos[1] + y, curPos[2] - z]

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

    #xz_angle = ((math.atan2(point2['z'] - point1['z'], point2['x'] - point1['x'])) + math.pi/2)
    #yz_angle = ((math.atan2(point2['z'] - point1['z'], point2['y'] - point1['y'])) + math.pi/2)
    xz_angle = (math.atan2(point2['z'], point2['x']) - math.atan2(point1['z'], point1['x']))
    yz_angle = (math.atan2(point2['z'], point2['y']) - math.atan2(point1['z'], point1['y'])) % (math.pi)

    yz_angle -= math.pi/2

    return [xz_angle, yz_angle]

  def distance_between_joints(self, point1, point2):
    return math.sqrt(math.pow(point2['x'] - point1['x'], 2) + math.pow(point2['y'] - point1['y'], 2)
                     + math.pow(point2['z'] - point1['z'], 2))

  def get_vector_between_joints(self, point1, point2):
    vector = [point2['x'] - point1['x'], point2['y'] - point1['y'], point2['z'] - point1['z']]
    length = np.linalg.norm(vector)
    normal_vector = vector / length
    return normal_vector

  def rotation_matrix_X(self, target, v):
    [tx, ty, tz] = -target
    [v1, v2, v3] = -v
    cos_alpha = (ty*v[1] + tz*v[2]) / (math.pow(v2, 2) - math.pow(v3, 2))
    sin_alpha = ((v2*cos_alpha) - ty) / v3
    return [[1, 0, 0], [0, cos_alpha, -sin_alpha], [0, sin_alpha, cos_alpha]]

  def rotation_matrix_Y(self, target, v):
    [tx, ty, tz] = -target
    [v1, v2, v3] = -v
    cos_beta = (v1*tx + v3*tz) / (math.pow(v1, 2) + math.pow(v3, 2))
    sin_beta = (tx-v1*cos_beta) / v3
    return [[cos_beta, 0, sin_beta], [0, 1, 0], [-sin_beta, 0, cos_beta]]

  def rotation_matrix_Z(self, target, v):
    [tx, ty, tz] = -target
    [v1, v2, v3] = -v
    cos_gamma = (v1*tx + v2*ty) / (math.pow(v1, 2) + math.pow(v2, 2))
    sin_gamma = (ty - v2*cos_gamma) / v1
    return [[cos_gamma, -sin_gamma, 0], [sin_gamma, cos_gamma, 0], [0, 0, 1]]

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


