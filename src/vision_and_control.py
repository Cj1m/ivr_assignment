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
from sympy import sin, cos, Matrix, symbols, MatMul, pprint
import message_filters


class vision:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing2', anonymous=True)

        # Create new topic for publishing rotation matrices to
        self.joint1_publisher = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.joint2_publisher = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.joint3_publisher = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.joint4_publisher = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        # Create a new topic for publishing the end effector position
        self.ee_x_position_pub = rospy.Publisher("ee_finder/x_position_estimate", Float64, queue_size=10)
        self.ee_y_position_pub = rospy.Publisher("ee_finder/y_position_estimate", Float64, queue_size=10)
        self.ee_z_position_pub = rospy.Publisher("ee_finder/z_position_estimate", Float64, queue_size=10)

        # Subscribe to camera 1 output processed by image1.py
        self.image_sub1 = message_filters.Subscriber("/image_topic1", Image)

        # Subscribe to camera 2 output processed by image2.py
        self.image_sub2 = message_filters.Subscriber("/image_topic2", Image)
        self.target_x_sub = rospy.Subscriber("/target_finder/x_position_estimate", Float64, self.update_target_x)
        self.target_y_sub = rospy.Subscriber("/target_finder/y_position_estimate", Float64, self.update_target_y)
        self.target_z_sub = rospy.Subscriber("/target_finder/z_position_estimate", Float64, self.update_target_z)

        # Synchronize subscriptions into one callback
        ts = message_filters.TimeSynchronizer([self.image_sub1, self.image_sub2], 1)
        ts.registerCallback(self.callback)

        # Initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        # Get the time
        self.time_trajectory = rospy.get_time()
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')

        # Each joints coordinates
        self.joint1_pos = {"x": 0, "y": 0, "z": 0}
        self.joint23_pos = {"x": 0, "y": 0, "z": 0}
        self.joint4_pos = {"x": 0, "y": 0, "z": 0}
        self.jointEE_pos = {"x": 0, "y": 0, "z": 0}

        # Colour threshold dictionaries
        self.upper_threshold = {'yellow': (5, 255, 255), 'blue': (255, 5, 5), 'green': (5, 255, 5), 'red': (5, 5, 255)}
        self.lower_threshold = {'yellow': (0, 10, 10), 'blue': (10, 0, 0), 'green': (0, 10, 0), 'red': (0, 0, 10)}

        # Symbols construction
        # alpha =   link1Angle
        # beta  =   link2_angle[1]
        # gamma =   link2_angle[0]
        # phi   =   link3_angle[1]
        self.alpha, self.beta, self.gamma, self.phi = symbols('alpha beta gamma phi')

        # Target position
        self.target_position = [0, 0, 7]

        # Initialize errors
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')

        # Global record of angles for control section
        self.link1_actual_angle = 0
        self.link2_actual_angle = [0, 0]
        self.link3_actual_angle = [0, 0]

    # Receive target x position
    def update_target_x(self, data):
        self.target_position[0] = data.data

    # Receive target y position
    def update_target_y(self, data):
        self.target_position[1] = data.data

    # Receive target z position
    def update_target_z(self, data):
        self.target_position[2] = data.data

    # Receive data from camera 1 and camera 2
    def callback(self, data1, data2):

        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.waitKey(1)

        # Get each joints co-ordinates from camera 1
        image1_joint1 = self.get_joint_coordinates(self.cv_image1, "yellow")
        image1_joint23 = self.get_joint_coordinates(self.cv_image1, "blue")
        image1_joint4 = self.get_joint_coordinates(self.cv_image1, "green")
        image1_jointEE = self.get_joint_coordinates(self.cv_image1, "red")

        # Get each joints co-ordinates from camera 2
        image2_joint1 = self.get_joint_coordinates(self.cv_image2, "yellow")
        image2_joint23 = self.get_joint_coordinates(self.cv_image2, "blue")
        image2_joint4 = self.get_joint_coordinates(self.cv_image2, "green")
        image2_jointEE = self.get_joint_coordinates(self.cv_image2, "red")

        # Compare co-ordinates obtained from camera 1 and camera 2
        # Update each joints global variable position
        self.set_coordinates(image1_joint1, image2_joint1, self.joint1_pos)
        self.set_coordinates(image1_joint23, image2_joint23, self.joint23_pos)
        self.set_coordinates(image1_joint4, image2_joint4, self.joint4_pos)
        self.set_coordinates(image1_jointEE, image2_jointEE, self.jointEE_pos)

        # Get length of each link
        link1_dist_pixels = self.distance_between_joints(self.joint1_pos, self.joint23_pos)
        link3_dist_pixels = self.distance_between_joints(self.joint23_pos, self.joint4_pos)
        link4_dist_pixels = self.distance_between_joints(self.joint4_pos, self.jointEE_pos)
        pixel_in_metres = 2 / link1_dist_pixels

        # Get vector from joint1 to end effector
        self.jointEE_pos_in_metres = {"x": (self.jointEE_pos['x'] - self.joint1_pos['x']) * pixel_in_metres,
                                      "y": (self.jointEE_pos['y'] - self.joint1_pos['y']) * pixel_in_metres,
                                      "z": (self.joint1_pos['z'] - self.jointEE_pos['z']) * pixel_in_metres}

        link2_angle_estimated = self.get_angle_between_points(self.joint23_pos, self.joint4_pos)
        link3_angle_estimated = np.subtract(self.get_angle_between_points(self.joint4_pos, self.jointEE_pos),
                                            link2_angle_estimated)
        # Translation matrices for each link
        a12 = np.matrix([[1, 0, 0, 0],
                         [0, math.cos(link2_angle_estimated[1]), -math.sin(link2_angle_estimated[1]), 0],
                         [0, math.sin(link2_angle_estimated[1]), math.cos(link2_angle_estimated[1]),
                          link1_dist_pixels * pixel_in_metres],
                         [0, 0, 0, 1]])
        a23 = np.matrix([[math.cos(link2_angle_estimated[0]), 0, math.sin(link2_angle_estimated[0]), 0],
                         [0, 1, 0, 0],
                         [-math.sin(link2_angle_estimated[0]), 0, math.cos(link2_angle_estimated[0]), 0],
                         [0, 0, 0, 1]])

        a34 = np.matrix([[1, 0, 0, 0],
                         [0, math.cos(link3_angle_estimated[1]), -math.sin(link3_angle_estimated[1]), 0],
                         [0, math.sin(link3_angle_estimated[1]), math.cos(link3_angle_estimated[1]),
                          link3_dist_pixels * pixel_in_metres],
                         [0, 0, 0, 1]])
        p4 = [0, 0, 2, 1]

        # Get link 1 angles
        link1Angle = self.find_Z_angle(self.jointEE_pos, a12, a23, a34, p4)

        # Translation matrices in SymPy form
        a12SymPy = Matrix([[1, 0, 0, 0],
                           [0, cos(self.alpha), -sin(self.alpha), 0],
                           [0, sin(self.alpha), cos(self.alpha), 2],
                           [0, 0, 0, 1]])
        a23SymPy = Matrix([[cos(self.beta), 0, sin(self.beta), 0],
                           [0, 1, 0, 0],
                           [-sin(self.beta), 0, cos(self.beta), 0],
                           [0, 0, 0, 1]])
        a34SymPy = Matrix([[1, 0, 0, 0],
                           [0, cos(self.gamma), -sin(self.gamma), 0],
                           [0, sin(self.gamma), cos(self.gamma), 3],
                           [0, 0, 0, 1]])
        a01SymPy = Matrix([[cos(self.phi), -sin(self.phi), 0, 0],
                           [sin(self.phi), cos(self.phi), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        p4 = Matrix([0, 0, link4_dist_pixels * pixel_in_metres, 1])

        combined_translation_matrices = a01SymPy * a12SymPy * a23SymPy * a34SymPy * p4

        # For controlling the robot use actual angles of joints
        link2_angle = self.link2_actual_angle
        link3_angle = self.link3_actual_angle

        # Publish the x, y and z coordinates for the end effector
        ee_x = Float64()
        ee_x.data = self.jointEE_pos_in_metres['x']
        ee_y = Float64()
        ee_y.data = self.jointEE_pos_in_metres['y']
        ee_z = Float64()
        ee_z.data = self.jointEE_pos_in_metres['z']
        try:
            self.ee_x_position_pub.publish(ee_x)
            self.ee_z_position_pub.publish(ee_z)
            self.ee_y_position_pub.publish(ee_y)
        except CvBridgeError as e:
            print(e)

        # Calculate Jacobian matrix [Replace int vector with joint angle states ]
        jacobian = self.get_jacobian([link1Angle, link2_angle[1], link2_angle[0], link3_angle[1]],
                                     combined_translation_matrices)

        # Begins forward kinematic control
        forward_control = self.get_PID_desired_angles(jacobian,
                                                      [link1Angle, link2_angle[1], link2_angle[0], link3_angle[1]])
        forward_control = [(forward_control[0]), (forward_control[1]),
                          (forward_control[2]), (forward_control[3])]
        # Move the robot
        try:
            self.link1_actual_angle = forward_control[0]
            self.link2_actual_angle[1] = forward_control[1]
            self.link3_actual_angle[1] = forward_control[2]
            self.joint1_publisher.publish(Float64(data=forward_control[0]))
            self.joint2_publisher.publish(Float64(data=forward_control[1]))
            self.joint3_publisher.publish(Float64(data=forward_control[2]))
            self.joint4_publisher.publish(Float64(data=forward_control[3]))
        except CvBridgeError as e:
            print(e)

    """Algorithm works when accurate angle measurements for joints 2,3,4 are used"""
    def find_Z_angle(self, EE, a12, a23, a34, p4):

        # Set closest angle and closest dist to values that will be overwritten
        closest_angle = 99999
        closest_dist = 99999

        # Find end-effector position relative to joint 1
        actual_EE = {"x": EE['x'] - self.joint1_pos['x'], "y": EE['y'] - self.joint1_pos['y'],
                     "z": self.joint1_pos['z'] - EE['z']}
        actual_EE_vector = [actual_EE['x'], actual_EE['y'], actual_EE['z']]

        # Rotate through all z angles in 2pi, subbing them into joint1 rotation matrix
        # Return angle of z that has closest estimated end effector position
        for cur_z_degrees in range(-180, 180):
            cur_z = cur_z_degrees * (math.pi / 180)
            a01 = np.matrix([[math.cos(cur_z), -math.sin(cur_z), 0, 0],
                             [math.sin(cur_z), math.cos(cur_z), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
            estimated_EE = np.matmul(np.matmul(np.matmul(np.matmul(a01, a12), a23), a34), p4)
            estimated_EE = ([estimated_EE[0, 0], estimated_EE[0, 1], estimated_EE[0, 2]])

            # euclidean distance of estimated end effector to actual end effector position
            if (np.linalg.norm(np.subtract(np.array(estimated_EE), np.array(actual_EE_vector))) < closest_dist):
                closest_angle = cur_z_degrees
                closest_dist = np.linalg.norm(np.subtract(np.array(estimated_EE), np.array(actual_EE_vector)))

        return (closest_angle * (math.pi / 180))

    # Applies SymPy function to calculate the jacobian then subs in the angles respectively
    def get_jacobian(self, joint_angles, forward_kinematics):
        return forward_kinematics.jacobian([self.alpha, self.beta, self.gamma, self.phi]).subs(
            [(self.alpha, joint_angles[0]), (self.beta, joint_angles[1]), (self.gamma, joint_angles[2]),
             (self.phi, joint_angles[3])])

    def get_PID_desired_angles(self, jacobian, joint_angles):

        J = np.array(jacobian).astype(np.float64)

        # P gain
        K_p = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])

        # D gain
        K_d = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])

        # estimate time step
        cur_time = np.array([rospy.get_time()])
        dt = cur_time - self.time_previous_step
        self.time_previous_step = cur_time

        # robot end-effector position
        pos = [self.jointEE_pos_in_metres['x'], self.jointEE_pos_in_metres['y'], self.jointEE_pos_in_metres['z']]

        # desired trajectory
        pos_d = self.target_position

        # estimate derivative of error
        self.error_d = ((np.subtract(pos_d, pos)) - self.error) / dt

        # estimate error
        self.error = np.subtract(pos_d, pos)

        q = joint_angles  # estimate initial value of joints'

        # Remove joint 3 from q and J (unnecessary extra degree of freedom)
        J = np.delete(J, 3, 0)

        J_inv = np.linalg.pinv(J)  # calculating the psudeo inverse of Jacobian
        dot_errorDgain = np.dot(K_p, self.error.transpose())
        dot_differenceInErrorDgain = np.dot(K_d, np.add(self.error_d.transpose(), dot_errorDgain))

        dq_d = np.dot(J_inv, dot_differenceInErrorDgain)  # control input (angular velocity of joints)
        q_d = q + (dt * dq_d)  # control input (angular position of joints)
        print ("Q: " + str(q_d))
        return q_d

    # Takes the translation matrices and computes the estimated position of the end effector
    def get_EE_with_forward_kinematics(self, link_angles):
        a01SymPy = Matrix([[cos(link_angles[0]), -sin(link_angles[0]), 0, 0],
                           [sin(link_angles[0]), cos(link_angles[0]), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        a12SymPy = Matrix([[1, 0, 0, 0],
                           [0, cos(link_angles[1]), -sin(link_angles[1]), 0],
                           [0, sin(link_angles[1]), cos(link_angles[1]), 2],
                           [0, 0, 0, 1]])
        a23SymPy = Matrix([[cos(link_angles[2]), 0, sin(link_angles[2]), 0],
                           [0, 1, 0, 0],
                           [-sin(link_angles[2]), 0, cos(link_angles[2]), 0],
                           [0, 0, 0, 1]])
        a34SymPy = Matrix([[1, 0, 0, 0],
                           [0, cos(link_angles[3]), -sin(link_angles[3]), 0],
                           [0, sin(link_angles[3]), cos(link_angles[3]), 3],
                           [0, 0, 0, 1]])
        p4 = Matrix([0, 0, 2, 1])

        estimated_EE = a01SymPy * a12SymPy * a23SymPy * a34SymPy * p4
        estimated_EE = np.array([estimated_EE[0], estimated_EE[1], estimated_EE[2]]).astype(np.float64)

        return estimated_EE

    # Applies trigonometry to given point using angles to calculate position of next point
    def get_next_point_pos(self, curPos, linkLength, angle):

        x = math.sin(angle[0]) * math.cos(angle[1]) * linkLength
        y = math.sin(angle[1]) * linkLength
        z = math.cos(angle[0]) * math.cos(angle[1]) * linkLength

        return [curPos[0] + x, curPos[1] + y, curPos[2] - z]

    # Combine both images coordinates to get accurate 3D coordinates
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

    # Threshold the image for the appropriate colour thresholds and return the center of the blob
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

        xz_angle = ((math.atan2(point2['z'] - point1['z'], point2['x'] - point1['x'])) + math.pi / 2)
        yz_angle = ((math.atan2(point2['z'] - point1['z'], point2['y'] - point1['y'])) + math.pi / 2)

        return [xz_angle, yz_angle * -1]

    # Appies euclidean distance between the two points passed to it
    def distance_between_joints(self, point1, point2):
        return math.sqrt(math.pow(point2['x'] - point1['x'], 2) + math.pow(point2['y'] - point1['y'], 2)
                         + math.pow(point2['z'] - point1['z'], 2))

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
