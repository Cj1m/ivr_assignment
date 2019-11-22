#!/usr/bin/env python

import roslib
import sys
import rospy
import cv2
import math
import numpy as np
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError
import message_filters


class target_finder:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('target_finder', anonymous=True)
        # Subscribe to camera 1 output processed by image1.py
        self.image_sub1 = message_filters.Subscriber("/image_topic1", Image)
        # Subscribe to camera 2 output processed by image2.py
        self.image_sub2 = message_filters.Subscriber("/image_topic2", Image)

        self.target_x_position_pub = rospy.Publisher("target_finder/x_position_estimate", Float64, queue_size=10)
        self.target_y_position_pub = rospy.Publisher("target_finder/y_position_estimate", Float64, queue_size=10)
        self.target_z_position_pub = rospy.Publisher("target_finder/z_position_estimate", Float64, queue_size=10)
        self.complete_target1_pub = rospy.Publisher("target_finder/target_view_1", Image, queue_size=1)
        self.complete_target2_pub = rospy.Publisher("target_finder/target_view_2", Image, queue_size=1)

        self.base_frame_position = [0, 0, 0]

        # Synchronize subscriptions into one callback
        ts = message_filters.TimeSynchronizer([self.image_sub1, self.image_sub2], 1)
        ts.registerCallback(self.callback)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # Get the time
        self.time_trajectory = rospy.get_time()

        self.rectangle_c1_template = cv2.inRange(cv2.imread('c1_rectangle_target.png', 1), (200, 200, 200), (255, 255, 255))

        self.rectangle_c2_template = cv2.inRange(cv2.imread('c2_rectangle_target.png', 1), (200, 200, 200), (255, 255, 255))
        self.sphere_template = cv2.inRange(cv2.imread('sphere_target.png', 1), (200, 200, 200), (255, 255, 255))

        self.rectangle = {"x": 0, "y": 0, "z": 0}
        self.sphere = {"x": 0, "y": 0, "z": 0}



    # Receive data from camera 1 and camera 2
    def callback(self, data1, data2):

        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data1, "bgr8")
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data2, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Set position of base frame
        if self.base_frame_position == [0, 0, 0]:
            base_frame_xzcoords = self.get_base_frame_position(self.cv_image1)
            base_frame_yzcoords = self.get_base_frame_position(self.cv_image2)
            self.base_frame_position = [base_frame_xzcoords[0], base_frame_yzcoords[0], base_frame_yzcoords[1]]

        # Threshold image 1 and image 2 for target and object
        complete_target1 = self.threshold_targets(self.cv_image1)
        complete_target2 = self.threshold_targets(self.cv_image2)

        # Erode and dilate images to improve accuracy
        kernel = np.ones((2, 2), np.uint8)
        complete_target2 = cv2.erode(complete_target2, kernel, iterations=6)
        complete_target2 = cv2.dilate(complete_target2, kernel, iterations=8)

        # Get center of target
        cv2.waitKey(3)
        target_centers1 = self.find_centers(complete_target1)
        target_centers2 = self.find_centers(complete_target2)

        # Get object and target coordinates
        sphere_xz, rectangle_xz = self.get_shape_coordinates(target_centers1, complete_target1, self.rectangle_c1_template)
        sphere_yz, rectangle_yz = self.get_shape_coordinates(target_centers2, complete_target2, self.rectangle_c2_template)

        joint1_pos = {'y': 399, 'x': 399, 'z': 532}
        pixels_in_meters = 0.0392156862745

        # When the object/target is in site update its coordinates relative to the base frame in metres
        if sphere_xz != [0, 0]:
            self.sphere["x"] = (sphere_xz[0] - self.base_frame_position[0]) * pixels_in_meters
            self.sphere["z"] = (self.base_frame_position[2] - sphere_xz[1]) * pixels_in_meters

        if sphere_yz[0] > 0:
            self.sphere["y"] = (sphere_yz[0]- self.base_frame_position[1]) * pixels_in_meters
        if rectangle_xz != [0, 0]:
            self.rectangle["x"] = rectangle_xz[0]
            self.rectangle["z"] = rectangle_xz[1]

        if rectangle_yz != [0, 0]:
            self.rectangle["y"] = rectangle_yz[0]

        # Publish the x, y and z coordinates for the target
        sphere_x = Float64()
        sphere_x.data = self.sphere['x']
        sphere_y = Float64()
        sphere_y.data = self.sphere['y']
        sphere_z = Float64()
        sphere_z.data = self.sphere['z']
        try:
            self.target_x_position_pub.publish(sphere_x)
            self.target_z_position_pub.publish(sphere_z)
            self.target_y_position_pub.publish(sphere_y)
        except CvBridgeError as e:
            print(e)

    # Returns the coordinates for the sphere and the rectangle
    def get_shape_coordinates(self, centers, image, rectangle_template):

        sphere = [0, 0]
        rectangle = [0, 0]

        # Applies chamfer matching to the target/object to templates
        for center in centers:
            cropped_target = self.cropTarget(image, center, [32, 32])

            dist_transformSphere = cv2.distanceTransform(cv2.bitwise_not(self.sphere_template), cv2.DIST_L2, 0)
            dist_transformRect = cv2.distanceTransform(cv2.bitwise_not(rectangle_template), cv2.DIST_L2, 0)

            rectangle_chamfer_distance = np.sum(dist_transformRect * cropped_target)
            sphere_chamfer_distance = np.sum(dist_transformSphere * cropped_target)

            # Assigns center of target/object to coordinates of closest matching template
            if sphere_chamfer_distance < rectangle_chamfer_distance:
                sphere = [center[0], center[1]]
            else:
                rectangle = [center[0], center[1]]

        return sphere, rectangle

    # Returns a cropped image around the object
    def cropTarget(self, image, center, shape):
        return image[center[1] - shape[1] / 2: center[1] + shape[1] / 2,
               center[0] - shape[0] / 2: center[0] + shape[0] / 2]

    # Returns the center of the object by contouring the image
    def find_centers(self, complete_target):
        cnts, hierarchy = cv2.findContours(complete_target, cv2.RETR_CCOMP,
                                cv2.CHAIN_APPROX_TC89_L1)
        centers = []
        for c in cnts:
           M = cv2.moments(c)
           if M["m00"] != 0:
               cX = int(M["m10"] / M["m00"])
               cY = int(M["m01"] / M["m00"])
               centers.append((cX, cY))
        return centers

    # Returns a thresholded image that includes just the target and object
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

    # Returns the center of the baseframe from the perspective of the passed image
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


