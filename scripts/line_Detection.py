#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from std_msgs.msg import Int32, Int32MultiArray
from sensor_msgs.msg import Image
from decoder import decodeImage


LINE_DETECTION_NODE_NAME = 'line_detection_node'
CAMERA_TOPIC_NAME = 'camera_rgb'
CENTROID_TOPIC_NAME = '/centroid'


def video_detection(data):

    # experimentally found values from find_camera_values.py
    Hue_low = rospy.get_param("/Hue_low")
    Hue_high = rospy.get_param("/Hue_high")
    Saturation_low = rospy.get_param("/Saturation_low")
    Saturation_high = rospy.get_param("/Saturation_high")
    Value_low = rospy.get_param("/Value_low")
    Value_high = rospy.get_param("/Value_high")
    green_filter = rospy.get_param("/green_filter")
    start_height = rospy.get_param("/camera_start_height")
    bottom_height = rospy.get_param("/camera_bottom_height")
    left_width = rospy.get_param("/camera_left_width")
    right_width = rospy.get_param("/camera_right_width")
    
    # Image processing from rosparams
    frame = decodeImage(data.data, data.height, data.width)
    height, width, channels = frame.shape
    new_width = int(right_width - left_width)

    img = frame[start_height:bottom_height, left_width:right_width]
    orig = img.copy()

    # changing color space to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # setting threshold limits for custom color filter
    lower = np.array([Hue_low, Saturation_low, Value_low])
    upper = np.array([Hue_high, Saturation_high, Value_high])

    # creating mask
    mask = cv2.inRange(hsv, lower, upper)

    m = cv2.moments(mask, False)
    centroid_and_frame_width = []
    try:
        cx, cy = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])
    except ZeroDivisionError:
        cy, cx = int(height / 2), int(width / 2)

    # Publish Message
    mid_x = cx
    mid_y = cy
    cv2.circle(img, (mid_x, mid_y), 7, (255, 0, 0), -1)
    centroid_and_frame_width.append(mid_x)
    centroid_and_frame_width.append(new_width)
    pub.publish(data=centroid_and_frame_width)

    # plotting results
    try:
        cv2.imshow("original", orig)
        cv2.imshow("yellow mask", mask)
        cv2.imshow("plotting centroid", img)
        cv2.waitKey(1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node(LINE_DETECTION_NODE_NAME, anonymous=False)
    camera_sub = rospy.Subscriber(CAMERA_TOPIC_NAME, Image, video_detection)
    pub = rospy.Publisher(CENTROID_TOPIC_NAME, Int32MultiArray, queue_size=2)
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        rospy.spin()
        rate.sleep()
