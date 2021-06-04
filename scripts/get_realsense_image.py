#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from decoder import decodeImage

MY_CAMERA_NODE_NAME = 'my_camera_node'
CAMERA_TOPIC_NAME = 'camera/color/image_raw'

def get_camera_image(data):
    frame = decodeImage(data.data, data.height, data.width)
    height, width, channels = frame.shape

    img = cv2.cvtColor(frame[0:height, 0:width], cv2.COLOR_RGB2BGR)
    
    try:
        cv2.imshow('img', img)
        cv2.waitKey(1)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node(MY_CAMERA_NODE_NAME, anonymous=False)
    camera_sub = rospy.Subscriber(CAMERA_TOPIC_NAME, Image, get_camera_image)
    rate = rospy.Rate(60)
    while not rospy.is_shutdown():
        rospy.spin()
        rate.sleep()
