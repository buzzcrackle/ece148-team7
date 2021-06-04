#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool

CLASSIFICATION_NODE_NAME = 'classification_sub_node'
CLASSIFICATION_TOPIC_NAME = '/classification'


def callback(data):
    # output_start= rospy.get_param("/Throttle_max_reverse")
    # output_end = rospy.get_param("/Throttle_max_forward")
    # Throttle_neutral = rospy.get_param("/Throttle_neutral")
    #
    # input_start = -1
    # input_end = 1
    #
    # input_throttle = data.data
    # normalized_throttle = output_start + (input_throttle - input_start) * ((output_end - output_start) / (input_end - input_start))
    classification_data = data.data
    if classification_data == False:
        throttle_float = 0
    elif classification_data == True:
        throttle_float = 1.0
    throttle_pub.publish(throttle_float)




def listener():
    rospy.init_node(CLASSIFICATION_NODE_NAME, anonymous=False)
    throttle_pub = rospy.Publisher(THROTTLE_TOPIC_NAME, Float32, queue_size=1)
    rospy.Subscriber(CLASSIFICATION_TOPIC_NAME, Bool, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()