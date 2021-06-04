#!/usr/bin/env python
import rospy
import cv2
from std_msgs.msg import Float32, Int32, Int32MultiArray, Bool
from sensor_msgs.msg import Image
import numpy as np
from decoder import decodeImage
import time
import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import torchvision.transforms as transforms
import PIL.Image
# from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
# from jetcam.usb_camera import USBCamera
# from jetcam.csi_camera import CSICamera
# from jetcam.utils import bgr8_to_jpeg
# import matplotlib.pyplot as plt
# import ipywidgets
# from IPython.display import display
# import utils

CLASSIFICATION_NODE_NAME = 'classification_node'
CLASSIFICATION_TOPIC_NAME = '/classification'
CAMERA_TOPIC_NAME = 'camera_rgb'

global classification_bool
classification_bool = Bool()
# global steering_float, throttle_float
# steering_float = Float32()
# throttle_float = Float32()
def preprocess(image):
    global device
    device = torch.device('cuda')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def execute(image):
    data = preprocess(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    # image_w.value = bgr8_to_jpeg(image[:, ::-1, :])
    # cv2.imshow(image)
    return()

def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )
        else:    
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )
    return kpoint

def execute(img):
    image_vec = []
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    for i in range(counts[0]):
        keypoints = get_keypoint(objects, i, peaks)
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                x = keypoints[j][2]
                y = keypoints[j][1]
                point = [j,x,y]
                image_vec.append(point)
    #draw_objects(img, counts, objects, peaks)
    return image_vec




def classification_determination(data):
    # kp = 0.3
    global classification_bool
    classification_bool = Bool()
    # Image processing from rosparams
    frame = decodeImage(data.data, 224, 224)
    
    image = preprocess(frame)
    pose_data = execute(image)
    print(pose_data)
    classification_bool=True
    

    classification_pub.publish(classification_bool)


if __name__ == '__main__':
    with open('/home/jetson/projects/catkin_ws/src/ece148/scripts/human_pose.json', 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

    MODEL_WEIGHTS = '/home/jetson/projects/catkin_ws/src/ece148/scripts/resnet18_baseline_att_224x224_A_epoch_249.pth'

    model.load_state_dict(torch.load(MODEL_WEIGHTS))

    WIDTH = 224
    HEIGHT = 224

    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

    OPTIMIZED_MODEL = '/home/jetson/projects/catkin_ws/src/ece148/scripts/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

    t0 = time.time()
    torch.cuda.current_stream().synchronize()
    for i in range(50):
        y = model_trt(data)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()

    print(50.0 / (t1 - t0))

    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')

    # -------------------train model-------------------
    # call model trainging script using training set of 100 images 50 stop 50 go and 
    # vector of 50 0's and 50 1's
    


    rospy.init_node(CLASSIFICATION_NODE_NAME, anonymous=False)
    camera_sub = rospy.Subscriber(CAMERA_TOPIC_NAME, Image, classification_determination)
    classification_pub = rospy.Publisher(CLASSIFICATION_TOPIC_NAME, Bool, queue_size=1)
    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        rospy.spin()
        rate.sleep()
