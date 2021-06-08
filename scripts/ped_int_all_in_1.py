#!/usr/bin/env python
# from torch._C import T
from cv_bridge import CvBridge
import glob
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import csv
import pandas as pd
import random
from collections import defaultdict
import pickle
import rospy
import cv2
from std_msgs.msg import Float32, Int32, Int32MultiArray, Bool, String
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
from joblib import dump, load
from jetcam.usb_camera import USBCamera
# from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg

# camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=30)


PED_INT_NODE_NAME = 'ped_int'
CAMERA_TOPIC_NAME = 'camera_rgb'
CLASSIFICATION_TOPIC_NAME = '/classification'
TYPE_TOPIC_NAME = '/type'

global classification_bool
classification_bool = Bool()
global type_str
type_str = String()
# global steering_float, throttle_float
# steering_float = Float32()
# throttle_float = Float32()
def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
# def preprocess(image):
#     global device
#     device = torch.device('cuda')
#     image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
#     image = cv2.resize(np.float32(image), dsize=(224, 224), interpolation=cv2.INTER_AREA)
#     image = PIL.Image.fromarray(image)
#     image = transforms.functional.to_tensor(image).to(device)
#     image.sub_(mean[:, None, None]).div_(std[:, None, None])
#     return image[None, ...]

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

def execute(change):
    img = change['new']
    try: 
        type_str = ' | made_imgmsg | '
        bridge = CvBridge()
        rgb = bridge.cv2_to_imgmsg(img)
        camera_pub.publish(rgb)
    except TypeError:
        pass
    dict_key = ""
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    image_vec = []
    classification_bool = True
    for i in range(counts[0]):
        keypoints = get_keypoint(objects, i, peaks)
        dict_key = ""
        header = ""
        image_vec = []
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                if j in range(0,5):
                    continue
                dict_key += str(j)
                header += str(j) + '_x,' + str(j) + '_y,'
#                 x = keypoints[j][2]
#                 y = keypoints[j][1]
                image_vec.append(keypoints[j][2])
                image_vec.append(keypoints[j][1])
    if len(image_vec)==26:
        inpt = torch.tensor([image_vec], dtype=torch.float32).to(device) 
        raw_out = model_pose(inpt)
        pred_prob = raw_out.item()
        if pred_prob < 0.5:
            print("Prediction = crossing")
            classification_bool = False
        else:
            print("Prediction = not_crossing")
            classification_bool = True
    # type_str += dict_key
    # if dict_key in model_dict:
    #     curr_model = model_dict[dict_key]
    #     prediction = curr_model.predict([image_vec])[0][0]
    #     if prediction == 1:
    #         type_str += ' | predicted is 1'
    #         classification_bool = False
    #     else:
    #         type_str += ' | predicted is 2'
    #         classification_bool = True
    #     classification_pub.publish(classification_bool)
    
    type_pub.publish(type_str)
    
    
    classification_pub.publish(classification_bool)

    #draw_objects(img, counts, objects, peaks)
    # def execute2(img):
    # img = img['new']
    # data = preprocess(img)
    # cmap, paf = model_trt(data)
    # cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    # counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    # for i in range(counts[0]):
    #     keypoints = get_keypoint(objects, i, peaks)
    #     dict_key = ""
    #     header = ""
    #     image_vec = []
        # for j in range(len(keypoints)):
        #     if keypoints[j][1]:
        #         dict_key += str(j)
        #         header += str(j) + '_x,' + str(j) + '_y,'
#                 x = keypoints[j][2]
#                 y = keypoints[j][1]
                # image_vec.append(keypoints[j][2])
                # image_vec.append(keypoints[j][1])
        
#         print(str(image_vec))
        # writepath = folder + '/' + dict_key + '.csv'
        # if not os.path.exists(writepath):
        #     with open(writepath,'w+') as fd:
        #         fd.write(header[:-1] + "\n")
        
        # with open(writepath,'a') as fd:
        #     fd.write(str(image_vec)[1:-1] + "\n")
        
    #draw_objects(img, counts, objects, peaks)
#     return data_dict




class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = torch.nn.Linear(num_features, 8)  # 4-(8-8)-1
        self.hid2 = torch.nn.Linear(8, 8)
        self.oupt = torch.nn.Linear(8, 1)

        torch.nn.init.xavier_uniform_(self.hid1.weight) 
        torch.nn.init.zeros_(self.hid1.bias)
        torch.nn.init.xavier_uniform_(self.hid2.weight) 
        torch.nn.init.zeros_(self.hid2.bias)
        torch.nn.init.xavier_uniform_(self.oupt.weight) 
        torch.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = torch.tanh(self.hid1(x)) 
        z = torch.tanh(self.hid2(z))
        z = torch.sigmoid(self.oupt(z)) 
        return z



    
    

if __name__ == '__main__':

    
    

    
    
    with open('/home/jetson/projects/catkin_ws/src/ece148/scripts/human_pose.json', 'r') as f:
        human_pose = json.load(f)

    topology = trt_pose.coco.coco_category_to_topology(human_pose)

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    
    # MODEL_WEIGHTS = '/home/jetson/projects/catkin_ws/src/ece148/scripts/resnet18_baseline_att_224x224_A_epoch_249.pth'

    # model.load_state_dict(torch.load(MODEL_WEIGHTS))
    HEIGHT = 224
    WIDTH = 224
    

    data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

    # model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)

    OPTIMIZED_MODEL = '/home/jetson/projects/catkin_ws/src/ece148/scripts/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'

    # torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

    t0 = time.time()
    torch.cuda.current_stream().synchronize()
    for i in range(50):
        y = model_trt(data)
    torch.cuda.current_stream().synchronize()
    t1 = time.time()

    global device
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    device = torch.device('cuda')
    
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    num_features = 26
    path = "/home/jetson/projects/catkin_ws/src/ece148/scripts/banknote_sd_model.pth"
    global model_pose
    model_pose = Net()  # later . . 
    model_pose.load_state_dict(torch.load(path))
    model_pose.eval()

    # from trt_pose.parse_objects import ParseObjects
    parse_objects = ParseObjects(topology)
    rospy.init_node(PED_INT_NODE_NAME, anonymous=False)
    classification_pub = rospy.Publisher(CLASSIFICATION_TOPIC_NAME, Bool, queue_size=1)
    type_pub = rospy.Publisher(TYPE_TOPIC_NAME, String, queue_size=1)
    camera = USBCamera(width=WIDTH, height=HEIGHT, capture_fps=30)
    camera.running = True
    camera_pub = rospy.Publisher(CAMERA_TOPIC_NAME, Image, queue_size=10)
    rate = rospy.Rate(10)
    type_pub.publish(type_str)
    while not rospy.is_shutdown():
        #construct msg
        
        # classification_bool = Bool()
        # type_str = String()
        
        # type_str = 'in classification function'
        # type_pub.publish(type_str)
        
        # bridge = CvBridge()

        # image = bridge.imgmsg_to_cv2(data, "rgb8")
        # type_str = 'changed image'
        # type_pub.publish(type_str)
        # Image processing from rosparams
        # image = decodeImage(data.data, data.height, data.width)
        # image = bridge.imgmsg_to_cv2(image)


        # image = preprocess(image)
        # img = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        # image = preprocess(img)

        execute({'new': camera.value}) #execute(image)
        # print(pose_data)
        camera.observe(execute, names='value')

        # type_str = str(pose_data)
        # type_pub.publish(type_str)
        
        # classification_bool=True
        

        classification_pub.publish(classification_bool)

        rospy.spin()
        rate.sleep()
    camera.unobserve_all()
