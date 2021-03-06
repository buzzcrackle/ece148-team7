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
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
from jetcam.usb_camera import USBCamera
# from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg

def setup():

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

    return 0

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
    cv2.imshow(image)
    return (counts, objects, peaks)