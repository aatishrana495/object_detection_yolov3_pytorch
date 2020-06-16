from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from IPython.display import Image, display

import cv2
import torchvision.transforms as transforms



def detect(img):
    print(img.shape)
    convert=transforms.ToTensor()
    input_imgs=convert(img).cuda()
    detections=model(input_imgs.unsqueeze(0))
    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    print("detections")
    print(detections[0])
    print(len(detections))
    print(torch.is_tensor(detections[0]))

    frm=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    print(frm.shape)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Create plot
    plt.figure()
    fig, ax = plt.subplots(1)
    ## ax.imshow(img)

    if torch.is_tensor(detections[0]):
        detections = rescale_boxes(detections[0], opt.img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("predicted class number")
            print(int(cls_pred))
            #if(int(cls_pred) < 1 or int(cls_pred) > 5):
            #    continue
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            #!cls_pred=cls_pred%8
            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )
            cv2.line(frm, (x1, y1), (x2, y1), color[int(cls_pred)], 2)
            cv2.line(frm, (x2, y1), (x2, y2), color[int(cls_pred)], 2)
            cv2.line(frm, (x2, y2), (x1, y2), color[int(cls_pred)], 2)
            cv2.line(frm, (x1, y2), (x1, y1), color[int(cls_pred)], 2)
            cv2.putText(frm, classes[int(cls_pred)], (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color[int(cls_pred)], 2, cv2.LINE_AA)

    # plt.axis("off")
    # plt.gca().xaxis.set_major_locator(NullLocator())
    # plt.gca().yaxis.set_major_locator(NullLocator())
    #plt.show(block =True)
    #cv2.imshow("functoan",frm)
    return frm
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="C:\\Users\\asus\\Downloads\\bucket.mp4", help="path to video")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt3_49.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--output_video", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    result = cv2.VideoWriter(opt.output_video,
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (800, 800))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        print("## loading ",(opt.weights_path))
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        print("## loadiing checkpoint_model")
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode
    classes = load_classes(opt.class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    print("\nPerforming object detection:")
    prev_time = time.time()
    
    if(opt.video_path!="-1"):
        vdo=cv2.VideoCapture(opt.video_path)
        while vdo.isOpened():
            print("Inside vdo")
            res,frame=vdo.read()
            k=cv2.waitKey(50)
            if(k=='p'):
                break
            image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            dim=(416,416)
            image = detect(cv2.resize(image, dim))
            result.write(cv2.resize(image,(800,800)))
            cv2.imshow("display",cv2.resize(image,(800,800)))
            current_time = time.time()
            fps=1/(current_time-prev_time)
            inference_time = datetime.timedelta(seconds=current_time - prev_time)
            prev_time = current_time
            print("\t+ Batch 1, Inference Time: %s" % (inference_time))
            print("FPS : %d" % (int(fps)))
            
    elif(opt.video_path=="-1"):
        print("Node handle")
        #nodeHandle
        
    else:
        print("## INVALID PATH GIVEN ##")

result.release()
    
    
    
    
    
    
