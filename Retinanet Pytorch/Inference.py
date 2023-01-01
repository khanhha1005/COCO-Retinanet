import numpy as np
import time
import os
import csv
import cv2
import argparse


import matplotlib.pyplot  as plt

import torch
import numpy as np
import time
import os
import cv2
from Model import model


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    # color = COLORS[COCO_INSTANCE_CATEGORY_NAMES.index(classes[i])]

    # cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    # cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.putText(image, caption, (int(b[0]), int(b[1]-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, 
                lineType=cv2.LINE_AA)
def detect_image(img_path,model_path):
    labels = {0: 'person',1: 'bicycle',2: 'car',3: 'motorcycle',4: 'airplane',5: 'bus',6: 'train',7: 'truck',8: 'boat',9: 'traffic light',10: 'fire hydrant',11: 'stop sign',12: 'parking meter',13: 'bench',14: 'bird',15: 'cat',16: 'dog',17: 'horse',18: 'sheep',19: 'cow',20: 'elephant',21: 'bear',22: 'zebra',23: 'giraffe',24: 'backpack',25: 'umbrella',26: 'handbag',27: 'tie',28: 'suitcase',29: 'frisbee',30: 'skis',31: 'snowboard',32: 'sports ball',33: 'kite',34: 'baseball bat',35: 'baseball glove',36: 'skateboard',37: 'surfboard',38: 'tennis racket',39: 'bottle',40: 'wine glass',41: 'cup',42: 'fork',43: 'knife',44: 'spoon',45: 'bowl',46: 'banana',47: 'apple',48: 'sandwich',49: 'orange',50: 'broccoli',51: 'carrot',52: 'hot dog',53: 'pizza',54: 'donut',55: 'cake',56: 'chair',57: 'couch',58: 'potted plant',59: 'bed',60: 'dining table',61: 'toilet',62: 'tv',63: 'laptop',64: 'mouse',65: 'remote',66: 'keyboard',67: 'cell phone',68: 'microwave',69: 'oven',70: 'toaster',71: 'sink',72: 'refrigerator',73: 'book',74: 'clock',75: 'vase',76: 'scissors', 77: 'teddy bear',78: 'hair drier',79: 'toothbrush'}
    image =  cv2.imread(img_path)
    # Create the model
    model = model.resnet50(num_classes=80, pretrained=True)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(parser.model_path))
        model = torch.nn.DataParallel(model).cuda()

    model.training = False
    model.eval()

    image_orig = image.copy()

    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():

        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda()

        st = time.time()
        scores, classification, transformed_anchors = model(image.cuda().float())
        print('Elapsed time: {}'.format(time.time() - st))
        idxs = np.where(scores.cpu() > 0.5)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]

            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            label_name = labels[int(classification[idxs[0][j]])]
            score = scores[j]
            print(label_name)
            caption = '{} {:.3f}'.format(label_name, score)
            # draw_caption(img, (x1, y1, x2, y2), label_name)
            draw_caption(image_orig, (x1, y1, x2, y2), caption)
            cv2.rectangle(image_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
        cv2.imshow('detections', image_orig)
        cv2.waitKey(0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser = parser.parse_args()

    detect_image(parser.image_dir, parser.model_path, parser.class_list)
