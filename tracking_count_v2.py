import argparse
import numpy as np
import sys
import cv2
import time
import os
import queue
from functools import partial
import threading
from datetime import datetime
from collections import deque

import hydra
import torch
import argparse
import time
import torch
from numpy import random

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from processing import preprocess, postprocess 
from yolov8_utils import process_output
from triton_model import connect_triton_server, TritonModel


object_counter = {}

object_counter1 = {}

# line = [(100, 500), (1050, 500)]
# line = [(1200, 300), (2300, 550)]
line = [(960, 100), (960, 1000)]

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
NAMES = ['vehicle', 'human', 'license_plate']

def init_tracker():
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
    return deepsort
##########################################################################################
def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str
def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    global object_counter, object_counter1
    cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
          direction = get_direction(data_deque[id][0], data_deque[id][1])
          if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
              cv2.line(img, line[0], line[1], (255, 255, 255), 3)
              if "South" in direction:
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
              if "North" in direction:
                if obj_name not in object_counter1:
                    object_counter1[obj_name] = 1
                else:
                    object_counter1[obj_name] += 1
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    
    #4. Display Count in top right corner
    for idx, (key, value) in enumerate(object_counter1.items()):
        cnt_str = str(key) + ":" +str(value)
        cv2.line(img, (width - 500,25), (width,25), [85,45,255], 40)
        cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (width - 150, 65 + (idx*40)), (width, 65 + (idx*40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)

    for idx, (key, value) in enumerate(object_counter.items()):
        cnt_str1 = str(key) + ":" +str(value)
        cv2.line(img, (20,25), (500,25), [85,45,255], 40)
        cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
        cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,255], 30)
        cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
    return img

def display_count(img):
    global object_counter, object_counter1
    height, width, _ = img.shape
    cv2.line(img, line[0], line[1], (46,162,112), 3)

    for idx, (key, value) in enumerate(object_counter1.items()):
        cnt_str = str(key) + ":" +str(value)
        cv2.line(img, (width - 500,25), (width,25), [85,45,255], 40)
        cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        cv2.line(img, (width - 150, 65 + (idx*40)), (width, 65 + (idx*40)), [85, 45, 255], 30)
        # print(f"cv2.putText(img, {cnt_str})")
        cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)

    for idx, (key, value) in enumerate(object_counter.items()):
        cnt_str1 = str(key) + ":" +str(value)
        cv2.line(img, (20,25), (500,25), [85,45,255], 40)
        cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
        cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,255], 30)
        # print(f"    cv2.putText(img, {cnt_str1})")
        cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    return img

def update_tracker_and_render(preds, frame):
    global deepsort, NAMES
    preds = torch.from_numpy(preds)

    # write
    xywh_bboxs = []
    confs = []
    oids = []
    outputs = []
    for *xyxy, conf, cls in reversed(preds):
        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
        xywh_bboxs.append(xywh_obj)
        confs.append([conf.item()])
        oids.append(int(cls))
    xywhs = torch.Tensor(xywh_bboxs)
    confss = torch.Tensor(confs)

    if xywhs.numel() > 0:
        outputs = deepsort.update(xywhs, confss, oids, frame)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            rendered_img = draw_boxes(frame, bbox_xyxy, NAMES, object_id, identities)
            return rendered_img
        else:
            return display_count(frame)
    else:
        return display_count(frame)


def get_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input video file')
    parser.add_argument('-a',
                        '--async',
                        dest="async_set",
                        action="store_true",
                        required=False,
                        default=False,
                        help='Use asynchronous inference API')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=False,
                        default='yolov8',
                        help='Inference model name, default yolov8')
    parser.add_argument('-x',   
                        '--model-version',
                        type=str,
                        required=False,
                        default="",
                        help='Version of model. Default is to use latest version.')
    parser.add_argument('--width',
                        type=int,
                        required=False,
                        default=768,
                        help='Inference model input width, default 640')
    parser.add_argument('--height',
                        type=int,
                        required=False,
                        default=768,
                        help='Inference model input height, default 640')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL, default localhost:8001')
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        required=False,
                        default='',
                        help='Output video file name.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=16,
                        help='Batch size. Default is 4.')
    parser.add_argument('-f',
                        '--fps',
                        type=float,
                        required=False,
                        default=None,
                        help='Video output fps, default video original FPS')
    parser.add_argument('-i',
                        '--model-info',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Print model status, configuration and statistics')
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose client output')
    return parser.parse_args()


def frame_generator(cap, batch_size=1):
    while True:
        frames = []
    
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if frames:
            yield frames
        else:
            break
 
def yolov8_preprocess(frames):
    input_image_buffer = []
    for frame in frames:
        input_image_buffer.append(preprocess(frame, [FLAGS.height, FLAGS.width]))
    input_image_buffer = [np.stack(input_image_buffer, axis=0)]
    return input_image_buffer

def yolov8_postprocess(prediction, frame):
    prediction = np.expand_dims(prediction, axis=0)
    detections = process_output(prediction, frame.shape, [FLAGS.width, FLAGS.height], conf_threshold=0.25, iou_threshold=0.45)
    rendered_img = update_tracker_and_render(detections, frame)
    return rendered_img

def completion_callback(infer_status, result, error):
    infer_status.put((result, error))


def postprocess_thread():
    print("Starting postprocess thread")
    processed_count = 0
    current_batch = 0

    while processed_count < sent_count or not sent_count:
        result, error = infer_status.get()
        if error:
            print("inference failed: " + str(error))
            sys.exit(1)
   
        predictions = result.as_numpy(model.output_names[0])
        request_id = int(result.get_response().id)
        predictions_dict[request_id] = predictions
        print(f"Results received from batch {current_batch}")

        while current_batch in predictions_dict:
            batch_end = time.time()
            for frame, prediction in zip(input_frames[current_batch], predictions_dict[current_batch]):
                rendered_frame = yolov8_postprocess(prediction, frame)
                out.write(rendered_frame)        
            batch_process_end = time.time()
            print(f"Postprocessed batch {request_id}, took {batch_process_end-batch_end:.3f} s")
            print(f"Writing batch {current_batch} to output")
            current_batch += 1
        processed_count += 1

    print("All postprocessing complete")


if __name__ == '__main__':
    FLAGS = get_flags()
    triton_client = connect_triton_server(FLAGS.url)
    model = TritonModel(FLAGS.model_name, triton_client, model_version=FLAGS.model_version)
    deepsort = init_tracker()
    infer_status = queue.Queue()
    output_dict = {}
    predictions_dict = {}
    input_frames = {}

    print("Running in video mode")
    if not FLAGS.input:
        print("FAILED: no input video")
        sys.exit(1)

    print("Opening input video stream...")
    cap = cv2.VideoCapture(FLAGS.input)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) if not FLAGS.fps else FLAGS.fps
    
    if not cap.isOpened():
        print(f"FAILED: cannot open video {FLAGS.input}")
        sys.exit(1)


    print("Opening output video stream...")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    filename = os.path.splitext(os.path.basename(FLAGS.input))[0]
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_filename = FLAGS.out if FLAGS.out else f"output/{filename}_bs{FLAGS.batch_size}_{FLAGS.model_name}_{now}.mp4"
    print(f"Output video: {out_filename}")
    out = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height))

    sent_count = 0
    print("Invoking inference...")
    start_time = time.time()

    postprocess_thread = threading.Thread(target=postprocess_thread)
    postprocess_thread.start()

    for frames in frame_generator(cap, FLAGS.batch_size):
        print(f"Sending batch {sent_count}...")

        input_image_buffer = yolov8_preprocess(frames)
        input_frames[sent_count] = frames

        output = model.async_infer(input_image_buffer, partial(completion_callback, infer_status), request_id=str(sent_count))

        sent_count += 1

    print("All requests sent")

    postprocess_thread.join()
    cap.release()
    out.release()
    end_time = time.time()

    print(f"Took {end_time-start_time:.3f}s in total")
    print(f"{total_frames/(end_time-start_time):.3f} fps")
    print("Done!")

