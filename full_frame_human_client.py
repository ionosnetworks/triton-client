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

# import hydra
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

NAMES = ['vehicle', 'human', 'license_plate']
 
class Track:
    def __init__(self, id) -> None:
        self.id = id
        self.frames = []
        self.attributes = []
        self.width, self.height = 0, 0

    def add_frame(self, original_frame, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        frame = original_frame[y1:y2, x1:x2, :]
        self.width = max(self.width, frame.shape[1])
        self.height = max(self.height, frame.shape[0])
        self.frames.append(frame)

    def save(self, save_folder='output/test', fps=5):
        if len(self.frames) < 5: return
        if not os.path.exists(save_folder): 
            os.makedirs(save_folder)
        filename = os.path.join(save_folder, f'{self.id}.mp4')
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (self.width, self.height))
        for frame in self.frames:
            y_pos = (self.height - frame.shape[0]) // 2
            x_pos = (self.width - frame.shape[1]) // 2
            frame = cv2.copyMakeBorder(frame, y_pos, y_pos, x_pos, x_pos, cv2.BORDER_CONSTANT, value=(127,127,127))
            out.write(frame)       
        out.release()


def init_tracker():
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
    return deepsort

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

def update_tracks(preds, frame):
    global deepsort, NAMES
    preds = torch.from_numpy(preds)

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
            for idx, id in enumerate(identities):
                if NAMES[object_id[idx]] != "human": continue
                if id not in track_dict:
                    track_dict[id] = Track(id)
                track_dict[id].add_frame(frame, bbox_xyxy[idx])

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
    rendered_img = update_tracks(detections, frame)
    return rendered_img

def completion_callback(infer_status, result, error):
    infer_status.put((result, error))

def postprocess_thread():
    global out_folder
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
        print(f"Results received from batch {request_id}")

        while current_batch in predictions_dict:
            batch_end = time.time()
            for frame, prediction in zip(input_frames[current_batch], predictions_dict[current_batch]):
                rendered_frame = yolov8_postprocess(prediction, frame)
                # out.write(rendered_frame)        
            batch_process_end = time.time()
            print(f"Postprocessed batch {current_batch}, took {batch_process_end-batch_end:.3f} s")
            print(f"Writing batch {current_batch} to output")
            current_batch += 1
        processed_count += 1
    
    for track in track_dict.values():
        track.save(save_folder=out_folder)

    print(f"All human clips saved to {out_folder}")


if __name__ == '__main__':
    FLAGS = get_flags()
    triton_client = connect_triton_server(FLAGS.url)
    model = TritonModel(FLAGS.model_name, triton_client, model_version=FLAGS.model_version)
    deepsort = init_tracker()
    infer_status = queue.Queue()
    output_dict = {}
    predictions_dict = {}
    input_frames = {}
    track_dict = {}

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
    out_folder = "output/" + os.path.splitext(os.path.basename(out_filename))[0]
    if not os.path.exists("output"):
        os.makedirs("output")
    # print(f"Output video: {out_filename}")
    # out = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height))

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
    # out.release()
    end_time = time.time()

    folder_path =  out_folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.mp4')):
            command = f'python human_client_v3.py -u {FLAGS.url} --folder {out_folder+"/track_res"} {file_path}'
            print(command)
            os.system(command)

    print(f"Took {end_time-start_time:.3f}s in total")
    print(f"{total_frames/(end_time-start_time):.3f} fps")
    print("Done!")

