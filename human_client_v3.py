#!/usr/bin/env python

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
from collections import defaultdict

from processing import preprocess, postprocess
from render import visualize_detection as visualize
from triton_model import connect_triton_server, TritonModel
from PIL import Image

from preprocess_input import preprocess_input

def preprocess(img, input_shape):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    image = img.resize(input_shape, Image.LANCZOS).convert('RGB')
    inference_input = preprocess_input(np.array(image).astype(np.float32).transpose(2, 0, 1))
    return inference_input

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
                        default='human_attributes',
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
                        default=128,
                        help='Inference model input width, default 640')
    parser.add_argument('--height',
                        type=int,
                        required=False,
                        default=256,
                        help='Inference model input height, default 640')
    parser.add_argument('--confidence',
                        type=float,
                        required=False,
                        default=0.7,
                        help='confidence threshold, default 0.7')
    parser.add_argument('-cm', '--consecutive-match',
                        type=int,
                        required=False,
                        default=5,
                        help='minimum consecutive match, default 5')
    parser.add_argument('--iou_threshold',
                        type=float,
                        required=False,
                        default=0.45,
                        help='IoU threshold, default 0.45')
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
    parser.add_argument('-fd',
                        '--folder',
                        type=str,
                        required=False,
                        default='',
                        help='Output folder name.')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=4,
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
 
def human_preprocess(frames):
    input_image_buffer = []
    for frame in frames:
        input_image_buffer.append(preprocess(frame, [FLAGS.width, FLAGS.height]))
    input_image_buffer = [np.stack(input_image_buffer, axis=0)]
    return input_image_buffer



def show_attributes(frames, grouped_attributes_list):
    global prev_count, counter
    blank_image = np.zeros((768, 768, 3), np.uint8)
    blank_image.fill(255)
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 2
    red_color = (0, 0, 255)
    black_color = (0, 0, 0)
    text_size, _ = cv2.getTextSize("dp", font, font_scale, thickness)
    dh = text_size[1]
    rendered_frames = []
    cur_count = set()
    for frame, grouped_attributes in zip(frames, grouped_attributes_list):
        width, height = frame.shape[1], frame.shape[0]
        blank_image[768 - height:, 768 - width:] = frame
        frame = blank_image.copy()
        h = dh 
        for group, attrs in grouped_attributes.items():
            for attr in attrs:
                counter[group, attr] += 1
                cur_count.add((group, attr))

        shown_group = set()
        for group, attr in sorted(list(counter)):
            if counter[group, attr] >= FLAGS.consecutive_match:
                if group not in shown_group:
                    cv2.putText(frame, f"{group}:", (5, h), font, font_scale, black_color, thickness)
                    h += dh
                    shown_group.add(group)
                cv2.putText(frame, f"{attr}", (5, h), font, font_scale, red_color, thickness)
                h += dh 

        for group, attr in prev_count.difference(cur_count):
            if counter[group, attr] < FLAGS.consecutive_match:
                counter[group, attr] = 0

        rendered_frames.append(frame) 
        prev_count = cur_count

    return rendered_frames   

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

        batch_end = time.time()
        bs = len(result.as_numpy(model.output_names[0]))
        grouped_attrs_list = [ defaultdict(list) for _ in range(bs)]
        for output_name in model.output_names:
            res = result.as_numpy(output_name)
            attribute_name = output_name[:-8]
            attribute_name_split = attribute_name.split('_')
            group, attr = attribute_name_split[0].lower(), '_'.join(attribute_name_split[1:]).lower()
            if bs > 1:
                res = np.squeeze(res)
            for i in range(bs):
                if res[i][0] <= res[i][1]:
                    grouped_attrs_list[i][group].append(attr)

        request_id = int(result.get_response().id)
        # frames = input_frames[request_id]
        predictions_dict[request_id] = grouped_attrs_list
        # print(grouped_attrs_list)
        # rendered_frames = show_attributes(frames, grouped_attrs_list)
        # batch_process_end = time.time()
        print(f"Received batch {request_id}")

        while current_batch in predictions_dict:
            frames = input_frames[current_batch]
            grouped_attrs_list = predictions_dict[current_batch]
            rendered_frames = show_attributes(frames, grouped_attrs_list)
            print(f"Writing batch {current_batch} to output")
            for rendered_frame in rendered_frames:
                out.write(rendered_frame)
            current_batch += 1
        processed_count += 1

    print("All postprocessing complete")


if __name__ == '__main__':
    FLAGS = get_flags()
    triton_client = connect_triton_server(FLAGS.url)
    model = TritonModel(FLAGS.model_name, triton_client, model_version=FLAGS.model_version)
    infer_status = queue.Queue()
    output_dict = {}
    predictions_dict = {}
    input_frames = {}
    counter = defaultdict(int)
    prev_count = set()

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
    out_folder = FLAGS.folder if FLAGS.folder else "output"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    filename = os.path.splitext(os.path.basename(FLAGS.input))[0]
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_filename = FLAGS.out if FLAGS.out else f"{out_folder}/{filename}_bs{FLAGS.batch_size}_{FLAGS.model_name}_{now}.mp4"
    print(f"Output video: {out_filename}")
    out = cv2.VideoWriter(out_filename, fourcc, fps, (768, 768))

    sent_count = 0
    print("Invoking inference...")
    start_time = time.time()

    postprocess_thread = threading.Thread(target=postprocess_thread)
    postprocess_thread.start()

    for frames in frame_generator(cap, FLAGS.batch_size):
        print(f"Sending batch {sent_count}...")

        input_image_buffer = human_preprocess(frames)
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






