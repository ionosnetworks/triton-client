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

from processing import preprocess, postprocess 
from yolov8_utils import process_output
from render import visualize_detection as visualize
from triton_model import connect_triton_server, TritonModel
from lpr_utils import lpr_inference

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
    parser.add_argument('--confidence',
                        type=float,
                        required=False,
                        default=0.7,
                        help='confidence threshold, default 0.7')
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

def yolov8_postprocess(predictions, frames):
    detected_objects_list = []
    prediction = np.stack(predictions, axis=0)
    frame = frames[0]
    detections = process_output(prediction, frame.shape, [FLAGS.width, FLAGS.height], FLAGS.confidence, FLAGS.iou_threshold, batched=True)
    for i in range(len(frames)):
        frame = frames[i]
        detected_objects = postprocess(detections[i], frame.shape)
        detected_objects_list.append(detected_objects)
    return detected_objects_list

def completion_callback(infer_status, result, error):
    infer_status.put((result, error))


def postprocess_thread(lpr_model):
    print("Starting postprocess thread")
    processed_count = 0
    current_batch = 0

    while processed_count < sent_count or not sent_count:
        result, error = infer_status.get()
        if error:
            print("inference failed: " + str(error))
            sys.exit(1)

        batch_end = time.time()
        predictions = result.as_numpy(model.output_names[0])
        request_id = int(result.get_response().id)
        frames = input_frames[request_id]
        rendered_frames = []
        detected_objects_list = yolov8_postprocess(predictions, frames)
        lpr_inference(detected_objects_list, frames, lpr_model)

        for idx, frame in enumerate(frames):
            detected_objects = detected_objects_list[idx]
            rendered_frame = visualize(frame, detected_objects, verbose=FLAGS.verbose)
            rendered_frames.append(rendered_frame)
            
        batch_process_end = time.time()
        print(f"Postprocessed batch {request_id}, took {batch_process_end-batch_end:.3f} s")
        output_dict[request_id] = rendered_frames

        while current_batch in output_dict:
            for frame in output_dict[current_batch]:
                out.write(frame)
            print(f"Writing batch {current_batch} to output")
            current_batch += 1
        processed_count += 1

    print("All postprocessing complete")


if __name__ == '__main__':
    FLAGS = get_flags()
    triton_client = connect_triton_server(FLAGS.url)
    model = TritonModel(FLAGS.model_name, triton_client, model_version=FLAGS.model_version)
    lpr_model = TritonModel("lprnet_tao", triton_client)
    infer_status = queue.Queue()
    output_dict = {}
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
    if not os.path.exists("output"):
        os.makedirs("output")
    out = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height))

    sent_count = 0
    print("Invoking inference...")
    start_time = time.time()

    postprocess_thread = threading.Thread(target=postprocess_thread, args=(lpr_model,))
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






