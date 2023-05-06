#!/usr/bin/env python

import argparse
import numpy as np
import sys
import cv2
import time
import os

from processing import preprocess 
from yolov8_utils import process_output, postprocess
from render import visualize_detection
from triton_model import connect_triton_server, TritonModel


def get_flags():
    parser = argparse.ArgumentParser()

    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input video file')
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
 
if __name__ == '__main__':
    FLAGS = get_flags()
    triton_client = connect_triton_server(FLAGS.url)
    model = TritonModel(FLAGS.model_name, triton_client, model_version=FLAGS.model_version)

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
    out_filename = FLAGS.out if FLAGS.out else f"output/{filename}_bs{FLAGS.batch_size}_output.mp4"
    print(f"Output video: {out_filename}")
    out = cv2.VideoWriter(out_filename, fourcc, fps, (frame_width, frame_height))

    counter = 0
    print("Invoking inference...")
    start_time = time.time()

    for frames in frame_generator(cap, FLAGS.batch_size):
        print(f"Processing batch {counter}...")
        batch_start = time.time()

        input_image_buffer = []
        for frame in frames:
            input_image_buffer.append(preprocess(frame, [FLAGS.height, FLAGS.width]))
        input_image_buffer = [np.stack(input_image_buffer, axis=0)]

        output = model(input_image_buffer)
        
        batch_end = time.time()
        print(f"Finished request, batch size {FLAGS.batch_size},took {batch_end-batch_start:.3f} s"  )

        predictions = output[model.output_names[0]]
        for idx, prediction in enumerate(predictions):
            frame = frames[idx]
            prediction = np.expand_dims(prediction, axis=0)
            detections = process_output(prediction, frame.shape, [FLAGS.width, FLAGS.height], conf_threshold=0.25, iou_threshold=0.45)
            detected_objects = postprocess(detections, frame.shape)
            if FLAGS.verbose:
                print(f"Batch {counter} Frame {idx}: {len(detected_objects)} objects")
            rendered_frame = visualize_detection(frame, detected_objects, verbose=FLAGS.verbose)
            out.write(rendered_frame)
            
        batch_process_end = time.time()
        print(f"Postprocessed batch {counter}, took {batch_process_end-batch_end:.3f} s")

        counter += 1

    cap.release()
    out.release()
    end_time = time.time()

    print(f"Took {end_time-start_time:.3f}s in total")
    print(f"{total_frames/(end_time-start_time):.3f} fps")
    print("Done!")






