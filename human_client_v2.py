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
from torchvision import transforms
from render import visualize_detection as visualize
from triton_model import connect_triton_server, TritonModel
import torch

def preprocess(imgs, input_shape, letter_box=True):
    if letter_box:
        new_imgs = []
        for img in imgs:
            img_h, img_w, _ = img.shape
            new_h, new_w = input_shape[0], input_shape[1]
            offset_h, offset_w = 0, 0
            if (new_w / img_w) <= (new_h / img_h):
                new_h = int(img_h * new_w / img_w)
                offset_h = (input_shape[0] - new_h) // 2
            else:
                new_w = int(img_w * new_h / img_h)
                offset_w = (input_shape[1] - new_w) // 2
            resized = cv2.resize(img, (new_w, new_h))
            img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
            img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
            new_imgs.append(img)
        imgs = new_imgs
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(input_shape, interpolation=transforms.InterpolationMode.BICUBIC),
        # transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensors = [transform(img) for img in imgs]
    img_tensors = torch.cat(img_tensors, dim=0).view(-1, 3, input_shape[0], input_shape[1])
    input_batch = img_tensors.to("cpu")
    # print(input_batch.shape)
    
    return [input_batch.numpy()]

 

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
 
 
def show_attributes(frames, grouped_attributes_list):
    blank_image = np.zeros((768, 768, 3), np.uint8)
    blank_image.fill(255)
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 2
    red_color = (0, 0, 255)
    black_color = (0, 0, 0)
    text_size, _ = cv2.getTextSize("dpl", font, font_scale, thickness)
    dh = text_size[1] + 2
  
    rendered_frames = []
    for frame, grouped_attributes in zip(frames, grouped_attributes_list):
        width, height = frame.shape[1], frame.shape[0]
        blank_image[768 - height:, 768 - width:] = frame
        frame = blank_image.copy()
        h = dh 
        for group, attrs in grouped_attributes.items():
            cv2.putText(frame, f"{group}:", (5, h), font, font_scale, black_color, thickness)
            h += dh
            for attr in attrs:
                cv2.putText(frame, f"{attr}", (5, h), font, font_scale, red_color, thickness)
                h += dh 
        rendered_frames.append(frame)  

    return rendered_frames   

def completion_callback(infer_status, result, error):
    infer_status.put((result, error))


def postprocess_thread():
    print("Starting postprocess thread")
    processed_count = 0
    current_batch = 0
    counter = defaultdict(int)

    while processed_count < sent_count or not sent_count:
        result, error = infer_status.get()
        if error:
            print("inference failed: " + str(error))
            sys.exit(1)

        batch_end = time.time()
        bs = len(result.as_numpy(model.output_names[0]))
        grouped_attrs_list = [defaultdict(list) for _ in range(bs)]
        for output_name in model.output_names:
            res = result.as_numpy(output_name)
            attribute_name = output_name[:-8]
            attribute_name_split = attribute_name.split('_')
            group, attr = attribute_name_split[0].lower(), '_'.join(attribute_name_split[1:]).lower()
            res = np.squeeze(res)
            # print(res)
            for i in range(bs):
                if res[i][0] < res[i][1]:
                    grouped_attrs_list[i][group].append(attr)
                    counter[attr] += 1
                else:
                    counter[attr] = 0

        request_id = int(result.get_response().id)
        frames = input_frames[request_id]
        # print(grouped_attrs_list)
        rendered_frames = show_attributes(frames, grouped_attrs_list, counter)
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
    if not os.path.exists("output"):
        os.makedirs("output")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    filename = os.path.splitext(os.path.basename(FLAGS.input))[0]
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_filename = FLAGS.out if FLAGS.out else f"output/{filename}_bs{FLAGS.batch_size}_{FLAGS.model_name}_{now}.mp4"
    print(f"Output video: {out_filename}")
    out = cv2.VideoWriter(out_filename, fourcc, fps, (768, 768))

    sent_count = 0
    print("Invoking inference...")
    start_time = time.time()

    postprocess_thread = threading.Thread(target=postprocess_thread)
    postprocess_thread.start()

    for frames in frame_generator(cap, FLAGS.batch_size):
        print(f"Sending batch {sent_count}...")

        input_image_buffer = preprocess(frames, [FLAGS.height, FLAGS.width])
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






