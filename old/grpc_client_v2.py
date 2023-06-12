#!/usr/bin/env python

import argparse
import numpy as np
import sys
import cv2
import time
import os

import tritonclient.grpc as grpcclient
from processing import preprocess 
from yolov8_utils import process_output, postprocess
from render import visualize_detection
from triton_client import connect_triton_server

def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        choices=['dummy', 'image', 'video'],
                        default='dummy',
                        help='Run mode. \'dummy\' will send an emtpy buffer to the server to test if inference works. \'image\' will process an image. \'video\' will process a video.')
    parser.add_argument('input',
                        type=str,
                        nargs='?',
                        help='Input file to load from in image or video mode')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=False,
                        default='yolov8',
                        help='Inference model name, default yolov7')
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
                        default=0.25,
                        help='confidence threshold, default 0.25')
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
                        help='Write output into file instead of displaying it')
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
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds, default no timeout')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-r',
                        '--root-certificates',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded root certificates, default none')
    parser.add_argument('-p',
                        '--private-key',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded private key, default is none')
    parser.add_argument('-c',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none')
    parser.add_argument('-b',
                        '--batch-size',
                        type=int,
                        required=False,
                        default=16,
                        help='Batch size. Default is 16.')
    return parser.parse_args()


def frame_generator(cap, batch_size=1):
    while True:
        frames = []
    
        for i in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if frames:
            yield frames
        else:
            break

def set_batch_io(batched_data, input_name, output_name, dtype="FP32"):
    for input_name in INPUT_NAMES:
        inputs = [grpcclient.InferInput(input_name, batched_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_data)

    outputs = []
    for output_name in OUTPUT_NAMES:
        outputs.append(grpcclient.InferRequestedOutput(output_name))

    return inputs, outputs 


if __name__ == '__main__':

    INPUT_NAMES = ["images"]
    OUTPUT_NAMES = ["output0"]

    FLAGS = get_flags()
    triton_client = connect_triton_server(FLAGS)
     
 
    # VIDEO MODE
    if FLAGS.mode == 'video':
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
                input_image_buffer.append(preprocess(frame, [FLAGS.width, FLAGS.height]))
            input_image_buffer = np.stack(input_image_buffer, axis=0)

            inputs, outputs = set_batch_io(input_image_buffer, INPUT_NAMES, OUTPUT_NAMES)


            response = triton_client.infer(model_name=FLAGS.model_name,
                                          inputs=inputs,
                                          outputs=outputs,
                                          request_id=str(counter),
                                          client_timeout=FLAGS.client_timeout)
            
            
            this_id = response.get_response().id
            batch_end = time.time()
            print(f"Finished request {this_id}, batch size {FLAGS.batch_size},took {batch_end-batch_start:.3f} s"  )

            for output_name in OUTPUT_NAMES:
                result = response.as_numpy(output_name)
                if FLAGS.verbose:
                    print(f"Received result buffer \"{output_name}\" of size {result.shape}")
                    print(f"Naive buffer sum: {np.sum(result)}")
            
 
            predictions = response.as_numpy(OUTPUT_NAMES[0])
            for idx, prediction in enumerate(predictions):
                frame = frames[idx]
                prediction = np.expand_dims(prediction, axis=0)
                detections = process_output(prediction, frame.shape, [FLAGS.width, FLAGS.height], FLAGS.confidence, FLAGS.iou_threshold)
                detected_objects = postprocess(detections, frame.shape)
                if FLAGS.verbose:
                    print(f"Batch {counter} Frame {idx}: {len(detected_objects)} objects")
                rendered_frame = visualize_detection(frame, detected_objects)
                out.write(rendered_frame)
            batch_process_end = time.time()
            print(f"Postprocessed batch {counter}, took {batch_process_end-batch_end:.3f} s")

            counter += 1

        cap.release()
        out.release()
        end_time = time.time()
 


#     # IMAGE MODE
#     if FLAGS.mode == 'image':
#         print("Running in 'image' mode")
#         if not FLAGS.input:
#             print("FAILED: no input image")
#             sys.exit(1)

#         print("Creating buffer from image file...")
#         input_image = cv2.imread(str(FLAGS.input))
#         if input_image is None:
#             print(f"FAILED: could not load input image {str(FLAGS.input)}")
#             sys.exit(1)

#         input_image_buffer = preprocess(input_image, [FLAGS.width, FLAGS.height])
#         input_image_buffer = np.expand_dims(input_image_buffer, axis=0)
#         inputs[0].set_data_from_numpy(input_image_buffer)

#         print("Invoking inference...")
#         results = triton_client.infer(model_name=FLAGS.model,
#                                       inputs=inputs,
#                                       outputs=outputs,
#                                       client_timeout=FLAGS.client_timeout)


#         num_dets = results.as_numpy(OUTPUT_NAMES[0])
#         det_boxes = results.as_numpy(OUTPUT_NAMES[1])
#         det_scores = results.as_numpy(OUTPUT_NAMES[2])
#         det_classes = results.as_numpy(OUTPUT_NAMES[3])
#         detected_objects = postprocess_v7(num_dets, det_boxes, det_scores, det_classes, input_image.shape[1], input_image.shape[0], [FLAGS.width, FLAGS.height])


#         print(f"Detected objects: {len(detected_objects)}")
#         rendered_image = visualize_detection(input_image, detected_objects, labels=COCOLabels)

#         if FLAGS.out:
#             cv2.imwrite(FLAGS.out, rendered_image)
#             print(f"Saved result to {FLAGS.out}")

#         else:
#             cv2.imshow('image', rendered_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()







if FLAGS.model_info:
    statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
    if len(statistics.model_stats) != 1:
        print("FAILED: get_inference_statistics")
        sys.exit(1)
    print(statistics)

print(f"Took {end_time-start_time:.3f}s in total")
print(f"{total_frames/(end_time-start_time):.3f} fps")
print("Done!")



