#!/usr/bin/env python

import argparse
import numpy as np
import sys
import cv2

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from processing import preprocess 
from yolov8_utils import process_output, postprocess
from render import visualize_detection
 
from triton_client import connect_triton_server

INPUT_NAMES = ["images"]
OUTPUT_NAMES = ["output0"]

if __name__ == '__main__':
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
                        '--model',
                        type=str,
                        required=False,
                        default='yolov8',
                        help='Inference model name, default yolov7')
    parser.add_argument('--width',
                        type=int,
                        required=False,
                        default=768,
                        help='Inference model input width, default 768')
    parser.add_argument('--height',
                        type=int,
                        required=False,
                        default=768,
                        help='Inference model input height, default 768')
    parser.add_argument('--confidence',
                        type=float,
                        required=False,
                        default=0.25,
                        help='confidence threshold, default 0.25')
    parser.add_argument('--iou_threshold',
                        type=float,
                        required=False,
                        default=0.5,
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
                        default=24.0,
                        help='Video output fps, default 24.0 FPS')
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
    parser.add_argument('-x',
                        '--certificate-chain',
                        type=str,
                        required=False,
                        default=None,
                        help='File holding PEM-encoded certicate chain default is none')
    

    FLAGS = parser.parse_args()
    triton_client = connect_triton_server(FLAGS)
    
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, FLAGS.width, FLAGS.height], "FP32"))
    inputs[0].set_data_from_numpy(np.ones(shape=(1, 3, FLAGS.width, FLAGS.height), dtype=np.float32))
    outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))

    # DUMMY MODE
    if FLAGS.mode == 'dummy':
        print("Running in 'dummy' mode")
        print("Creating emtpy buffer filled with ones...")
        print("Invoking inference...")
        results = triton_client.infer(model_name=FLAGS.model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=FLAGS.client_timeout)

        for output in OUTPUT_NAMES:
            result = results.as_numpy(output)
            print(f"Received result buffer \"{output}\" of size {result.shape}")
            print(f"Naive buffer sum: {np.sum(result)}")


    # IMAGE MODE
    if FLAGS.mode == 'image':
        print("Running in 'image' mode")
        if not FLAGS.input:
            print("FAILED: no input image")
            sys.exit(1)

        print("Creating buffer from image file...")
        input_image = cv2.imread(str(FLAGS.input))
        if input_image is None:
            print(f"FAILED: could not load input image {str(FLAGS.input)}")
            sys.exit(1)
        input_image_buffer = preprocess(input_image, [FLAGS.width, FLAGS.height])
        input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

        inputs[0].set_data_from_numpy(input_image_buffer)

        print("Invoking inference...")
        results = triton_client.infer(model_name=FLAGS.model,
                                      inputs=inputs,
                                      outputs=outputs,
                                      client_timeout=FLAGS.client_timeout)

        for output in OUTPUT_NAMES:
            result = results.as_numpy(output)
            print(f"Received result buffer \"{output}\" of size {result.shape}")
            print(f"Naive buffer sum: {np.sum(result)}")

        prediction = results.as_numpy(OUTPUT_NAMES[0])
        detections = process_output(prediction, input_image.shape, [FLAGS.width, FLAGS.height], FLAGS.confidence, FLAGS.iou_threshold)
        detected_objects = postprocess(detections, input_image.shape)

        print(f"Detected objects: {len(detected_objects)}")
        rendered_image = visualize_detection(input_image, detected_objects)

        if FLAGS.out:
            cv2.imwrite(FLAGS.out, rendered_image)
            print(f"Saved result to {FLAGS.out}")

        else:
            cv2.imshow('image', rendered_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    # VIDEO MODE
    if FLAGS.mode == 'video':
        print("Running in 'video' mode")
        if not FLAGS.input:
            print("FAILED: no input video")
            sys.exit(1)

        print("Opening input video stream...")
        cap = cv2.VideoCapture(FLAGS.input)
        if not cap.isOpened():
            print(f"FAILED: cannot open video {FLAGS.input}")
            sys.exit(1)

        counter = 0
        out = None
        print("Invoking inference...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("failed to fetch next frame")
                break

            if counter == 0 and FLAGS.out:
                print("Opening output video stream...")
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                out = cv2.VideoWriter(FLAGS.out, fourcc, FLAGS.fps, (frame.shape[1], frame.shape[0]))

            input_image_buffer = preprocess(frame, [FLAGS.width, FLAGS.height])
            input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

            inputs[0].set_data_from_numpy(input_image_buffer)

            results = triton_client.infer(model_name=FLAGS.model,
                                          inputs=inputs,
                                          outputs=outputs,
                                          client_timeout=FLAGS.client_timeout)

            prediction = results.as_numpy(OUTPUT_NAMES[0])
            detections = process_output(prediction, frame.shape, [FLAGS.width, FLAGS.height], FLAGS.confidence, FLAGS.iou_threshold)
            detected_objects = postprocess(detections, frame.shape)

            print(f"Frame {counter}: {len(detected_objects)} objects")
            rendered_frame = visualize_detection(frame, detected_objects)
            counter += 1

            if FLAGS.out:
                out.write(rendered_frame)
            else:
                cv2.imshow('image', rendered_frame)
                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()
        if FLAGS.out:
            out.release()
        else:
            cv2.destroyAllWindows()

if FLAGS.model_info:
    statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
    if len(statistics.model_stats) != 1:
        print("FAILED: get_inference_statistics")
        sys.exit(1)
    print(statistics)

print("Done")