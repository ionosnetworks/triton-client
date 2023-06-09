#!/usr/bin/env python

import argparse
import numpy as np
import sys
import cv2

import tritonclient.grpc as grpcclient
from processing import preprocess, postprocess_v7 
from render import visualize_detection
from triton_client import connect_triton_server
from labels import COCOLabels
from tritonclient.utils import InferenceServerException

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
    parser.add_argument(
                        '-x',
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
                        default='2.tcp.ngrok.io:11563',
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
                        default=8,
                        help='Batch size. Default is 1.')
    return parser.parse_args()


def setup_model_io(INPUT_NAMES, OUTPUT_NAMES, FLAGS):
    inputs = []
    outputs = []

    for input_name in INPUT_NAMES:
        inputs.append(grpcclient.InferInput(input_name, [FLAGS.batch_size, 3, FLAGS.width, FLAGS.height], "FP32"))

    for output_name in OUTPUT_NAMES:
        outputs.append(grpcclient.InferRequestedOutput(output_name))

    return inputs, outputs

def requestGenerator(batched_image_data, input_name, output_name, dtype, FLAGS):

    # Set the input data
    inputs = [grpcclient.InferInput(input_name, batched_image_data.shape, dtype)]
    inputs[0].set_data_from_numpy(batched_image_data)

    outputs = [grpcclient.InferRequestedOutput(output_name)]

    yield inputs, outputs, FLAGS.model_name, FLAGS.model_version



if __name__ == '__main__':
    INPUT_NAMES = ["images"]
    OUTPUT_NAMES = ["output0"]
    input_name = INPUT_NAMES[0]
    output_name = OUTPUT_NAMES[0]
    dtype = "FP32"

    FLAGS = get_flags()
    triton_client = connect_triton_server(FLAGS)
    # inputs, outputs = setup_model_io(INPUT_NAMES, OUTPUT_NAMES, FLAGS)
    

    image_data = []
    for i in range(FLAGS.batch_size * 5):
        image_data.append(np.random.randn(3, FLAGS.width, FLAGS.height).astype(np.float32))
    # Send requests of FLAGS.batch_size images. If the number of
    # images isn't an exact multiple of FLAGS.batch_size then just
    # start over with the first images until the batch is filled.
    requests = []
    responses = []
    # result_filenames = []
    request_ids = []
    image_idx = 0
    last_request = False
    sent_count = 0

    while not last_request:
        repeated_image_data = []

        for idx in range(FLAGS.batch_size):
            repeated_image_data.append(image_data[image_idx])
            image_idx = (image_idx + 1) % len(image_data)
            if image_idx == 0:
                last_request = True

        batched_image_data = np.stack(repeated_image_data, axis=0)


        # Send request
        try:
            for inputs, outputs, model_name, model_version in requestGenerator(
                    batched_image_data, input_name, output_name, dtype, FLAGS):
                sent_count += 1
                response = triton_client.infer(FLAGS.model_name,
                                        inputs,
                                        request_id=str(sent_count),
                                        model_version=FLAGS.model_version,
                                        outputs=outputs)
                responses.append(response)
                this_id = response.get_response().id
                print("Request {}, batch size {}".format(this_id, FLAGS.batch_size))
                for output_name in OUTPUT_NAMES:
                    result = response.as_numpy(output_name)
                    print(f"Received result buffer \"{output_name}\" of size {result.shape}")
                    print(f"Naive buffer sum: {np.sum(result)}")
                

        except InferenceServerException as e:
            print("inference failed: " + str(e))
            sys.exit(1)


 










#     # DUMMY MODE
#     if FLAGS.mode == 'dummy':
#         print("Running in 'dummy' mode")
#         print("Creating emtpy buffer filled with ones...")
#         print("Invoking inference...")
#         inputs[0].set_data_from_numpy(np.ones(shape=(1, 3, FLAGS.width, FLAGS.height), dtype=np.float32))

#         results = triton_client.infer(model_name=FLAGS.model,
#                                       inputs=inputs,
#                                       outputs=outputs,
#                                       client_timeout=FLAGS.client_timeout)

#         for output_name in OUTPUT_NAMES:
#             result = results.as_numpy(output_name)
#             print(f"Received result buffer \"{output_name}\" of size {result.shape}")
#             print(f"Naive buffer sum: {np.sum(result)}")




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


#     # VIDEO MODE
#     if FLAGS.mode == 'video':
#         print("Running in 'video' mode")
#         if not FLAGS.input:
#             print("FAILED: no input video")
#             sys.exit(1)

#         print("Opening input video stream...")
#         cap = cv2.VideoCapture(FLAGS.input)
#         if not cap.isOpened():
#             print(f"FAILED: cannot open video {FLAGS.input}")
#             sys.exit(1)

#         counter = 0
#         out = None
#         print("Invoking inference...")
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("failed to fetch next frame")
#                 break

#             if counter == 0 and FLAGS.out:
#                 print("Opening output video stream...")
#                 fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#                 out = cv2.VideoWriter(FLAGS.out, fourcc, FLAGS.fps, (frame.shape[1], frame.shape[0]))

#             input_image_buffer = preprocess(frame, [FLAGS.width, FLAGS.height])
#             input_image_buffer = np.expand_dims(input_image_buffer, axis=0)

#             inputs[0].set_data_from_numpy(input_image_buffer)

#             results = triton_client.infer(model_name=FLAGS.model,
#                                           inputs=inputs,
#                                           outputs=outputs,
#                                           client_timeout=FLAGS.client_timeout)

#             num_dets = results.as_numpy("num_dets")
#             det_boxes = results.as_numpy("det_boxes")
#             det_scores = results.as_numpy("det_scores")
#             det_classes = results.as_numpy("det_classes")
#             detected_objects = postprocess_v7(num_dets, det_boxes, det_scores, det_classes, frame.shape[1], frame.shape[0], [FLAGS.width, FLAGS.height])

#             print(f"Frame {counter}: {len(detected_objects)} objects")
#             rendered_frame = visualize_detection(frame, detected_objects, labels=COCOLabels)
#             counter += 1

#             if FLAGS.out:
#                 out.write(rendered_frame)
#             else:
#                 cv2.imshow('image', rendered_frame)
#                 if cv2.waitKey(1) == ord('q'):
#                     break

#         cap.release()
#         if FLAGS.out:
#             out.release()
#         else:
#             cv2.destroyAllWindows()

# if FLAGS.model_info:
#     statistics = triton_client.get_inference_statistics(model_name=FLAGS.model)
#     if len(statistics.model_stats) != 1:
#         print("FAILED: get_inference_statistics")
#         sys.exit(1)
#     print(statistics)

# print("Done")



