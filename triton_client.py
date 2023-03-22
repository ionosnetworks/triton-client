#!/usr/bin/env python

import sys
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

def set_flags(**args):
    FLAGS = {}
    FLAGS.model = 'unifiednet'
    FLAGS.width = 640
    FLAGS.height = 640
    FLAGS.url = "localhost:8001"
    FLAGS.fps = 24.0
    FLAGS.model_info = False
    FLAGS.verbose = False
    FLAGS.client_timeout = None
    FLAGS.ssl = False
    FLAGS.root_certificates = None
    FLAGS.private_key = None
    FLAGS.certificate_chain = None
    if args:
        for key, value in args.items():
            if key in FLAGS:
                FLAGS[key] = value
    return FLAGS


def connect_triton_server(FLAGS):
    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    # Health check
    if not triton_client.is_server_live():
        print("FAILED : is_server_live")
        sys.exit(1)

    if not triton_client.is_server_ready():
        print("FAILED : is_server_ready")
        sys.exit(1)

    if not triton_client.is_model_ready(FLAGS.model):
        print("FAILED : is_model_ready")
        sys.exit(1)

    if FLAGS.model_info:
        # Model metadata
        try:
            metadata = triton_client.get_model_metadata(FLAGS.model)
            print(metadata)
        except InferenceServerException as ex:
            if "Request for unknown model" not in ex.message():
                print("FAILED : get_model_metadata")
                print("Got: {}".format(ex.message()))
                sys.exit(1)
            else:
                print("FAILED : get_model_metadata")
                sys.exit(1)

        # Model configuration
        try:
            config = triton_client.get_model_config(FLAGS.model)
            if not (config.config.name == FLAGS.model):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)
    return triton_client


def triton_infer(image_input, FLAGS=None):

    if not FLAGS:
        FLAGS = set_flags()
    INPUT_NAMES = ["images"]
    OUTPUT_NAMES = ["output0"]
    inputs = []
    outputs = []
    for INPUT_NAME in INPUT_NAMES:
        inputs.append(grpcclient.InferInput(INPUT_NAME, [1, 3, FLAGS.width, FLAGS.height], "FP32"))
    for OUTPUT_NAME in OUTPUT_NAMES:
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAME))


    inputs[0].set_data_from_numpy(image_input)

    triton_client = connect_triton_server(FLAGS)
    print("Invoking inference...")
    results = triton_client.infer(model_name=FLAGS.model,
                                    inputs=inputs,
                                    outputs=outputs,
                                    client_timeout=FLAGS.client_timeout)

    model_outputs = []
    for OUTPUT_NAME in OUTPUT_NAMES:
        result = results.as_numpy(OUTPUT_NAME)
        model_outputs.append(result)
        print(f"Received result buffer \"{OUTPUT_NAME}\" of size {result.shape}")
        print(f"Naive buffer sum: {sum(result)}")
    return model_outputs
