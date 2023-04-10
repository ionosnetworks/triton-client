#!/usr/bin/env python

import sys
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import argparse

def set_flags(**args):
    FLAGS = argparse.Namespace()
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

    if not triton_client.is_model_ready(FLAGS.model_name):
        print("FAILED : is_model_ready")
        sys.exit(1)
    
    print("server and model ready!")

    if FLAGS.model_info:
        # Model metadata
        try:
            metadata = triton_client.get_model_metadata(FLAGS.model_name)
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
            config = triton_client.get_model_config(FLAGS.model_name)
            if not (config.config.name == FLAGS.model_name):
                print("FAILED: get_model_config")
                sys.exit(1)
            print(config)
        except InferenceServerException as ex:
            print("FAILED : get_model_config")
            print("Got: {}".format(ex.message()))
            sys.exit(1)
    return triton_client

