import tritonclient.grpc as grpcclient

def simple_triton_model(model_name, input_datas, url="localhost:8001"):

    triton_client = grpcclient.InferenceServerClient(url)

    model_meta = triton_client.get_model_metadata(model_name)
    input_names = [input.name for input in model_meta.inputs]
    output_names = [output.name for output in model_meta.outputs]
    input_datatypes = [input.datatype for input in model_meta.inputs]

    output_dict = {}
    inputs = []

    for input_data, input_name, input_datatype in zip(input_datas, input_names, input_datatypes):
        input = grpcclient.InferInput(input_name, input_data.shape, input_datatype)
        inputs.append(input)
        inputs[-1].set_data_from_numpy(input_data)

    outputs = triton_client.infer(model_name=model_name, inputs=inputs)

    for output_name in output_names:
        output_dict[output_name] = outputs.as_numpy(output_name)
            
    return output_dict