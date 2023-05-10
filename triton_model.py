import tritonclient.grpc as grpcclient

def connect_triton_server(url="localhost:8001"):
    import sys
    # Create server context
    try:
        triton_client = grpcclient.InferenceServerClient(url)
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

    print("server ready!")

    return triton_client
 

class TritonModel:
    def __init__(self, model_name, triton_client, model_version="", preprocess=None, postprocess=None, verbose=False):
        self.model_name = model_name
        self.model_version = model_version
        self.triton_client = triton_client
        self.model_meta = triton_client.get_model_metadata(model_name)
        self.config = triton_client.get_model_config(model_name).config
        self.input_names = [input.name for input in self.model_meta.inputs]
        self.output_names = [output.name for output in self.model_meta.outputs]
        self.input_datatypes = [input.datatype for input in self.model_meta.inputs]
        self.output_datatypes = [output.datatype for output in self.model_meta.outputs]
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.verbose = verbose

    def __call__(self, input_datas, **kwargs):
        """input_datas: list of numpy arrays corresponding to the input names"""

        output_dict = {}
        inputs = []

        for input_data, input_name, input_datatype in zip(input_datas, self.input_names, self.input_datatypes):
            input = grpcclient.InferInput(input_name, input_data.shape, input_datatype)
            inputs.append(input)
            inputs[-1].set_data_from_numpy(input_data)

        outputs = self.triton_client.infer(model_name=self.model_name, inputs=inputs, model_version=self.model_version, **kwargs)

        for output_name in self.output_names:
            output_dict[output_name] = outputs.as_numpy(output_name)
                        
        return output_dict
    

    def infer(self, input_datas, **kwargs):
        return self.__call__(self, input_datas, **kwargs)
    

    def async_infer(self, input_datas, callback, **kwargs):
        """input_datas: list of numpy arrays corresponding to the input names"""

        inputs = []

        for input_data, input_name, input_datatype in zip(input_datas, self.input_names, self.input_datatypes):
            input = grpcclient.InferInput(input_name, input_data.shape, input_datatype)
            inputs.append(input)
            inputs[-1].set_data_from_numpy(input_data)

        self.triton_client.async_infer(self.model_name, inputs, callback, model_version=self.model_version, **kwargs)

    
 
 