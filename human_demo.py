# %%
import tritonclient.grpc as grpcclient
from triton_model import TritonModel
import numpy as np

# %%
model_name = "human_attributes"
triton_client = grpcclient.InferenceServerClient('10.1.10.119:8001')
model_config = triton_client.get_model_config(model_name).config

# %%
bs = 4

model = TritonModel(model_name, triton_client)
input_datas = [np.random.randn(bs, 3, 224, 224).astype(np.float32)]
output = model(input_datas)
 

groups = ['AGE', 'GENDER', 'APPAREL_LOWER', 'FACESHAPE', 'FACEFEATURES', 'PHYSIQUE', 'HERITAGE',\
        'HAIR_LENGTH', 'HAIR_COLOR', 'HAIR_TYPE','HAIR_STYLE','APPAREL_STYLE', 'APPAREL_UPPER_SLEEVELENGTH',\
        'APPAREL_UPPER_COLOR', 'APPAREL_LOWER_COLOR', 'APPAREL_LOWER_BOTTOMLENGTH','APPAREL_UPPER_TOPSTYLE', \
        'APPAREL_UPPER_COVERINGS', 'APPAREL_LOWER_BOTTOMSTYLE', 'FOOTWEAR', 'POSSESSIONS', 'ROLE']

 # %%

results = np.zeros((bs, 224))

all_attributes = [[] for _ in range(bs)]

for idx, output_name in enumerate(model.output_names):
    res = output[output_name]
    res = np.squeeze(res)
    print(np.argmax(res, axis=1))
    results[:, idx] = np.argmax(res, axis=1)

for idx, result in enumerate(results):
    print(result)
    for i in result:
        if i == 1:
            all_attributes[idx].append(model.output_names[idx][:-8])

for i in range(bs):
    print(all_attributes[i])
 
# %%
from collections import defaultdict
grouped_attrs_list = [ defaultdict(list) for _ in range(bs)]
for idx, output_name in enumerate(model.output_names):
    res = output[output_name]
    attribute_name = output_name[:-8]
    attribute_name_split = attribute_name.split('_')
    group, attr = attribute_name_split[0].lower(), '_'.join(attribute_name_split[1:]).lower()
    res = np.squeeze(res)
    for i in range(bs):
        if res[i][0] > res[i][1]:
            grouped_attrs_list[i][group].append(attr)
       
for i in range(bs):
    for groups in grouped_attrs_list[i]:
        print(groups)
 


#  all_attributes = set()
# for attr, val in output.items():
#     attr = attr.replace('/Softmax', '')
#     attr_groups = attr.split('_')[0]
#     val = np.squeeze(val)
#     if val[0] > val[1]:
#         all_attributes.add(attr)
# all_attributes = sorted(list(all_attributes)) 
# print(all_attributes)
 
   


# %%


model_meta = triton_client.get_model_metadata(model_name)
model_config = triton_client.get_model_config(model_name).config

input_names = [input.name for input in model_meta.inputs]
output_names = [output.name for output in model_meta.outputs]
input_datatypes = [input.datatype for input in model_meta.inputs]
# %%

output_dict = {}
inputs = []
outputs = []

for input_data, input_name, input_datatype in zip(input_datas, input_names, input_datatypes):
    input = grpcclient.InferInput(input_name, input_data.shape, input_datatype)
    inputs.append(input)
    inputs[-1].set_data_from_numpy(input_data)

# for output_name in output_names:
#     output = grpcclient.InferRequestedOutput(output_name)
#     outputs.append(output)

# outputs = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
outputs = triton_client.infer(model_name=model_name, inputs=inputs)

for output_name in output_names:
    output_dict[output_name] = outputs.as_numpy(output_name)
    result = output_dict[output_name]
    print(f"Received result buffer \"{output_name}\" of size {result.shape}")
    print(f"Naive buffer sum: {np.sum(result)}")





# %%

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

# %%

input_datas = [np.random.randn(1, 3, 768, 768).astype(np.float32)]
model_name = "yolov8"
url = "6.tcp.ngrok.io:17001"
output = simple_triton_model(model_name, input_datas, url=url)