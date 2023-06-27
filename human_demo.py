# %%
import tritonclient.grpc as grpcclient
from triton_model import TritonModel
import numpy as np
import cv2
from processing import preprocess, postprocess
from PIL import Image
import torch
from torchvision import transforms


# %%
model_name = "human_attributes"
triton_client = grpcclient.InferenceServerClient('learn04.prem.infra.livereachmedia.com:8001')
triton_client = grpcclient.InferenceServerClient('10.1.10.119:8001')

model_config = triton_client.get_model_config(model_name).config

# %%


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
    print(input_batch.shape)
    
    return [input_batch.numpy()]

# %%
from preprocess_input import preprocess_input
def preprocess(img, input_shape):
    image = img.resize(input_shape, Image.ANTIALIAS).convert('RGB')
    inference_input = preprocess_input(np.array(image).astype(np.float32).transpose(2, 0, 1))
    return [np.expand_dims(inference_input, axis=0)]

img = Image.open('human/test.png')
# img = cv2.imread('test/human.jpg')
preprocessed_img = preprocess(img, (128, 256))

# %%
# preprocessed_img = cv2.resize(img, (224, 224)).transpose((2, 0, 1)).astype(np.float32)
# preprocessed_img = [np.expand_dims(preprocessed_img, axis=0)]
model = TritonModel(model_name, triton_client)
output = model(preprocessed_img)
output['AGE_16_30/Softmax']
output['GENDER_FEMALE/Softmax']

# %%
bs = 4
input_datas = [np.random.randint(256, size=(bs, 3, 224, 224)).astype(np.float32)]
output = model(input_datas)
 

groups = ['AGE', 'GENDER', 'APPAREL_LOWER', 'FACESHAPE', 'FACEFEATURES', 'PHYSIQUE', 'HERITAGE',\
        'HAIR_LENGTH', 'HAIR_COLOR', 'HAIR_TYPE','HAIR_STYLE','APPAREL_STYLE', 'APPAREL_UPPER_SLEEVELENGTH',\
        'APPAREL_UPPER_COLOR', 'APPAREL_LOWER_COLOR', 'APPAREL_LOWER_BOTTOMLENGTH','APPAREL_UPPER_TOPSTYLE', \
        'APPAREL_UPPER_COVERINGS', 'APPAREL_LOWER_BOTTOMSTYLE', 'FOOTWEAR', 'POSSESSIONS', 'ROLE']



 # %%

results = np.zeros((bs, 224))
# %%
result = output
bs = 1
grouped_attrs_list = [ defaultdict(list) for _ in range(bs)]
for output_name in model.output_names:
    res = result[output_name]
    attribute_name = output_name[:-8]
    attribute_name_split = attribute_name.split('_')
    group, attr = attribute_name_split[0].lower(), '_'.join(attribute_name_split[1:]).lower()
    res = np.squeeze(res)
    for i in range(bs):
        if res[i][0] < res[i][1]:
            grouped_attrs_list[i][group].append(attr)
# %%
from collections import defaultdict
grouped_attrs_list = [ defaultdict(list) for _ in range(bs)]

for idx, output_name in enumerate(model.output_names):
    res = output[output_name]
    attribute_name = output_name[:-8]
    attribute_name_split = attribute_name.split('_')
    group, attr = attribute_name_split[0].lower(), '_'.join(attribute_name_split[1:]).lower()
    # res = np.squeeze(res)
    for i in range(bs):
        if res[i][0] < res[i][1]:
            grouped_attrs_list[i][group].append(attr)
       
  
 

 
   


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

def yolo_preprocess(img, input_shape, letter_box=False):
    # img = img.astype(np.float32)
    # mean = [103.939, 116.779, 123.68]
    # img -= mean
    if letter_box:
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
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    # img /= 255.0
    return img



