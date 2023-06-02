# %%
import tritonclient.grpc as grpcclient
from triton_model import TritonModel
import numpy as np
import cv2
# %%

triton_client = grpcclient.InferenceServerClient('2.tcp.ngrok.io:18508')
model_config = triton_client.get_model_config("lprnet_tao").config
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
         'U', 'V', 'W', 'X', 'Y', 'Z', '', '']
# %%
model_name = "lprnet_tao"
img = cv2.imread("plate.png")
resized = cv2.resize(img, (96, 48))
image = np.transpose(resized, (2, 0, 1)).astype(np.float32)
image /= 255
# %%
model = TritonModel(model_name, triton_client)
# input_datas = [np.zeros((1, 3, 48, 96)).astype(np.float32)]
input_datas = [np.expand_dims(image, axis=0)]

# %%
output = model(input_datas)
pred = output['tf_op_layer_ArgMax']

decode(pred, CHARS)




# %%

# with open('us_lp_characters.txt', 'r') as file:
#     # Read the contents of the file and store it in a list
#     lines = file.read().splitlines()

# CHARS = lines  + ['']

# %%

def decode(preds, CHARS):
    # greedy decode
    pred_labels = list()
    labels = list()
    for i in range(preds.shape[0]):
        pred = preds[i, :]
        pred_label = pred
        no_repeat_blank_label = list()
        pre_c = pred_label[0]
        for c in pred_label: # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        pred_labels.append(no_repeat_blank_label)
        
    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)
    
    return labels



def inference(file_path, model=model):
    img = cv2.imread(file_path)
    resized = cv2.resize(img, (96, 48))
    image = np.transpose(resized, (2, 0, 1)).astype(np.float32)
    image /= 255
    input_datas = [np.expand_dims(image, axis=0)]
    output = model(input_datas)
    pred = output['tf_op_layer_ArgMax']
    labels = decode(pred, CHARS)
    print(labels)




# %%
