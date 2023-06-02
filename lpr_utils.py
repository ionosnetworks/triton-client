import cv2
import numpy as np

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
         'U', 'V', 'W', 'X', 'Y', 'Z', '', '']

def decode(preds, CHARS=CHARS):
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


def lpr_single_inference(image, model):
    resized = cv2.resize(image, (96, 48))
    image = np.transpose(resized, (2, 0, 1)).astype(np.float32)
    image /= 255
    input_datas = [np.expand_dims(image, axis=0)]
    output = model(input_datas)
    pred = output['tf_op_layer_ArgMax']
    label = decode(pred, CHARS)
    return label[0]

def lpr_batch_inference(images, model, batch_size=16):
    image_buffer = []
    recognition_res = []

    for image in images:
        resized = cv2.resize(image, (96, 48))
        image = np.transpose(resized, (2, 0, 1)).astype(np.float32)
        image /= 255
        image_buffer.append(image)

    i = 0 
    while i < len(image_buffer):
        input_datas = [np.stack(image_buffer[i:i+batch_size])]
        output = model(input_datas)
        pred = output['tf_op_layer_ArgMax']
        label = decode(pred, CHARS)
        recognition_res.extend(label)
        i += batch_size

    return recognition_res



def lpr_inference(detected_objects_list, frames, lpr_model):
    crop_queue = []
    locations = []

    for idx in range(len(frames)):
        frame = frames[idx]
        detected_objects = detected_objects_list[idx]
        for idy, box in enumerate(detected_objects):
            if box.classID == 2:
                x1, y1, x2, y2 = box.box()
                crop = frame[y1:y2, x1:x2, :]
                crop_queue.append(crop)
                locations.append((idx, idy))
    
    if not crop_queue: return
    
    recognition_res = lpr_batch_inference(crop_queue, lpr_model)

    for loc, label in zip(locations, recognition_res):
        idx, idy = loc
        detected_objects_list[idx][idy].plate = label
    
   




