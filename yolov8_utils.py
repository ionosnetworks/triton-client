import cv2
import numpy as np
from ultralytics.yolo.utils import ops
import torch
from boundingbox import BoundingBox

# def prepare_input(image, input_shape, stride, pt):
#     input_tensor = LetterBox(input_shape, auto=pt, stride=stride)(image=image)
#     input_tensor = input_tensor.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#     input_tensor = np.ascontiguousarray(input_tensor).astype(np.float32)  # contiguous
#     input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
#     input_tensor = input_tensor[None].astype(np.float32)
#     return input_tensor


def process_output(detections, 
                   ori_shape, 
                   input_shape, 
                   conf_threshold, 
                   iou_threshold,
                   classes=None,
                   agnostic=False,
                   max_det=300,
                   batched=False,
                   ):
    detections = torch.from_numpy(detections.copy()).to("cpu")
    detections = ops.non_max_suppression(detections,
                                          conf_thres=conf_threshold,
                                          iou_thres=iou_threshold,
                                          classes=classes,
                                          agnostic=agnostic,
                                          max_det=max_det,
                                          )

    for i in range(len(detections)): 
        # Extract boxes from predictions
        detections[i][:, :4] = ops.scale_boxes(input_shape, detections[i][:, :4], ori_shape).round()

    
    return [detection.cpu().numpy()  for detection in detections] if batched else detections[0].cpu().numpy()


# def rescale_boxes(boxes, ori_shape, input_shape):

#     input_height, input_width = input_shape
#     img_height, img_width = ori_shape
#     # Rescale boxes to original image dimensions
#     input_shape = np.array(
#         [input_width, input_height, input_width, input_height])
#     boxes = np.divide(boxes, input_shape, dtype=np.float32)
#     boxes *= np.array([img_width, img_height, img_width, img_height])
#     return boxes


def postprocess(detections, image_shape):
    img_w, img_h = image_shape[1], image_shape[0]
    det_boxes = detections[:,:4]
    det_scores = detections[:,4]
    det_classes = detections[:,5]
    boxes = det_boxes.astype(int)
    scores = det_scores
    classes = det_classes.astype(int)

    detected_objects = []
    for box, score, label in zip(boxes, scores, classes):
        detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], img_w, img_h))
    return detected_objects

