# %%
import tritonclient.grpc as grpcclient
import numpy as np
import cv2
# %%
client = grpcclient.InferenceServerClient(url="2.tcp.ngrok.io:18508")
# %%
cap = cv2.VideoCapture("test.mp4")
def frame_generator(cap, batch_size=1):
    while True:
        frames = []
    
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        if frames:
            yield frames
        else:
            break


# %%
for frames in frame_generator(cap, batch_size=8):
 
    image_data = np.stack(frames, axis=0)
    break


input_tensors = [grpcclient.InferInput("detection_preprocessing_input", image_data.shape, "UINT8")]
input_tensors[0].set_data_from_numpy(image_data)
results = client.infer(model_name="detection_preprocessing", inputs=input_tensors)
output_data = results.as_numpy("detection_preprocessing_output") 
# %%
