# Triton Inference Server Instructions

Make sure docker container is started by running the following command

```
docker run --gpus all --rm --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v ~/triton/models:/models nvcr.io/nvidia/tritonserver:22.06-py3 tritonserver --model-repository=/models --strict-model-config=false
```

And install tritonclient[all] opencv-python packages

```bash
pip3 install tritonclient[all] opencv-python
pip3 install "numpy<1.24.0"
```

## How to run model in your code

Example client can be found in client.py. It can run dummy input, images and videos.

```bash
python3 client.py image test.jpg -u URL_FROM_NGROK
```

```
$ python3 client.py --help
usage: client.py [-h] [-m MODEL] [--width WIDTH] [--height HEIGHT] [-u URL] [-o OUT] [-f FPS] [-i] [-v] [-t CLIENT_TIMEOUT] [-s] [-r ROOT_CERTIFICATES] [-p PRIVATE_KEY] [-x CERTIFICATE_CHAIN] {dummy,image,video} [input]

positional arguments:
  {dummy,image,video}   Run mode. 'dummy' will send an emtpy buffer to the server to test if inference works. 'image' will process an image. 'video' will process a video.
  input                 Input file to load from in image or video mode

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Inference model name, default yolov7
  --width WIDTH         Inference model input width, default 640
  --height HEIGHT       Inference model input height, default 640
  -u URL, --url URL     Inference server URL, default localhost:8001
  -o OUT, --out OUT     Write output into file instead of displaying it
  -f FPS, --fps FPS     Video output fps, default 24.0 FPS
  -i, --model-info      Print model status, configuration and statistics
  -v, --verbose         Enable verbose client output
  -t CLIENT_TIMEOUT, --client-timeout CLIENT_TIMEOUT
                        Client timeout in seconds, default no timeout
  -s, --ssl             Enable SSL encrypted channel to the server
  -r ROOT_CERTIFICATES, --root-certificates ROOT_CERTIFICATES
                        File holding PEM-encoded root certificates, default none
  -p PRIVATE_KEY, --private-key PRIVATE_KEY
                        File holding PEM-encoded private key, default is none
  -x CERTIFICATE_CHAIN, --certificate-chain CERTIFICATE_CHAIN
                        File holding PEM-encoded certicate chain default is none
```
