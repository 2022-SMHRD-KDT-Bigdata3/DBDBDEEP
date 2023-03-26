# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Builds ultralytics/yolov5:latest image on DockerHub https://hub.docker.com/r/ultralytics/yolov5
# Image is CUDA-optimized for YOLOv5 single/multi-GPU training and inference

# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# FROM docker.io/pytorch/pytorch:latest
FROM pytorch/pytorch:latest

# Downloads to user config dir
# ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
        && apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev libsndfile1-dev -y \
        && pip3 install pyaudio
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

# 파이썬 패키지 설치
COPY requirements.txt .
# mkl-fft 패키지 제거
RUN sed -i '/mkl-fft==1.3.1/d' requirements.txt
RUN sed -i '/mkl-random==1.2.2/d' requirements.txt
RUN sed -i '/mkl-service==2.4.0/d' requirements.txt
# RUN alias python=python3

# Security updates
# https://security.snyk.io/vuln/SNYK-UBUNTU1804-OPENSSL-3314796
RUN apt upgrade --no-install-recommends -y openssl

# Create working directory
RUN rm -rf /usr/src/app && mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
# COPY . /usr/src/app  (issues as not a .git directory)
COPY . .
# Install pip packages
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -r requirements.txt albumentations comet gsutil notebook \
    coremltools onnx onnx-simplifier onnxruntime 'openvino-dev>=2022.3'
    # tensorflow tensorflowjs \

# Set environment variables
ENV OMP_NUM_THREADS=1

# Cleanup
ENV DEBIAN_FRONTEND teletype


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -f utils/docker/Dockerfile -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# DockerHub tag update
# t=ultralytics/yolov5:latest tnew=ultralytics/yolov5:v6.2 && sudo docker pull $t && sudo docker tag $t $tnew && sudo docker push $tnew

# Clean up
# sudo docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/ultralytics/yolov5:latest

# Start the Flask app
CMD ["python3", "detect2.py"]
# 베이스 이미지 설정
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /yolov5

# 필요한 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update \
        && apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev libsndfile1-dev -y \
        && pip3 install pyaudio
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

# 파이썬 패키지 설치
COPY requirements.txt .
# mkl-fft 패키지 제거
RUN sed -i '/mkl-fft==1.3.1/d' requirements.txt
RUN sed -i '/mkl-random==1.2.2/d' requirements.txt
RUN sed -i '/mkl-service==2.4.0/d' requirements.txt
RUN pip install --upgrade pip
RUN pip install pygobject
RUN pip install --no-cache-dir --verbose -r requirements.txt

# 소스 코드 복사
COPY . .

# Flask 애플리케이션 실행
CMD ["python", "detect2.py"]
