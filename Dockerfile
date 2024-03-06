FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel AS gpu

WORKDIR /workspace

# Necessary linux packages
RUN apt update
RUN apt install build-essential -y
RUN DEBIAN_FRONTEND=noninteractive apt install tzdata -y
RUN apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 unzip wget -y

# Install dependencies from pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
