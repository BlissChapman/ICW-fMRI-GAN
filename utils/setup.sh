#!/bin/bash

# Install python
sudo apt-get install python3-pip python3-dev build-essential
sudo pip3 install --upgrade pip
sudo pip3 install --upgrade virtualenv

# Install CUDA
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-8-0 -y

# Enable persistence mode
sudo nvidia-smi -pm 1

# Verify CUDA install
sudo nvidia-smi

# CUDA optimizations for wussy gpu
sudo nvidia-smi -pm 1
sudo nvidia-smi -ac 2505,875
sudo nvidia-smi --auto-boost-default=DISABLED

# Install project dependencies
sudo pip3 install -r requirements.txt

# Install Torch
sudo pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl
sudo pip3 install torchvision

# Install Screen
sudo apt-get install screen

# Restart machine
sudo reboot
