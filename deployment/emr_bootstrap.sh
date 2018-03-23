#!/bin/bash


# enable debugging & set strict error trap
sudo yum -y install -v python36 python36-pip

sudo python3.6 -m pip install tensorflow-gpu
sudo python3.6 -m pip install pandas
sudo python3.6 -m pip install boto3
sudo python3.6 -m pip install scipy
sudo python3.6 -m pip install numpy

export PYTHONPATH='/home/hadoop/CVAE/'
