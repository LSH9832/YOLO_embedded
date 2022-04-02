# 基于python的yolo算法嵌入式部署
## 极简代码，结构清晰，修改方便，快速部署


## 环境安装

先在相应的设备上安装好TensorRT或者是OpenVINO的环境

```
pip3 install -r requirements.txt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python3 setup.py install
cd ..
```

## 运行demo

先打开main.py修改一下相应的参数，比如视频源，置信度阈值、MNS阈值、是否使用多进程进行推理任务等

```
python3 main.py
```

检测器结构请见文件夹models中的文件，具体使用方法请见predict.py
