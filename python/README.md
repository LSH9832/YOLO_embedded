# 基於python的yolo算法嵌入式部署
## 極簡代碼，結構清晰，修改方便，快速部署


## Preparing

```
pip3 install -r requirements.txt
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python3 setup.py install
cd ..
```

## demo

first modify main.py with your model, source, and other settings

```
python3 main.py
```
view predict.py and directory models for more usage
