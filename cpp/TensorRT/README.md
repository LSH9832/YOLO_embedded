# yolo TensorRT

主要给Nvidia Jetson系列嵌入式设备使用的，程序使用tensorrt推理模型，并将结果发送到服务器，接收demo见DataServer.py。
网络输入输出适配于yolox。

```
input [1, 3, input_size, input_size] RGB image
                  ↓
output [m, 5 + n] matrix → [m: total num of grid cell, 5 + n: cx, cy, w, h, conf_obj, conf_class_1, ... , conf_class_n]
```

环境配置为：

- CUDA10.2
- CUDnn8.0.0
- TensorRT7.1.3 (安装这个版本的TensorRT，在使用本项目已经给出的4个时保证不会报错)
- OpenCV
- libcurl

## 安装
```
git clone https://github.com/LSH9832/yoloTRT.git
cd yolo_result_provider && sh setup.sh
```

## 使用
```
./detect -e          /path/to/your/tensorrt_engine_file, 在model文件夾中已經提供了四個engine文件
         -no-push    stop pushing rtmp stream，列举此选项将不会进行推流
         -post       post json result(default false) 列举此选项将向服务器发送json数据，具体见DataServer.py
         -show       show image(default false), 列举此选项将在本地端显示图像
         -repeat     repeat playing video(default false)，如果是图像源是本地视频，列举此项将重复播放
         -v          /path/to/your/video_file or rtsp/rtmp stream 图像源，可以是本地视频，也可以是rtsp/rtmp视频流
         -conf       confidence threshold between 0-1(default 0.25)，置信度阈值
         -nms        NMS threshold between 0-1(default 0.45) NMS阈值
         -size       input size of images, wrong size will cause error(default 640, 416 if tiny-model or nano-model)
         -ip         rtmp server ip(default 127.0.0.1)
         -port       rtmp server port(default 1935)
         -post-port  request server port(default 80)
         -fps        stream rate(default 30)
         -b          bitrate(default 4000000)
         -name       rtmp name(default live/test)  example: rtmp://ip:port/live/test
         -post-name  post json result name(default detect_result) example: http://ip:postport/post-name?data
         -clsfile    class file name(default classes.txt), 类别文件，见文件夾classes中的两个文件

# for more details
./detect -h
```
### 如果觉得每次在命令行中输入这么多参数太麻烦，可以运行run.py文件代替detect,只需要在run.py文件中将相应的参数进行修改即可。
