# YOLO_embedded： YOLO算法快速嵌入式部署
- for Ubuntu
- OpenVINO & TensorRT

本项目提供c++和python两种语言，详情请见各个文件夹下的README.md

## 安装OpenVINO

- [点此](https://www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download-previous-versions.html?operatingsystem=linux&distributions=webdownload)进入官网选择版本进行下载，然后打开install_openvino.sh将相应的文件和文件夹名称改为对应版本名称
- 选择一个合适的路径，将下载的压缩包和install_openvino.sh放在该路径下
- 运行install_openvino.sh

## 安装TensorRT

- [点此](https://developer.nvidia.com/nvidia-tensorrt-download)进入官网，注册登录后选择合适的版本进行下载，由于不同平台安装方式不同，在此不加赘述。Jetson系列的嵌入式设备可以直接使用其刷机软件安装tensorrt
