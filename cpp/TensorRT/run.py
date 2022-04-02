import os


ip = "127.0.0.1"                            # 推流的服务器地址
port = 1935                                 # 推流的服务器端口
post_port = 12345                           # 服务器接收json端口
name = "live/test"                          # 推流的名称
post_name = "detect_result"                 # 发送json数据名称（http://ip:port/post_name?data）
fps = 30                                    # 推流最大帧率

# 视频流地址（本地文件，RTSP/RTMP视频流）
source = "~/Videos/test.mp4"

engine_file = "../../files/engine/coco80s.engine"         # TensorRT模型文件
class_file = "../../files/txt/coco_classes.txt"           # 类别名称文件，一行一个名称
size = 640                                  # s、m、l用640, tiny、nano用416
conf = 0.20                                 # 置信度阈值
nms = 0.4                                   # NMS阈值
bitrate = 6000000                           # 比特率越高，圖像越清晰，帶寬要求越高

# ultra params
repeat = True                               # 如果推流的是本地视频，可以选择视频播放完毕后是否重复
show = True                                 # 是否在本地显示
push = False                                # 是否推流
post = False                                # 是否发送json数据


###########################################################################################
"""        do not edit the following code unless you know what you are doing            """
###########################################################################################

this_dir = os.path.dirname(os.path.abspath(__file__))
ultra_command = ""
if repeat:
    ultra_command += "-repeat "
if show:
    ultra_command += "-show "
if not push:
    ultra_command += "-no-push "
if post:
    ultra_command += "-post "

file = "./build/detect"
file = os.path.join(this_dir, file)

command = "%s " \
          "-e %s " \
          "-clsfile %s " \
          "-size %d " \
          "-v %s " \
          "-conf %.2f " \
          "-nms %.2f " \
          "-ip %s " \
          "-port %d " \
          "-name %s " \
          "-fps %d " \
          "-post-port %d " \
          "-post-name %s " \
          "-b %d " \
          "%s" % (
              file,
              engine_file,
              class_file,
              size,
              source,
              conf,
              nms,
              ip,
              port,
              name,
              fps,
              post_port,
              post_name,
              bitrate,
              ultra_command
          )

print(command)
os.system(command)
