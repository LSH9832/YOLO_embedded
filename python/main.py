from predict import Predicter


def openvino_run(source, multiprocess=False):
    from models.OpenVINO import Detector

    openvino_detector = Detector(xml_file="../files/xml/yolox_nano.xml", bin_file="../files/bin/yolox_nano.bin", device="CPU")

    openvino_detector.load_class_from_file("../files/txt/coco_classes.txt")
    openvino_detector.set_conf_thres(0.2)
    openvino_detector.set_nms_thres(0.45)

    pre = Predicter(openvino_detector, source)
    pre.run_multiple() if multiprocess else pre.run_single()


def tensorrt_run(source, multiprocess=False):
    from models.TensorRT import Detector

    tensorrt_detector = Detector(trt_file="../files/pth/yolox_tiny.pth", input_size=416)

    tensorrt_detector.load_class_from_file("../files/txt/coco_classes.txt")

    tensorrt_detector.set_conf_thres(0.2)
    tensorrt_detector.set_nms_thres(0.45)

    pre = Predicter(tensorrt_detector, source)
    pre.run_multiple() if multiprocess else pre.run_single()


if __name__ == '__main__':
    cam_source = "rtmp://192.168.1.108/ch1/main/av_stream" # 0, "rtsp://xxx", "/home/$USER/Videos/test.mp4"
    
    # openvino_run(cam_source, multiprocess=True)
    tensorrt_run(cam_source, multiprocess=True)
