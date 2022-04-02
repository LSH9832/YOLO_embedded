import cv2


def show(my_dict, detector, source):
    import cv2
    from time import time

    is_webcam = source.startswith("rtmp://") or source.startswith("rtsp://")

    cam = cv2.VideoCapture(source)
    fpss = []
    result = []
    frames = []


    print('wait for model-loading')
    while not my_dict['run']:
        cam.grab() if is_webcam else None

    detector.fps()
    while cam.isOpened():
        t1 = time()
        success, frame = cam.read()
        if success:

            frames.append(frame)

            my_dict['img'] = detector.preprocess(frame, detector.get_input_size())
            my_dict['updated'] = True

            fpss.append(detector.fps())
            fpss = fpss[1:] if len(fpss) > 10 else fpss
            now_mean_fps = sum(fpss) / len(fpss)
            print('\rplay_fps=%.2f, inference_fps=%.2f' % (now_mean_fps, my_dict['pre_fps']), end='')

            while len(frames) > 1:
                frame = frames.pop(0)

            if my_dict['update_result']:
                result = my_dict['result']
                my_dict['update_result'] = False
                if result is not None and len(result):
                    result = detector.postprocess(result, detector.get_input_size())
            if result is not None and len(result):
                detector.plot_result(frame, result)

            cv2.imshow('test', frame)

            if cv2.waitKey(1) == 27:
                cam.release()
                break
        else:
            try:
                cam.release()
            except:
                pass
            if is_webcam:
                cam.open(source)
            else:
                break

        # while not time() - t1 >= 0.03:
        #     pass

    print('')
    my_dict['run'] = False
    cv2.destroyAllWindows()


def detect(my_dict, detector):
    detector.load()
    my_dict['run'] = True
    while my_dict['run']:                               # 如果程序仍需要运行
        if my_dict['updated']:                          # 如果图像已经更新
            img = my_dict['img']                        # 获取图像
            my_dict['updated'] = False                  # 设置图像状态为未更新
            detector.fps()                              # 开始计时
            result = detector.inference(img)            # 推理
            my_dict['pre_fps'] = detector.fps()         # 结束计时并计算FPS
            my_dict['result'] = result                  # 存储结果
            my_dict['update_result'] = True             # 设置结果状态为已更新


class Predicter:

    def __init__(self, detector, source):
        self.__source = source
        self.__detector = detector

    def run_single(self):
        self.__detector.load()
        cam = cv2.VideoCapture(self.__source)
        self.__detector.fps()
        while cam.isOpened():
            success, image = cam.read()
            if success:

                # get and plot result
                result = self.__detector.predict(image)
                if result is not None and len(result):
                    self.__detector.plot_result(image, result)

                # show
                cv2.imshow("result", image)
                if cv2.waitKey(1) == 27:
                    cv2.destroyAllWindows()
                    cam.release()
                    break
            print("\r%.2f fps" % self.__detector.fps(), end="")
        print("")

    def run_multiple(self):
        from multiprocessing import Process, Manager, freeze_support
        freeze_support()
        d = Manager().dict()

        d['run'] = False
        d['pre_fps'] = 0

        d['img'] = None
        d['updated'] = False
        d['result'] = []
        d['update_result'] = False

        processes = [Process(target=show, args=(d, self.__detector, self.__source)),
                     Process(target=detect, args=(d, self.__detector))]
        [process.start() for process in processes]
        [process.join() for process in processes]
