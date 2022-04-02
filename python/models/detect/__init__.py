from .visual import draw_bb
from .nms import multiclass_nms

from time import time
import numpy as np
import cv2


class BaseDetector(object):

    _class_names = []
    __last_time = time()
    __nms_thres = 0.45
    __conf_thres = 0.1
    r=1


    def __init__(self, name_list=None):
        self._class_names = name_list

    def set_conf_thres(self, conf_thres):
        self.__conf_thres = conf_thres

    def set_nms_thres(self, nms_thres):
        self.__nms_thres = nms_thres

    def load_class_from_file(self, file_name="./coco_classes.txt"):
        import os
        assert os.path.exists(file_name), "file %s not exist!" % file_name
        classes = open(file_name).read().split('\n')
        if not len(classes[-1]):
            classes.pop(-1)
        self._class_names = classes

    @staticmethod
    def _xywh2xyxy(boxes, center=True):
        """Get coordinates (x0, y0, x1, y0) from model output (x, y, w, h)"""
        all_boxes = []
        for x, y, w, h in boxes:
            if center:
                x0, y0 = (x - 0.5 * w), (y - 0.5 * h)
                x1, y1 = (x + 0.5 * w), (y + 0.5 * h)
            else:
                x0, y0 = x, y
                x1, y1 = x + w, y + h
            all_boxes.append([x0, y0, x1, y1])
        return all_boxes

    @staticmethod
    def _xyxy2xywh(boxes, center=True):
        boxes = np.array(boxes)
        x, y, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = x2 - x
        h = y2 - y
        if center:
            x = (0.5 * (x + x2)).astype(int)
            y = (0.5 * (x + x2)).astype(int)

        boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] = x, y, w, h
        return boxes

    def fps(self):
        now_time = time()
        this_fps = 1./(now_time - self.__last_time)
        self.__last_time = now_time
        return this_fps

    def plot_result(self, img, result):
        draw_bb(img=img, pred=result, names=self._class_names)

    def nms(self, boxes, confs):
        return multiclass_nms(boxes=boxes, scores=confs, score_thr=self.__conf_thres, nms_thr=self.__nms_thres)

    def preprocess(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        self.r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * self.r), int(img.shape[0] * self.r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * self.r), : int(img.shape[1] * self.r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

    def postprocess(self, outputs, img_size, p6=False):

        grids = []
        expanded_strides = []

        if not p6:
            strides = [8, 16, 32]
        else:
            strides = [8, 16, 32, 64]

        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        predictions = outputs[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= self.r
        result = self.nms(boxes_xyxy, scores)
        return result
