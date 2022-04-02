import torch

from .detect import BaseDetector


class Detector(BaseDetector):

    def __init__(self, trt_file, input_size=640):
        super().__init__()
        self.__trt_file = trt_file
        self.__input_size = (input_size, input_size)
    
    def load(self):
        from torch2trt import TRTModule
        self.__model = TRTModule()
        self.__model.load_state_dict(torch.load(self.__trt_file))
    
    def get_input_size(self):
        return self.__input_size

    def encode(self, img):
        img = self.preprocess(img, self.__input_size)
        return img

    def inference(self, img):
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.cuda()
        result = self.__model(img)
        return result.cpu().numpy()

    def decode(self, result):
        return self.postprocess(result, self.__input_size, p6=False)

    def predict(self, img):
        return self.decode(self.inference(self.encode(img)))
