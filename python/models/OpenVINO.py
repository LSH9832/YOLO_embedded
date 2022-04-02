from openvino.inference_engine import IECore

from .detect import BaseDetector


class Detector(BaseDetector):
    __ie = IECore()
    xml_file = "./yolox_nano.xml"
    bin_file = "./yolox_nano.bin"
    __exec_net = None

    def __init__(self, xml_file, bin_file, device="CPU"):
        super().__init__()
        self.xml_file = xml_file
        self.bin_file = bin_file
        self.__net = self.__ie.read_network(model=self.xml_file, weights=self.bin_file)

        if len(self.__net.input_info) != 1:
            print('Sample supports only single input topologies')
            exit(-1)
        if len(self.__net.outputs) != 1:
            print('Sample supports only single output topologies')
            exit(-1)

        self.__input_blob = next(iter(self.__net.input_info))
        self.__out_blob = next(iter(self.__net.outputs))

        self.__net.input_info[self.__input_blob].precision = 'FP32'
        self.__net.outputs[self.__out_blob].precision = 'FP16'

        self.__num_of_classes = max(self.__net.outputs[self.__out_blob].shape)
        self.__device = device
        _, _, self.__h, self.__w = self.__net.input_info[self.__input_blob].input_data.shape
    
    def load(self):
        self.__exec_net = self.__ie.load_network(network=self.__net, device_name=self.__device)
    
    def get_input_size(self):
        return (self.__h, self.__w)
    
    def encode(self, img):
        return self.preprocess(img, (self.__h, self.__w))

    def inference(self, img):
        return self.__exec_net.infer(inputs={self.__input_blob: img})[self.__out_blob]

    def decode(self, result):
        return self.postprocess(result, (self.__h, self.__w), p6=False)

    def predict(self, img):
        return self.decode(self.inference(self.encode(img)))
