import cv2


class Colors:
    def __init__(self):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness

    c1, c2 = (int(x[0]), int((x[1]))), (int(x[2]), int((x[3])))
    # print(c1,c2)
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], (c1[1] - t_size[1] - 3) if (c1[1] - t_size[1] - 3) > 0 else (c1[1] + t_size[1] + 3)
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2) if (c1[1] - t_size[1] - 3) > 0 else (c1[0], c2[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def draw_bb(img, pred, names, type_limit=None, line_thickness=2):
    if type_limit is None:
        type_limit = names
    if pred is not None:
        if len(pred[0]) == 6:
            for *xyxy, conf, cls in pred:
                if names[int(cls)] in type_limit:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img, label=label, color=colors(int(cls), True), line_thickness=line_thickness)
        elif len(pred[0]) == 7:
            for *xyxy, conf0, conf1, cls in pred:
                conf = conf0 * conf1
                if names[int(cls)] in type_limit:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img, label=label, color=colors(int(cls), True), line_thickness=line_thickness)
