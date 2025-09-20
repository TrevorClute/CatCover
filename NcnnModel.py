import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_SIZE = 640


def letter_box(img: np.uint8):
    h0, w0 = img.shape[:2]
    r = min(IMAGE_SIZE/h0, IMAGE_SIZE/w0)
    new_unpad = (int(round(w0*r)), int(round(h0*r)))
    im_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = IMAGE_SIZE - new_unpad[0], IMAGE_SIZE - new_unpad[1]
    top, bottom = dh//2, dh - dh//2
    left, right = dw//2, dw - dw//2
    im_padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return im_padded





class NcnnNodel:
    model = YOLO("./models/v3-cls/best_ncnn_model/", task="classify")
    def __init__(self):
        pass
    def predict(self, img):
        predictions = []
        img = letter_box(img)
        results = self.model.predict(source=img, imgsz=640, save=False) 
        result = results[0]
        names = result.names
        top5 = result.probs.top5
        top5conf = result.probs.top5conf
        names_and_conf = []
        for i in range(len(top5)):
            name = names[top5[i]]
            conf = top5conf[i].item()
            names_and_conf.append({"name":name, "conf":conf})
        return names_and_conf


# def main():
#     model = NcnnNodel()
#     image = cv2.imread("imgs/{'name': 'marbles', 'conf': 0.9162629842758179} -- 2025-09-16 21:01:37.jpg")
#     model.predict(image)
# main()
