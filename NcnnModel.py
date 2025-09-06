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
    model = YOLO("./best_ncnn_model/", task="detect")
    def __init__(self):
        pass
    def predict(self, img):
        predictions = []
        img = letter_box(img)
        results = self.model.predict(source=img, imgsz=640, conf=0.84, save=False)
        result = results[0]
        for box in result.boxes:
            conf = float(box.conf[0])
            id = int(box.cls[0])
            name = result.names[id]
            predictions.append({"name":name, "conf":conf})
        return predictions









