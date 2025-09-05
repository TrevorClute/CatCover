import onnx
import onnxruntime
import cv2
import numpy as np


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


sess = onnxruntime.InferenceSession(
    "best.onnx", providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
outputs = sess.get_outputs()
inp_name = inputs[0].name
out_name = outputs[0].name

img = cv2.imread("first_frame.jpg")
lb = letter_box(img)
x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(
    2, 0, 1)[None].astype(np.float32) / 255

pred = sess.run([out_name], {inp_name: x})

pred = pred[0]              # drop batch dim → (N, 85)
obj_conf = pred[:, 4]       # (N,)
cls_conf = pred[:, 5:]      # (N, num_classes)
cls_ids  = cls_conf.argmax(1)        # best class per row
cls_score = cls_conf.max(1)          # confidence of that class
final_score = obj_conf * cls_score   # combine obj + class

print(final_score)
