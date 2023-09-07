import numpy as np
import tensorflow as tf
import pathlib
import cv2
from paddleocr import PaddleOCR
from importlib import resources as impresources
import numpy.typing as npt
from typing import Any


def load_interpreter(model_path: str):
    return tf.lite.Interpreter(model_path=model_path)


INTERPRETER_PATH = str(impresources.files("car_plate_detection") / "model")
INTERPRETER = load_interpreter(f"{INTERPRETER_PATH}/yolov7_tiny_model_float16.tflite")
OCR = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)


def letterbox(
    im, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)


def crop_image(orig_image, x0, y0, x1, y1):
    return orig_image[y0:y1, x0:x1]


def detect_plate(image_mat, interpreter=INTERPRETER)->tuple[Any,float]:
    """Detect car plate in an image using TFLite model
    Return only the cropped car plate and score
    """

    image = image_mat.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    processed_img = image.astype(np.float32)
    processed_img /= 255

    # Allocate tensors.
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    interpreter.set_tensor(input_details[0]["index"], processed_img)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    outputs = interpreter.get_tensor(output_details[0]["index"])

    # ori_images = [image_mat.copy()]

    # crop the plate to return
    if len(outputs) == 0:
        return None
    batch_id, x0, y0, x1, y1, cls_id, score = outputs[0]

    # scale xs and ys
    box = np.array([x0, y0, x1, y1])
    box -= np.array(dwdh * 2)
    box /= ratio
    box = box.round().astype(np.int32).tolist()

    plate = crop_image(image_mat, *box)

    return plate,float(score)


def ocr_plate(plate, ocr)->tuple[str,float]:
    texts = ocr.ocr(plate, cls=False, det=True)

    max_height = 0
    tallest_text = ""
    score = 0
    for box, text in texts[0]:
        height = abs(box[0][1] - box[2][1])
        if height > max_height:
            tallest_text = text[0]
            score = text[1]

        max_height = height

    return tallest_text, float(score)


def detect_and_extract(image_mat, interpreter=INTERPRETER, ocr=OCR)->tuple[str,float,float]:
    plate,detection_score = detect_plate(image_mat, interpreter)
    plate_text,ocr_score = ocr_plate(plate, ocr)

    return plate_text,detection_score,ocr_score


# if __name__ == '__main__':
#     # load the interpreter
#     test_path = str(impresources.files('car_plate_detection')/'model')
#     interpreter = load_interpreter(f"{test_path}/yolov7_tiny_model_float16.tflite")

#     # load an image and run inference
#     image = cv2.imread(f"{test_path}/image2.jpg")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     plate = detect_and_extract(image,interpreter)
#     1
