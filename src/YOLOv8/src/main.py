from yolo_utils import yolov8
from vietocr_utils import vietocr
from define_angle import find_angle

from PIL import Image
import cv2



if __name__=="__main__":
    test_case = "../testcase.jpg"

    # load model
    detect_receipt = yolov8(model_path="../weights/detect_receipt.pt")
    detect_fields = yolov8(model_path="../weights/detect_fields.pt")
    # detect_fields = yolov8(model_path="/CS338/lynm/yolov8/runs/detect/train/weights/best.pt")
    vietocr_instance = vietocr(model_path="../weights/vietocr.pth")

    # PHASE 1: DETECT RECEIPT
    img = Image.open(test_case)
    img = img.crop(detect_receipt.inference(img, threshold=0.5)[0][0])


    # PHASE 2: ROTATE RECEIPT
    angle = find_angle(vietocr_instance, img.width, img.height, img)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=True)

    # PHASE 3: DETECT 4 FIELDS
    boxes, class_id = detect_fields.inference(img, threshold=0.1)

    # PHASE 4: USING VIETOCT TO RECOGNIZE
    output = {
        "0": "", 
        "1": "", 
        "2": "", 
        "3": "", 
    }
    for cls, box in zip(class_id, boxes):
        sub_img = img.crop(box)
        text, _ = vietocr_instance.inference(sub_img)
        output[f"{cls}"] += text
        
    print(output)

