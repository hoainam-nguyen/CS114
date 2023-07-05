import numpy as np
import cv2 
import io 
import codecs
from PIL import Image
import base64

def base64_to_image(byte64_str: str, grayscale: bool = False) -> Image:
    bytearr = codecs.decode(codecs.encode(byte64_str, encoding="ascii"), encoding="base64")

    if grayscale:
        return Image.open(io.BytesIO(bytearr)).convert("L")

    return Image.open(io.BytesIO(bytearr)).convert("RGB")

def image_to_base64(image: Image) -> str:
	image = image.convert("RGB")
	image_byte_array = io.BytesIO()
	image.save(image_byte_array, format='JPEG')
	image_base64 = base64.b64encode(image_byte_array.getvalue()).decode('utf-8')
	return image_base64


def crop_background(image, grayscale=False):
	"""Crop black background only"""
	if not type(image).__module__ == np.__name__:
		img_arr = io.imread(image, True)
	else:
		img_arr = image
	gray = img_arr[:,:,0] if img_arr.ndim > 2 else img_arr
	_, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
	x, y, w, h = cv2.boundingRect(thresholded)
	output = (gray if grayscale else image)[y:y+h, x:x+w]
	return output

def plot_bboxes(img0, bboxes):
	"""Save image"""
	image = np.array(img0)
	for box in bboxes:
		poly = np.array(box).astype(np.int32).reshape((-1))
		poly = poly.reshape(-1, 2)
		image = np.ascontiguousarray(img0, dtype=np.uint8)
		cv2.polylines(image, [poly.reshape((-1, 1, 2))],
					True, color=(0, 0, 255), thickness=2)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	
	return Image.fromarray(image).convert("RGB")
