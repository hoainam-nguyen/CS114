from PIL import Image

def find_angle(vietocr_instance, width, height, crop_img): 
    # bbox co dang: x_center, y_center, w, h
    if height > width: 
        # prob_0 = vietocr_instance.inference(crop_img)
        # prob_180 = vietocr_instance.inference(crop_img.rotate(180, resample=Image.BILINEAR, expand=True))
        return 0 #if prob_0 >= prob_180 else 180
    else: 
        _, prob_90 = vietocr_instance.inference(crop_img.rotate(90, resample=Image.BILINEAR, expand=True))
        _, prob_270 = vietocr_instance.inference(crop_img.rotate(270, resample=Image.BILINEAR, expand=True))
        return 90 if prob_90 >= prob_270 else 270