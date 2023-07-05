import json
import logging

import numpy as np
from fastapi import APIRouter
from PIL import Image

from src.chatgpt.tools import ChatTooLs
from src.pipiline import Baseline
from src.utils.image_utils import base64_to_image, image_to_base64, plot_bboxes
from src.utils.models import *


router = APIRouter(
    prefix="/api/v1",
    tags=["Application"]
    )

def response_error():
    return ResponseModel(
        status_code=500, msg="Failed", data={}
    )

# rmbg (remove backround) 
@router.post('/rmbg')
async def rmbg(req: RMBGSchema) -> ResponseModel:
    try:
        img = base64_to_image(req.image_base64)        
        removed_img, cropped_img = Baseline.rmbg(np.array(img))

        cropped_base64 = image_to_base64(Image.fromarray(cropped_img).convert("RGB"))
        removed_base64 = image_to_base64(Image.fromarray(removed_img).convert("RGB"))

    except Exception as err:
        logging.error(err)
        return response_error()
    
    return ResponseModel(
        status_code=200, msg="Finish", data={"cropped_base64": cropped_base64, "removed_base64": removed_base64}
    )


# gpt_extraction 
@router.post('/gpt_extraction')
async def gpt_extraction(req: GptExtractionSchema) -> ResponseModel:

    try:
        output = ChatTooLs.get_completion(
            system_message=ChatTooLs().system_message_wordlevel,
            user_prompt=req.content
        )
        content = output[0]

    except Exception as err:
        logging.error(err)
        return response_error()
    
    return ResponseModel(
        status_code=200, msg="Finish", data={"content": content}
    )

# e2e
@router.post('/e2e')
async def e2e(req: E2ESchema) -> ResponseModel:
    # NOTE: dummy config
    debug = False

    store_dict = json.load(open("data/store.json"))
    method_map = {
        "yolo_rotate_yolo_vietocr": 0,
        "rmbg_rotate_yolo_vietocr": 1,
        "rmbg_rotate_craft_vietocr_pick": 2,
        "rmgb_rotate_craft_vietocr_gpt_lines": 3,
        "rmgb_rotate_craft_vietocr_gpt_boxes": 4,

    }

    method = method_map[req.method]

    try:
        img0 = base64_to_image(req.image_base64)
        removed_img, cropped_img = Baseline.rmbg(np.array(img0))

        # METHOD 0:
        if method==0:
            boxes = Baseline.receipt_detecter(img0, False)[0][0]
            cropped_img = img0.crop(boxes)
            angle = Baseline.find_angle(cropped_img)
            rotated_img = cropped_img.rotate(angle, resample=Image.BILINEAR, expand=True)
            rotated_img = np.array(rotated_img)
            cropped_img = np.array(cropped_img)
            bboxes, labels = Baseline.fields_detecter(Image.fromarray(rotated_img).convert("RGB"))
            ploted_img = plot_bboxes(rotated_img.copy(), bboxes.copy())
            texts, text_lines = Baseline.text_recognize(rotated_img, np.array(bboxes), False, debug)
            bboxes = bboxes.tolist() if not isinstance(bboxes, list) else bboxes
            
            result_dict = Baseline.post_process(labels, texts, bboxes, store_dict)
            extracted = {}
            for key, value in result_dict.items():
                extracted[key] = ' | '.join(value["value"])

        # METHOD 01
        if method==1:
            resized_img = Baseline.resize_img(cropped_img.copy())
            bboxes = Baseline.detect_bboxes(resized_img.copy())
            rotated_img, bboxes = Baseline.check_rotation(resized_img.copy(), bboxes)
            bboxes, labels = Baseline.fields_detecter(Image.fromarray(rotated_img).convert("RGB"))
            ploted_img = plot_bboxes(rotated_img.copy(), bboxes.copy())
            texts, text_lines = Baseline.text_recognize(rotated_img, np.array(bboxes), False, debug)
            bboxes = bboxes.tolist() if not isinstance(bboxes, list) else bboxes

            result_dict = Baseline.post_process(labels, texts, bboxes, store_dict)
            extracted = {}
            for key, value in result_dict.items():
                extracted[key] = ' | '.join(value["value"])

        # METHOD 2
        if method==2:
            resized_img = Baseline.resize_img(cropped_img.copy())
            bboxes = Baseline.detect_bboxes(resized_img.copy())
            rotated_img, bboxes = Baseline.check_rotation(resized_img.copy(), bboxes)
            ploted_img = plot_bboxes(rotated_img.copy(), bboxes.copy())
            texts, text_lines = Baseline.text_recognize(rotated_img, np.array(bboxes), debug)
            bboxes = bboxes.tolist() if not isinstance(bboxes, list) else bboxes

            annotations = []
            j = 1
            for boxes, text in zip(bboxes, texts):
                box_str = []
                for box in boxes:
                    box = list(map(int, box))
                    if text and len(text):
                        box = [str(b) for b in box]
                        box = ','.join(box)
                        box_str.append(box)

                annotations.append(
                    f"{j},{','.join(box_str)},{text}"
                )
            
            _img = np.array(Image.fromarray(rotated_img).convert("RGB"))
            result_dict = Baseline.pick_extraction(annotations, _img)
            
            result_dict = Baseline.post_prcess_result_dict(result_dict, store_dict)
            extracted = {}
            for key, value in result_dict.items():
                extracted[key] = ' | '.join(value["value"])
            
        # METHOD 3
        if method==3:
            resized_img = Baseline.resize_img(cropped_img.copy())
            bboxes = Baseline.detect_bboxes(resized_img.copy())
            rotated_img, bboxes = Baseline.check_rotation(resized_img.copy(), bboxes)
            ploted_img = plot_bboxes(rotated_img.copy(), bboxes.copy())
            texts, text_lines = Baseline.text_recognize(rotated_img, np.array(bboxes), debug)
            bboxes = bboxes.tolist() if not isinstance(bboxes, list) else bboxes
            _extracted = Baseline.gpt_extraction(text_lines, opt="line")
            try:
                _extracted = json.loads(_extracted)                
                extracted = {
                    "ADDRESS": _extracted.get("ADDRESS"), 
                    "SELLER": _extracted.get("SELLER"), 
                    "TIMESTAMP": _extracted.get("TIMES"), 
                    "TOTAL_COST": _extracted.get("COST")
                }

            except:
                extracted = {
                    "ADDRESS": -1, "SELLER": -1, "TIMESTAMP": -1, "TOTAL_COST": -1  
                }

        # METHOD 4
        if method==4:
            resized_img = Baseline.resize_img(cropped_img.copy())
            bboxes = Baseline.detect_bboxes(resized_img.copy())
            rotated_img, bboxes = Baseline.check_rotation(resized_img.copy(), bboxes)
            ploted_img = plot_bboxes(rotated_img.copy(), bboxes.copy())
            texts, text_lines = Baseline.text_recognize(rotated_img, np.array(bboxes), debug)
            bboxes = bboxes.tolist() if not isinstance(bboxes, list) else bboxes
            _extracted = Baseline.gpt_extraction(texts, opt="word")
            try:
                _extracted = json.loads(_extracted)                
                extracted = {
                    "ADDRESS": _extracted.get("ADDRESS"), 
                    "SELLER": _extracted.get("SELLER"), 
                    "TIMESTAMP": _extracted.get("TIMES"), 
                    "TOTAL_COST": _extracted.get("COST_NAME") + " | " + _extracted.get("COST_VALUE"),
                }
            except:
                extracted = {
                    "ADDRESS": -1, "SELLER": -1, "TIMESTAMP": -1, "TOTAL_COST": -1  
                }
    
        if debug:
            try:
                Image.fromarray(removed_img).convert("RGB").save("tmp/debug/removed_img.png")
                Image.fromarray(cropped_img).convert("RGB").save("tmp/debug/cropped_img.png")
                Image.fromarray(rotated_img).convert("RGB").save("tmp/debug/rotated_img.png")
                ploted_img.save("tmp/debug/ploted_img.png")
                with open('tmp/text.txt', 'w') as f:
                    for text in texts:
                        f.write(text+'\n')
            except:
                pass 

        outputs = dict(
            extracted = extracted,
            bboxes = bboxes,
            texts = texts,
            images = dict(
                removed_img = image_to_base64(Image.fromarray(removed_img)),
                cropped_img = image_to_base64(Image.fromarray(cropped_img)),
                rotated_img = image_to_base64(Image.fromarray(rotated_img)),
                ploted_img =  image_to_base64(ploted_img),
                used_img = image_to_base64(Image.fromarray(rotated_img))
            )
        )
    except Exception as err:
        logging.error(err)
        return response_error()

    return ResponseModel(
        status_code=200, msg="Finish", data=outputs
    )