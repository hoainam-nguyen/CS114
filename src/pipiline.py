from typing import List

import cv2
import json
import numpy as np
from PIL import Image
from rembg import remove

from src.const import RESOURCE_MAP
from src.CRAFT import net
from src.utils.image_utils import (base64_to_image, crop_background,
                                   image_to_base64, plot_bboxes)
from src.utils.rotation import align_box, rotate_90
from src.chatgpt.tools import ChatTooLs
from src.utils.speller import fix_result_by_dictionary, fix_result_by_rule_based


class Baseline():
    @staticmethod 
    def find_angle(img): 
        if img.height > img.width: 
            return 0 
        else: 
            _, prob_90 = RESOURCE_MAP["ocr_model"].predict(img.rotate(90, resample=Image.BILINEAR, expand=True), return_prob=True)
            _, prob_270 = RESOURCE_MAP["ocr_model"].predict(img.rotate(270, resample=Image.BILINEAR, expand=True), return_prob=True)
            return 90 if prob_90 >= prob_270 else 270

    @staticmethod
    def resize_img(img: np.ndarray) -> np.ndarray:
        height = 1920
        width = int(img.shape[1]*(height/img.shape[0]))
        return cv2.resize(img, (width, height))  

    @staticmethod
    def check_rotation(img0: np.ndarray, bboxes: List[List[float]]):
        is_rotated, img1, _ = rotate_90(img0, bboxes)
        bboxes = Baseline.detect_bboxes(img1)
        img2, is_aligned = align_box(img1, bboxes, skew_threshold=1)  # align image with threshold = 1 degree
        img3, is_flip = Baseline.flip_classify(img2)
        bboxes = Baseline.detect_bboxes(img3) \
            if (is_aligned or is_flip) else bboxes

        return img3, bboxes 
    
    
    @staticmethod
    def text_recognize(img0: np.ndarray, bboxes: List[List[float]], use_sort: bool=True, debug: bool=False):
        # img0 = np.array(Image.fromarray(img0).convert("RGB"))
        def _xyxy(box):
            return [
                min(box[:,0]),
                min(box[:,1]),               
                max(box[:,0]),
                max(box[:,1]),               
            ]
        
        texts = []
        incline = {'prev_height': 0, 'prev_line': -1}
        use_incline = True 
        infos = []
        if use_sort:
            bboxes = sorted(bboxes, key=lambda x: x[1][1])
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = list(map(int, _xyxy(box)))
            cropped = img0.copy()[y1:y2, x1:x2]
            
            if debug:
                _img = Image.fromarray(cropped).convert("RGB")
                _img.save(f"tmp/debug/cropped_im{i}.png")
            # text recognize
            try:
                text = RESOURCE_MAP['ocr_model'].predict(
                    Image.fromarray(cropped).convert("RGB")
                )
            except Exception as err:
                text = ""

            if use_incline:  # try to make text output keep it line
                current_height = sum(box[:,1])/4
                per_diff = abs(1-incline['prev_height']/current_height)
                if per_diff < 0.02:  # different to be same line
                    infos[incline['prev_line']].append(text)
                else:
                    infos.append([text])
                    incline['prev_line'] += 1
                incline['prev_height'] = current_height

            texts.append(text)

        return texts, infos

    @staticmethod
    def flip_classify(img0: np.ndarray):
        img = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (128,128))
        img = np.array(img).reshape(128*128)
        predicted = RESOURCE_MAP["flip_classifier"].predict([img])
        if predicted == 0:
            return cv2.rotate(img0, cv2.ROTATE_180), True
        return img0, False

    @staticmethod
    def rmbg(img: np.ndarray) -> List[np.ndarray]:
        '''Remove and crop background'''
        removed_img = remove(img)
        cropped_img = crop_background(removed_img)
        return removed_img, cropped_img

    @staticmethod
    def detect_bboxes(img: np.ndarray) -> List[List[float]]:
        # bboxes = RESOURCE_MAP["craft_detect_model"].predict(img)

        if img.shape[0] == 2: img = img[0]
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4: img = img[:, :, :3]
        img = np.array(img)

        bboxes, _, _ = net.test_net(
            RESOURCE_MAP["craft_detect_model"], 
            img, 
            0.7, 0.4, 0.4, 
            "cuda:0", 
            False, 
            RESOURCE_MAP["refinenet_model"]
        )

        return bboxes

    @staticmethod
    def gpt_extraction(texts: List[List[str]], opt: str="line"):
        if opt=="word":
            content = "\n".join(texts)
            system_message = ChatTooLs().system_message_wordlevel

        if opt=="line":
            texts = [' | '.join(text) for text in texts]
            content = '\n'.join(texts)
            system_message = ChatTooLs().system_message_linelevel

        resp = ChatTooLs.get_completion(
            system_message=system_message,
            user_prompt=content
        )[0]
        try:
            res = json.loads(json.dumps(resp))
        except Exception as err:
            res = {}
        return res
    
    @staticmethod
    def pick_extraction(annotations: List[str], img0: np.ndarray):
        
        output, position = RESOURCE_MAP["pick_extracter"].predict(
            annotations, img0
        )

        # res = {
        #     "SELLER": [],
        #     "ADDRESS": [],
        #     "TIMESTAMP": [],
        #     "TOTAL_COST": []
        # }

        result_dict = {
            "ADDRESS": {"value": [], "bboxes": []}, 
            "SELLER": {"value": [], "bboxes": []}, 
            "TIMESTAMP": {"value": [], "bboxes": []}, 
            "TOTAL_COST": {"value": [], "bboxes": []}
        }

        for p, out in zip(position, output):
            try:
                box_str = annotations[p].split(',')[1:-1]
                box_str = ','.join(box_str)

                result_dict[out['entity_name']]['value'].append(out['text'])
                result_dict[out['entity_name']]['bboxes'].append(box_str)

            except:
                pass
        return result_dict 

    @staticmethod
    def fields_detecter(img0, convert_box: bool=True):
        output = RESOURCE_MAP["fields_detecter"].predict(img0, convert_box=convert_box)
        return output
    
    @staticmethod
    def receipt_detecter(img0, convert_box: bool=True):
        output = RESOURCE_MAP["receipt_detecter"].predict(img0, convert_box=convert_box)
        return output


    @staticmethod
    def post_prcess_result_dict(result_dict, store_dict):
        fix_result_by_dictionary(result_dict, store_dict)
        fix_result_by_rule_based(result_dict, None, store_dict)

        return result_dict
    
    @staticmethod
    def post_process(labels, texts, bboxes, store_dict):
        result_dict = {
            "ADDRESS": {"value": [], "bboxes": []}, 
            "SELLER": {"value": [], "bboxes": []}, 
            "TIMESTAMP": {"value": [], "bboxes": []}, 
            "TOTAL_COST": {"value": [], "bboxes": []}
        }
                    
        for label, text, boxes in zip(labels, texts, bboxes):

            box_str = []
            for box in boxes:
                box = list(map(int, box))
                if text and len(text):
                    box = [str(b) for b in box]
                    box = ','.join(box)
                    box_str.append(box)
            box_str = ','.join(box_str)

            result_dict[label]["value"].append(text)
            result_dict[label]["bboxes"].append(box_str)
        
        fix_result_by_dictionary(result_dict, store_dict)
        fix_result_by_rule_based(result_dict, None, store_dict)

        return result_dict