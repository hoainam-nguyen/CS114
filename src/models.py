import os

import torch
from ultralytics import YOLO

import src.PICK.model.pick as pick_arch_module
from src.CRAFT import model, net
from src.CRAFT.net import model_setup
from src.PICK.data_utils.documents import Document
from src.PICK.data_utils.pick_dataset import BatchCollateFn
from src.PICK.utils.util import (bio_tags_to_spans2, iob_index_to_str,
                                 text_index_to_str)
from src.vietocr import Config, Predictor


class CraftModel:
    def __init__(self, model, pretrained:str, device: str="cuda:0"):
        
        self.model = model
        self.cuda = True if "cuda" in device else False
        self.pretrained = pretrained 

    def load_model(self):
        return model_setup(self.model, self.pretrained, self.cuda)
        
    # def predict(self, img):
    #     bboxes, _, _ = net.test_net(self.model, img, 0.7, 0.4, 0.4, self.cuda, False, self.refine_net)
    #     return bboxes


class VietOCRModel():
    def __init__(self, config_name: str, device: str="cuda:0") -> None:
        self.config = Config.load_config_from_name(config_name)
        self.device = device

    def load_model(self):
        self.predictor = Predictor(self.config)
        return self.predictor
    
    # def predict(self, img):
    #     return self.predictor.predict(img)



class PICKModel():
    def __init__(self, pretrained: str, device: str="cuda:0") -> None:
        self.device = torch.device(device)

        import sys
        sys.path.append(
            os.path.join(os.environ["PWD"], "src/PICK")
        )
        
        self.checkpoint = torch.load(pretrained, map_location=device)
        self.config = self.checkpoint['config']
        self.state_dict = self.checkpoint['state_dict']
        self.monitor_best = self.checkpoint['monitor_best']
        
        self.model = self.config.init_obj('model_arch', pick_arch_module)
        self.model = self.model.to(device)
        self.model.load_state_dict(self.state_dict)
        self.model.eval()
    
    def predict(self, annotations, img0):
        doc = Document(
            boxes_and_transcripts_file=annotations,
            image_file=img0,
            image_index=0,
            training=False
        )
        
        input_data_item = BatchCollateFn(training=False)([doc])
        for key, input_value in input_data_item.items():
            if key in ["text_length"]:
                continue
            if input_value is not None:
                input_data_item[key] = input_value.to(self.device)

        with torch.no_grad():
            output = self.model(**input_data_item)

        logits = output['logits']
        new_mask = output['new_mask']
        image_indexs = input_data_item['image_indexs']  # (B,)
        text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
        mask = input_data_item['mask']
        text_length = input_data_item['text_length']
        boxes_coors = input_data_item['boxes_coordinate'].cpu().numpy()[0]
        
        best_paths = self.model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
        predicted_tags = []
        for path, score in best_paths:
            predicted_tags.append(path)

        # convert iob index to iob string
        decoded_tags_list = iob_index_to_str(predicted_tags)
        # union text as a sequence and convert index to string
        decoded_texts_list = text_index_to_str(text_segments, mask)
        for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
            # List[ Tuple[str, Tuple[int, int]] ]
            # spans = bio_tags_to_spans(decoded_tags, [])
            spans, line_pos_from_bottom = bio_tags_to_spans2(decoded_tags, text_length.cpu().numpy())
            # spans = sorted(spans, key=lambda x: x[1][0])

            entities = []  # exists one to many case
            for entity_name, range_tuple in spans:
                entity = dict(entity_name=entity_name,
                                text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                entities.append(entity)     
            
            return entities, line_pos_from_bottom


class Yolov8Model():
    def __init__(self, pretrained) -> None:
        self.model = YOLO(pretrained)

    def predict(self, img0, convert_box: bool=True):
        output = self.model.predict(source=img0)

        bboxes = [list(map(int, coordinate.boxes.xyxy[0].cpu().numpy())) for coordinate in output[0]]
        labels = [output[0].names[int(coordinate.boxes.cls.cpu().numpy()[0])] for coordinate in output[0]]
        
        if len(labels) and "class" in labels[0]:
            map_label = {
                "class_0": "SELLER",
                "class_1": "ADDRESS",
                "class_2": "TIMESTAMP",
                "class_3": "TOTAL_COST"
            }
            _labels = []
            for _label in labels:
                _labels.append(map_label[_label])
            labels = _labels
        if not convert_box:
            return bboxes, labels
        
        corners = []
        for box in bboxes:
            x1, y1, x2, y2 = box
            corners.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            
        return corners, labels

    