from ultralytics import YOLO


class yolov8:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self._load_model()
    
    def _load_model(self):
        model = YOLO(self.model_path)
        return model
    
    def inference(self, image, threshold):
        # '''
        # image: Duong dan den anh test hoac anh test da duoc doc bang opencv
        # '''
        '''
        Params:
            image: List[str, np.ndarray] Đường đã ảnh nguồn hoặc mảng được đọc từ opencv
        '''
        raw_information_predict = self.model.predict(source=image, conf=threshold, save=True)
        
        # boxes = []
        # class_id = []
        # for coordinate in raw_information_predict[0]:
        #     boxes.append(coordinate.boxes.xyxy[0].cpu().numpy().astype(int))
        #     class_id.append(coordinate.boxes.cls.cpu().numpy().astype(int))

        # boxes = [[int(i) for i in a] for a in boxes]
        # NOTE: boxes = [list(map(int, box)) for box in boxes]
        # class_id = [a[0] for a in class_id]

        boxes = [list(map(int, coordinate.boxes.xyxy[0].cpu().numpy())) \
                for coordinate in raw_information_predict[0]]
        class_id = [int(coordinate.boxes.cls.cpu().numpy()[0]) \
                    for coordinate in raw_information_predict[0]]


        return boxes, class_id
    



# if __name__=="__main__":
#     yolov8_instance = yolov8(model_path="model_yolov8.pt", threshold=0.5)
#     boxes = yolov8_instance.inference(image="test-case.jpg")
#     print(boxes)
    # img = cv2.imread('test-case.jpg')
    # boxes = yolov8_instance.inference(image = img)
    # print(boxes)