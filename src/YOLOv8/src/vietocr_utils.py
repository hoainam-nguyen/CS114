import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor


class vietocr: 
    def __init__(self, model_path: str):
        self.base_config = 'vgg_transformer'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
        self.trained_model = model_path
        self.detector = self._load_model()

    def _load_model(self):
        config = Cfg.load_config_from_name(self.base_config)
        config['device'] = self.device
        config['vocab'] = self.vocab
        config['weights'] = self.trained_model
        detector = Predictor(config)
        return detector

    def inference(self, image) -> tuple:
        text, prob = self.detector.predict(image , return_prob = True)
        return text, prob