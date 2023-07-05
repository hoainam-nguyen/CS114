import os 
import time
import logging 
import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.models import VietOCRModel, CraftModel, PICKModel, Yolov8Model
from src.CRAFT.model import CRAFT, RefineNet
from src.const import RESOURCE_MAP

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


app = FastAPI(
    description="Project for CS338 | UIT",
    title="MC-OCR system",
    docs_url="/"
)

from routers.application import router as application_router 
app.include_router(application_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

def load_model():

    logging.info("Loading OCR model"); t1 = time.time()
    RESOURCE_MAP["ocr_model"] = VietOCRModel(
        config_name="vgg_seq2seq",
        device="cuda:0"
    ).load_model()
    logging.info("(Done .) {} s".format(time.time() - t1) )
    
    logging.info("Loading Craft detecter model"); t1 = time.time()
    RESOURCE_MAP["craft_detect_model"] = CraftModel(
        model=CRAFT(),
        pretrained="weights/craft_mlt_25k.pth",
        device="cuda:0"
    ).load_model()
    logging.info("(Done .) {} s".format(time.time() - t1) )

    logging.info("Loading Refinenet model"); t1 = time.time()
    RESOURCE_MAP["refinenet_model"] = CraftModel(
        model=RefineNet(),
        pretrained="weights/craft_refiner_CTW1500.pth",
        device="cuda:0"
    ).load_model()
    logging.info("(Done .) {} s".format(time.time() - t1) )

    logging.info("Loading Flip classifier model"); t1 = time.time()
    with open("weights/rotate_180.pkl", 'rb') as f:
        RESOURCE_MAP["flip_classifier"] = pickle.load(f)
        f.close()
    logging.info("(Done .) {} s".format(time.time() - t1) )

    logging.info("Loading YOLO model"); t1 = time.time()
    RESOURCE_MAP["fields_detecter"] = Yolov8Model(
        pretrained="weights/yolo_detect_fields.pt"
    )
    logging.info("(Done .) {} s".format(time.time() - t1) )

    logging.info("Loading YOLO model"); t1 = time.time()
    RESOURCE_MAP["receipt_detecter"] = Yolov8Model(
        pretrained="weights/yolo_detect_receipt.pt"
    )
    logging.info("(Done .) {} s".format(time.time() - t1) )

    logging.info("Loading PICK model"); t1 = time.time()
    RESOURCE_MAP["pick_extracter"] = PICKModel(
        pretrained="weights/model_best_pick.pth"
    )
    logging.info("(Done .) {} s".format(time.time() - t1) )


def main():
    uvicorn.run(
        app=app,
        host='0.0.0.0',
        port=5012
    )

if __name__=="__main__":
    load_model()
    main()