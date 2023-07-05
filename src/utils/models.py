from pydantic import BaseModel

from typing import Any

class ResponseModel(BaseModel):
    status_code: int 
    msg: str
    data: Any 

class RMBGSchema(BaseModel):
    image_base64: str

class GptExtractionSchema(BaseModel):
    content: str

class E2ESchema(BaseModel):
    method: str
    image_base64: str

