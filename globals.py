import os
from dotenv import load_dotenv

load_dotenv("../../.env")


class AppConfig:
    def __init__(self):
        self.LLM_MODEL = os.getenv("LLM_MODEL")
        self.LLM_API_KEY = os.getenv("LLM_API_KEY")

        self.FACE_PROTOTXT = "models/face_detection/deploy.prototxt"
        self.FACE_MODEL = (
            "models/face_detection/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )


config = AppConfig()
