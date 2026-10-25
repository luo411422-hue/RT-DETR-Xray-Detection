# backend/inference.py
import threading, cv2, numpy as np, yaml
from yolo import yolov5, yolov7, yolov8, rtdetr   # 直接用你现有的类

class _Hub:
    def __init__(self):
        self.lock = threading.Lock()
        self.cfg = None
        self.model = None

    def ready(self):
        return self.model is not None

    def load_from_cfg(self, cfg: dict):
        with self.lock:
            self.cfg = cfg
            t = cfg.get("model_type")
            if t == "yolov5":
                self.model = yolov5(**cfg)
            elif t == "yolov7":
                self.model = yolov7(**cfg)
            elif t == "yolov8":
                self.model = yolov8(**cfg)
            elif t == "rtdetr":
                self.model = rtdetr(**cfg)
            else:
                raise ValueError(f"未知模型类型: {t}")

    def infer_image(self, bgr: np.ndarray) -> np.ndarray:
        image_det, result = self.model(bgr)   # 已经画好框
        return image_det

    def infer_stream(self, bgr: np.ndarray) -> np.ndarray:
        image_det, result = self.model(bgr)
        return image_det

ModelHub = _Hub()
