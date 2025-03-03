from ultralytics import YOLO

#YOLO configuration https://docs.ultralytics.com/usage/cfg/#train-settings
class YoloDetector:
    #YOLO configuration ref here https://docs.ultralytics.com/usage/cfg/
    def __init__(self, model_path, device, imgsz, classList, confidence):
        if device is not None:
            self.model = YOLO(model_path, device)
        else:
            self.model = YOLO(model_path)
        #self.classList = ["person"]
        self.imgsz = imgsz
        self.classList = classList
        self.confidence = confidence

    def detect(self, image):
        if self.imgsz is not None and self.imgsz > 0:
            results = self.model.predict(image, conf=self.confidence, imgsz=self.imgsz)
        else:
            results = self.model.predict(image, conf=self.confidence)
        result = results[0]
        detections = self.make_detections(result)
        return detections

    def make_detections(self, result):
        boxes = result.boxes
        detections = []
        for box in boxes:
          x1, y1, x2, y2 = box.xyxy[0]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          w, h = x2 - x1, y2 - y1
          class_number = int(box.cls[0])

          if result.names[class_number] not in self.classList:
            continue
          conf = box.conf[0]
          detections.append((([x1, y1, w, h]), class_number, conf))
        return detections
