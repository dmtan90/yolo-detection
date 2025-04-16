from ultralytics import YOLO

#YOLO configuration https://docs.ultralytics.com/usage/cfg/#train-settings
class YoloDetector:
    #YOLO configuration ref here https://docs.ultralytics.com/usage/cfg/
    def __init__(self, model_path, device, imgsz, classList, confidence = 0.1, iou = 0.1, tracker = False, embedder = "bytetrack"):
        if device is not None:
            self.model = YOLO(model_path, device)
        else:
            self.model = YOLO(model_path)
        #self.classList = ["person"]
        self.imgsz = imgsz
        self.classList = classList
        self.confidence = confidence
        self.iou = iou  # Store the IOU threshold
        self.tracker = tracker # Enable tracker
        self.embedder = embedder

    def detect(self, image):
        results = None
        if self.imgsz is not None and self.imgsz > 0:
            if self.tracker:
                embedder = self.embedder + ".yaml"
                results = self.model.track(image, conf=self.confidence, imgsz=self.imgsz, iou=self.iou, persist=True, tracker=embedder)
            else:    
                results = self.model.predict(image, conf=self.confidence, imgsz=self.imgsz, iou=self.iou)
        else:
            if self.tracker:
                embedder = self.embedder + ".yaml"
                results = self.model.track(image, conf=self.confidence, iou=self.iou, persist=True, tracker=embedder)
            else:    
                results = self.model.predict(image, conf=self.confidence, iou=self.iou)
            #results = self.model.predict(image, conf=self.confidence, iou=self.iou)
        result = results[0]
        detections = self.make_detections(result)
        return detections

    def make_detections(self, result):
        boxes = result.boxes
        detections = []
        if self.tracker and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            #class_ids = boxes.cls.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                class_number = int(box.cls[0])

                if result.names[class_number] not in self.classList:
                    continue
                conf = box.conf[0]
                detections.append((([x1, y1, w, h]), class_number, conf, track_id))
        else:    
            for box in boxes:
              x1, y1, x2, y2 = box.xyxy[0]
              x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
              w, h = x2 - x1, y2 - y1
              class_number = int(box.cls[0])

              if result.names[class_number] not in self.classList:
                continue
              conf = box.conf[0]
              detections.append((([x1, y1, w, h]), class_number, conf, -1))
        return detections
