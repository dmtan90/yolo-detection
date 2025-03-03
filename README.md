# YoloV10X and Deep SORT based Object Detection and Tracking
This demo implements object detection and tracking using Ultralytics YoloV10 and Deep SORT.

- [camos.py](./camos.py) - main file that uses detector and tracker on a video feed, image or std(java processbuilder)
- [tracker.py](./tracker.py) - tracker script using Deep SORT, need to install embedder libraries before using

usage: camos.py [-h] [--streamId STREAMID] [--mode {video,image,std}] [--input INPUT]
                [--model {models/yolov10n.pt,models/yolov10s.pt,models/yolov10m.pt,models/yolov11n.pt,models/yolov11s.pt,models/yolov11m.pt}] [--imgsz IMGSZ]
                [--device DEVICE] [--threshold THRESHOLD] [--tracker | --no-tracker]
                [--embedder {mobilenet,torchreid,clip_RN50,clip_RN101,clip_RN50x4,clip_RN50x16,clip_ViT-B/32,clip_ViT-B/16}] [--embedder_gpu | --no-embedder_gpu]

