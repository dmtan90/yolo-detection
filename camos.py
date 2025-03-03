import cv2
import time
import json
import sys
import os.path
import torch
import argparse
from datetime import datetime
from yolo_detector import YoloDetector
from tracker import Tracker

HOME_DIR = "/usr/local/antmedia/yolov10/"
MODEL_PATH = HOME_DIR + "models/yolov11n.pt"
MODEL_PATHS = ["models/yolov10n.pt", "models/yolov10s.pt", "models/yolov10m.pt", "models/yolov11n.pt", "models/yolov11s.pt", "models/yolov11m.pt"]
VIDEO_PATH = HOME_DIR + "assets/football.mp4"
VIDEO_OUTPUT_PATH = HOME_DIR + "assets/football_output.mp4"
IMAGE_PATH = HOME_DIR + "assets/sample.png"
STREAM_ID = "Camera_1"
MODE = "video"
IMGSZ = None
DEVICE = None
THRESHOLD = 0.1
TRACKER_ENABLE = False
TRACKER_EMBEDDER = "mobilenet"
TRACKER_EMBEDDER_GPU = False

#STD_TIMEOUT = 10 * 60* 1000; #10 minutes
COCO_NAMES_FILE = HOME_DIR + "coco.names"  # Replace with your file path
#DEVICE = "cpu"# or "cuda"
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Check for CUDA device and set it
#print(f'Using device: {DEVICE}')

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--streamId',
        default=STREAM_ID,
        help='streamId of camera',
    )
    parser.add_argument(
        '--mode',
        default=MODE,
        help='input mode to run inference',
        choices=[
            "video",
            "image",
            "std"
        ]
    )
    parser.add_argument(
        '--input',
        default=VIDEO_PATH,
        help='path to input video or image',
    )
    parser.add_argument(
        '--model',
        default=MODEL_PATH,
        help='model path',
        choices=MODEL_PATHS
    )
    parser.add_argument(
        '--imgsz',
        default=IMGSZ,
        help='image resize, 640 will resize images to 640x640',
        type=int
    )
    parser.add_argument(
        '--device',
        default=DEVICE,
        help='specifies the computational device(s) for running: single CUDA GPU (device=cuda), multiple CUDA GPU (device=cuda:0), CPU (device=cpu), or MPS for Apple silicon (device=mps).',
    )
    parser.add_argument(
        '--threshold',
        default=THRESHOLD,
        help='score threshold to filter out detections',
        type=float
    )
    parser.add_argument(
        '--tracker',
        default=TRACKER_ENABLE,
        help='Enable DeepSORT tracking',
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        '--embedder',
        default=TRACKER_EMBEDDER,
        help='type of feature extractor to use for DeepSORT. You need to install external models by yourself before using it such as pip install git+https://github.com/openai/CLIP.git',
        choices=[
            "mobilenet",
            "torchreid",
            "clip_RN50",
            "clip_RN101",
            "clip_RN50x4",
            "clip_RN50x16",
            "clip_ViT-B/32",
            "clip_ViT-B/16"
        ]
    )
    parser.add_argument(
        '--embedder_gpu',
        default=TRACKER_EMBEDDER_GPU,
        help='Use GPU for DeepSORT',
        action=argparse.BooleanOptionalAction
    )
    
    args = parser.parse_args()
    
    print(f"streamId: {str(args.streamId)} mode: {str(args.mode)} input: {str(args.input)} device: {str(args.device)} model: {str(args.model)} imgsz: {str(args.imgsz)} threshold: {str(args.threshold)} tracker: {str(args.tracker)} embedder: {str(args.embedder)} embedder_gpu: {str(args.embedder_gpu)}")
    sys.stdout.flush() # Flush data to Java in STD mode
    
    return args

def getCurrentMs():
    current_time = datetime.now()
    ms = current_time.timestamp() * 1000;
    return ms;

def sleep_milliseconds(milliseconds):
    """Sleeps for the specified number of milliseconds."""
    seconds = milliseconds / 1000.0
    time.sleep(seconds)

def load_class_names(file_path):
    """Loads class names from a file."""
    try:
        with open(file_path, "r") as f:
            class_names = [line.strip() for line in f]
        return class_names
    except FileNotFoundError:
        print(f"Error: Class names file not found at {file_path}")
        return None

def calculate_iou(detect_box, tracker_box):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1_min, y1_min, w1, h1 = detect_box
    x2_min, y2_min, x2_max, y2_max = tracker_box

    x1_max = x1_min + w1
    y1_max = y1_min + h1

    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def get_detection_from_tracker(tracker_map, detections, tracking_id, track_box):
    # Find the detection that matches the track's most recent bounding box
    best_match = None
    best_match_iou = 0

    for det in detections:
        det_box = det[0] # assuming detection bounding box is the first element

        # Calculate IOU (Intersection over Union)
        iou = calculate_iou(det_box, track_box)

        if iou > best_match_iou:
            best_match_iou = iou
            best_match = det
    
    #if best_match is None and tracking_id in tracker_map:
    #    best_match = tracker_map[tracking_id]
    
    #tracker_map[tracking_id] = best_match
    
    if best_match is not None:
        tracker_map[tracking_id] = best_match
        
    return best_match

def post_detection_process(tracker_map, frame, tracker, class_names, detections, tracking_ids, boxes):
    data = [];
    if len(tracking_ids) == 0:#first frame
        for detection in detections:
            # Extract values
            bbox, class_id, confidence_tensor = detection
            x_min, y_min, width, height = bbox
            confidence = confidence_tensor.item()  # Extract the float value from the tensor
            className = class_names[class_id]
            x = x_min
            y = y_min
            w = int(width)
            h = int(height)
            item = {
                "trackerId": -1, 
                "classId": int(class_id), 
                "className": className, 
                "confidence": round(confidence, 2), 
                "bbox": [x, y, w, h], 
                "originBox": [x, y, w, h],
                "meta": ""
            }
            #print(f"item: {(json.dumps(item))}")
            data.append(item)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(frame, f"{str(-1)} - {className} - {confidence:.2f}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:    
        for i, (tracking_id, tracker_box) in enumerate(zip(tracking_ids, boxes)):
            detection = get_detection_from_tracker(tracker_map, detections, tracking_id, tracker_box)
            if detection is None:
                continue
                
            class_id = -1
            className = None;
            x1 = int(tracker_box[0])
            y1 = int(tracker_box[1])
            w1 = int(tracker_box[2] - tracker_box[0])
            h1 = int(tracker_box[3] - tracker_box[1])
            
            x2 = x1
            y2 = y1
            w2 = w1
            h2 = h1
            
            if detection is not None:
                # Extract values
                bbox, class_id, confidence_tensor = detection
                x_min, y_min, width, height = bbox
                confidence = confidence_tensor.item()  # Extract the float value from the tensor
                className = class_names[class_id]
                # Convert bounding box
                #x_min = int(x_center - width / 2)
                #y_min = int(y_center - height / 2)
                #x_max = int(x_center + width / 2)
                #y_max = int(y_center + height / 2)
                
                x2 = x_min
                y2 = y_min
                w2 = int(width)
                h2 = int(height)
            
            item = {
                "trackerId": int(tracking_id), 
                "classId": int(class_id), 
                "className": className, 
                "confidence": round(confidence, 2), 
                #"bbox": [x1, y1, w1, h1], 
                "bbox": [x2, y2, w2, h2],
                "meta": ""
            }
            #print(f"item: {(json.dumps(item))}")
            data.append(item)
            
            cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 1)
            #cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
              
            cv2.putText(frame, f"{str(tracking_id)} - {className} - {confidence:.2f}", 
                (int(tracker_box[0]), int(tracker_box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return data

def processFrame(tracker_map, detector, tracker, class_names, streamId, frame, writeFrame=False):
    if frame is not None:
        detectMs = getCurrentMs()
        detections = detector.detect(frame)
        detectMs = getCurrentMs() - detectMs
        
        trackMs = getCurrentMs()
        tracking_ids = []
        boxes = []
        if tracker is not None:
            tracking_ids, boxes = tracker.track(detections, frame)
        trackMs = getCurrentMs() - trackMs
        
        postMs = getCurrentMs()
        data = post_detection_process(tracker_map, frame, tracker, class_names, detections, tracking_ids, boxes);
        postMs = getCurrentMs() - postMs
        
        print(f"Data: {(json.dumps(data))}")
        sys.stdout.flush() # Flush data to Java in STD mode
        print(f"Detection time {int(detectMs)}ms - Tracking time {int(trackMs)}ms - Post proccess {int(postMs)}ms")
        sys.stdout.flush() # Flush data to Java in STD mode
        if writeFrame is True:
            cv2.imwrite(HOME_DIR + f"assets/{streamId}_{str(getCurrentMs())}.png",frame )
    else: 
        print("Error: Unable to open image file.")
        sys.stdout.flush() # Flush data to Java in STD mode

def processVideo(tracker_map, detector, tracker, class_names, streamId, path):
    print(f"Run in video mode")
    sys.stdout.flush() # Flush data to Java in STD mode
    if path is None:
        path = VIDEO_PATH
        
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        sys.stdout.flush() # Flush data to Java in STD mode
        exit()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.perf_counter()    
        processFrame(tracker_map, detector, tracker, class_names, streamId, frame, False)
        end_time = time.perf_counter()
        fps = 1 / (end_time - start_time)
        print(f"Current fps: {fps}")
        sys.stdout.flush() # Flush data to Java in STD mode
        #print(f"Data: {(json.dumps(data))}")
        writer.write(frame)
        #cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    writer.release()
    
def processImage(tracker_map, detector, tracker, class_names, streamId, path):
    print(f"Run in image mode")
    sys.stdout.flush() # Flush data to Java in STD mode
    if path is None:
        path = IMAGE_PATH
    if os.path.isfile(path):
        frame = cv2.imread(path)
        processFrame(tracker_map, detector, tracker, class_names, streamId, frame, True)
    else:
        print(f"File {path} is not existed")
        sys.stdout.flush() # Flush data to Java in STD mode
        
        
def main():
    args = parseArgs()
    sys.stdout.flush() # Flush data to Java in STD mode
    # Example usage:
    tracker_map = {};
    #tracker_map.clear()
    class_names = load_class_names(COCO_NAMES_FILE)
    detector = YoloDetector(model_path=args.model, device=args.device, imgsz=args.imgsz, classList=class_names, confidence=args.threshold)
    tracker = None
    if args.tracker:
        tracker = Tracker(embedder=args.embedder, embedder_gpu=args.embedder_gpu)
    
    startMs = getCurrentMs()
    
    if args.mode == 'video':
        processVideo(tracker_map, detector, tracker, class_names, args.streamId, args.input)
    elif args.mode == "image":
        processImage(tracker_map, detector, tracker, class_names, args.streamId, args.input)
    elif args.mode == "std":
        print(f"Run in std mode")
        sys.stdout.flush() # Flush data to Java in STD mode
        counter = 1;
        while True:
            currentMs = getCurrentMs()
            #IMPORTANT => Do not delete below line: This line indicates the service is running and the Java code can send the 
            print(f"Ready to send command") 
            sys.stdout.flush() # Flush data to Java in STD mode
            imagePath = input(f"{str(counter)}.Input image path: ") 
            sys.stdout.flush() # Flush data to Java in STD mode
            print(f"Received input: {imagePath}")
            sys.stdout.flush() # Flush data to Java in STD mode
            try:
                if imagePath is None or imagePath == "":
                    continue
                processImage(tracker_map, detector, tracker, class_names, args.streamId, imagePath)
            except UnicodeDecodeError:
                print("Error decoding bytes to UTF-8 string. The input data might not be UTF-8 encoded.")
                sys.stdout.flush() # Flush data to Java in STD mode
                # Handle the error appropriately, perhaps try a different encoding.
            counter += 1
        print("Exited")
        sys.stdout.flush() # Flush data to Java in STD mode
if __name__ == "__main__":
  main()
