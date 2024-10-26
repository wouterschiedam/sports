from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import numpy as np

def load_local_model(model_path: str) -> YOLO:
    # Loads a YOLO model from a local path
    return YOLO(model_path)

PLAYER_DETECTION_MODEL_PATH = "./runs/detect/train9/weights/best.pt";
PLAYER_DETECTION_MODEL = load_local_model(PLAYER_DETECTION_MODEL_PATH);

SOURCE_VIDEOS = [
    "./content/08fd33_0.mp4",
     "./content/0bfacc_0.mp4",
     "./content/121364_0.mp4",
     "./content/2e57b9_0.mp4",
     "./content/573e61_0.mp4"
];

PLAYER_ID = 2
STRIDE = 30

frame_generator = sv.get_video_frames_generator(
    source_path=SOURCE_VIDEOS[0], stride=STRIDE)

crops = []
for frame in tqdm(frame_generator, desc='collecting crops'):

    result = PLAYER_DETECTION_MODEL(frame)[0];

    # Extract bounding boxes, confidence scores, and class labels from results
    boxes = result.boxes.xyxy.cpu().numpy();  # Bounding boxes
    confidences = result.boxes.conf.cpu().numpy();  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy().astype(int);  # Class IDs
    # Create Detections object
    
    detections = sv.Detections(
        xyxy=boxes,
        confidence=confidences,
        class_id=class_ids
    );

    detections = detections.with_nms(threshold=0.5, class_agnostic=True)
    detections = detections[detections.class_id == PLAYER_ID]

    players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

     
