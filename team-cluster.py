from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
import numpy as np
import torch
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH, map_location=DEVICE)
EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

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

    # Apply Non-Maximum Suppression without reassigning detections
    nms_detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Filter by class_id for PLAYER_ID directly on the numpy arrays
    player_mask = nms_detections.class_id == PLAYER_ID
    filtered_boxes = nms_detections.xyxy[player_mask]

    # Crop images based on the filtered bounding boxes
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in filtered_boxes]
    crops += players_crops

sv.plot_images_grid(crops[:100], grid_size=(10, 10))


BATCH_SIZE = 32

crops = [sv.cv2_to_pillow(crop) for crop in crops]
batches = chunked(crops, BATCH_SIZE)
data = []
with torch.no_grad():
    for batch in tqdm(batches, desc='embedding extraction'):
        inputs = EMBEDDINGS_PROCESSOR(images=batch, return_tensors="pt").to(DEVICE)
        outputs = EMBEDDINGS_MODEL(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
        data.append(embeddings)

data = np.concatenate(data)

