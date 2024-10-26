import os
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
import numpy as np

# if using nvidia gpu
#os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

def load_local_model(model_path: str) -> YOLO:
    # Loads a YOLO model from a local path
    return YOLO(model_path)

from ultralytics import YOLO

PLAYER_DETECTION_MODEL_PATH = "./runs/detect/train9/weights/best.pt";
PLAYER_DETECTION_MODEL = load_local_model(PLAYER_DETECTION_MODEL_PATH);
BALL_ID = 0;

SOURCE_VIDEOS = [
    "./content/08fd33_0.mp4",
     "./content/0bfacc_0.mp4",
     "./content/121364_0.mp4",
     "./content/2e57b9_0.mp4",
     "./content/573e61_0.mp4"
];

TARGET_VIDEO_PATH = [
    "./content/08fd33_0_result.mp4",
    "./content/0bfacc_0_result.mp4",
    "./content/121364_0_result.mp4",
    "./content/2e57b9_0_result.mp4",
    "./content/573e61_0_result.mp4"
];

class_names = ['ball', 'goalkeeper', 'player', 'referee'];

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    thickness=2
);

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
);

triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFD700"),
    base=20, height=17
);

tracker = sv.ByteTrack();
tracker.reset()


video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEOS[1]);
video_sink = sv.VideoSink(TARGET_VIDEO_PATH[1], video_info=video_info);
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEOS[1]);


with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):
        
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

        
        # Ensure class IDs and confidences are valid arrays before proceeding
        if detections.class_id is not None and detections.confidence is not None:
            # Filter for ball and other detections and create new Detections objects
            ball_mask = detections.class_id == BALL_ID
            if ball_mask.any():  # Check if there are any ball detections
                ball_detections = sv.Detections(xyxy=detections.xyxy[ball_mask], confidence=detections.confidence[ball_mask], class_id=detections.class_id[ball_mask]);
            else:
                ball_detections = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.empty(0), class_id=np.empty(0, dtype=int));

            non_ball_mask = detections.class_id != BALL_ID
            if non_ball_mask.any():  # Check if there are any non-ball detections
                all_detections = sv.Detections(xyxy=detections.xyxy[non_ball_mask], confidence=detections.confidence[non_ball_mask], class_id=detections.class_id[non_ball_mask]);
            else:
                all_detections = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.empty(0), class_id=np.empty(0, dtype=int));


            # labels = [
            #     f"{class_names[class_id]} {confidence:.2f}"
            #     for class_id, confidence in zip(detections.class_id, detections.confidence)
            # ];

        else:
            print("No labels found for frame. Using default labels...")
            ball_detections = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.empty(0), class_id=np.empty(0, dtype=int));
            all_detections = sv.Detections(xyxy=np.empty((0, 4)), confidence=np.empty(0), class_id=np.empty(0, dtype=int));

        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)

        if all_detections.class_id is not None:
            all_detections.class_id -= 1

        all_detections = tracker.update_with_detections(detections=all_detections)

        if all_detections.tracker_id is not None:
            labels = [
                f"#{tracker_id}"
                for tracker_id
                in all_detections.tracker_id
            ];
        else:
            labels = [];

        # Annotate the frame with detections
        annotated_frame = ellipse_annotator.annotate(frame.copy(), all_detections);
        annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections);
        annotated_frame = label_annotator.annotate(annotated_frame, all_detections, labels=labels);
        video_sink.write_frame(annotated_frame);

