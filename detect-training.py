from ultralytics import YOLO

# Laad het model
model = YOLO("./runs/detect/train8/weights/best.pt"); 

# Train het model
# epochs = aantal iteraties
trained_results = model.train(data="datasets/football-players-detection/data.yaml", epochs=50, batch=6, imgsz=1280)

metrics = model.val()
