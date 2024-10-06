from ultralytics import YOLO


model = YOLO("yolov10s-seg.yaml")
# Train the model with 2 GPUs
results = model.train(data="Compete_segment.yaml",pretrained='yolov8s-seg.pt',epochs=100, device=[0,1,2,3,4,5,6,7])
