from ultralytics import YOLO
import os 
os.environ["NCCL_P2P_DISABLE"] = "1" 


model_filename = "runs/detect/tld_yolov10_x_sgd/weights/best.pt"
model = YOLO(model_filename)

# Validate the model
metrics = model.val(batch=1, imgsz=1280, device="0")  # no arguments needed, dataset and settings remembered
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
