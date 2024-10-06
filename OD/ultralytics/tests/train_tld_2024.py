from ultralytics import YOLO
import os 
os.environ["NCCL_P2P_DISABLE"] = "1" 



# Load a model
model = YOLO('yolov10x.pt')  # load a pretrained model (recommended for training)

#########################################################
# Train the model
model.train(data='../ultralytics/datasets/tld_2024.yaml', 
            name="tld_yolov10_x_sgd",
            epochs=30, 
            imgsz=1280,  
            device="0,1,2,3,4,5,6,7", 
            batch=32,
            cache=True, 
            pretrained=True, 
            lr0 = 0.001,
            fliplr = 0.0, 
            mosaic = 1.0,
            optimizer='SGD',
            close_mosaic=5)