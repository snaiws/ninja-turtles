from ultralytics import YOLO
import os 
import glob

def get_file_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension.lstrip('.')

if __name__ == "__main__":
    test_db_path = "../datasets/TLD_2024/test"
    test_res_path = "%s/predictions"%(test_db_path)

    if not os.path.exists(test_res_path):
        os.makedirs(test_res_path)

    img_exts = ["jpg","bmp","png"]
    img_files = list()
    for img_ext in img_exts:
        img_files += glob.glob("%s/images/*.%s"%(test_db_path, img_ext))

    img_files.sort() 

    model_filename = "runs/detect/tld_yolov10_x_sgd/weights/best.pt"
    model = YOLO(model_filename)


    for img_filename in img_files:
        result = model.predict(img_filename, imgsz=1280, conf=0.001, iou=0.6)[0]
        # result = model.predict(img_filename)[0]
        
        img_ext = get_file_extension(img_filename)
        txt_filename = img_filename.replace(img_ext, "txt")
        txt_filename = txt_filename.replace("images","predictions")
        boxes = result.boxes 
        num_obj = len(boxes.cls)

        with open(txt_filename, 'w') as f1:
            for obj_idx in range(num_obj):
                cls_id = int(boxes.cls[obj_idx])
                cs = boxes.conf[obj_idx]
                xywhn = boxes.xywhn[obj_idx] 
                # class_id norm_center_x norm_center_y norm_w norm_h confidence_score
                f1.write("%d %lf %lf %lf %lf %lf\n"%(cls_id, xywhn[0], xywhn[1],xywhn[2],xywhn[3], cs))

                # xywh = boxes.xywh[obj_idx]
                # f1.write("%d %lf %lf %lf %lf %lf\n"%(cls_id, cs, xywh[0], xywh[1],xywh[2],xywh[3]))


        if num_obj == 0:
            print(txt_filename)
