import os 
import glob
import shutil

from ultralytics import YOLO



def get_file_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension.lstrip('.')

def predict_OD(model, params, path_image_test, path_bbox_pred):
    result = model.predict(path_image_test, **params)[0]
    boxes = result.boxes 
    num_obj = len(boxes.cls)

    os.makedirs(os.path.dirname(path_bbox_pred), exist_ok = True)
    with open(path_bbox_pred, 'w') as f1:
        for obj_idx in range(num_obj):
            cls_id = int(boxes.cls[obj_idx])
            cs = boxes.conf[obj_idx]
            xywhn = boxes.xywhn[obj_idx] 
            # class_id norm_center_x norm_center_y norm_w norm_h confidence_score
            f1.write("%d %lf %lf %lf %lf %lf\n"%(cls_id, xywhn[0], xywhn[1], xywhn[2], xywhn[3], cs))
            
    return True

def make_predset(path_model, params, path_test, path_pred):
    """
    pred 데이터셋 만드는 함수
    path_model : 모델 경로
    path_test : yolo 형태의 테스트 데이터셋 디렉토리 경로
    path_pred : yolo 형태의 예측 데이터셋(예정) 디렉토리 경로
    """
    # 예측데이터 겹치기 방지
    if os.path.exists(path_pred):
        raise

    model = YOLO(path_model)

    paths_image_test = list(filter(lambda x: '.jpg' in x or '.png' in x or '.gif' in x, glob.glob(path_test+"/**", recursive = True)))

    for path_image_test in paths_image_test:
        path_bbox_pred = path_image_test.replace('images','labels').replace('.jpg','.txt').replace('.png','.txt').replace('.gif','.txt')
        path_bbox_pred = path_bbox_pred.replace(path_test, path_pred)
        predict_OD(model, params, path_image_test, path_bbox_pred)
        path_image_pred = path_image_test.replace(path_test, path_pred)
        os.makedirs(os.path.dirname(path_image_pred), exist_ok = True)
        shutil.copy(path_image_test, path_image_pred)

if __name__ == "__main__":
    path_model = "/workspace/Storage/ninja-turtles/Models/OD/reimple/best.pt"
    path_test = "/workspace/Storage/ninja-turtles/Data/OD/test_labeled/0.0.2"
    path_pred = "/workspace/Storage/ninja-turtles/Data/OD/pred_OD_reimple_test1"
    params = {
        "imgsz" : 1280,
        "conf" : 0.001,
        "iou" : 0.6
    }
    make_predset(path_model, params, path_test, path_pred)