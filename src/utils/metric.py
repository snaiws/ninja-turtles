import os

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def eval_mAP50_COCO(path_gt_COCO:str, path_pred_COCO:str) -> float:
    """
    두 데이터셋(gt_file, pred_file)을 COCO 방식의 mAP50으로 비교하는 함수
    알고리즘에 따라 mAP50의 결과가 다를 수 있으므로 COCO방식으로 맞추는 것이 좋다.
    입력:
        path_gt_COCO (str): Ground Truth annotation 파일 경로 (COCO 형식 JSON)
        path_pred_COCO (str): Prediction 결과 파일 경로 (COCO 형식 JSON)

    출력:
        float: mAP@IoU=threshold_mAP 점수
    """
    # COCO API를 사용해 Ground Truth와 Prediction 파일 로드
    coco_gt = COCO(path_gt_COCO) # json 경로만 받음
    coco_pred = coco_gt.loadRes(path_pred_COCO) # np.ndarray와 str을 받는데, gt를 path로 받으니 str로
    
    # COCO 평가 객체 생성
    coco_eval = COCOeval(coco_gt, coco_pred, iouType='bbox')
    
    # 평가 수행
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # mAP50 값 추출
    mAP50 = coco_eval.stats[1]  # stats[1]은 IoU=0.5에서의 mAP

    return mAP50

def yolo_to_COCO(path_yolo:str, path_COCO:str, pred=False) -> bool:
    """
    yolo 방식으로 저장된 label 디렉토리를 통해 COCO json 생성
    입력:
        path_yolo : yolo 방식으로 저장된 label 디렉토리, 바로 하위에 images, labels 디렉토리 존재
        path_COCO : COCO josn 저장 경로
    출력:
        작업 성공 여부
    """

    # COCO 형식의 기본 구조
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    path_labels = os.path.join(path_yolo, 'labels')
    path_images = os.path.join(path_yolo, 'images')
    path_labels = [os.path.join(path_labels, x) for x in os.listdir(path_labels)]
    path_images = [os.path.join(path_images, x) for x in os.listdir(path_images)]

    
    # 카테고리 정보 추가
    
    class_names = ['veh_go', 'veh_goLeft', 'veh_noSign', 'veh_stop', 
        'veh_stopLeft', 'veh_stopWarning', 'veh_warning', 
        'ped_go', 'ped_noSign', 'ped_stop', 'bus_go', 
        'bus_noSign', 'bus_stop', 'bus_warning']
    for idx, class_name in enumerate(class_names):
        coco_format["categories"].append({
            "id": idx,
            "name": class_name
        })
        
    annotation_id = 1
    for path_image, path_label in zip(path_images, path_labels):
        im = cv2.imread(path_image)
        image_height, image_width, _ = im.shape
        # 이미지 정보 추가 (가정: 파일명에 이미지 ID를 포함시키는 방식 사용)
        image_id = int(os.path.splitext(os.path.basename(path_label))[0])
        coco_format["images"].append({
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": f"{image_id}.jpg"
        })
        
        # YOLO 파일 읽기 및 변환
        with open(path_label, 'r') as file:
            for line in file:
                values = list(map(float, line.split()))
                class_id, x_center, y_center, bbox_width, bbox_height = values[:5]

                
                # YOLO 좌표에서 COCO 좌표로 변환
                x_min = x_center * image_width
                y_min = y_center * image_height
                bbox_width = bbox_width * image_width
                bbox_height = bbox_height * image_height
                
                # COCO 형식의 annotation 추가
                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                }
                if pred:
                    ann['score'] = values[-1]
                coco_format["annotations"].append(ann)
                annotation_id += 1

    with open(path_COCO, 'w') as f:
        if pred:
            json.dump(coco_format['annotations'], f, indent=4)
        else:
            json.dump(coco_format, f, indent=4)

    return True

if __name__ == "__main__":
    path_yolo = "/workspace/Storage/ninja-turtles/Data/OD/test_labeled/0.0.2"
    path_COCO = "/workspace/Storage/ninja-turtles/Data/OD/test_labeled/0.0.2/COCO.json"
    yolo_to_COCO(path_yolo = path_yolo, path_COCO = path_COCO)
    path_yolo = "/workspace/Storage/ninja-turtles/Data/OD/pred_OD_baseline_test_sample1"
    path_COCO = "/workspace/Storage/ninja-turtles/Data/OD/pred_OD_baseline_test_sample1/COCO.json"
    yolo_to_COCO(path_yolo = path_yolo, path_COCO = path_COCO, pred = True)


    # # Ground Truth와 Prediction JSON 파일 경로 설정
    gt_file_path = "/workspace/Storage/ninja-turtles/Data/OD/test_labeled/0.0.2/COCO.json"
    pred_file_path = "/workspace/Storage/ninja-turtles/Data/OD/pred_OD_baseline_test_sample1/COCO.json"

    # # mAP50 계산
    mAP50_score = eval_mAP50_COCO(gt_file_path, pred_file_path)
    print(f"mAP@IoU=0.5: {mAP50_score}")