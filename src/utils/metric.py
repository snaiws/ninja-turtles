import os
import copy

import json
import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



class COCO_handler:
    def __init__(self, path_gt_yolo:str, path_pred_yolo:str, path_gt_COCO:str, path_pred_COCO:str):
        self.path_gt_yolo = path_gt_yolo
        self.path_pred_yolo = path_pred_yolo
        self.path_gt_COCO = path_gt_COCO
        self.path_pred_COCO = path_pred_COCO
        self.class_names = ['veh_go', 'veh_goLeft', 'veh_noSign', 'veh_stop', 
            'veh_stopLeft', 'veh_stopWarning', 'veh_warning', 
            'ped_go', 'ped_noSign', 'ped_stop', 'bus_go', 
            'bus_noSign', 'bus_stop', 'bus_warning']
        self.class_names = dict(enumerate(self.class_names))
        if os.path.exists(path_gt_COCO):
            self.yolo_to_COCO(path_yolo = path_gt_yolo, path_COCO = path_gt_COCO, class_names = self.class_names)
        if os.path.exists(path_pred_COCO):
            self.yolo_to_COCO(path_yolo = path_pred_yolo, path_COCO = path_pred_COCO, class_names = self.class_names, pred = True)

        self.coco_gt, self.coco_eval = self.get_COCO(path_gt_COCO, path_pred_COCO)


    def get_COCO(self, path_gt_COCO, path_pred_COCO):
        # COCO API를 사용해 Ground Truth와 Prediction 파일 로드
        coco_gt = COCO(path_gt_COCO) # json 경로만 받음
        coco_pred = coco_gt.loadRes(path_pred_COCO) # np.ndarray와 str을 받는데, gt를 path로 받으니 str로
        
        # COCO 평가 객체 생성
        coco_eval = COCOeval(coco_gt, coco_pred, iouType='bbox')
        return coco_gt, coco_eval

    def eval_mAP50_COCO(self) -> float:
        """
        두 데이터셋(gt_file, pred_file)을 COCO 방식의 mAP50으로 비교하는 함수
        알고리즘에 따라 mAP50의 결과가 다를 수 있으므로 COCO방식으로 맞추는 것이 좋다.
        입력:
            path_gt_COCO (str): Ground Truth annotation 파일 경로 (COCO 형식 JSON)
            path_pred_COCO (str): Prediction 결과 파일 경로 (COCO 형식 JSON)

        출력:
            float: mAP@IoU=threshold_mAP 점수
        """
        coco_gt = self.coco_gt
        coco_eval = copy.deepcopy(self.coco_eval)
        # 평가 수행
        mAP50 = {}
        # mAP50 값 추출
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP50['total'] = coco_eval.stats[1]  # stats[1]은 IoU=0.5에서의 mAP
    
        for cat_id in coco_gt.getCatIds():
            # 카테고리별 평가 설정
            cat_name = coco_gt.loadCats(cat_id)[0]['name']
            coco_eval.params.catIds = [cat_id]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            mAP50[cat_name] = coco_eval.stats[1]

        return mAP50


    def yolo_to_COCO(self, path_yolo:str, path_COCO:str, class_names, pred=False) -> bool:
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
        
        for idx in class_names:
            coco_format["categories"].append({
                "id": idx,
                "name": class_names[idx]
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
    path_gt_yolo = "/workspace/Storage/ninja-turtles/Data/OD/test_labeled/0.0.2"
    path_gt_COCO = "/workspace/Storage/ninja-turtles/Data/OD/test_labeled/0.0.2/COCO.json"
    
    path_pred_yolo = "/workspace/Storage/ninja-turtles/Data/OD/pred_OD_baseline_test_sample1"
    path_pred_COCO = "/workspace/Storage/ninja-turtles/Data/OD/pred_OD_baseline_test_sample1/COCO.json"
    
    cocomong = COCO_handler(path_gt_yolo, path_pred_yolo, path_gt_COCO, path_pred_COCO)

    # # mAP50 계산
    mAP50_score = cocomong.eval_mAP50_COCO()
    print(f"mAP@IoU=0.5: {mAP50_score}")