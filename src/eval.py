import os
import shutil

from utils.bbox_visualization import draw_bbox
from utils.metric import COCO_handler
from utils.zip import zip_folder



def eval_analysis(path_gt_yolo, path_pred_yolo, path_gt_COCO, path_pred_COCO):
    
    cocomong = COCO_handler(path_gt_yolo, path_pred_yolo, path_gt_COCO, path_pred_COCO)

    # mAP50 계산
    mAP50_score = cocomong.eval_mAP50_COCO()
    print(f"mAP@IoU=0.5: {mAP50_score}")
    
    # 분석을 위한 이미지세트 만들기
    coco_eval = cocomong.coco_eval
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    ## 기본 임시 디렉토리 준비
    path_temp = './analysis'
    path_zip = './analysis.zip'
    for category_id in cocomong.class_names:
        # 객체탐지 특성 상 TN은 신경쓰지 않는다.
        category = cocomong.class_names[category_id]
        path = os.path.join(path_temp, category, 'FP')
        os.makedirs(path, exist_ok = True)
        path = os.path.join(path_temp, category, 'FN')
        os.makedirs(path, exist_ok = True)
        path = os.path.join(path_temp, category, 'TP')
        os.makedirs(path, exist_ok = True)

    ### iou결과별 처리(이미지-카테고리 조합 case)
    for image_id, category_id in coco_eval.ious:
        category = cocomong.class_names[category_id]
        path_image_gt = os.path.join(cocomong.path_gt_yolo, 'images', str(image_id)+'.jpg')
        path_bbox_gt = os.path.join(cocomong.path_gt_yolo, 'labels', str(image_id)+'.txt')
        path_bbox_pred = os.path.join(cocomong.path_pred_yolo, 'labels', str(image_id)+'.txt')
        
        # case의 ground truth별 처리
        for gt in coco_eval.ious[(image_id, category_id)]:
            # FP
            if len(gt) == 0:
                # pred에 해당클래스가 있다면 FP case 추가
                with open(path_bbox_pred, 'r') as f:
                    preds_classes = [int(x.split()[0]) for x in f.readlines()]
                if category_id in preds_classes:
                    path_image_FP_gt = os.path.join(path_temp, category, 'FP', str(image_id), str(image_id)+'_gt'+'.jpg')
                    path_image_FP_pred = os.path.join(path_temp, category, 'FP', str(image_id), str(image_id)+'_pred'+'.jpg')
                    draw_bbox(path_image = path_image_gt, path_bbox = path_bbox_gt, dict_label = cocomong.class_names, path_output = path_image_FP_gt)
                    draw_bbox(path_image = path_image_gt, path_bbox = path_bbox_pred, dict_label = cocomong.class_names, path_output = path_image_FP_pred)
            # FN
            elif sum(list(map(lambda x: x if x>=0.5 else 0, gt))) == 0:
                path_image_FN_gt = os.path.join(path_temp, category, 'FN', str(image_id), str(image_id)+'_gt'+'.jpg')
                path_image_FN_pred = os.path.join(path_temp, category, 'FN', str(image_id), str(image_id)+'_pred'+'.jpg')
                draw_bbox(path_image = path_image_gt, path_bbox = path_bbox_gt, dict_label = cocomong.class_names, path_output = path_image_FN_gt)
                draw_bbox(path_image = path_image_gt, path_bbox = path_bbox_pred, dict_label = cocomong.class_names, path_output = path_image_FN_pred)
            # TP
            elif sum(list(map(lambda x: x if x>=0.5 else 0, gt))) > 0:
                path_image_TP_gt = os.path.join(path_temp, category, 'TP', str(image_id), str(image_id)+'_gt'+'.jpg')
                path_image_TP_pred = os.path.join(path_temp, category, 'TP', str(image_id), str(image_id)+'_pred'+'.jpg')
                draw_bbox(path_image = path_image_gt, path_bbox = path_bbox_gt, dict_label = cocomong.class_names, path_output = path_image_TP_gt)
                draw_bbox(path_image = path_image_gt, path_bbox = path_bbox_pred, dict_label = cocomong.class_names, path_output = path_image_TP_pred)
            # error
            else:
                raise

    #### zip 생성, 임시폴더 삭제
    zip_folder(path_temp, path_zip)
    shutil.rmtree(path_temp)

    return mAP50_score


if __name__ == "__main__":
    path_gt_yolo = "/workspace/Storage/ninja-turtles/Data/OD/test_labeled/0.0.2"
    path_gt_COCO = "/workspace/Storage/ninja-turtles/Data/OD/test_labeled/0.0.2/COCO.json"
    path_pred_yolo = "/workspace/Storage/ninja-turtles/Data/OD/pred_OD_baseline_test_sample1"
    path_pred_COCO = "/workspace/Storage/ninja-turtles/Data/OD/pred_OD_baseline_test_sample1/COCO.json"

    result = eval_analysis(path_gt_yolo, path_pred_yolo, path_gt_COCO, path_pred_COCO)

    print(result)