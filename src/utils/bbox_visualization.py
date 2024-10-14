import cv2
import matplotlib.pyplot as plt



def convert_to_absol(shape, bbox):
    x,y,w,h = bbox
    l1, l2, _ = shape
    x_ = int(x*l2)
    y_ = int(y*l1)
    w_ = int(w*l2)
    h_ = int(h*l1)
    return (x_, y_, w_, h_)


def draw_bbox(path_image:str, path_bbox:str, dict_label:dict, path_output:str=None) -> None:
    """
    이미지 위에 bounding box를 그려서 리턴하는 함수.

    Parameters:
    - path_image (str): 원본 이미지 파일 path.
    - path_bbox (str): (label, x, y, width, height) 형식의 bounding box 좌표 목록이 담긴 txt파일 path.
    - path_output (str, optional): 결과 이미지를 저장할 경로. 제공되지 않으면 plot.
    
    Returns:
    - image_with_bbox: bounding box가 그려진 이미지.

    todo:
    - object detection 데이터 모델로 class화
    """
    # import
    image = cv2.imread(path_image)
    with open(path_bbox, 'r') as f:
        bboxes = f.readlines()
    
    # Draw bounding box
    for bbox in bboxes:
        # parsing
        label, x, y, w, h = list(map(float, bbox.split(' ')))
        label = dict_label[int(label)]
        x, y, w, h = convert_to_absol(image.shape, (x,y,w,h))
        # draw
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        label_x, label_y = x, y - 10  # 라벨이 bounding box 위에 위치하도록 함
        cv2.rectangle(image, (label_x, label_y - label_size[1]), (label_x + label_size[0], label_y), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 결과 이미지 표시
    if path_output is None:
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)  # 키 입력 대기
        cv2.destroyAllWindows()
    # 이미지 저장 (옵션)
    else:
        cv2.imwrite(path_output, image)

    return image



# 예시
if __name__ == "__main__":
    path_image = '/home/snaiws/Storage/ninja-turtles/Data/raw/train/images/00000000.jpg'
    path_bbox = '/home/snaiws/Storage/ninja-turtles/Data/raw/train/labels/00000000.txt'
    path_label = "/home/snaiws/Storage/ninja-turtles/Data/raw/train/classes.txt"
    with open(path_label, 'r') as f:
        dict_label = dict(enumerate([x.strip() for x in f.readlines()]))
    draw_bbox(path_image = path_image, path_bbox = path_bbox, dict_label = dict_label, path_output="output_image.jpg")