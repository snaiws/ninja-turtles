import cv2 
import glob


def display():
    db_path = "/media/ssd_8tb/myGit/ultralytics/datasets/ETRITLR"
    db_type = "train"
    img_list = glob.glob("%s/%s/images/*.jpg"%(db_path, db_type))
    class_name_list = list() 
    class_name_file = "%s/%s/classes.txt"%(db_path, db_type)
    with open(class_name_file, 'r') as f1:
        lines = f1.readlines()
    
    for line in lines:
        line = line.rstrip() 
        class_name_list.append(line)


    img_list.sort() 

     

    for img_filename in img_list:
        img = cv2.imread(img_filename)
        gt_filename = img_filename.replace("jpg", "txt")
        gt_filename = gt_filename.replace("images","labels")
        with open(gt_filename, 'r') as f1:
            gts = f1.readlines()

        img_h = img.shape[0]
        img_w = img.shape[1]

        for gt in gts:
            splited = gt.rstrip()
            label = gt.split(' ')
            class_id = int(label[0])
            norm_obj_center_x = float(label[1])
            norm_obj_center_y = float(label[2])
            norm_obj_w = float(label[3])
            norm_obj_h = float(label[4])
            
            norm_obj_left = norm_obj_center_x - norm_obj_w/2
            norm_obj_right = norm_obj_center_x + norm_obj_w/2
            norm_obj_top = norm_obj_center_y - norm_obj_h/2
            norm_obj_bot = norm_obj_center_y + norm_obj_h/2

            left = int(norm_obj_left * img_w)
            right = int(norm_obj_right * img_w)
            top = int(norm_obj_top * img_h)
            bot = int(norm_obj_bot * img_h)
            class_name = class_name_list[class_id]
            cv2.rectangle(img, (left,top), (right,bot), (255,0,0), 2)
            cv2.putText(img, class_name, (left,top-5), 1, 0.5, (0,255,0), 1)
        cv2.imshow("image", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    display()