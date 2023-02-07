import numpy as np
import cv2


def get_image_context_bbox(image_path, bbox):
    original_img = cv2.imread(image_path)
    x, y, w, h = bbox
    cropped_image = original_img[y:y+h, x:x+w]
    return original_img, cropped_image

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def get_motion_img(img_path_list, boxes_list):
    w, h, c = cv2.imread(img_path_list[0]).shape
    motion_img = np.zeros((w, h, c), dtype=np.int16)
    line_motion_img = np.zeros((w, h, c), dtype=np.int16)
    
    center_points = []

    first = boxes_list[0][2] * boxes_list[0][3]
    last = boxes_list[-1][2] * boxes_list[-1][3]

    if last < first:
        print("IS REVERSED")
        img_path_list = img_path_list[::-1]
        boxes_list = boxes_list[::-1]


    prev_box = []
    for idx, img_path in enumerate(img_path_list):
        img = cv2.imread(img_path)

        if len(prev_box) == 0:
            x, y, w, h = boxes_list[idx]
            prev_box = [x, y, x+w, y+h]
        else:
            x, y, w, h = boxes_list[idx]
            curr_box = [x, y, x+w, y+h]
            if bb_intersection_over_union(prev_box, curr_box) > 0.5:
                continue
            else:
                prev_box = curr_box

        x, y, w, h = boxes_list[idx]


        motion_img[y:y+h, x:x+w, :] = img[y:y+h, x:x+w, :]
        center_points.append((int(x+ w/2),int(y + h/2)))
    center_points = np.array(center_points)
    
    cv2.drawContours(line_motion_img, [center_points], 0, (255,255,255), 160)
    return line_motion_img, motion_img