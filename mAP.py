import torch
from collections import Counter
import os
from IoU import intersection_over_union
from utils import mAP_plot

def mean_average_precision(
    true_boxes, pred_boxes, iou_threshold=0.5, box_format="corners"):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    detections = []
    ground_truths = []

    # Go through all predictions and targets,
    # and only add the ones that belong to the
    # current class c
    for detection in pred_boxes:
        detections.append(detection)

    for true_box in true_boxes:
        ground_truths.append(true_box)

    # find the amount of bboxes for each training example
    # Counter here finds how many ground truth bboxes we get
    # for each training example, so let's say img 0 has 3,
    # img 1 has 5 then we will obtain a dictionary with:
    # amount_bboxes = {0:3, 1:5}
    amount_bboxes = Counter([gt[0] for gt in ground_truths])

    # We then go through each key, val in this dictionary
    # and convert to the following (w.r.t same example):
    # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
    for key, val in amount_bboxes.items():
        amount_bboxes[key] = torch.zeros(val)

    # sort by box probabilities which is index 2
    detections.sort(key=lambda x: x[2], reverse=True)
    TP = torch.zeros((len(detections)))
    FP = torch.zeros((len(detections)))
    total_true_bboxes = len(ground_truths)
    

    for detection_idx, detection in enumerate(detections):
        # Only take out the ground_truths that have the same
        # training idx as detection
        ground_truth_img = [
            bbox for bbox in ground_truths if bbox[0] == detection[0]
        ]

        num_gts = len(ground_truth_img)
        best_iou = 0

        for idx, gt in enumerate(ground_truth_img):
            iou = intersection_over_union(
                torch.tensor(detection[3:]),
                torch.tensor(gt[3:]),
                box_format=box_format,
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

            # 값 확인하고 싶을때 확인할것
            # if iou>0:
            #     print(f'iou         : {iou}')
            #     print(f'detection   : {detection[3:]}')
            #     print(f'ground_truth: {gt[3:]}')
            #     print(f'best_iou    : {best_iou}\n')

        if best_iou > iou_threshold:
            # only detect ground truth detection once
            if amount_bboxes[detection[0]][best_gt_idx] == 0:
                # true positive and add this bounding box to seen
                TP[detection_idx] = 1
                amount_bboxes[detection[0]][best_gt_idx] = 1
            else:
                FP[detection_idx] = 1

        # if IOU is lower then the detection is a false positive
        else:
            FP[detection_idx] = 1

    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    print(f'TP_cuscum             :{TP_cumsum}')
    print(f'FP_cumsum             :{FP_cumsum}')

    recalls = TP_cumsum / (total_true_bboxes + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    # precision에는 1부터 시작하여 떨어지는구조
    # recall   에는 0부터 시작하여 상승하는구조

    mAP_plot(recalls, precisions)

    # torch.trapz for numerical integration
    average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)



if __name__ == "__main__":

    # txt 가져오기
    gt_path = './gt_labels/'
    # tinaface 평가시 
    label_path = './tinaface_labels/'
    
    # yoloR 평가시 
    # label_path = './yoloR_labels/'

    gt   = os.listdir(gt_path)
    pred = os.listdir(label_path)

    gt_arr   = []
    pred_arr = []

    for gt_ in gt:
        lines_gt = open(gt_path+gt_)
        for line in lines_gt:
            line = line.split()
            line[1:] = list(map(float, line[1:]))
            gt_arr.append(line)

    for pred_ in pred:
        lines_pred = open(label_path+pred_)
        for line in lines_pred:
            line = line.split()
            line[1:] = list(map(float, line[1:]))
            pred_arr.append(line)
    # print(gt_arr)
    # print(pred_arr)
    print(f'amount of ground truth: {len(gt_arr)}')
    mAP = mean_average_precision(gt_arr, pred_arr)
    print(f'mAP                   : {round(mAP.item(),3)*100}')
