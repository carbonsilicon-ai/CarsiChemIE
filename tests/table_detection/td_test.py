import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np

def main(args):
    annFile = args.annFile
    resFile = args.resFile
    
    # Load ground truth COCO annotations
    cocoGt = COCO(annFile)

    categories = cocoGt.loadCats(cocoGt.getCatIds())
    print("categories:", categories)

    # 初始化 COCOeval 对象
    cocoDt = cocoGt.loadRes(resFile)
    
    ## config
    category_idx = args.category_idx
    area_idx = 0

    # Initialize COCOeval object
    cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')

    # Set IoU thresholds range from [0.5, 0.55, ..., 0.95]
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    cocoEval.params.iouThrs = iou_thresholds

    # Run evaluation
    cocoEval.evaluate()
    cocoEval.accumulate()

    # Extract precision, recall, and F1 score for each IoU threshold
    precision = cocoEval.eval['precision']  # 5D array: [TxRxKxAxM]
    recall = cocoEval.eval['recall']        # 4D array: [TxKxAxM]

    # Define category and area limitations (assuming no area and max detections limitation)
    category_idx = 0  # Assume calculating for the first category
    area_idx = 0      # Use default 'all' area
    max_dets_idx = 2  # Use default maximum detections

    # Store precision, recall, and F1 score for each IoU threshold
    results = []

    for i, iou_thr in enumerate(iou_thresholds):
        # Get precision and recall at the current IoU threshold
        precision_iou = precision[i, :, category_idx, area_idx, max_dets_idx]
        recall_iou = recall[i, category_idx, area_idx, max_dets_idx]

        # Remove NaN values
        valid_precision = precision_iou[~np.isnan(precision_iou)]
        valid_recall = recall_iou if not np.isnan(recall_iou) else 0.0

        # Calculate F1 score
        if valid_precision.size > 0:
            precision_mean = np.mean(valid_precision)
            f1_score = 2 * (precision_mean * valid_recall) / (precision_mean + valid_recall) if (precision_mean + valid_recall) > 0 else 0.0
        else:
            precision_mean = 0.0
            f1_score = 0.0

        # Store results
        results.append({
            "IoU": iou_thr,
            "Precision": precision_mean,
            "Recall": valid_recall,
            "F1": f1_score
        })

    # Output results
    for result in results:
        print(f"IoU: {result['IoU']:.2f} | Precision: {result['Precision']:.3f} | Recall: {result['Recall']:.3f} | F1: {result['F1']:.3f}")

    

def parse_args():
    parse = argparse.ArgumentParser(description='evalation for `Table Detection`(TD)')  # 2、Create parameter object
    parse.add_argument('--annFile', type=str, default="data/label/dataset.json", help='The path of file with true labels.')
    parse.add_argument('--resFile', type=str, default="data/test/yolo9c.json", help='The path to save the predicted labels.')
    parse.add_argument('--category_idx', type=int, help='The category_idx for evalation', default=0)
    args = parse.parse_args()  #
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    




