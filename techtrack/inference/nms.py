import cv2
import numpy as np

class NMS:
    def __init__(self, score_threshold=0.5, nms_iou_threshold=0.5):
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def filter(self, bboxes, class_ids, scores):
        indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=self.score_threshold, nms_threshold=self.nms_iou_threshold)

        # Extract the indices of the filtered boxes
        indices = indices.flatten() if len(indices) > 0 else []
        
        filtered_bboxes = [bboxes[i] for i in indices]
        filtered_class_ids = [class_ids[i] for i in indices]
        filtered_scores = [scores[i] for i in indices]

        return filtered_bboxes, filtered_class_ids, filtered_scores
