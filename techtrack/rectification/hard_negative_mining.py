import os
import cv2
import json
import sys

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
import numpy as np
import shutil
from typing import List, Dict, Tuple
from inference.nms import NMS


class HardNegativeMining:
    def __init__(self, annotation_dir: str, prediction_dir: str):
        self.prediction_dir = prediction_dir
        self.annotation_dir = annotation_dir
        self.predictions = self.load_data(prediction_dir)
        self.annotations = self.load_data(annotation_dir)

    def load_data(self, dir_path: str) -> Dict[str, List[Tuple[int, List[float]]]]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"The directory '{dir_path}' does not exist.")

        data = {}
        for filename in os.listdir(dir_path):
            if filename.endswith('.txt'):
                with open(os.path.join(dir_path, filename)) as f:
                    yolo_data = []
                    for line in f:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        yolo_data.append((int(class_id), [x_center, y_center, width, height]))
                    data[filename[:-4]] = yolo_data  # Use filename without '.txt' as key
        return data

    def compute_iou(self, box1: List[float], box2: List[float]) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2

        iou = inter_area / float(box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
        return iou

    def compute_loss(self, predictions, annotations, iou_threshold=0.5, lambda_class=1.0, lambda_fn=1.0):
        total_loss = 0
        matched_annotations = set()
        for pred_class_id, pred_bbox in predictions:
            best_iou = 0
            best_annot_idx = None
            class_mismatch = False

            # Find the best matching annotation for this prediction
            for i, (annot_class_id, annot_bbox) in enumerate(annotations):
                if i in matched_annotations:
                    continue  # Skip already matched annotations

                iou = self.compute_iou(pred_bbox, annot_bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_annot_idx = i
                    class_mismatch = pred_class_id != annot_class_id

            if best_annot_idx is not None:
                matched_annotations.add(best_annot_idx)
                total_loss += (1 - best_iou)  # IoU loss
                if class_mismatch:
                    total_loss += lambda_class  # Class mismatch penalty

        # Penalize false negatives (annotations that were not matched)
        for i, (annot_class_id, annot_bbox) in enumerate(annotations):
            if i not in matched_annotations:
                total_loss += lambda_fn  # False negative penalty

        return total_loss


    def sample_hard_negatives(self, num_samples_percentage, iou_threshold=0.5, lambda_class=0.2, lambda_fn=1.0):
        """
        Sample hard negatives based on a percentage of total samples.
        
        Args:
            num_samples_percentage (float): Percentage of samples to select (0.1 = 10%)
            iou_threshold (float): IoU threshold for matching predictions
            lambda_class (float): Weight for classification loss
            lambda_fn (float): Weight for false negative loss
        
        Returns:
            list: List of copied file names
        """
        losses = []
        image_files = []
        self.num_samples_percentage = num_samples_percentage
        
        if self.predictions is None:
            self.predictions = self.load_data(self.prediction_dir)
            self.annotations = self.load_data(self.annotation_dir)
        
        # Create output directory
        output_dir = f'../storage/hard_negatives_{self.num_samples_percentage}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute losses for all valid samples
        for filename in self.predictions:
            prediction = self.predictions[filename]
            annotation = self.annotations.get(filename)
            
            if annotation:
                loss = self.compute_loss(prediction, annotation, iou_threshold, lambda_class, lambda_fn)
                losses.append(loss)
                image_files.append(filename)
        
        # Calculate number of samples to select based on percentage
        num_samples = int(len(image_files) * num_samples_percentage)
        if num_samples == 0:
            num_samples = 1  # Ensure at least one sample is selected
        
        # Sort and select top N% samples
        sorted_indices = np.argsort(losses)[::-1]  # Sort losses in descending order
        selected_files = [image_files[i] for i in sorted_indices[:num_samples]]
        
        # Copy both images and annotations to output directory
        copied_files = []
        for file in selected_files:
            # Copy image file
            img_src_path = os.path.join(self.annotation_dir, f"{file}.jpg")
            img_dst_path = os.path.join(output_dir, f"{file}.jpg")
            
            # Copy annotation file (YOLO format .txt)
            ann_src_path = os.path.join(self.annotation_dir, f"{file}.txt")
            ann_dst_path = os.path.join(output_dir, f"{file}.txt")
            
            # Check if both image and annotation files exist
            if os.path.exists(img_src_path) and os.path.exists(ann_src_path):
                # Copy both files
                shutil.copy2(img_src_path, img_dst_path)
                shutil.copy2(ann_src_path, ann_dst_path)
                copied_files.append(f"{file}.jpg")
            else:
                missing_files = []
                if not os.path.exists(img_src_path):
                    missing_files.append("image")
                if not os.path.exists(ann_src_path):
                    missing_files.append("annotation")
                print(f"Warning: {', '.join(missing_files)} file(s) not found for {file}")
        
        print(f"Saved {len(copied_files)} hard negative samples ({num_samples_percentage*100:.1f}%) to {output_dir}")
        return copied_files
    
    def compute_complexity_score(self, annotation: List[Tuple[int, List[float]]]) -> float:
        # Compute a complexity score based on the number and size of objects.
        num_objects = len(annotation)
        total_area = sum(bbox[2] * bbox[3] for _, bbox in annotation)  
        return num_objects * total_area
    
    def predict_and_save(self, model, iou_threshold=0.5):
        nms = NMS(nms_iou_threshold=iou_threshold)
        os.makedirs(f'{self.prediction_dir}_{self.num_samples}', exist_ok=True)

        for image_file in os.listdir(self.annotation_dir):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(self.annotation_dir, image_file)
                prediction_file = os.path.join(self.prediction_dir, image_file.replace('.jpg', '.txt'))

                # Load image
                image = cv2.imread(image_path)
                original_height, original_width = image.shape[:2]

                # Model prediction
                predictions = model.predict(image)
                bboxes, class_ids, scores = model.post_process(predictions, score_threshold=0.5)
                filtered_bboxes, filtered_class_ids, filtered_scores = nms.filter(bboxes, class_ids, scores)

                # Save predictions to a text file in the specified output directory
                with open(prediction_file, 'w') as f:
                    for class_id, score, bbox in zip(filtered_class_ids, filtered_scores, filtered_bboxes):
                        # Calculate center and dimensions
                        x_center = bbox[0] + bbox[2] / 2  
                        y_center = bbox[1] + bbox[3] / 2  
                        width = bbox[2]  
                        height = bbox[3]  

                        # Normalize the coordinates by the dimensions of the original image
                        x_center_normalized = x_center / original_width
                        y_center_normalized = y_center / original_height
                        width_normalized = width / original_width
                        height_normalized = height / original_height

                        # Write in YOLO format
                        f.write(f"{class_id} {x_center_normalized:.6f} {y_center_normalized:.6f} "
                                f"{width_normalized:.6f} {height_normalized:.6f}\n")
        
        self.predictions = self.load_data(self.prediction_dir)
        
if __name__ == "__main__":
    num_samples_percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for i in num_samples_percentage:   
        hnm = HardNegativeMining(annotation_dir="../logistics", prediction_dir="../storage/predictions")
        hard_negatives = hnm.sample_hard_negatives(num_samples_percentage=i)