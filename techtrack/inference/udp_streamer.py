import cv2
import numpy as np
import socket
from inference.nms import NMS
from inference.object_detection import Model

class UDPStreamer:
    def __init__(self, url='udp://127.0.0.1:23000'):
        self.url = url
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def stream(self, model=None):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                break  

            if model:
                nms = NMS(nms_iou_threshold=0.4)
                predictions = model.predict(frame)
                bboxes, class_ids, scores = model.post_process(predictions, score_threshold=0.5)
                filtered_bboxes, filtered_class_ids, filtered_scores = nms.filter(bboxes, class_ids, scores)

                # Draw bounding boxes and labels on the original frame
                for bbox, class_id, score in zip(filtered_bboxes, filtered_class_ids, filtered_scores):
                    x, y, w, h = bbox
                    label = f"{model.classes[class_id]}: {score:.2f}"
                    
                    # Draw bounding box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            yield frame


    def close(self):
        self.cap.release()

if __name__ == "__main__":
    streamer = UDPStreamer()

    object_detector = Model(config_path="../yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg", 
                  weights_path="../yolo_model_2/yolov4-tiny-logistics_size_416_2.weights", 
                  class_names_path="../yolo_model_2/logistics.names")

    for frame in streamer.stream(object_detector):
        cv2.imshow('UDP Stream', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    streamer.close()
    cv2.destroyAllWindows()