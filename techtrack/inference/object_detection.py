import cv2
import numpy as np
from inference.preprocessing import Preprocessing
from inference.nms import NMS

class Model:
    def __init__(self, config_path, weights_path, class_names_path):
        # Load the YOLO model configuration and weights
        self.net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load class names from the specified file
        with open(class_names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
    
    def predict(self, frame, input_size=416):
        self.frame_width = frame.shape[1]  
        self.frame_height = frame.shape[0]  

        # Preprocess the frame into a blob suitable for YOLO model input
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get the names of YOLO output layers
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Forward pass to get the predictions
        outputs = self.net.forward(output_layers)
        return outputs
    
    def post_process(self, predict_output, score_threshold):
        bboxes = []
        class_ids = []
        scores = []
        
        for output in predict_output:
            for detection in output:
                scores_per_class = detection[5:]
                class_id = np.argmax(scores_per_class)
                confidence = scores_per_class[class_id]
                
                if confidence > score_threshold:
                    # YOLO outputs are relative to the input size
                    center_x = int(detection[0] * self.frame_width)
                    center_y = int(detection[1] * self.frame_height)
                    w = int(detection[2] * self.frame_width)
                    h = int(detection[3] * self.frame_height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    bboxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    scores.append(float(confidence))
        
        return bboxes, class_ids, scores

if __name__ == "__main__":
    # Initialize the Model
    model = Model(config_path="../yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg", 
                  weights_path="../yolo_model_2/yolov4-tiny-logistics_size_416_2.weights", 
                  class_names_path="../yolo_model_2/logistics.names")

    # Initialize the Preprocessing 
    preprocessor = Preprocessing()

    # Initialize the NMS
    nms = NMS(nms_iou_threshold=0.4)

    # Process video frames
    for frame in preprocessor.capture_video(filename="../test_videos/worker-zone-detection.mp4", drop_rate=10):
        if frame is None:
            break

        # Predict
        predictions = model.predict(frame)

        # Post-process the predictions 
        bboxes, class_ids, scores = model.post_process(predictions, score_threshold=0.5)

        # Apply NMS 
        filtered_bboxes, filtered_class_ids, filtered_scores = nms.filter(bboxes, class_ids, scores)

        # Draw bounding boxes and labels on the original frame
        for bbox, class_id, score in zip(filtered_bboxes, filtered_class_ids, filtered_scores):
            x, y, w, h = bbox
            label = f"{model.classes[class_id]}: {score:.2f}"
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display the processed frame with detections
        cv2.imshow("YOLO Object Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
