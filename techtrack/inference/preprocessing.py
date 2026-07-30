import cv2
import numpy as np
import socket
import struct
import os
class Preprocessing:
    def __init__(self):
        self.filename = None
        self.drop_rate = 1
        self.cap = None
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.buffer_size = 65536
        self.url = None

    def open_video(self, source):
        self.url = source

        if source.startswith('udp://'):
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            print("Test", self.cap.isOpened())
        else:
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {source}")
        self.frame_count = 0

    def close_video(self):
        if self.cap:
            self.cap.release()
        if self.udp_socket:
            self.udp_socket.close()

    def capture_video(self, source, drop_rate):
        self.open_video(source)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            if self.frame_count % drop_rate == 0:
                yield frame
        
        self.close_video()

    def resize_frame(self, frame, target_size=(416, 416)):
        return cv2.resize(frame, target_size)

    def stream_video(self, url):
        self.open_video(url)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                break
            
            # Display the frame
            cv2.imshow('Video Stream', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
        
        self.close_video()
        cv2.destroyAllWindows()

    def setup_udp_socket(self, ip='127.0.0.1', port=23000):
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.bind((ip, port))

    def receive_udp_stream(self, buffer_size=65536):
        self.buffer_size = buffer_size
        if not self.udp_socket:
            raise ValueError("UDP socket not set up. Call setup_udp_socket() first.")
        
        data, _ = self.udp_sock.recvfrom(self.buffer_size)
        npdata = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(npdata, 1)
        return frame
    
    def udp_stream(self, ip='127.0.0.1', port=23000):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                yield frame
            else:
                break
        
        self.close_video()
        cv2.destroyAllWindows()

    def save_new_detections(self, frame, bboxes, class_ids, scores, classes, output_folder, frame_count, prev_detections):
        current_detections = set(class_ids)
        new_detections = current_detections - prev_detections
        
        if new_detections:
            os.makedirs(output_folder, exist_ok=True)
            yolo_data = []

            # Draw bounding boxes and labels for new detections
            for bbox, class_id, score in zip(bboxes, class_ids, scores):
                if class_id in new_detections:
                    x, y, w, h = bbox
                    label = f"{classes[class_id]}: {score:.2f}"
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # YOLO format
                    x_center = (x + w / 2) / frame.shape[1]
                    y_center = (y + h / 2) / frame.shape[0]
                    width = w / frame.shape[1]
                    height = h / frame.shape[0]
                    
                    yolo_data.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
            # Save the frame with new detections
            filename = os.path.join(output_folder, f"new_detection_frame_{frame_count}.jpg")
            cv2.imwrite(filename, frame)

            txt_filename = filename.replace(".jpg", ".txt")
            with open(txt_filename, 'w') as f:
                f.write("\n".join(yolo_data))

            print(f"New detection(s) saved: {[classes[class_id] for class_id in new_detections]}")
        
        return current_detections

    def save_image(self, image, path: str):
        if not path.endswith('.jpg'):
            path += '.jpg'

        os.makedirs(os.path.dirname(path), exist_ok=True)

        cv2.imwrite(path, image)

if __name__ == "__main__":
    preprocessor = Preprocessing()
    
    # preprocessor.stream_video("../test_videos/worker-zone-detection.mp4")
    preprocessor.open_video('udp://127.0.0.1:23002')
    preprocessor.udp_stream()

