import cv2
import os
import argparse
from inference.preprocessing import Preprocessing
from inference.object_detection import Model
from inference.nms import NMS
from inference.udp_streamer import UDPStreamer
from rectification.hard_negative_mining import HardNegativeMining
from rectification.augmentation import Augmentation

def main(args):
    # Initialize components
    video_processor = Preprocessing()
    nms = NMS(nms_iou_threshold=0.4)
    object_detector = Model(config_path="yolo_model_2/yolov4-tiny-logistics_size_416_2.cfg", 
                  weights_path="yolo_model_2/yolov4-tiny-logistics_size_416_2.weights", 
                  class_names_path="yolo_model_2/logistics.names")

    if args.mode == 'inference':
        base_name = os.path.basename(args.input).replace('.mp4', '')
        output_folder = os.path.join(args.output, f'{base_name}_detection')
        prev_detections = set()
        frame_count = 0

        for frame in video_processor.capture_video(args.input, drop_rate=10):
            if frame is None:
                break
            
            frame_count += 1
            predictions = object_detector.predict(frame)
            bboxes, class_ids, scores = object_detector.post_process(predictions, score_threshold=0.5)
            filtered_bboxes, filtered_class_ids, filtered_scores = nms.filter(bboxes, class_ids, scores)
            
            # Save new detections
            prev_detections = video_processor.save_new_detections(
                frame, filtered_bboxes, filtered_class_ids, filtered_scores, 
                object_detector.classes, output_folder, frame_count, prev_detections
            )
    
    elif args.mode == 'rectification':
        # Perform hard negative mining
        hard_negative_miner = HardNegativeMining(annotation_dir=args.input, prediction_dir=args.output)

        user_input = input("Do you need to predict and save the files to the directory? (Y/N): ")
        if user_input.lower() == 'y':
            hard_negative_miner.predict_and_save(object_detector, iou_threshold=0.5)

        hard_negatives_name = hard_negative_miner.sample_hard_negatives(num_samples=args.top_n)
        
        images = []
        for i, img_name in enumerate(hard_negatives_name):
            img = cv2.imread(f"{args.input}/{img_name}")
            images.append(img)

            cv2.imshow('Image Window', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            output_folder = os.path.join(args.output, 'hard_negatives')
            os.makedirs(output_folder, exist_ok=True)
            video_processor.save_image(img, f"{args.output}/hard_negatives/hard_negatives_{i}.jpg")

        # Perform augmentation on hard negatives
        augmenter = Augmentation()
        augmented_images = augmenter.augment(images)

        for i, img in enumerate(augmented_images):
            output_folder = os.path.join(args.output, 'hard_negatives')
            os.makedirs(output_folder, exist_ok=True)
            video_processor.save_image(img, f"{args.output}/augmented/augmented_{i}.jpg")

    elif args.mode == 'stream':
        # Start UDP streaming
        udp_streamer = UDPStreamer(f'udp://{args.host}:{args.port}')
        
        for frame in udp_streamer.stream(object_detector):
            cv2.imshow('UDP Stream', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        udp_streamer.close()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TechTrack Object Detection System")
    parser.add_argument('mode', choices=['inference', 'rectification', 'stream'], help='Operation mode')
    parser.add_argument('--input', help='Input file or directory')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--top_n', type=int, default=10, help='Top N hard negatives to select')
    parser.add_argument('--host', default='127.0.0.1', help='UDP host')
    parser.add_argument('--port', type=int, default=23000, help='UDP port')
    
    args = parser.parse_args()
    main(args)