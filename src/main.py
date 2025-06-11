import os
import cv2
import time
import argparse
from typing import List
from .detector import DrivingDetector
from .assessment import DrivingAssessment
from .config import Config
from .image_processing import preprocess_image, draw_lanes

def process_image(image_path: str, config: Config) -> None:
    detector = DrivingDetector(config)
    assessor = DrivingAssessment(config)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    image = cv2.resize(image, (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT))
    processed_image = preprocess_image(image)
    
    obj_results = detector.detect_objects(image)
    lane_results = detector.detect_lanes(processed_image)
    
    if lane_results is not None:
        image = draw_lanes(image, lane_results)
    
    detector_results = {
        'lane_departure': detector.check_lane_departure(image.shape[1] // 2),
        'speed': 75,  # Simulated speed
        'aggressive_turn': False
    }
    
    abnormalities = assessor.assess_behavior(detector_results)
    
    print(f"Assessment for {image_path}:")
    if abnormalities:
        print("Abnormal driving behaviors detected:")
        for event in abnormalities:
            print(f"- {event.type} (severity: {event.severity})")
    else:
        print("No abnormal driving behaviors detected.")
    
    cv2.imshow('Driving Assessment', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Abnormal Driving Detection")
    parser.add_argument('-i', '--images', nargs='+', help="List of image paths")
    args = parser.parse_args()
    
    config = Config()
    
    if args.images:
        for img_path in args.images:
            process_image(img_path, config)
            time.sleep(1)
    else:
        sample_images = [
            os.path.join(config.IMAGE_DIR, "normal_driving_1.jpg"),
            os.path.join(config.IMAGE_DIR, "aggressive_driving_1.jpg"),
            os.path.join(config.IMAGE_DIR, "drowsy_driver_1.jpg")
        ]
        for img_path in sample_images:
            process_image(img_path, config)
            time.sleep(1)

if __name__ == "__main__":
    main()