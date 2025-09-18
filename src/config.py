import os
from dataclasses import dataclass

@dataclass
class Config:
    # Path configurations
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = os.path.join(BASE_DIR, 'data')
    IMAGE_DIR: str = os.path.join(DATA_DIR, 'sample_images')
    VIDEO_DIR: str = os.path.join(DATA_DIR, 'sample_videos')
    MODEL_DIR: str = os.path.join(BASE_DIR, 'models')
    
    # Model configuration
    YOLO_MODEL: str = os.path.join(MODEL_DIR, 'yolov8n.pt')
    BEHAVIOR_MODEL: str = os.path.join(MODEL_DIR, 'behavior_model.h5')
    
    # Detection thresholds
    LANE_DEPARTURE_THRESHOLD: float = 0.7
    SPEEDING_THRESHOLD: float = 1.2
    AGGRESSIVE_TURN_THRESHOLD: float = 0.8
    
    # Visualization settings
    DISPLAY_WIDTH: int = 800
    DISPLAY_HEIGHT: int = 600
    
    # Audio settings
    ALARM_SOUND: str = os.path.join(DATA_DIR, 'alarm.mp3')
