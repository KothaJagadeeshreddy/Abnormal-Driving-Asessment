from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
from .config import Config

@dataclass
class DrivingEvent:
    type: str
    severity: str
    timestamp: str
    speed: Optional[float] = None
    speed_limit: Optional[float] = None

class DrivingAssessment:
    def __init__(self, config: Config = None):
        self.config = config if config else Config()
        self.abnormal_events: List[DrivingEvent] = []
        self.start_time: datetime = datetime.now()
        
    def assess_behavior(self, detector_results: Dict[str, Any]) -> List[DrivingEvent]:
        """Assess driving behavior based on detector results"""
        abnormalities: List[DrivingEvent] = []
        
        # Check for lane departure
        if detector_results.get('lane_departure', False):
            abnormalities.append(DrivingEvent(
                type='lane_departure',
                severity='high',
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        # Check for speeding
        speed_limit = 60  # Example speed limit (km/h)
        current_speed = detector_results.get('speed', 0)
        if current_speed > speed_limit * self.config.SPEEDING_THRESHOLD:
            abnormalities.append(DrivingEvent(
                type='speeding',
                severity='medium',
                speed=current_speed,
                speed_limit=speed_limit,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        # Check for aggressive maneuvers
        if detector_results.get('aggressive_turn', False):
            abnormalities.append(DrivingEvent(
                type='aggressive_turn',
                severity='medium',
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
        
        self.abnormal_events.extend(abnormalities)
        return abnormalities
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a summary report of the driving assessment"""
        events_by_type = {}
        for event in self.abnormal_events:
            events_by_type[event.type] = events_by_type.get(event.type, 0) + 1
            
        return {
            'start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_abnormal_events': len(self.abnormal_events),
            'events_by_type': events_by_type,
            'event_details': self.abnormal_events
        }

    def reset(self) -> None:
        """Reset the assessment tracker"""
        self.abnormal_events = []
        self.start_time = datetime.now()