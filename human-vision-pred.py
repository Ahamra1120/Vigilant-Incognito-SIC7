import cv2
import numpy as np
from datetime import datetime
import time
import paho.mqtt.client as mqtt
import json
from collections import defaultdict
import warnings
import os
from ultralytics import YOLO
import hashlib

warnings.filterwarnings('ignore')

# MQTT Configuration
MQTT_SERVER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "sic7/teamincognito/alarm"
MQTT_TOPIC1 = "sic7/teamincognito/sensors"

# Thresholds
DETECTION_THRESHOLD = 0.5
TRACKING_TIME_THRESHOLD = 5.0  # 5 seconds for each person
STATIONARY_THRESHOLD = 20  # Pixels movement to consider as "staying"

class PeopleTracker:
    def __init__(self, model_path='yolo11n.pt'):
        """Initialize YOLO People Tracker with individual tracking"""
        print("Loading YOLO model for person tracking...")
        
        try:
            self.model = YOLO(model_path)
            self.model.conf = DETECTION_THRESHOLD
            print("✓ Model loaded successfully!")
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
        
        self.person_class_id = 0
        
        # Tracking dictionaries
        self.tracked_persons = {}  # {person_id: {data}}
        self.alerted_persons = set()  # IDs of persons already alerted
        self.next_person_id = 1
        
        # MQTT Client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.connect_mqtt()
        
        self.is_running = True
        self.output_video = None
        
    def connect_mqtt(self):
        """Connect to MQTT broker"""
        try:
            self.mqtt_client.connect(MQTT_SERVER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()
            print(f"✓ Connected to MQTT: {MQTT_SERVER}:{MQTT_PORT}")
        except Exception as e:
            print(f"✗ MQTT connection failed: {e}")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            print("✓ MQTT connection established!")
        else:
            print(f"✗ MQTT connection failed with code {rc}")
    
    def send_mqtt_alert(self, person_id, position, duration):
        """
        Send alert for individual person staying too long
        Format 1: Plain text 's'
        Format 2: JSON with person details
        """
        try:
            # 1. Kirim plain text 's'
            
            # 2. Kirim JSON detail
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = {
                "alert_type": "person_staying",
                "person_id": person_id,
                "duration_seconds": round(duration, 1),
                "position": position,
                "timestamp": timestamp,
                "message": f"Person {person_id} stayed for {duration:.1f} seconds at position {position}"
            }
            
            json_message = json.dumps(message)
            self.mqtt_client.publish(MQTT_TOPIC, json_message, qos=0)
            
            print(f"🚨 ALERT: Person {person_id} stayed for {duration:.1f}s at {position}")

            
            self.mqtt_client.publish(MQTT_TOPIC1, "s", qos=0)
            
        except Exception as e:
            print(f"Failed to send MQTT alert: {e}")
    
    def calculate_person_hash(self, bbox):
        """Create unique hash for a person based on position and size"""
        x1, y1, x2, y2 = bbox
        # Normalize coordinates to grid (10x10 grid)
        grid_x = int((x1 + x2) / 2 / 100)  # Adjust 100 based on your resolution
        grid_y = int((y1 + y2) / 2 / 100)
        width = x2 - x1
        height = y2 - y1
        
        # Create hash string
        hash_str = f"{grid_x}_{grid_y}_{int(width)}_{int(height)}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]
    
    def match_existing_person(self, bbox, current_time):
        """Match detection to existing tracked person or create new one"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        best_match_id = None
        min_distance = float('inf')
        
        for person_id, data in self.tracked_persons.items():
            # Calculate distance between centers
            dist_x = abs(center_x - data['last_center_x'])
            dist_y = abs(center_y - data['last_center_y'])
            distance = np.sqrt(dist_x**2 + dist_y**2)
            
            # Check if this is the same person (within threshold)
            if distance < STATIONARY_THRESHOLD * 2:  # Matching threshold
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = person_id
        
        if best_match_id is not None:
            # Update existing person
            person_data = self.tracked_persons[best_match_id]
            person_data['last_seen'] = current_time
            person_data['last_center_x'] = center_x
            person_data['last_center_y'] = center_y
            person_data['bbox'] = bbox
            
            # Calculate duration
            duration = current_time - person_data['first_seen']
            person_data['duration'] = duration
            
            # Check if needs alert
            if (duration >= TRACKING_TIME_THRESHOLD and 
                best_match_id not in self.alerted_persons):
                position = f"({int(center_x)},{int(center_y)})"
                self.send_mqtt_alert(best_match_id, position, duration)
                self.alerted_persons.add(best_match_id)
            
            return best_match_id, self.tracked_persons[best_match_id]
        else:
            # Create new person
            person_id = self.next_person_id
            self.tracked_persons[person_id] = {
                'person_id': person_id,
                'first_seen': current_time,
                'last_seen': current_time,
                'last_center_x': center_x,
                'last_center_y': center_y,
                'bbox': bbox,
                'duration': 0.0,
                'color': tuple(np.random.randint(0, 255, 3).tolist())  # Random color
            }
            self.next_person_id += 1
            print(f"👤 New person detected: ID {person_id}")
            return person_id, self.tracked_persons[person_id]
    
    def cleanup_old_tracks(self, current_time, max_age=10.0):
        """Remove tracks that haven't been seen for a while"""
        to_remove = []
        for person_id, data in self.tracked_persons.items():
            if current_time - data['last_seen'] > max_age:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            if person_id in self.alerted_persons:
                self.alerted_persons.remove(person_id)
            del self.tracked_persons[person_id]
    
    def process_frame(self, frame, frame_timestamp):
        """Process single frame for person tracking"""
        if frame is None:
            return frame, 0
        
        # Run inference
        results = self.model(frame, verbose=False)[0]
        
        processed_frame = frame.copy()
        current_time = time.time()
        
        # Process each detection
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls == self.person_class_id and conf >= DETECTION_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)
                
                # Match or create person track
                person_id, person_data = self.match_existing_person(bbox, current_time)
                
                # Draw bounding box with person-specific color
                color = person_data['color']
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw person ID and duration
                duration = person_data['duration']
                label = f"ID:{person_id} {duration:.1f}s"
                
                # Color code based on alert status
                if person_id in self.alerted_persons:
                    label_color = (0, 0, 255)  # Red for alerted
                    box_color = (0, 0, 255)
                elif duration >= TRACKING_TIME_THRESHOLD:
                    label_color = (0, 165, 255)  # Orange for threshold reached
                    box_color = (0, 165, 255)
                else:
                    label_color = (0, 255, 0)  # Green for normal
                    box_color = color
                
                # Update box color if needed
                if person_id in self.alerted_persons or duration >= TRACKING_TIME_THRESHOLD:
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), box_color, 3)
                
                # Draw label background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(processed_frame, 
                            (x1, y1 - label_height - baseline - 5),
                            (x1 + label_width, y1),
                            box_color, -1)
                
                # Draw label text
                cv2.putText(processed_frame, label,
                          (x1, y1 - baseline - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Cleanup old tracks
        self.cleanup_old_tracks(current_time)
        
        # Add statistics overlay
        active_count = len(self.tracked_persons)
        alerted_count = len(self.alerted_persons)
        
        # Draw info panel
        cv2.putText(processed_frame, f"Active Persons: {active_count}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(processed_frame, f"Alerted: {alerted_count}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Threshold: {TRACKING_TIME_THRESHOLD}s", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw legend
        cv2.putText(processed_frame, "Legend:", (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(processed_frame, "Green: Tracking", (frame.shape[1] - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(processed_frame, "Orange: >5s", (frame.shape[1] - 150, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(processed_frame, "Red: Alert Sent", (frame.shape[1] - 150, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Add timestamp
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, timestamp_str, 
                   (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return processed_frame, active_count
    
    def setup_video_writer(self, width, height, fps, output_path="tracking_output.mp4"):
        """Setup video writer for saving results"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"📹 Output will be saved to: {output_path}")
        return output_path
    
    def process_video_stream(self, video_source=0, save_output=True):
        """Main processing loop for person tracking"""
        print(f"\n📂 Processing: {video_source}")
        print(f"⚡ Alert: Person staying > {TRACKING_TIME_THRESHOLD} seconds")
        print(f"📏 Stationary threshold: {STATIONARY_THRESHOLD} pixels")
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"❌ Cannot open video source: {video_source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            fps = 30  # Default for webcam
        
        print(f"📊 Video: {width}x{height} @ {fps:.1f} FPS")
        if total_frames > 0:
            print(f"📈 Total frames: {total_frames}")
        
        # Setup output video
        output_file = None
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"person_tracking_{timestamp}.mp4"
            self.setup_video_writer(width, height, fps, output_file)
        
        start_time = time.time()
        frame_counter = 0
        last_log_time = start_time
        
        print("\n▶️ Starting person tracking...")
        print("   Press Ctrl+C to stop\n")
        
        try:
            while self.is_running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ End of video")
                    break
                
                frame_counter += 1
                frame_timestamp = time.time()
                
                # Process frame
                processed_frame, active_count = self.process_frame(frame, frame_timestamp)
                
                # Save frame to output video
                if self.output_video is not None:
                    self.output_video.write(processed_frame)
                
                # Log progress periodically
                current_time = time.time()
                if current_time - last_log_time >= 2.0:  # Log every 2 seconds
                    elapsed = current_time - start_time
                    fps_actual = frame_counter / elapsed
                    
                    # Get persons close to threshold
                    near_threshold = []
                    for person_id, data in self.tracked_persons.items():
                        if (TRACKING_TIME_THRESHOLD - 1 <= data['duration'] < TRACKING_TIME_THRESHOLD and
                            person_id not in self.alerted_persons):
                            near_threshold.append(person_id)
                    
                    log_msg = f"📊 Frame: {frame_counter}"
                    if total_frames > 0:
                        log_msg += f"/{total_frames}"
                    log_msg += f" | Active: {active_count} | FPS: {fps_actual:.1f}"
                    
                    if near_threshold:
                        log_msg += f" | Near threshold: {near_threshold}"
                    
                    print(log_msg)
                    last_log_time = current_time
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            # Cleanup
            cap.release()
            if self.output_video is not None:
                self.output_video.release()
                print(f"\n💾 Video saved: {output_file}")
            
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
            elapsed_total = time.time() - start_time
            print(f"\n📊 Tracking Summary:")
            print(f"   Frames processed: {frame_counter}")
            print(f"   Total time: {elapsed_total:.1f}s")
            print(f"   Average FPS: {frame_counter/elapsed_total:.1f}")
            print(f"   Total persons detected: {self.next_person_id - 1}")
            print(f"   Alerts sent: {len(self.alerted_persons)}")
            print(f"   MQTT disconnected")

def main():
    """Main function"""
    print("=" * 60)
    print("INDIVIDUAL PERSON TRACKING WITH MQTT ALERTS")
    print("=" * 60)
    print(f"Alert triggers when a person stays > {TRACKING_TIME_THRESHOLD} seconds")
    print("Each person tracked individually with unique ID")
    print()
    
    # Input options
    print("Input Options:")
    print("1. 📷 Webcam (default)")
    print("2. 📁 Video file")
    print("3. 🌐 RTSP stream")
    
    choice = input("\nSelect option (1-3) [1]: ").strip() or "1"
    
    # Load model
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    print(f"\n📥 Loading model from: {model_url}")
    
    try:
        tracker = PeopleTracker(model_url)
        
        if choice == "2":
            video_path = input("Enter video file path: ").strip()
            if video_path and os.path.exists(video_path):
                tracker.process_video_stream(video_path, save_output=True)
            else:
                print("❌ File not found. Using webcam instead.")
                tracker.process_video_stream(0, save_output=True)
        
        elif choice == "3":
            rtsp_url = input("Enter RTSP URL: ").strip()
            if rtsp_url:
                tracker.process_video_stream(rtsp_url, save_output=True)
            else:
                print("Using webcam")
                tracker.process_video_stream(0, save_output=True)
        
        else:
            # Webcam
            print("Using webcam...")
            tracker.process_video_stream(0, save_output=True)
            
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

if __name__ == "__main__":
    main()
