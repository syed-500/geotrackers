import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
import os


class WorkerTrackingSystem:
    def __init__(self):
        # Model Configuration
        self.YOLO_MODEL_PATH = 'yolov8s.pt'
        self.VIDEO_SOURCE = 'bui.mp4'
        self.CONFIDENCE_THRESHOLD = 0.5
        self.IOU_THRESHOLD = 0.45
        self.MAX_WORKERS = 5

        # Tracking Parameters
        self.ACTIVITY_THRESHOLD = 3  # Seconds to consider worker as inactive if not detected
        self.MIN_DETECTION_FRAMES = 5  # Minimum frames to confirm a worker
        self.TIME_WINDOW = 60  # Time window for activity calculation (seconds)

        # Window Configuration
        self.WINDOW_WIDTH = 1280  # Fixed window width
        self.WINDOW_HEIGHT = 720  # Fixed window height

        # Initialize tracking state
        self.roi = None
        self.drawing = False
        self.start_point = None
        self.worker_states = {}
        self.last_save_time = time.time()
        self.SAVE_INTERVAL = 60  # Save stats every 60 seconds

        # Name assignment mode
        self.name_assignment_mode = False
        self.selected_worker_id = None

        # Load previous session data if exists
        self.session_file = 'worker_session.pkl'
        self.load_session()

        self._initialize_systems()

    def load_session(self):
        """Load previous session data including total times"""
        if os.path.exists(self.session_file):
            with open(self.session_file, 'rb') as f:
                session_data = pickle.load(f)
                self.worker_states = session_data.get('worker_states', {})
                # Initialize total_time if not present
                for worker in self.worker_states.values():
                    if 'total_time' not in worker:
                        worker['total_time'] = worker.get('active_time', 0)

    def save_session(self):
        """Save current session data"""
        with open(self.session_file, 'wb') as f:
            pickle.dump({'worker_states': self.worker_states}, f)

    def _initialize_systems(self):
        """Initialize YOLO, DeepSORT, and video capture"""
        # Initialize YOLO
        self.model = YOLO(self.YOLO_MODEL_PATH)
        self.model.conf = self.CONFIDENCE_THRESHOLD
        self.model.iou = self.IOU_THRESHOLD

        # Initialize DeepSORT with optimized parameters
        self.tracker = DeepSort(
            max_age=30,
            n_init=self.MIN_DETECTION_FRAMES,
            nms_max_overlap=0.7,
            embedder="mobilenet",
            embedder_gpu=True,
            half=True,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
        )

        # Initialize video capture
        self.cap = cv2.VideoCapture(self.VIDEO_SOURCE)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source {self.VIDEO_SOURCE}")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

    def handle_mouse_event(self, event, x, y, flags, param):
        """Handle both ROI drawing and worker selection"""
        if self.name_assignment_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                # Find closest worker to click
                closest_worker = None
                min_distance = float('inf')
                for track_id, state in self.worker_states.items():
                    if 'current_bbox' in state:
                        bbox = state['current_bbox']
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                        if distance < min_distance:
                            min_distance = distance
                            closest_worker = track_id

                if closest_worker is not None and min_distance < 50:  # 50 pixel threshold
                    self.selected_worker_id = closest_worker
                    self.prompt_name_input()
        else:
            # Regular ROI drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.roi = (self.start_point[0], self.start_point[1], x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.roi = (self.start_point[0], self.start_point[1], x, y)

    def prompt_name_input(self):
        """Prompt for worker name input"""
        if self.selected_worker_id is not None:
            # Create a window for name input
            cv2.namedWindow("Name Input", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Name Input", 400, 100)

            # Create a blank image for the input window
            img = np.zeros((100, 400, 3), np.uint8)
            cv2.putText(img, "Press any key and enter name in console", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Name Input", img)
            cv2.waitKey(0)
            cv2.destroyWindow("Name Input")

            # Get name from console
            name = input(f"Enter name for Worker {self.selected_worker_id}: ")
            if name.strip():
                if self.selected_worker_id in self.worker_states:
                    self.worker_states[self.selected_worker_id]['name'] = name
                else:
                    self.worker_states[self.selected_worker_id] = {
                        'name': name,
                        'active_time': 0,
                        'total_time': 0,
                        'last_seen': time.time(),
                        'is_active': True,
                        'position_history': [],
                        'activity_windows': []
                    }

    def is_within_roi(self, bbox):
        """Check if any part of bbox is within ROI"""
        if not self.roi:
            return False

        x1, y1, x2, y2 = bbox
        roi_x1, roi_y1, roi_x2, roi_y2 = self.roi

        # Check for any overlap between the bounding box and the ROI
        return not (x2 < roi_x1 or x1 > roi_x2 or y2 < roi_y1 or y1 > roi_y2)

    def update_worker_state(self, track_id, bbox):
        """Update worker state with improved activity tracking"""
        current_time = time.time()

        if track_id not in self.worker_states:
            self.worker_states[track_id] = {
                'name': f"Worker {track_id}",  # Default name until assigned
                'active_time': 0,
                'total_time': 0,
                'last_seen': current_time,
                'is_active': True,
                'position_history': [],
                'activity_windows': [],
                'current_bbox': bbox
            }

        state = self.worker_states[track_id]
        state['current_bbox'] = bbox  # Store current bbox for selection

        # Update position history
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        state['position_history'].append((center_x, center_y, current_time))

        # Keep only recent positions
        state['position_history'] = [pos for pos in state['position_history']
                                     if current_time - pos[2] <= self.TIME_WINDOW]

        # Calculate movement and activity
        if len(state['position_history']) > 1:
            positions = state['position_history']
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
            distance = np.sqrt(dx ** 2 + dy ** 2)

            # Update activity windows
            state['activity_windows'].append((current_time, distance > 2))
            state['activity_windows'] = [w for w in state['activity_windows']
                                         if current_time - w[0] <= self.TIME_WINDOW]

            # Update active and total time
            if state['is_active']:
                elapsed_time = current_time - state['last_seen']
                state['active_time'] += elapsed_time
                state['total_time'] += elapsed_time

        state['last_seen'] = current_time
        state['is_active'] = True

    def draw_interface(self, frame, tracks):
        """Draw the tracking interface with enhanced visualization"""
        # Resize frame to fixed window size while maintaining aspect ratio
        frame = self.resize_frame(frame, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        # Calculate scaling factors
        scale_x = self.WINDOW_WIDTH / self.frame_width
        scale_y = self.WINDOW_HEIGHT / self.frame_height

        # Draw ROI
        if self.roi:
            scaled_roi = (
                int(self.roi[0] * scale_x),
                int(self.roi[1] * scale_y),
                int(self.roi[2] * scale_x),
                int(self.roi[3] * scale_y)
            )
            cv2.rectangle(frame, (scaled_roi[0], scaled_roi[1]), (scaled_roi[2], scaled_roi[3]),
                          (255, 0, 0), 2)
            cv2.putText(frame, "Working Area", (scaled_roi[0], scaled_roi[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw mode indicator
        mode_text = "Name Assignment Mode" if self.name_assignment_mode else "ROI Mode"
        cv2.putText(frame, f"Mode: {mode_text} (Press 'n' to toggle)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw worker tracking boxes and information
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)

            state = self.worker_states.get(track_id, {
                'name': f"Worker {track_id}",
                'active_time': 0,
                'total_time': 0,
                'activity_windows': []
            })

            # Calculate activity percentage
            if state['activity_windows']:
                active_windows = sum(1 for _, is_active in state['activity_windows'] if is_active)
                activity_percentage = (active_windows / len(state['activity_windows'])) * 100
            else:
                activity_percentage = 0

            # Color based on selection and ROI
            if track_id == self.selected_worker_id and self.name_assignment_mode:
                color = (255, 255, 0)  # Yellow for selected worker
            else:
                color = (0, 255, 0) if self.is_within_roi(bbox) else (0, 0, 255)

            # Adjust bounding box coordinates based on resizing
            scaled_bbox = (
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            )

            # Draw bounding box
            cv2.rectangle(frame, (scaled_bbox[0], scaled_bbox[1]), (scaled_bbox[2], scaled_bbox[3]), color, 2)

            # Draw worker information
            info_text = f"{state['name']} - {activity_percentage:.1f}% active"
            cv2.putText(frame, info_text, (scaled_bbox[0], scaled_bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw activity dashboard
        self._draw_dashboard(frame)

        return frame  # Return the resized and annotated frame

    def _draw_dashboard(self, frame):
        """Draw activity dashboard with detailed statistics"""
        # Background for dashboard
        dashboard_width = 350
        dashboard_height = min(len(self.worker_states) * 60 + 40, 300)
        cv2.rectangle(frame, (frame.shape[1] - dashboard_width, 0),
                      (frame.shape[1], dashboard_height),
                      (0, 0, 0), -1)

        # Title
        cv2.putText(frame, "Worker Activity Dashboard",
                    (frame.shape[1] - dashboard_width + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Worker statistics
        y_pos = 50
        for track_id, state in self.worker_states.items():
            if time.time() - state.get('last_seen', 0) > self.ACTIVITY_THRESHOLD:
                continue

            # Format active time
            active_time = state['active_time']
            active_hours = int(active_time // 3600)
            active_minutes = int((active_time % 3600) // 60)
            active_seconds = int(active_time % 60)

            # Format total time
            total_time = state['total_time']
            total_hours = int(total_time // 3600)
            total_minutes = int((total_time % 3600) // 60)
            total_seconds = int(total_time % 60)

            # Draw worker name
            cv2.putText(frame, f"{state['name']}",
                        (frame.shape[1] - dashboard_width + 10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw active time
            cv2.putText(frame, f"Active: {active_hours:02d}:{active_minutes:02d}:{active_seconds:02d}",
                        (frame.shape[1] - dashboard_width + 10, y_pos + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw total time
            cv2.putText(frame, f"Total: {total_hours:02d}:{total_minutes:02d}:{total_seconds:02d}",
                        (frame.shape[1] - dashboard_width + 10, y_pos + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            y_pos += 60

    def resize_frame(self, frame, window_width, window_height):
        """
        Resize frame to fit within window dimensions while maintaining aspect ratio.
        Adds black padding (letterboxing) if necessary.
        """
        frame_height, frame_width = frame.shape[:2]
        aspect_ratio_frame = frame_width / frame_height
        aspect_ratio_window = window_width / window_height

        if aspect_ratio_frame > aspect_ratio_window:
            # Frame is wider than window
            new_width = window_width
            new_height = int(window_width / aspect_ratio_frame)
        else:
            # Frame is taller than window
            new_height = window_height
            new_width = int(window_height * aspect_ratio_frame)

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Create a black background
        canvas = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # Compute top-left corner for centered placement
        x_offset = (window_width - new_width) // 2
        y_offset = (window_height - new_height) // 2

        # Place the resized frame onto the canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

        return canvas

    def run(self):
        """Main processing loop"""
        cv2.namedWindow("Worker Tracking System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Worker Tracking System", self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        cv2.setMouseCallback("Worker Tracking System", self.handle_mouse_event)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video stream or cannot fetch the frame.")
                    break

                # Run detection and tracking
                results = self.model(frame, stream=True)
                detections = []

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].item()
                        class_id = int(box.cls[0].item())

                        if class_id == 0 and conf >= self.CONFIDENCE_THRESHOLD:  # Person class
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            detections.append(([x1, y1, bbox_width, bbox_height], conf, class_id))

                # Update tracker
                tracks = self.tracker.update_tracks(detections, frame=frame)

                # Update worker states
                for track in tracks:
                    if track.is_confirmed():
                        bbox = track.to_tlbr()
                        if self.is_within_roi(bbox):
                            self.update_worker_state(track.track_id, bbox)

                # Draw interface and get the annotated frame
                annotated_frame = self.draw_interface(frame, tracks)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    # Toggle name assignment mode
                    self.name_assignment_mode = not self.name_assignment_mode
                    self.selected_worker_id = None
                    print(f"{'Entered' if self.name_assignment_mode else 'Exited'} name assignment mode")
                elif key == ord('s'):
                    # Save current session
                    self.save_session()
                    print("Session saved")

                # Show the frame
                cv2.imshow("Worker Tracking System", annotated_frame)

                # Save statistics periodically
                if time.time() - self.last_save_time >= self.SAVE_INTERVAL:
                    self.save_session()
                    self.last_save_time = time.time()

        finally:
            self.save_session()
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    tracking_system = WorkerTrackingSystem()
    tracking_system.run()
