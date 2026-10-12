import cv2 
import numpy as np
from pathlib import Path
from ultralytics import YOLO 

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov8s-pose.pt"

class PlayerPoseDetector: 
    # Lowered confidence from 0.3 to 0.15 to better capture the far-side player 
    def __init__(self, model_path=DEFAULT_MODEL_PATH, conf=0.15, imgsz=1280): 
        """Initializes the YOLO model once to keep video processing fast.""" 
        self.model = YOLO(str(model_path)) 
        self.conf = conf 
        self.imgsz = imgsz 
        
        # YOLOv8 COCO pose mapping (17 keypoints) 
        self.keypoint_names = [ 
            "Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear", 
            "L-Should", "R-Should", "L-Elb", "R-Elb", "L-Wrist", "R-Wrist", 
            "L-Hip", "R-Hip", "L-Knee", "R-Knee", "L-Ankle", "R-Ankle" 
        ] 

        # Court dimensions including margins based on court_reference.py[cite: 7]
        self.ref_x_min = 0
        self.ref_x_max = 1117 + (274 * 2) # court_width + right_left_border * 2[cite: 7]
        self.ref_y_min = 0
        self.ref_y_max = 2408 + (549 * 2) # court_height + top_bottom_border * 2[cite: 7]

    def is_in_playing_area(self, foot_x, foot_y, inv_matrix):
        """Projects image coordinates to reference court and checks against boundaries."""
        if inv_matrix is None:
            return True # Fallback to drawing everyone if the court matrix isn't provided
            
        # Format point for perspectiveTransform
        pt_array = np.array([[[foot_x, foot_y]]], dtype=np.float32)
        ref_pt = cv2.perspectiveTransform(pt_array, inv_matrix)[0][0]
        
        # Check if transformed coordinates fall inside the extended court plane
        return (self.ref_x_min <= ref_pt[0] <= self.ref_x_max) and \
               (self.ref_y_min <= ref_pt[1] <= self.ref_y_max)

    def process_frame(self, frame, matrix_trans=None):
        """Runs inference on a single frame, filters by court area, and draws keypoints."""
        # YOLO requires the image as a numpy array, which OpenCV frames already are
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False) 
        result = results[0] 
        
        # If no people are detected, return the original frame 
        if result.keypoints is None: 
            return frame 
            
        keypoints = result.keypoints.xy.cpu().numpy() 
        
        # Calculate inverse homography matrix once per frame
        inv_matrix = np.linalg.inv(matrix_trans) if matrix_trans is not None else None
        
        for person_idx, person_kpts in enumerate(keypoints): 
            valid_kpts = [kpt for kpt in person_kpts if kpt[0] > 0 and kpt[1] > 0]
            if not valid_kpts:
                continue
                
            # Determine player foot position (average of visible ankles, or lowest body part)
            ankles = [person_kpts[15], person_kpts[16]] # L-Ankle, R-Ankle 
            valid_ankles = [a for a in ankles if a[0] > 0]
            
            if valid_ankles:
                foot_x = np.mean([a[0] for a in valid_ankles])
                foot_y = np.mean([a[1] for a in valid_ankles])
            else:
                foot_x = np.mean([kpt[0] for kpt in valid_kpts])
                foot_y = max([kpt[1] for kpt in valid_kpts])
                
            # Spatial Filter: Skip if foot coordinates project outside the court boundaries
            if not self.is_in_playing_area(foot_x, foot_y, inv_matrix):
                continue

            for kpt_idx, kpt in enumerate(person_kpts): 
                x, y = int(kpt[0]), int(kpt[1]) 
                
                # YOLO returns [0., 0.] for obscured/out-of-frame points 
                if x > 0 and y > 0: 
                    # Draw a solid red dot 
                    cv2.circle(frame, (x, y), radius=4, color=(0, 0, 255), thickness=-1) 
                    
                    # Label the point with the person ID and body part name 
                    label = f"P{person_idx} {self.keypoint_names[kpt_idx]}" 
                    cv2.putText(frame, label, (x + 5, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1) 
                    
        return frame 

    def process_video(self, input_path, output_path, court_detector=None):
        """Reads a video, processes it frame-by-frame, and writes the output.""" 
        cap = cv2.VideoCapture(input_path) 
        if not cap.isOpened(): 
            raise ValueError(f"Error opening video stream or file: {input_path}") 
            
        fps = int(cap.get(cv2.CAP_PROP_FPS)) 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) 
        
        print(f"Processing {total_frames} frames from {input_path}...") 
        frame_count = 0 
        
        while True: 
            ret, frame = cap.read() 
            if not ret: 
                break 
            
            matrix_trans = None
            # Extract the matrix if a CourtDetector instance was passed in
            if court_detector is not None:
                _, _, matrix_trans = court_detector.detect_frame(frame, use_refine_kps=False, use_homography=True)
                
            annotated_frame = self.process_frame(frame, matrix_trans=matrix_trans)
            out.write(annotated_frame) 
            
            frame_count += 1 
            if frame_count % 30 == 0: 
                print(f"Processed {frame_count}/{total_frames} frames...") 
                
        cap.release() 
        out.release() 
        print(f"Successfully saved annotated video to {output_path}") 
