import cv2
from court_detector import CourtDetector
from player_detector import PlayerPoseDetector

# 1. Initialize both models once to load them into the GPU
print("Loading Court PyTorch model...")
court_detector = CourtDetector(weights_path='Deuce-Ex-Machina/Court Detector/model_tennis_court_det.pt')

print("Loading YOLO Player model...")
# LOWERED CONFIDENCE TO 0.15 HERE TO CATCH THE FAR PLAYER
pose_detector = PlayerPoseDetector(model_path="yolov8s-pose.pt", conf=0.15)

# 2. Setup video input and output
input_file = "Deuce-Ex-Machina/Test_media/Clipto AI Video Downloader CRAZY 70-Shot Rally! _ Australian Open 2023.mp4"
output_file = "output_combined.mp4"

cap = cv2.VideoCapture(input_file)
if not cap.isOpened():
    raise ValueError(f"Error: Could not open video at {input_file}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# We force the output to 1280x720 so the court math aligns perfectly
width, height = 1280, 720
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 3. State variables for Court Smoothing (Exponential Moving Average)
frame_count = 0
detect_interval = 5
smoothing_factor = 0.2
current_keypoints = None 
target_keypoints = None
latest_matrix = None # ADDED VARIABLE TO CACHE THE HOMOGRAPHY MATRIX

print(f"Processing {total_frames} frames...")

# 4. Master Video Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # --- COURT PIPELINE ---
    # Update target coordinates at the specified interval
    if frame_count % detect_interval == 0 or target_keypoints is None:
        # UNPACK 3 VARIABLES TO GET THE MATRIX
        _, target_keypoints, latest_matrix = court_detector.detect_frame(frame, use_refine_kps=False, use_homography=True)
        if current_keypoints is None:
            current_keypoints = [{"id": kp["id"], "x": kp["x"], "y": kp["y"]} for kp in target_keypoints]

    # Smoothly interpolate current coordinates toward target coordinates
    target_dict = {kp["id"]: kp for kp in target_keypoints}
    for curr in current_keypoints:
        if curr["id"] in target_dict:
            targ = target_dict[curr["id"]]
            curr["x"] += smoothing_factor * (targ["x"] - curr["x"])
            curr["y"] += smoothing_factor * (targ["y"] - curr["y"])

    # Resize the raw frame to 1280x720 before drawing so the coordinates match the math
    frame_resized = cv2.resize(frame, (width, height))
    
    # --- PLAYER PIPELINE ---
    # PASS THE CACHED MATRIX INTO THE PLAYER DETECTOR TO FILTER BY HOMOGRAPHY
    frame_annotated = pose_detector.process_frame(frame_resized, matrix_trans=latest_matrix)
    
    # --- COMBINED DRAWING ---
    # Draw the smoothed court points on the same frame that YOLO just annotated
    for kp in current_keypoints:
        cv2.circle(frame_annotated, (int(kp["x"]), int(kp["y"])), radius=5, color=(0, 0, 255), thickness=-1)
        
    out.write(frame_annotated)
    frame_count += 1
    
    if frame_count % 30 == 0:
        print(f"Processed {frame_count}/{total_frames} frames...")

# 5. Clean up
cap.release()
out.release()
print(f"Successfully saved combined video to {output_file}")