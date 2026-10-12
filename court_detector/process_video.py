import cv2  

class CourtVideoProcessor:  
    def __init__(self, court_detector, player_detector):
        """
        Takes already initialized detector instances.
        This prevents reloading the neural network weights for every new video.
        """
        self.court_detector = court_detector
        self.player_detector = player_detector

    def process(self, input_path, output_path, detect_interval=5, smoothing_factor=0.2):  
        """Processes a video and saves the annotated output to the disk."""  
        cap = cv2.VideoCapture(input_path)  
        if not cap.isOpened():  
            raise ValueError(f"Error: Could not open video at {input_path}")  

        fps = int(cap.get(cv2.CAP_PROP_FPS))  
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
        
        # Force the output to the 1280x720 coordinate space used by the model  
        width, height = 1280, 720  
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  

        print(f"Processing {total_frames} frames from {input_path}...")  
        
        frame_count = 0  
        current_keypoints = None  
        target_keypoints = None  
        latest_matrix = None

        while True:  
            ret, frame = cap.read()  
            if not ret:  
                break  
                
            # 1. Update TARGET coordinates and Homography Matrix at the specified interval
            if frame_count % detect_interval == 0 or target_keypoints is None:  
                # Unpack the 3 variables based on our court_detector.py update
                _, target_keypoints, latest_matrix = self.court_detector.detect_frame(frame, use_refine_kps=False, use_homography=True)
                
                # Snap instantly on the very first frame  
                if current_keypoints is None:  
                    current_keypoints = [{"id": kp["id"], "x": kp["x"], "y": kp["y"]} for kp in target_keypoints]  

            # 2. Smoothly interpolate CURRENT coordinates toward TARGET coordinates  
            target_dict = {kp["id"]: kp for kp in target_keypoints}  
            
            for curr in current_keypoints:  
                if curr["id"] in target_dict:  
                    targ = target_dict[curr["id"]]  
                    curr["x"] += smoothing_factor * (targ["x"] - curr["x"])  
                    curr["y"] += smoothing_factor * (targ["y"] - curr["y"])  
                
            # 3. Draw Court Keypoints
            output_frame = cv2.resize(frame, (width, height))  
            for kp in current_keypoints:  
                cv2.circle(output_frame, (int(kp["x"]), int(kp["y"])), radius=5, color=(0, 0, 255), thickness=-1)  
            
            # 4. Process Players (using the cached homography matrix to filter out non-players)
            output_frame = self.player_detector.process_frame(output_frame, matrix_trans=latest_matrix)
                
            # 5. Write Frame
            out.write(output_frame)  
            frame_count += 1  
            
            if frame_count % 100 == 0:  
                print(f"Processed {frame_count}/{total_frames} frames...")  

        cap.release()  
        out.release()  
        print(f"Successfully saved to {output_path}")  