import torch
import cv2
import numpy as np
import torch.nn.functional as F
from tracknet import BallTrackerNet
from postprocess import postprocess, refine_kps
from homography import get_trans_matrix, refer_kps

class CourtDetector:
    def __init__(self, weights_path):
        """Initializes the architecture and loads weights into memory once."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = BallTrackerNet(out_channels=15)
        
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def detect(self, image_path, use_refine_kps=True, use_homography=True):
        """Processes an image from a file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        return self.detect_frame(image, use_refine_kps, use_homography)

    def detect_frame(self, image, use_refine_kps=False, use_homography=True):
        """Processes a raw OpenCV frame array directly from memory."""
        # Force the base image to match the author's native dataset resolution (1280x720)
        image = cv2.resize(image, (1280, 720))
        
        # The neural net still requires the 640x360 downscaled version
        img = cv2.resize(image, (640, 360))
        
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)
        
        with torch.no_grad():
            out = self.model(inp.float().to(self.device))[0]
            pred = F.sigmoid(out).detach().cpu().numpy()
            
        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
            
            if use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
                x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
                
            points.append((x_pred, y_pred))
            
        if use_homography:
            matrix_trans = get_trans_matrix(points)
            if matrix_trans is not None:
                points_transformed = cv2.perspectiveTransform(refer_kps, matrix_trans)
                points = [np.squeeze(x) for x in points_transformed]

        keypoints = []
        for i, pt in enumerate(points):
            if pt[0] is not None: 
                keypoints.append({
                    "id": i,
                    "x": int(pt[0]),
                    "y": int(pt[1])
                })
                
        return image, keypoints, matrix_trans