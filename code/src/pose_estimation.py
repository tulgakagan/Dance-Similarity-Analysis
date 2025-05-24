import mediapipe as mp
import cv2
import numpy as np
import os
from tqdm import tqdm

def extract_pose_from_frames(frames_dir, model_complexity=1, is_avatar=False, batch_size=16):
    """
    Extract pose keypoints from frames using MediaPipe Pose.

    Args:
        frames_dir (str): Directory containing the frame images
        model_complexity (int): MediaPipe model complexity (0, 1, or 2)
        is_avatar (bool): Whether the frames contain an avatar/tutorial video
        batch_size (int): Number of frames to process in one batch
        
    Returns:
        np.ndarray: Array of keypoints with shape (n_frames, 33, 3)
    """
    if is_avatar:
        static_image_mode = False
        min_detection_confidence = 0.25 
        min_tracking_confidence  = 0.25
        model_complexity         = 2
    else:
        static_image_mode = True
        min_detection_confidence = 0.5
        min_tracking_confidence  = 0.5

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )

    # Get all image files (jpg and png)
    frame_files = sorted([f for f in os.listdir(frames_dir) 
                         if f.lower().endswith((".jpg", ".png"))])
    
    if not frame_files:
        print(f"No frame files found in directory: {frames_dir}")
        return np.array([])
    
    keypoints_all = []
    
    # Set description based on video type
    desc = "Extracting avatar poses" if is_avatar else "Extracting human poses"
    
    # Process frames in batches to improve performance
    for i in range(0, len(frame_files), batch_size):
        batch_files = frame_files[i:i+batch_size]
        
        for file in tqdm(batch_files, desc=f"{desc} (batch {i//batch_size + 1}/{(len(frame_files)+batch_size-1)//batch_size})"):
            image_path = os.path.join(frames_dir, file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image file: {image_path}")
                # Use zeros for failed image reads
                keypoints = np.zeros((33, 3))
                keypoints_all.append(keypoints)
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                keypoints = np.array([[lm.x, lm.y, lm.visibility] for lm in landmarks])
            else:
                # If pose detection fails, use zeros
                keypoints = np.zeros((33, 3))

            keypoints_all.append(keypoints)

    pose.close()
    keypoints_all = np.array(keypoints_all)  # shape: (n_frames, 33, 3)
    return keypoints_all