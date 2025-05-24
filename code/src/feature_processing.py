import numpy as np
import logging


def normalize_poses(keypoints_all: np.ndarray) -> np.ndarray:
    """Normalize poses by centering at hips and scaling by shoulder width."""
    if keypoints_all.ndim != 3 or keypoints_all.shape[2] < 2:
        logging.error("Invalid keypoints_all shape for normalization. Expected (num_frames, 33, 2+).")
        return np.array([])

    keypoints_copy = np.copy(keypoints_all) 
    num_frames = keypoints_copy.shape[0]
    normalized_poses = np.zeros((num_frames, 33, 2))

    for i in range(num_frames):
        keypoints = keypoints_copy[i]

        # Ensure keypoint indices are valid for the data
        l_shoulder = keypoints[11][:2]
        r_shoulder = keypoints[12][:2]
        l_hip = keypoints[23][:2]
        r_hip = keypoints[24][:2]

        center = (l_hip + r_hip) / 2.0
        scale = np.linalg.norm(l_shoulder - r_shoulder)
        
        # Add epsilon for numerical stability if scale is zero
        if scale < 1e-6: 
            logging.warning(f"Frame {i}: Shoulder width is close to zero. Normalization might be unstable.")
            scale = 1e-6 

        normalized_poses[i] = (keypoints[:, :2] - center) / scale
    
    logging.info(f"Normalized {num_frames} pose frames.")
    return normalized_poses

def _calculate_angle_between_vectors(vec_a, vec_b):
    # Calculate cosine of the angle
    cosine_angle = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-6) # Add epsilon
    # Clip to avoid domain errors with arccos due to floating point inaccuracies
    angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) 
    return np.degrees(angle_rad)

def compute_joint_angles(normalized_pose_data: np.ndarray, joint_triplets: dict) -> dict:
    """
    Computes joint angles for each frame from normalized 2D pose data.

    Args:
        normalized_pose_data: Numpy array of shape (num_frames, 33, 2).
        joint_triplets: Dictionary defining the joint triplets for angle calculation.
                        e.g., {"left_elbow": (11, 13, 15)}

    Returns:
        A dictionary where keys are joint names and values are lists of angles (in degrees)
        for each frame.
    """
    if normalized_pose_data.ndim != 3 or normalized_pose_data.shape[2] != 2:
         logging.error("Invalid normalized_pose_data shape for angle computation. Expected (num_frames, 33, 2).")
         return {}

    num_frames = normalized_pose_data.shape[0]
    angles_all_frames = {name: [] for name in joint_triplets}

    for i in range(num_frames):
        frame_keypoints = normalized_pose_data[i]
        for name, (idx_a, idx_b, idx_c) in joint_triplets.items():
            # Ensure indices are valid for frame_keypoints
            if not (0 <= idx_a < 33 and 0 <= idx_b < 33 and 0 <= idx_c < 33):
                logging.warning(f"Invalid keypoint index for joint {name} in frame {i}. Skipping angle.")
                angles_all_frames[name].append(np.nan) # Or some other placeholder
                continue

            p_a = frame_keypoints[idx_a]
            p_b = frame_keypoints[idx_b]
            p_c = frame_keypoints[idx_c]

            # Create vectors from point b to a, and b to c
            vec_ba = p_a - p_b 
            vec_bc = p_c - p_b

            angle = _calculate_angle_between_vectors(vec_ba, vec_bc)
            angles_all_frames[name].append(angle)
    
    logging.info(f"Computed joint angles for {num_frames} frames for {len(joint_triplets)} joint types.")
    return angles_all_frames