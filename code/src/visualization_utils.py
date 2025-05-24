import cv2
import numpy as np
import os
import logging
from tqdm import tqdm

def create_angle_hud_frame(original_frame: np.ndarray, 
                           keypoints_frame_normalized: np.ndarray, # (33,3) normalized x,y,visibility
                           current_frame_angles: dict, # {"left_elbow": angle, ...} for this frame
                           joint_triplets_for_display: dict,
                           hud_margin_width: int = 180) -> np.ndarray:
    """Creates a frame with the original video on the left and an angle HUD on the right."""
    h, w = original_frame.shape[:2]
    
    # Create a wider canvas with margin on right for HUD
    canvas = np.ones((h, w + hud_margin_width, 3), dtype=np.uint8) * 255  # white background
    canvas[:, :w] = original_frame # Place original frame on the left
    
    # Draw skeleton on the image part (canvas[:, :w])
    # Points on skeleton are based on keypoints_frame_normalized relative to original_frame dimensions
    for i in range(keypoints_frame_normalized.shape[0]):
        if keypoints_frame_normalized[i, 2] > 0.5: # visibility
            pt_x = int(keypoints_frame_normalized[i, 0] * w)
            pt_y = int(keypoints_frame_normalized[i, 1] * h)
            cv2.circle(canvas, (pt_x, pt_y), 4, (0, 255, 0), -1)
    
    # Draw lines for specified triplets on the image part
    for _name, (a_idx, b_idx, c_idx) in joint_triplets_for_display.items():
        a, b, c = keypoints_frame_normalized[a_idx], keypoints_frame_normalized[b_idx], keypoints_frame_normalized[c_idx]
        if min(a[2], b[2], c[2]) < 0.5: # Check visibility
            continue
        a_xy = np.array([a[0] * w, a[1] * h]).astype(int)
        b_xy = np.array([b[0] * w, b[1] * h]).astype(int)
        c_xy = np.array([c[0] * w, c[1] * h]).astype(int)
        cv2.line(canvas, tuple(a_xy), tuple(b_xy), (255, 0, 0), 2)
        cv2.line(canvas, tuple(b_xy), tuple(c_xy), (255, 0, 0), 2)

    # Draw angle text in right margin (HUD)
    y_offset = 24
    line_height = 24
    for name, angle_value in current_frame_angles.items():
        # Ensure the joint name is in displayable format
        display_name = name.replace("_", " ").title()
        if not np.isnan(angle_value):
            text = f"{display_name}: {angle_value:.1f}"
            text_pos = (w + 10, y_offset) # Position in the margin
            cv2.putText(canvas, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            y_offset += line_height
    
    return canvas

def get_score_color(score: float) -> tuple:
    """Returns BGR color based on similarity score."""
    if score >= 85:
        return (0, 200, 0)  # Green
    elif score >= 70: # 
        return (0, 200, 200)
    else:
        return (0, 0, 255)


def create_comparison_frame(avatar_hud_frame: np.ndarray, 
                            user_hud_frame: np.ndarray, 
                            joint_similarity_scores: dict, # {"left_elbow": score, ...}
                            target_height: int = None) -> np.ndarray:
    """
    Combines avatar and user HUD frames side-by-side with a similarity score bar at the bottom.
    Based on logic from percent_match_hud_overlay.py.
    """
    # Resize avatar frame to match user frame height if target_height is specified or user_hud_frame is provided
    if target_height is None:
        target_height = user_hud_frame.shape[0]
    
    if avatar_hud_frame.shape[0] != target_height:
         scale_factor_avatar = target_height / avatar_hud_frame.shape[0]
         avatar_width_resized = int(avatar_hud_frame.shape[1] * scale_factor_avatar)
         avatar_resized = cv2.resize(avatar_hud_frame, (avatar_width_resized, target_height))
    else:
         avatar_resized = avatar_hud_frame

    if user_hud_frame.shape[0] != target_height:
         scale_factor_user = target_height / user_hud_frame.shape[0]
         user_width_resized = int(user_hud_frame.shape[1] * scale_factor_user)
         user_resized = cv2.resize(user_hud_frame, (user_width_resized, target_height))
    else:
         user_resized = user_hud_frame
         
    top_row = cv2.hconcat([avatar_resized, user_resized])
    combined_width = top_row.shape[1]

    # Create dynamic similarity row (bottom_row)
    bottom_row_height = 80
    bottom_row = np.ones((bottom_row_height, combined_width, 3), dtype=np.uint8) * 255 # White background
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Render text on bottom row
    x_offset = 20
    y_text_pos = 50 # y-position for text within bottom_row
    
    # Calculate spacing based on number of joints to display
    num_joints_to_display = len(joint_similarity_scores)
    text_spacing = 160 
    if num_joints_to_display * text_spacing > combined_width - x_offset :
        text_spacing = (combined_width - x_offset) / (num_joints_to_display +1)


    for joint_name, score_value in joint_similarity_scores.items():
        display_name = joint_name.replace("_", " ").title()
        text = f"{display_name}: {score_value}%"
        color = get_score_color(score_value)
        cv2.putText(bottom_row, text, (int(x_offset), y_text_pos), font, 0.55, color, 2, cv2.LINE_AA)
        x_offset += text_spacing
    
    final_frame = cv2.vconcat([top_row, bottom_row])
    return final_frame


def save_frames_to_video(input_frame_dir: str, output_video_path: str, fps: int):
    """
    Reads frames from a directory and saves them as an MP4 video.
    Based on frames_to_video from human_video_visualization.py
    """
    frame_files = sorted([f for f in os.listdir(input_frame_dir) if f.lower().endswith((".jpg", ".png"))])
    if not frame_files:
        logging.error(f"No image frames found in {input_frame_dir} to create video.")
        return

    sample_frame_path = os.path.join(input_frame_dir, frame_files[0])
    sample_frame = cv2.imread(sample_frame_path)
    if sample_frame is None:
        logging.error(f"Could not read sample frame: {sample_frame_path}")
        return
    
    h, w = sample_frame.shape[:2]

    # Use 'mp4v' fourcc for .mp4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    if not out_video.isOpened():
         logging.error(f"Failed to open VideoWriter for {output_video_path}")
         return

    for file_name in tqdm(frame_files, desc=f"Saving video {os.path.basename(output_video_path)}"):
        img_path = os.path.join(input_frame_dir, file_name)
        img = cv2.imread(img_path)
        if img is not None:
            out_video.write(img)
        else:
            logging.warning(f"Skipping problematic frame: {img_path}")
    
    out_video.release()
    logging.info(f"Video saved to: {output_video_path}")