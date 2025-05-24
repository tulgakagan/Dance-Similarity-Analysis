import cv2
import os
import logging

def extract_frames(video_path: str, output_dir: str, fps: int = 15):
    """
    Extracts frames from a video at a specified FPS.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the extracted frames.
        fps: Target frames per second to extract.
    """
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return 0

    #original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_fps = 30.0
    if original_fps == 0:
        logging.warning(f"Could not read original FPS from {video_path}. Assuming 30 FPS.")
        original_fps = 30 # Default fallback

    frame_interval = int(original_fps / fps)
    if frame_interval == 0: # Ensure frame_interval is at least 1
        frame_interval = 1
        logging.warning(f"Target FPS ({fps}) is higher than or equal to original FPS ({original_fps}). Extracting all frames.")


    frame_count, saved_count = 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    logging.info(f"Extracted {saved_count} frames from {video_path} into '{output_dir}' at approx {fps} FPS.")
    return saved_count