import os
import argparse
import logging
import shutil
import numpy as np
import cv2
import yaml
from tqdm import tqdm  # For progress bars
import multiprocessing as mp # For HUD frame generation

# Utility modules from the src directory
from frame_utils import extract_frames
from pose_estimation import extract_pose_from_frames
from feature_processing import normalize_poses, compute_joint_angles
from comparison_utils import (compute_windowed_cosine_similarity,
                              compute_correlation_similarity_score,
                              compute_mae_similarity_score)
from visualization_utils import (
    create_angle_hud_frame,
    create_comparison_frame,
    save_frames_to_video
)

# --- Hardcoded Path Configuration ---
BASE_DATA_DIR = "data"
OUTPUT_DIR = "output"
RAW_VIDEOS_DIR = os.path.join(BASE_DATA_DIR, "raw_videos")
TUTORIAL_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "tutorial")
USER_VIDEO_DIR = os.path.join(BASE_DATA_DIR, "user")

# Standard file naming suffixes
FRAMES_SUFFIX = "frames"
KEYPOINTS_SUFFIX = "keypoints.npy"
NORM_KEYPOINTS_SUFFIX = "normalized_keypoints.npy"
ANGLES_SUFFIX = "joint_angles.npy"
ANGLE_HUD_FRAMES_SUFFIX = "hud_frames"
ANGLE_HUD_VIDEO_SUFFIX = "hud_video.mp4"
FINAL_COMPARISON_VIDEO_NAME = "final_comparison.mp4"

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def ensure_dir(directory_path):
    os.makedirs(directory_path, exist_ok=True)
    logging.info(f"Ensured directory exists: {directory_path}")

def process_single_video(video_input_path: str, video_name_type: str, config: dict):
    """
    Processes a single video: frame extraction, pose estimation, feature processing.
    """
    # This is the actual filename without extension, e.g., "my_dance_tutorial"
    video_file_name_no_ext = os.path.splitext(os.path.basename(video_input_path))[0]
    
    base_output_dir = config["data_paths"]["base_output_dir"]
    # video_type_specific_output_dir is "output/tutorial" or "output/user"
    video_type_specific_output_dir = os.path.join(base_output_dir, video_name_type)
    ensure_dir(video_type_specific_output_dir)
    
    # video_processing_dir is "output/tutorial/my_dance_tutorial"
    video_processing_dir = os.path.join(video_type_specific_output_dir, video_file_name_no_ext)
    ensure_dir(video_processing_dir)

    if not video_input_path or not os.path.exists(video_input_path):
        logging.error(f"Failed to obtain video file for {video_name_type} from {video_input_path}")
        return None
    logging.info(f"Using video file: {video_name_type}")

    frames_dir = os.path.join(video_processing_dir, config["data_paths"]["frames_suffix"])
    ensure_dir(frames_dir)
    logging.info(f"Extracting frames for {video_name_type} to {frames_dir}...")
    num_extracted = extract_frames(video_input_path, frames_dir, fps=config["fps"])
    if num_extracted == 0:
        logging.error(f"No frames extracted for {video_name_type}. Stopping processing for this video.")
        return None

    logging.info(f"Extracting pose keypoints for {video_name_type} from {frames_dir}...")
    is_avatar = (video_name_type == "tutorial")
    if is_avatar:
        logging.info("Using enhanced avatar detection mode for tutorial video")
    raw_keypoints_data = extract_pose_from_frames(
        frames_dir,
        model_complexity=config["pose_model_complexity"],
        is_avatar=is_avatar,
        batch_size=config.get("batch_size", 16)
    )
    if raw_keypoints_data.size == 0:
        logging.error(f"Pose estimation failed for {video_name_type}. No keypoints extracted.")
        return None
    keypoints_path = os.path.join(video_processing_dir, config["data_paths"]["keypoints_suffix"])
    np.save(keypoints_path, raw_keypoints_data)
    logging.info(f"Saved raw keypoints for {video_name_type} to {keypoints_path}")

    logging.info(f"Normalizing poses for {video_name_type}...")
    normalized_keypoints_data = normalize_poses(raw_keypoints_data)
    if normalized_keypoints_data.size == 0:
        logging.error(f"Pose normalization failed for {video_name_type}.")
        return None
    norm_keypoints_path = os.path.join(video_processing_dir, config["data_paths"]["norm_keypoints_suffix"])
    np.save(norm_keypoints_path, normalized_keypoints_data)
    logging.info(f"Saved normalized keypoints for {video_name_type} to {norm_keypoints_path}")

    logging.info(f"Calculating joint angles for {video_name_type}...")
    joint_angles_data = compute_joint_angles(normalized_keypoints_data, config["joint_triplets"])
    if not joint_angles_data:
        logging.error(f"Joint angle calculation failed for {video_name_type}.")
        return None
    angles_path = os.path.join(video_processing_dir, config["data_paths"]["angles_suffix"])
    np.save(angles_path, joint_angles_data)
    logging.info(f"Saved joint angles for {video_name_type} to {angles_path}")

    return {
        "video_type": video_name_type, # "tutorial" or "user"
        "video_file_name_no_ext": video_file_name_no_ext, # Crucial key for logging
        "processing_dir": video_processing_dir,
        "original_frames_dir": frames_dir,
        "raw_keypoints_path": keypoints_path,
        "normalized_keypoints_path": norm_keypoints_path,
        "joint_angles_path": angles_path,
        "num_frames": raw_keypoints_data.shape[0]
    }

def _process_hud_frame_helper(args_tuple):
    frame_filename, original_frames_dir, keypoints_for_frame, \
    angles_for_frame_display, joint_triplets_config, \
    hud_margin_width_config, hud_frames_output_dir_for_frame = args_tuple
    
    original_frame_path = os.path.join(original_frames_dir, frame_filename)
    original_frame = cv2.imread(original_frame_path)
    if original_frame is None:
        logging.warning(f"Could not read original frame for HUD: {original_frame_path}")
        return False
        
    hud_frame = create_angle_hud_frame(
        original_frame, keypoints_for_frame, angles_for_frame_display,
        joint_triplets_config, hud_margin_width=hud_margin_width_config
    )
    output_path = os.path.join(hud_frames_output_dir_for_frame, frame_filename)
    cv2.imwrite(output_path, hud_frame)
    return True

def generate_angle_hud_video(video_data: dict, config: dict):
    video_name_type = video_data["video_type"]
    video_processing_dir = video_data["processing_dir"]
    original_frames_dir = video_data["original_frames_dir"]
    raw_keypoints_all_frames = np.load(video_data["raw_keypoints_path"])
    joint_angles_all_frames_dict = np.load(video_data["joint_angles_path"], allow_pickle=True).item()

    hud_frames_output_dir = os.path.join(video_processing_dir, config["data_paths"]["angle_hud_frames_suffix"])
    ensure_dir(hud_frames_output_dir)
    
    logging.info(f"Generating angle HUD frames for {video_name_type} in {hud_frames_output_dir}...")
    source_frame_files = sorted([f for f in os.listdir(original_frames_dir) if f.lower().endswith((".jpg", ".png"))])
    num_frames_to_process = min(len(source_frame_files), raw_keypoints_all_frames.shape[0])

    if len(source_frame_files) != raw_keypoints_all_frames.shape[0]:
        logging.warning(f"Mismatch in frame count ({len(source_frame_files)}) and keypoint count ({raw_keypoints_all_frames.shape[0]}) for {video_name_type}. Processing {num_frames_to_process} frames.")
    
    map_args = []
    for i in range(num_frames_to_process):
        frame_filename = source_frame_files[i]
        current_frame_angles_display = {
            joint: joint_angles_all_frames_dict[joint][i] if joint in joint_angles_all_frames_dict and i < len(joint_angles_all_frames_dict[joint]) else np.nan
            for joint in config["joint_triplets"].keys()
        }
        map_args.append((
            frame_filename, original_frames_dir, raw_keypoints_all_frames[i],
            current_frame_angles_display, config["joint_triplets"],
            config["hud_margin_width"], hud_frames_output_dir
        ))
    
    num_processors = min(mp.cpu_count(), config.get("max_workers_hud", mp.cpu_count()))
    logging.info(f"Using {num_processors} processes for HUD frame generation of {video_name_type}")
    
    successful_frames = 0
    with mp.Pool(processes=num_processors) as pool:
        results = list(tqdm(pool.imap(_process_hud_frame_helper, map_args), total=len(map_args), desc=f"Generating HUD frames for {video_name_type}"))
        successful_frames = sum(1 for r in results if r)

    logging.info(f"Generated {successful_frames}/{len(map_args)} HUD frames for {video_name_type}")
    if successful_frames == 0 and len(map_args) > 0:
        logging.error(f"No HUD frames were successfully generated for {video_name_type}.")
        return None, None

    hud_video_output_path = os.path.join(video_processing_dir, config["data_paths"]["angle_hud_video_suffix"])
    save_frames_to_video(hud_frames_output_dir, hud_video_output_path, config["fps"])
    logging.info(f"Angle HUD video for {video_name_type} saved to {hud_video_output_path}")
    return hud_frames_output_dir, hud_video_output_path

# --- Scoring Helper Functions ---
def _prepare_sequences_for_comparison(seq_a_raw: list, seq_b_raw: list):
    seq_a_np = np.array(seq_a_raw, dtype=np.float32)
    seq_b_np = np.array(seq_b_raw, dtype=np.float32)
    seq_a_filt = seq_a_np[~np.isnan(seq_a_np)]
    seq_b_filt = seq_b_np[~np.isnan(seq_b_np)]

    if seq_a_filt.size == 0 or seq_b_filt.size == 0:
        return (seq_a_filt.tolist() if seq_a_filt.size > 0 else None,
                seq_b_filt.tolist() if seq_b_filt.size > 0 else None), \
               (None, None)
    min_len = min(seq_a_filt.size, seq_b_filt.size)
    return (seq_a_filt.tolist(), seq_b_filt.tolist()), \
           (seq_a_filt[:min_len].tolist(), seq_b_filt[:min_len].tolist())

def calculate_overall_dance_similarity(avatar_angles_path: str, user_angles_path: str, joint_names: list, config: dict):
    try:
        avatar_angles_all_dict = np.load(avatar_angles_path, allow_pickle=True).item()
        user_angles_all_dict = np.load(user_angles_path, allow_pickle=True).item()
    except FileNotFoundError as e:
        logging.error(f"Angle file not found for overall similarity: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading angle files for overall similarity: {e}")
        return None

    all_scores_by_method = {
        "CosineSim": {"joints": {}, "average": 0.0, "_sum": 0.0, "_count": 0},
        "MAE": {"joints": {}, "average": 0.0, "_sum": 0.0, "_count": 0},
        "Correlation": {"joints": {}, "average": 0.0, "_sum": 0.0, "_count": 0}
    }
    mae_max_range = config.get("mae_max_angle_range", 180.0)
    cosine_window_size = config.get("cosine_window_size", 15)

    for joint_name in joint_names:
        if joint_name in avatar_angles_all_dict and joint_name in user_angles_all_dict:
            avatar_seq_raw = avatar_angles_all_dict[joint_name]
            user_seq_raw = user_angles_all_dict[joint_name]
            (av_filt, us_filt), (av_aligned, us_aligned) = _prepare_sequences_for_comparison(list(avatar_seq_raw), list(user_seq_raw))

            if av_filt and us_filt:
                # Cosine Similarity
                score_cos = compute_windowed_cosine_similarity(av_filt, us_filt, window_size=cosine_window_size)
                all_scores_by_method["CosineSim"]["joints"][joint_name] = score_cos
                all_scores_by_method["CosineSim"]["_sum"] += score_cos
                all_scores_by_method["CosineSim"]["_count"] += 1
            else:
                all_scores_by_method["CosineSim"]["joints"][joint_name] = 0.0
            
            if av_aligned and us_aligned:
                score_mae = compute_mae_similarity_score(av_aligned, us_aligned, mae_max_range)
                all_scores_by_method["MAE"]["joints"][joint_name] = score_mae
                all_scores_by_method["MAE"]["_sum"] += score_mae
                all_scores_by_method["MAE"]["_count"] += 1
            else: all_scores_by_method["MAE"]["joints"][joint_name] = 0.0
            
            if av_aligned and us_aligned:
                score_corr = compute_correlation_similarity_score(av_aligned, us_aligned)
                all_scores_by_method["Correlation"]["joints"][joint_name] = score_corr
                all_scores_by_method["Correlation"]["_sum"] += score_corr
                all_scores_by_method["Correlation"]["_count"] += 1
            else: all_scores_by_method["Correlation"]["joints"][joint_name] = 0.0
        else:
            logging.warning(f"Joint {joint_name} not found in one or both angle data for overall scoring.")
            for method_key in all_scores_by_method: all_scores_by_method[method_key]["joints"][joint_name] = 0.0
    
    for method_key in all_scores_by_method:
        if all_scores_by_method[method_key]["_count"] > 0:
            all_scores_by_method[method_key]["average"] = round(all_scores_by_method[method_key]["_sum"] / all_scores_by_method[method_key]["_count"], 1)
        del all_scores_by_method[method_key]["_sum"]; del all_scores_by_method[method_key]["_count"]
    return all_scores_by_method

def generate_comparison_video(avatar_data: dict, user_data: dict, avatar_hud_frames_dir: str, user_hud_frames_dir: str, config: dict):
    base_output_dir_final_video = config["data_paths"]["base_output_dir"]
    tut_vid_name_no_ext = avatar_data.get("video_file_name_no_ext", "TUTORIAL") # Defensive get
    usr_vid_name_no_ext = user_data.get("video_file_name_no_ext", "USER")     # Defensive get
    comparison_video_filename = f"comparison_{tut_vid_name_no_ext}_vs_{usr_vid_name_no_ext}.mp4"
    comparison_video_output_path = os.path.join(base_output_dir_final_video, comparison_video_filename)
    
    avatar_angles_all_dict = np.load(avatar_data["joint_angles_path"], allow_pickle=True).item()
    user_angles_all_dict = np.load(user_data["joint_angles_path"], allow_pickle=True).item()
    avatar_hud_frame_files = sorted([f for f in os.listdir(avatar_hud_frames_dir) if f.lower().endswith((".jpg", ".png"))])
    user_hud_frame_files = sorted([f for f in os.listdir(user_hud_frames_dir) if f.lower().endswith((".jpg", ".png"))])

    num_comp_frames = min(len(avatar_hud_frame_files), len(user_hud_frame_files), avatar_data["num_frames"], user_data["num_frames"])
    if num_comp_frames == 0: logging.error("No frames for comparison video."); return

    logging.info(f"Generating comparison video with {num_comp_frames} frames...")
    temp_avatar_f = cv2.imread(os.path.join(avatar_hud_frames_dir, avatar_hud_frame_files[0]))
    temp_user_f = cv2.imread(os.path.join(user_hud_frames_dir, user_hud_frame_files[0]))
    if temp_avatar_f is None or temp_user_f is None: logging.error("Cannot read sample HUD frames for dimensions."); return

    dummy_s = {joint: 0.0 for joint in config["joint_triplets"].keys()}
    target_h = config.get("output_target_height", temp_user_f.shape[0])
    first_comp_f = create_comparison_frame(temp_avatar_f, temp_user_f, dummy_s, target_height=target_h)
    out_h, out_w = first_comp_f.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(comparison_video_output_path, fourcc, config["fps"], (out_w, out_h))
    if not video_writer.isOpened(): logging.error(f"Failed to open VideoWriter for {comparison_video_output_path}"); return

    window_size = config["similarity_window_size"]
    mae_max_range = config.get("mae_max_angle_range", 180.0)
    joint_n_list = list(config["joint_triplets"].keys())

    for i in tqdm(range(num_comp_frames), desc="Generating comparison video frames"):
        avatar_hud = cv2.imread(os.path.join(avatar_hud_frames_dir, avatar_hud_frame_files[i]))
        user_hud = cv2.imread(os.path.join(user_hud_frames_dir, user_hud_frame_files[i]))
        if avatar_hud is None or user_hud is None: logging.warning(f"Skipping frame {i}, missing HUD."); continue

        current_dyn_scores = {}
        for joint in joint_n_list:
            av_seq_win = avatar_angles_all_dict[joint][max(0, i - window_size + 1) : i + 1]
            us_seq_win = user_angles_all_dict[joint][max(0, i - window_size + 1) : i + 1]
            # MAE for the dynamic score
            score = compute_mae_similarity_score(list(av_seq_win), list(us_seq_win), mae_max_range)
            current_dyn_scores[joint] = score
        
        comp_out_f = create_comparison_frame(avatar_hud, user_hud, current_dyn_scores, target_height=target_h)
        video_writer.write(comp_out_f)
    video_writer.release()
    logging.info(f"Final comparison video saved to: {comparison_video_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process and compare two dance videos.")
    parser.add_argument("--config", default="config/params.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    yaml_config = load_config(args.config)
    
    log_level_str = yaml_config.get("logging_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    setup_logging(log_level)
    
    config = {
        "fps": yaml_config["fps_extraction"],
        "pose_model_complexity": yaml_config["pose_model_complexity"],
        "similarity_window_size": yaml_config["similarity_window_size"],
        "batch_size": yaml_config.get("batch_size", 16),
        "cosine_window_size": yaml_config.get("cosine_window_size", 15),
        "joint_triplets": yaml_config["joint_triplets"],
        "hud_margin_width": yaml_config["angle_hud_margin_width"],
        "output_target_height": yaml_config.get("output_target_height", 480),
        "max_workers_hud": yaml_config.get("max_workers_hud", mp.cpu_count()),
        "mae_max_angle_range": yaml_config.get("mae_max_angle_range", 180.0),
        "data_paths": {
            "base_output_dir": OUTPUT_DIR, "raw_videos_dir": RAW_VIDEOS_DIR,
            "frames_suffix": FRAMES_SUFFIX, "keypoints_suffix": KEYPOINTS_SUFFIX,
            "norm_keypoints_suffix": NORM_KEYPOINTS_SUFFIX, "angles_suffix": ANGLES_SUFFIX,
            "angle_hud_frames_suffix": ANGLE_HUD_FRAMES_SUFFIX,
            "angle_hud_video_suffix": ANGLE_HUD_VIDEO_SUFFIX,
            "final_comparison_video_name": FINAL_COMPARISON_VIDEO_NAME
        },
        "cleanup_temp_frames": yaml_config.get("cleanup_temp_frames", False)
    }
    
    tut_vid_file = yaml_config["videos"]["tutorial"]
    usr_vid_file = yaml_config["videos"]["user"]
    tut_vid_path = os.path.join(TUTORIAL_VIDEO_DIR, tut_vid_file)
    usr_vid_path = os.path.join(USER_VIDEO_DIR, usr_vid_file)
    
    ensure_dir(config["data_paths"]["base_output_dir"])
    logging.info("Starting video processing pipeline...")

    logging.info("--- Processing Tutorial Video ---")
    tutorial_data = process_single_video(tut_vid_path, "tutorial", config)
    if not tutorial_data: logging.error("Failed to process tutorial video. Exiting."); return
    tut_hud_frames_dir, _ = generate_angle_hud_video(tutorial_data, config)
    if not tut_hud_frames_dir: logging.error("Failed to generate tutorial HUD video/frames. Exiting."); return

    logging.info("--- Processing User Video ---")
    user_data = process_single_video(usr_vid_path, "user", config)
    if not user_data: logging.error("Failed to process user video. Exiting."); return
    usr_hud_frames_dir, _ = generate_angle_hud_video(user_data, config)
    if not usr_hud_frames_dir: logging.error("Failed to generate user HUD video/frames. Exiting."); return

    if tutorial_data and user_data:
        joint_names = list(config["joint_triplets"].keys())
        logging.info("--- Calculating Overall Full Dance Similarities (for reporting) ---")
        all_overall_scores = calculate_overall_dance_similarity(
            tutorial_data["joint_angles_path"], user_data["joint_angles_path"], joint_names, config
        )
        
        if all_overall_scores:
            tut_name = tutorial_data.get('video_file_name_no_ext', 'TUTORIAL_VIDEO_NAME_MISSING')
            usr_name = user_data.get('video_file_name_no_ext', 'USER_VIDEO_NAME_MISSING')
            if 'video_file_name_no_ext' not in tutorial_data or 'video_file_name_no_ext' not in user_data:
                 logging.warning("Key 'video_file_name_no_ext' was missing from tutorial_data or user_data dict. Check process_single_video return.")

            logging.info(f"Reportable Overall Scores for '{tut_name}' vs '{usr_name}':")
            for method, scores_data in all_overall_scores.items():
                logging.info(f"  METHOD: {method.upper()}")
                for joint_name_key, score_val in scores_data["joints"].items():
                    logging.info(f"    {joint_name_key}: {score_val:.1f}%")
                logging.info(f"    OVERALL AVERAGE ({method.upper()}): {scores_data['average']:.1f}%")
        else: logging.info("Could not calculate overall similarity scores (function returned None).")

    logging.info("--- Generating Final Comparison Video (Dynamic Windowed MAE Scores) ---")
    generate_comparison_video(tutorial_data, user_data, tut_hud_frames_dir, usr_hud_frames_dir, config)

    if config.get("cleanup_temp_frames", False):
        logging.info("Cleaning up temporary frame directories...")
        temp_dirs = []
        if tutorial_data: temp_dirs.append(tutorial_data.get("original_frames_dir"))
        if user_data: temp_dirs.append(user_data.get("original_frames_dir"))
        if tut_hud_frames_dir and tut_hud_frames_dir != tutorial_data.get("original_frames_dir"): temp_dirs.append(tut_hud_frames_dir)
        if usr_hud_frames_dir and usr_hud_frames_dir != user_data.get("original_frames_dir"): temp_dirs.append(usr_hud_frames_dir)
        for dir_path in set(filter(None, temp_dirs)):
            if os.path.exists(dir_path):
                try: shutil.rmtree(dir_path); logging.info(f"Successfully removed: {dir_path}")
                except Exception as e: logging.error(f"Failed to remove {dir_path}: {e}")
    logging.info("Pipeline finished successfully!")

if __name__ == "__main__":
    main()
