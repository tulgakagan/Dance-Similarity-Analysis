# CVIA - Dance Similarity Analysis Project

**Team Name:** Koi-un
**Team Members:** Doruk Efe Kanber, Eren Karsavuranoglu, Tulga Kagan Temel

## Project Overview

This project implements a computer vision pipeline to analyze and compare human dance movements against a reference video. The system extracts 2D pose keypoints from both videos, calculates time-series data for various joint angles, and then employs multiple computational metrics to quantify the similarity between the performances.

The core contribution is a comparative analysis of three similarity metrics:
- **Windowed Cosine Similarity (with magnitude scaling)**
- **Mean Absolute Error (MAE)-based Similarity**
- **Correlation-based Similarity**

The pipeline outputs per-joint and overall average similarity scores for these metrics. Additionally, a side-by-side comparison video is generated, featuring a dynamic scoring HUD based on windowed MAE.

## Project Report

For a detailed explanation of the methodology, experiments, and conclusions, see the full academic-style report located in the root of this repository:

**[Report (PDF)](../report.pdf)**

This report covers:

* Problem formulation and motivation
* Methodological pipeline
* Detailed explanation of the similarity metrics
* Experiments comparing “Good” and “Bad” dance imitations
* Analysis of results and metric effectiveness
* Limitations and future work suggestions

## Directory Structure

```
code/
├── config/
│   └── params.yaml        # Main configuration file
├── data/
│   ├── tutorial/          # Place reference/avatar videos here
│   └── user/              # Place user imitation videos here
├── output/                # All generated output (processed data, videos) goes here
├── src/
│   ├── main_pipeline.py   # Main executable script
│   ├── comparison_utils.py
│   ├── feature_processing.py
│   ├── frame_utils.py
│   ├── pose_estimation.py
│   └── visualization_utils.py
├── README.md              # This file
└── requirements.txt       # Python dependencies

report.pdf  # (in the root directory)
```

*The `output/` directory will be created upon first run. Within `output/`, subdirectories like `output/tutorial/<video_name_no_ext>/` and `output/user/<video_name_no_ext>/` will store intermediate files for each video.*

## Setup Instructions

1.  **Environment Setup**:
    It is recommended to use a Python virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2.  **Install Dependencies**:
    Install all required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Place Data Files**:
    - Place your reference (avatar) video inside the `data/tutorial/` directory.
    - Place your user imitation video(s) inside the `data/user/` directory.

4.  **Configure Parameters**:
    - Open `config/params.yaml`.
    - Under the `videos:` section, update `tutorial:` and `user:` with the exact filenames of the videos you placed in the `data/` directories.
    - Ensure your `params.yaml` includes:
        - `similarity_window_size`: (e.g., `30`) This is used for the windowed MAE dynamic scores in the comparison video.
        - `cosine_window_size`: (e.g., `15`) This is used for the overall Windowed Cosine Similarity scores. If not present, the code defaults to 15. It's best to define it explicitly.
        - `mae_max_angle_range`: (e.g., `180.0`)
    - Review and adjust other parameters like `fps_extraction` or `pose_model_complexity` as needed.

## How to Run

All commands should be run from the root directory of the project.

To run the entire pipeline (processing videos from scratch to generate pose data, angle data, overall similarity scores using CosineSim, MAE, and Correlation, and the final comparison video with dynamic MAE scores), use:

```bash
python src/main_pipeline.py --config config/params.yaml
```

## Output

-   **Processed Data:** Intermediate files (extracted frames, keypoints, normalized keypoints, joint angles as `.npy` files, and individual HUD videos) are stored in subdirectories within `output/`, structured as:
    -   `output/tutorial/<avatar_video_name_no_ext>/`
    -   `output/user/<user_video_name_no_ext>/`
-   **Final Comparison Video:** A side-by-side video named `comparison_<avatar_video_name_no_ext>_vs_<user_video_name_no_ext>.mp4` is saved directly in the `output/` directory. This video includes dynamic MAE-based similarity scores on its HUD.
-   **Logged Scores:** The overall similarity scores (Windowed Cosine Similarity, MAE-based Similarity, and Correlation-based Similarity) for each joint and the average across joints are printed to the console when the pipeline completes.