# Tecdia Submission - Jumbled Video Frame Reconstruction

This repository contains my submission for the Tecdia Internship Challenge — a computer vision task to reconstruct a shuffled video from jumbled frames using feature-based matching, graph optimization, and optical flow smoothing.

---

## Objective

Reconstruct the correct temporal order of 300 shuffled frames (5 seconds @ 60 FPS, 1080p) into a smooth, original video sequence.

The challenge evaluates problem-solving ability, algorithmic design, and practical implementation of efficient reconstruction techniques.

---

## Approach Overview

### 1. Feature Extraction and Pair Scoring (`pair_scorer_mt.py`)
- Each frame is resized for faster computation.
- ORB (Oriented FAST and Rotated BRIEF) keypoints are extracted for each frame.
- Matches between consecutive frames are computed using BFMatcher.
- A combined similarity score is derived from ORB feature matches and SSIM (Structural Similarity Index).
- Multi-threading is used to accelerate pairwise scoring across 300 frames.

### 2. Sequence Solving (`solve_order_mt.py`)
- Builds a directed graph where nodes are frames and edges represent similarity scores.
- Uses Beam Search and iterative 2-Opt optimization to find the most likely chronological order.
- Outputs the final sequence to `output/order_refined.txt`.

### 3. Frame Writing (`write_ordered_frames.py`)
- Reorders frames based on the solved order and saves them sequentially in `frames_out/`.

### 4. Temporal Smoothing and Optical Flow (`smooth_motion_blur.py`)
- Applies motion-based smoothing between adjacent frames using optical flow.
- Reduces jitter and creates a more natural, cinematic transition between frames.
- Produces the final reconstructed video.

---

## Directory Structure
project/
│
├── pair_scorer_mt.py
├── solve_order_mt.py
├── write_ordered_frames.py
├── smooth_motion_blur.py
├── requirements.txt
├── README.md
│
├── frames_full/ # Original shuffled frames (input)
├── frames_out/ # Ordered frames (intermediate)
├── output/ # Final output and order files
│ ├── order_refined.txt
│ └── reconstructed_cinematic.mp4


---

## Installation

1. Clone or download the repository.
2. Navigate into the project directory.
3. Create a virtual environment and install dependencies.

```bash
python -m venv venv
venv\Scripts\activate        # For Windows
source venv/bin/activate     # For macOS/Linux

pip install -r requirements.txt

How to Run

Run the scripts in the following sequence:

# Step 1: Compute pairwise similarity between frames
python pair_scorer_mt.py

# Step 2: Solve the correct frame order
python solve_order_mt.py

# Step 3: Write frames in correct order
python write_ordered_frames.py

# Step 4: Apply smoothing and motion blur to final output
python smooth_motion_blur.py

The reconstructed video will be saved as:

output/reconstructed_cinematic.mp4

Requirements

The project requires Python 3.9+ and the following libraries:

opencv-python
numpy
pandas
scikit-image
tqdm


These will be installed automatically when running:

pip install -r requirements.txt
Output Files
File	Description
output/order_initial.txt	Initial order before refinement
output/order_refined.txt	Final refined frame order
output/reconstructed_cinematic.mp4	Smoothed reconstructed video
frames_out/	Directory containing ordered frames

----------------------------------------------------------------------------------------
Author

Khushi Wadhawan
B.Tech - Computer Science and Engineering
Vellore Institute of Technology

Submission for Tecdia Internship Selection Challenge

Notes

Large folders such as frames_full/ and frames_out/ are excluded from GitHub due to size constraints.

The scripts are modular — you can adjust feature extraction, similarity metrics, or optimization strategies independently.

Recommended to use a GPU-enabled environment for faster execution.
