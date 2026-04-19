import os
from shlex import split
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from math import atan2, asin, degrees

# STEP 7 FOR COLLECTED DATASET AND DAiSEE DATASET
# Description: This script processes the dataset videos to extract:
#   - Eye Aspect Ratio (EAR)
#   - Mouth Aspect Ratio (MAR)
#   - Head Pose (Yaw, Pitch, Roll)
# It also saves the processed frames (with overlay annotations)
# and exports the numerical features into CSV files per dataset split.

# ==================
# PATH CONFIGURATION
# ==================
# DATASET_ROOT = "DAiSEE_final"       # Folder containing input videos (train/val/test)
# OUTPUT_ROOT = "DAiSEE_features"        # Output folder for CSV feature files
# FRAMES_ROOT = "DAiSEE_frames_features" # Output folder for saved frames with overlays

DATASET_ROOT = "collected_final_3sec"       # Folder containing input videos (train/val/test)
OUTPUT_ROOT = "collected_features_origframe"        # Output folder for CSV feature files
FRAMES_ROOT = "collected_frames_features_origframe" # Output folder for saved frames with overlays

# Create the output directories if they don’t exist
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(FRAMES_ROOT, exist_ok=True)

# ================
# GLOBAL PARAMETERS
# =================
# Process every frame
FRAME_INTERVAL = 1          
#TARGET_SIZE = (224, 224)   # Resize frames for uniform resolution of 224x224 pixels

# ==============================
# INITIALIZE MEDIAPIPE FACE MESH
# ==============================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# FaceMesh configuration: detects and tracks one face per frame
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,      # Process video stream (not static images)
    max_num_faces=1,              # Only detect one face at a time
    refine_landmarks=True,        # Get additional landmarks (eyes, lips)
    min_detection_confidence=0.5, # Detection threshold
    min_tracking_confidence=0.5   # Tracking threshold
)

# ================
# HELPER FUNCTIONS
# ================

# Euclidean distance between two 2D/3D points
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# ======================
# EYE ASPECT RATIO (EAR)
# ======================
def compute_EAR(landmarks):
    # MediaPipe FaceMesh indices for left and right eyes
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Computes the EAR for one eye
    def eye_aspect_ratio(eye_points):
        A = euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
        B = euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
        C = euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
        return (A + B) / (2.0 * C)  # Ratio of vertical to horizontal eye distance

    # Compute EAR for both eyes and take the average
    left_EAR = eye_aspect_ratio(LEFT_EYE)
    right_EAR = eye_aspect_ratio(RIGHT_EYE)
    EAR = (left_EAR + right_EAR) / 2.0
    return EAR

# ========================
# MOUTH ASPECT RATIO (MAR)
# ========================
def compute_MAR(landmarks):
    # MediaPipe indices for outer mouth landmarks
    MOUTH = [78, 308, 191, 415, 80, 81, 82, 13, 312, 311, 310, 14]
    
    # Compute three vertical distances (upper to lower lip)
    A = euclidean(landmarks[81], landmarks[311])
    B = euclidean(landmarks[82], landmarks[310])
    C = euclidean(landmarks[13], landmarks[14])
    
    # Horizontal mouth width (corner-to-corner)
    D = euclidean(landmarks[78], landmarks[308])
    
    # MAR = mean vertical distance ÷ horizontal distance
    MAR = (A + B + C) / (3.0 * D)
    return MAR

# ====================
# HEAD POSE ESTIMATION
# ====================
def extract_head_pose(landmarks, image_shape):
    h, w = image_shape[:2]

    # Select key facial landmarks for 3D-2D mapping
    image_points = np.array([
        (landmarks[1][0]*w, landmarks[1][1]*h),     # Nose tip
        (landmarks[152][0]*w, landmarks[152][1]*h), # Chin
        (landmarks[263][0]*w, landmarks[263][1]*h), # Right eye corner
        (landmarks[33][0]*w, landmarks[33][1]*h),   # Left eye corner
        (landmarks[287][0]*w, landmarks[287][1]*h), # Right mouth corner
        (landmarks[57][0]*w, landmarks[57][1]*h)    # Left mouth corner
    ], dtype="double")

    # 3D model reference points approximating a human face
    model_points = np.array([
        (0.0, 0.0, 0.0),           # Nose tip
        (0.0, -63.6, -12.5),       # Chin
        (43.3, 32.7, -26.0),       # Right eye outer corner
        (-43.3, 32.7, -26.0),      # Left eye outer corner
        (28.9, -28.9, -24.1),      # Right mouth corner
        (-28.9, -28.9, -24.1)      # Left mouth corner
    ])

    # Approximate camera intrinsic parameters
    focal_length = w
    camera_matrix = np.array([[focal_length, 0, w/2],
                              [0, focal_length, h/2],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4,1))  # Assume no lens distortion

    # Solve the PnP problem: estimate head pose (rotation and translation)
    # Perspective-n-Point (PnP) algorithm to determine the 3D position and orientation of a person's head from a 2D image
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return (0,0,0), None, None

    # Convert rotation vector to Euler angles (Yaw, Pitch, Roll)
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = atan2(R[2,1], R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else:
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0
    yaw, pitch, roll = degrees(y), degrees(x), degrees(z)
    return (yaw, pitch, roll), rvec, tvec

# ================
# DRAWING OVERLAYS
# ================
def draw_overlays(frame, landmarks, image_shape, rvec, tvec, camera_matrix,
                  EAR=None, MAR=None, HEAD=None, alpha=0.6):
    # Draws mesh, feature landmarks, and text overlays on frame.
    h, w = image_shape[:2]
    overlay = frame.copy()

    # Draw entire facial mesh (tessellation)
    mp_drawing.draw_landmarks(
        image=overlay,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
    )

    # Draw eyes (green points)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    for idx in LEFT_EYE + RIGHT_EYE:
        x, y = int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)
        cv2.circle(overlay, (x, y), 2, (0, 200, 0), -1)

    # Draw mouth (red points)
    MOUTH = [78, 308, 191, 415, 80, 81, 82, 13, 312, 311, 310, 14]
    for idx in MOUTH:
        x, y = int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)
        cv2.circle(overlay, (x, y), 2, (0, 0, 200), -1)

    # Draw head pose 3D axes if available
    if rvec is not None and tvec is not None:
        axis_len = 50  # Axis line length in pixels
        axis_points = np.float32([
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len]
        ])
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, np.zeros((4, 1)))

        # Nose tip as the origin of the axes
        nose_tip = np.array([landmarks.landmark[1].x * w, landmarks.landmark[1].y * h], dtype=int)
        cv2.line(overlay, tuple(nose_tip), tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 2)  # X - Red
        cv2.line(overlay, tuple(nose_tip), tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 2)  # Y - Green
        cv2.line(overlay, tuple(nose_tip), tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 2)  # Z - Blue

    # Add EAR, MAR, and head pose text annotations
    if EAR is not None and MAR is not None and HEAD is not None:
        text_lines = [
            (f"EAR: {EAR:.3f}", (0, 255, 0)),   # green text
            (f"MAR: {MAR:.3f}", (0, 0, 255)),   # red text
            (f"Yaw: {HEAD[0]:.1f}", (255, 255, 255)),
            (f"Pitch: {HEAD[1]:.1f}", (255, 255, 255)),
            (f"Roll: {HEAD[2]:.1f}", (255, 255, 255))
        ]
        y0, dy = 20, 22
        for i, (line, color) in enumerate(text_lines):
            y = y0 + i * dy
            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(overlay, (5, y - text_h - 3), (10 + text_w, y + 3), (0, 0, 0), -1)
            cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 1, cv2.LINE_AA)

    # Blend overlay with the original frame (semi-transparent)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# ========================================
# MAIN FUNCTION TO PROCESS EACH VIDEO CLIP
# ========================================
def process_clip(video_path, clip_id, label, split):
    cap = cv2.VideoCapture(video_path)
    EAR_seq, MAR_seq, HEAD_seq = [], [], []
    frame_count = 0

    # Create folder for each clip under its label directory
    out_dir = os.path.join(FRAMES_ROOT, split, label, clip_id)
    os.makedirs(out_dir, exist_ok=True)

    # Read and process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every nth frame
        if frame_count % FRAME_INTERVAL == 0:
            # frame_resized = cv2.resize(frame, TARGET_SIZE) # Resizing removed
            
            # Use the original frame dimensions (h, w)
            h, w = frame.shape[:2]
            
            # Convert the original frame to RGB for Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # If a face is detected, extract features
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                lm = [(p.x, p.y, p.z) for p in face_landmarks.landmark]

                EAR = compute_EAR(lm)
                MAR = compute_MAR(lm)
                # Pass the original frame's shape to head pose estimation
                HEAD, rvec, tvec = extract_head_pose(lm, frame.shape) 

                # Store feature values per frame
                EAR_seq.append(EAR)
                MAR_seq.append(MAR)
                HEAD_seq.append(HEAD)

                # Define camera matrix for overlay based on the original frame's dimensions
                focal_length = w
                camera_matrix = np.array([[focal_length, 0, w / 2],
                                          [0, focal_length, h / 2],
                                          [0, 0, 1]], dtype="double")

                # Draw overlays and save processed frame.
                # Use the original 'frame' for drawing and saving.
                draw_overlays(frame, face_landmarks, frame.shape,
                              rvec, tvec, camera_matrix, EAR=EAR, MAR=MAR, HEAD=HEAD)
                frame_filename = f"{clip_id}_frame{frame_count:04d}.jpg"
                cv2.imwrite(os.path.join(out_dir, frame_filename), frame)

        frame_count += 1

    cap.release()

    # Return all computed feature sequences for this clip
    return {
        "ClipID": clip_id,
        "EAR_seq": EAR_seq,
        "MAR_seq": MAR_seq,
        "HeadPose_seq": HEAD_seq,
        "CustomLabel": label
    }

# =====================================
# PROCESS ALL SPLITS (TRAIN, VAL, TEST)
# =====================================
for split in ["train", "val", "test"]:
    print(f"\nProcessing {split} set...")
    rows = []
    split_dir = os.path.join(DATASET_ROOT, split)

    # Iterate through labels
    for label in os.listdir(split_dir):
        label_dir = os.path.join(split_dir, label)
        if not os.path.isdir(label_dir):
            continue

        # Iterate through all .mp4 and .avi video files
        for file in tqdm(os.listdir(label_dir)):
            if not (file.lower().endswith(".mp4") or file.lower().endswith(".avi")):
                continue

            clip_id = os.path.splitext(file)[0].replace(" ", "_")
            video_path = os.path.join(label_dir, file)
            rows.append(process_clip(video_path, clip_id, label, split))

    # Convert collected feature data into a DataFrame
    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_ROOT, f"{split}_features.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

print("\nAll splits processed and feature CSVs generated successfully!")
