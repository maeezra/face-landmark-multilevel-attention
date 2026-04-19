import os
import warnings
import cv2
import json
import time
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import List, Tuple
from collections import deque
from math import atan2, asin, degrees 
# Import necessary components from the training script to ensure model loading works
from advanced_model_clean import (
    load_and_process, 
    SliceFeatures,
    MaskFromInput,
    ExpandMaskTo4D,
    TemporalSelfAttention,
    ScaledResidual,
    set_seed,
    safe_inverse_transform
)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K # Import Keras backend for parameter count

# Suppress TensorFlow/MediaPipe C++ Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3' 

# Suppress Keras/TensorFlow Python Warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')

# SCRIPT 11
# Description: Real-time inference with option to use webcam or video file
# Based on advanced_model_clean.py and adapted for real-time use
# Loads the best advanced_model_tf.keras model for inference

# =============
# Configuration
# =============
#MODEL_DIR = "runs_advanced/20251104-164805" # Placeholder for the latest run directory
#MODEL_DIR = "runs_advanced/20251118-133410"
#MODEL_DIR = "runs_advanced/20251119-160056"
MODEL_DIR = "runs_advanced/20251122-090120"
MAX_SEQUENCE_LENGTH = 90
MODEL_PATH = os.path.join(MODEL_DIR, 'best_advanced_model_tf.keras')   # Path to the best model file
NORMALIZATION_STATS_PATH = os.path.join(MODEL_DIR, 'advanced_normalization_stats.json')   # Path to normalization stats
LABEL_ENCODER_CLASSES_PATH = os.path.join(MODEL_DIR, 'label_encoder_classes.json')   # Path to label encoder classes
INPUT_VIDEO_PATH = 'input.mp4'                 # Input video file path
OUTPUT_VIDEO_PATH = 'output_predictions.mp4'   # Output file for video with predictions

# CONFIGURATION FLAG: Set this to True to use webcam instead of video file
USE_WEBCAM = False # Default to False, will be set by the __main__ block
WEBCAM_ID = 0      # Default webcam ID (usually 0)


NUM_FEATURES = 11  # ADDED: 5 (Original) + 6 (Derivatives)

# --- NEW: History buffer for Head Pose (must store at least 2 previous points) ---
# We store (Yaw, Pitch, Roll) for the last three frames to calculate current V and A.
# The main feature_sequence_buffer will store the final 11 features.
HEAD_POSE_HISTORY = deque(maxlen=3) # Stores raw (Yaw, Pitch, Roll) tuples

# ========================
# MediaPipe Initialization
# ========================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ====================================================
# Constants for Feature Calculation (Landmark Indices)
# ====================================================
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 308, 191, 415, 80, 81, 82, 13, 312, 311, 310, 14]
PnP_POINTS_INDEXES = [1, 152, 263, 33, 287, 57] 

# Colors for visualization
ATTENTION_COLORS = {
    "Fatigued": (0, 0, 255),    # Red/BGR
    "Focused": (0, 255, 0),     # Green/BGR
    "Distracted": (255, 0, 0)   # Blue/BGR
}
DEFAULT_COLOR = (128, 128, 128) # Gray

# =============================
# Feature Calculation Functions 
# =============================

def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p1[:2] - p2[:2]) 

# Calculation of Eye Aspect Ratio (EAR)
def calculate_ear(landmarks: np.ndarray) -> float:
    def eye_aspect_ratio_calc(indices, all_landmarks):
        A = _distance(all_landmarks[indices[1]], all_landmarks[indices[5]])
        B = _distance(all_landmarks[indices[2]], all_landmarks[indices[4]])
        C = _distance(all_landmarks[indices[0]], all_landmarks[indices[3]])
        return (A + B) / (2.0 * C + 1e-8)
    left_EAR = eye_aspect_ratio_calc(LEFT_EYE_INDICES, landmarks)
    right_EAR = eye_aspect_ratio_calc(RIGHT_EYE_INDICES, landmarks)
    return (left_EAR + right_EAR) / 2.0

# Calculation of Mouth Aspect Ratio (MAR)
def calculate_mar(landmarks: np.ndarray) -> float:
    A = _distance(landmarks[81], landmarks[311])
    B = _distance(landmarks[82], landmarks[310])
    C = _distance(landmarks[13], landmarks[14])
    D = _distance(landmarks[78], landmarks[308])
    return (A + B + C) / (3.0 * D + 1e-8)

# Calculation of Head Pose Angles
def get_head_pose_angles(landmarks: np.ndarray, frame_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = frame_size
    image_points = np.array([
        landmarks[1][:2], landmarks[152][:2], landmarks[263][:2], 
        landmarks[33][:2], landmarks[287][:2], landmarks[57][:2]
    ], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (43.3, 32.7, -26.0),
        (-43.3, 32.7, -26.0), (28.9, -28.9, -24.1), (-28.9, -28.9, -24.1)
    ])
    focal_length = w
    camera_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4,1))
    # Solve PnP to get rotation and translation vectors
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    if not success:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), None, None, None
    R, _ = cv2.Rodrigues(rvec)
    # Calculate Euler angles from rotation matrix
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy < 1e-6:
        pitch = atan2(-R[1,2], R[1,1])
        yaw = atan2(-R[2,0], sy)
        roll = 0
    else:
        pitch = atan2(R[2,1], R[2,2]) 
        yaw = atan2(-R[2,0], sy)
        roll = atan2(R[1,0], R[0,0])
    yaw, pitch, roll = degrees(yaw), degrees(pitch), degrees(roll)
    return np.array([yaw, pitch, roll], dtype=np.float32), rvec, tvec, camera_matrix

# Feature Extraction Function
def extract_features(landmarks: np.ndarray, frame_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Extract EAR, MAR, and head pose features
    ear = calculate_ear(landmarks)
    mar = calculate_mar(landmarks)
    head_pose, rvec, tvec, camera_matrix = get_head_pose_angles(landmarks, frame_size)
    feature_vector = np.array([ear, mar, head_pose[0], head_pose[1], head_pose[2]], dtype=np.float32)
    return feature_vector, head_pose, rvec, tvec, camera_matrix

# Draw Features Overlays Function
def draw_features_overlays(frame, face_landmarks, frame_shape, rvec, tvec, camera_matrix,
                  EAR, MAR, HEAD, alpha=0.6):
    # Draws feature landmarks, head pose axes, and text overlays on frame.
    # Note: This function draws over the mesh which should be drawn first.
    h, w = frame_shape[:2]
    overlay = frame.copy()

    # Draw eyes (green points)
    for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
        if idx < len(face_landmarks):
            x, y = int(face_landmarks[idx][0]), int(face_landmarks[idx][1])
            cv2.circle(overlay, (x, y), 2, (0, 200, 0), -1)

    # Draw mouth (red points)
    for idx in MOUTH_INDICES:
        if idx < len(face_landmarks):
            x, y = int(face_landmarks[idx][0]), int(face_landmarks[idx][1])
            cv2.circle(overlay, (x, y), 2, (0, 0, 200), -1)

    # Draw head pose 3D axes
    if rvec is not None and tvec is not None:
        axis_len = 50
        axis_points = np.float32([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, np.zeros((4, 1)))
        nose_tip = np.array([face_landmarks[1][0], face_landmarks[1][1]], dtype=int)
        
        cv2.line(overlay, tuple(nose_tip), tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 2)  # X - Red
        cv2.line(overlay, tuple(nose_tip), tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 2)  # Y - Green
        cv2.line(overlay, tuple(nose_tip), tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 2)  # Z - Blue

    # Add EAR, MAR, and head pose text annotations
    text_lines = [
        (f"EAR: {EAR:.3f}", (0, 255, 0)),
        (f"MAR: {MAR:.3f}", (0, 0, 255)),
        (f"Yaw: {HEAD[0]:.1f}", (255, 255, 255)),
        (f"Pitch: {HEAD[1]:.1f}", (255, 255, 255)),
        (f"Roll: {HEAD[2]:.1f}", (255, 255, 255))
    ]
    y0, dy = 55, 22
    for i, (line, color) in enumerate(text_lines):
        y = y0 + i * dy
        (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (5, y - text_h - 3), (10 + text_w, y + 3), (0, 0, 0), -1) 
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

# ==============================
# Real-Time Processing Functions
# ==============================
# Load assets
def load_assets(model_path: str, stats_path: str, labels_path: str):
    print("Loading model and assets...")
    # Load model with custom objects
    custom_objects = {
        "SliceFeatures": SliceFeatures,
        "MaskFromInput": MaskFromInput,
        "ExpandMaskTo4D": ExpandMaskTo4D,
        "TemporalSelfAttention": TemporalSelfAttention,
        "ScaledResidual": ScaledResidual,
        "safe_inverse_transform": safe_inverse_transform
    }
    model = load_model(model_path, custom_objects=custom_objects, compile=False)

    # Load statistics for normalization
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    mean = np.array(stats['mean'], dtype=np.float32)
    std = np.array(stats['std'], dtype=np.float32)

    # This line ensures the vector is reliably (11,) size for the len() check
    mean = mean.ravel() 
    std = std.ravel()


    # --- START NEW: Verification Check ---
    expected_size = NUM_FEATURES # which is 11
    
    if len(mean) != expected_size or len(std) != expected_size:
        print("\n--- ❌ CRITICAL NORMALIZATION ERROR ❌ ---")
        print(f"Loaded 'mean' vector size: {len(mean)}")
        print(f"Loaded 'std' vector size: {len(std)}")
        print(f"The model requires {expected_size} features, but your mean/std files only contain {len(mean)} elements.")
        print("\nACTION REQUIRED: Retrain your advanced model with the 11 features to generate new mean.npy/std.npy files.")
        print("Exiting inference...")
        return
    else:
        print(f"Normalization vectors loaded successfully. Size: {len(mean)} features.")
    # --- END NEW: Verification Check ---




    # Load class labels
    with open(labels_path, 'r') as f:
        classes = json.load(f)
    le = {i: cls for i, cls in enumerate(classes)}

    return model, mean, std, le

# Preprocess sequence
def preprocess_sequence(feature_buffer: deque, mean: np.ndarray, std: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    # Convert feature buffer to numpy array
    seq = list(feature_buffer)
    # Convert to numpy array
    X_raw = np.array(seq, dtype=np.float32)
    # Determine number of features
    n_features = X_raw.shape[-1] if X_raw.ndim > 1 and X_raw.shape[0] > 0 else 5

    # Pad or truncate sequence to max_len
    if X_raw.shape[0] < max_len:
        padding = np.zeros((max_len - X_raw.shape[0], n_features), dtype=np.float32)
        X_padded = np.concatenate([X_raw, padding], axis=0)
    elif X_raw.shape[0] > max_len:
        X_padded = X_raw[-max_len:]
    else:
        X_padded = X_raw

    # Normalize sequence
    X_norm = (X_padded - mean) / (std + 1e-8)
    X_norm = np.nan_to_num(X_norm, nan=0.0)
    X_batch = X_norm.reshape(1, max_len, n_features)

    return X_batch, X_padded


# The main function for the end-to-end system implementation.
# Args:
#     input_source (str or int): Video file path (str) or webcam ID (int).
def run_end_to_end_system(input_source: str or int, model_path: str, stats_path: str, labels_path: str, max_len: int, output_path: str):
    is_webcam = isinstance(input_source, int)

    # 1. Load assets
    try:
        model, mean, std, le = load_assets(model_path, stats_path, labels_path)
    except Exception as e:
        print(f"Error loading model or assets: {e}")
        print("Please ensure you have run the training script successfully and updated MODEL_DIR.")
        return

    # 2. Verify model input features
    expected_n_features = model.input_shape[-1]
    if expected_n_features != NUM_FEATURES:
        print(f"Error: Model expects {expected_n_features} features, but the script is configured for {NUM_FEATURES} features. Ensure your trained model matches the script's NUM_FEATURES constant.")
        return
    
    # 3. Model Complexity Analysis
    try:
        print("\n--- Model Complexity Analysis ---")
        # Print a concise summary
        model.summary(line_length=120) 
        
        # Calculate parameter count
        trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
        non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])
        total_params = trainable_params + non_trainable_params
        print(f"Total Parameters: {total_params:,}")
        
        # Note on FLOPs: This requires an external library (e.g., `keras-flops`) and is typically
        # done in a separate script or setup. For quick analysis, parameter count is a good proxy.
        print("Note: FLOPs analysis requires an external library and is not included.")
    except Exception as e:
        print(f"Warning: Could not perform model complexity analysis. {e}")

    # 4. Initialize MediaPipe and Video Capture
    if is_webcam: 
        cap = cv2.VideoCapture(input_source)
    else:
        cap = cv2.VideoCapture(input_source)
        
    if not cap.isOpened():
        if is_webcam:
            print(f"Error: Could not open webcam with ID {input_source}. Check connections.")
        else:
            print(f"Error: Could not open video file {input_source}")
        return

    # 5. Retrieve video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter only for video file input
    out = None
    if not is_webcam:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 6. Initialize real-time state variables
    feature_buffer = deque(maxlen=max_len)
    current_prediction = "Initializing..."
    prediction_color = DEFAULT_COLOR
    frame_count = 0
    start_time = time.time()
    inference_times = []
    
    last_ear, last_mar, last_head_pose = 0.0, 0.0, np.array([0.0, 0.0, 0.0])
    last_rvec, last_tvec, last_camera_matrix = None, None, None

    
    # Consistency Tracking Variables
    prediction_log = []
    last_prediction_state = None
    state_switch_count = 0

    print(f"Starting processing at {fps:.2f} FPS... (Source: {'Webcam' if is_webcam else 'Video File'})")
    if is_webcam:
        print("Press 'q' or 'ESC' to exit the live feed.")

    # 7. Initialize MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        # 8. Processing loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Flip frame horizontally for a more natural webcam experience
            if is_webcam:
                 frame = cv2.flip(frame, 1)

            # 8.1 Convert frame to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            # 8.2 Process face landmarks and draw
            face_landmarks_pixel_coords = None
            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_TESSELATION, 
                        landmark_drawing_spec=None, 
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Custom drawing specification for green eye contours
                    GREEN_LINE_SPEC = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1) # BGR Green

                    # Draw Left Eye Contour (replace the default FACEMESH_CONTOURS call for a cleaner, greener look)
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_LEFT_EYE, # Use specific eye connections
                        landmark_drawing_spec=None,
                        connection_drawing_spec=GREEN_LINE_SPEC
                    )
                    # Draw Right Eye Contour
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_RIGHT_EYE, # Use specific eye connections
                        landmark_drawing_spec=None,
                        connection_drawing_spec=GREEN_LINE_SPEC
                    )
                    # Re-add other contours (Mouth, Eyebrows, etc.) with default style
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_lms,
                        connections=mp_face_mesh.FACEMESH_LIPS, 
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Extract pixel coordinates for feature calculation
                    face_landmarks_pixel_coords = np.array([
                        [lm.x * frame_width, lm.y * frame_height, lm.z]
                        for lm in face_lms.landmark
                    ])
                    break 

                # Feature Computation and Temporal Modeling
                if face_landmarks_pixel_coords is not None:
                    # Extract features from the current frame's landmarks (5 original features)
                    feature_vector_raw, HEAD, rvec, tvec, camera_matrix = extract_features(
                        face_landmarks_pixel_coords, 
                        (frame_height, frame_width)
                    )
                    # feature_buffer.append(feature_vector)
                    
                    # # Update last known feature values
                    # last_ear, last_mar = feature_vector[0], feature_vector[1]
                    # last_head_pose = head_pose
                    # last_rvec, last_tvec, last_camera_matrix = rvec, tvec, camera_matrix

                    # if len(feature_buffer) >= 5: # Only run inference after a few frames
                    #     X_batch, _ = preprocess_sequence(feature_buffer, mean, std, max_len)


                    EAR, MAR = feature_vector_raw[0], feature_vector_raw[1]
                    
                    # Derivatives
                    
                    # 1. Store raw Head Pose (Yaw, Pitch, Roll) for derivative calculation
                    # HEAD is a 3-element numpy array: [Yaw, Pitch, Roll]
                    HEAD_POSE_HISTORY.append(HEAD) 
                    
                    # Initialize V and A to zeros (used for the first two frames)
                    VEC_V = np.zeros(3, dtype=np.float32)
                    VEC_A = np.zeros(3, dtype=np.float32)

                    # 2. Calculate Velocity and Acceleration using history
                    if len(HEAD_POSE_HISTORY) >= 2:
                        P_curr = np.array(HEAD_POSE_HISTORY[-1])
                        P_prev = np.array(HEAD_POSE_HISTORY[-2])
                        VEC_V = P_curr - P_prev # Velocity is P_current - P_previous
                        
                        if len(HEAD_POSE_HISTORY) >= 3:
                            P_prev2 = np.array(HEAD_POSE_HISTORY[-3])
                            VEC_prev = P_prev - P_prev2
                            VEC_A = VEC_V - VEC_prev # Acceleration is V_current - V_previous
                    
                    # 3. Combine all 11 features for the model input
                    # ORDER MUST MATCH ADVANCED_MODEL_CLEAN.PY EXACTLY
                    current_features = np.concatenate([
                        [EAR, MAR],        # 2 Features (Log-transformed)
                        HEAD,                      # 3 Features (Position)
                        VEC_V,                     # 3 Features (Velocity)
                        VEC_A                      # 3 Features (Acceleration)
                    ]).astype(np.float32)

                    # 4. Append the full 11-feature vector to the main sequence buffer
                    feature_buffer.append(current_features)
                    
                    # END
                    
                    # Update last known feature values (using raw features for drawing visibility)
                    last_ear, last_mar = EAR, MAR 
                    last_head_pose = HEAD
                    last_rvec, last_tvec, last_camera_matrix = rvec, tvec, camera_matrix

                    # CHANGE TO: Require more frames of evidence before running the model
                    # 15 frames is 0.5 seconds, 30 frames is 1 second (safer)
                    if len(feature_buffer) >= 30: # UPDATED from 5 to 30
                        X_batch, _ = preprocess_sequence(feature_buffer, mean, std, MAX_SEQUENCE_LENGTH) # Use the new MAX_SEQUENCE_LENGTH


                        
                        # Run inference on the preprocessed batch
                        inf_start = time.time()
                        y_pred_probs = model.predict(X_batch, verbose=0)
                        inf_end = time.time()
                        inference_times.append(inf_end - inf_start)

                        # Determine the predicted class index and label
                        y_pred_idx = np.argmax(y_pred_probs[0])
                        current_prediction = le.get(y_pred_idx, "Unknown")
                        prediction_color = ATTENTION_COLORS.get(current_prediction, DEFAULT_COLOR)

                        # State Consistency Check
                        if last_prediction_state is not None and current_prediction != last_prediction_state:
                            state_switch_count += 1
                        last_prediction_state = current_prediction

            else:
                # No face detected: reset
                feature_buffer.clear()
                current_prediction = "No Face Detected"
                prediction_color = DEFAULT_COLOR
                last_ear, last_mar, last_head_pose = 0.0, 0.0, np.array([0.0, 0.0, 0.0])
                last_rvec, last_tvec, last_camera_matrix = None, None, None


                # Reset last_prediction_state when face is lost
                last_prediction_state = None

            # Log Prediction for every frame (even if "No Face Detected")
            log_entry = {
                'frame': frame_count,
                'time_sec': (time.time() - start_time),
                'prediction': current_prediction,
                'is_face_detected': (face_landmarks_pixel_coords is not None)
            }
            prediction_log.append(log_entry)

            
            # 8.3 Draw the detailed feature overlays (EAR, MAR, Head Pose axes/text) 
            if face_landmarks_pixel_coords is not None and last_rvec is not None:
                draw_features_overlays(
                    frame=frame, 
                    face_landmarks=face_landmarks_pixel_coords,
                    frame_shape=(frame_height, frame_width), 
                    rvec=last_rvec, 
                    tvec=last_tvec, 
                    camera_matrix=last_camera_matrix,
                    EAR=last_ear, 
                    MAR=last_mar, 
                    HEAD=last_head_pose
                )

            # 8.4 Overlay main prediction text 
            text = f"State: {current_prediction}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, prediction_color, 2, cv2.LINE_AA)

            # 8.5 Overlay a grid indicator
            cv2.rectangle(frame, (0, 0), (frame_width, frame_height), prediction_color, 10)

            if is_webcam:
                window_name = 'Multi-Level Attention Recognition'
                cv2.imshow(window_name, frame)
                
                # Exit the webcam

                # Wait for key press. If 'q' or 'ESC' is pressed, 'k' will hold the value.
                k = cv2.waitKey(5) & 0xFF

                # Check for key press ('q' or 'ESC') OR window closure ('x' button)
                if k in [ord('q'), 27] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break # Exit the while loop
            else:
                out.write(frame)

    # 9. Efficiency Metrics Calculation
    end_time = time.time()
    total_time = end_time - start_time
    total_frames = frame_count
    avg_fps = total_frames / (total_time + 1e-8)
    avg_latency_ms = (total_time / (total_frames + 1e-8)) * 1000

    avg_inference_latency_ms = 0
    if inference_times:
        avg_inference_latency_ms = (np.mean(inference_times)) * 1000

    # Calculate Switch Rate
    switch_rate_per_min = (state_switch_count / total_time) * 300 if total_time > 0 else 0

    print("\n--- System Summary ---")
    print(f"Total Frames Processed: {total_frames}")
    print(f"Total Processing Time: {total_time:.2f} seconds")
    print(f"Overall Throughput (FPS): {avg_fps:.2f} FPS")
    print(f"Average Processing Latency (per frame): {avg_latency_ms:.2f} ms")
    print(f"Average **Model Inference** Latency: {avg_inference_latency_ms:.2f} ms")
    print(f"Total State Switches: {state_switch_count}")
    print(f"State Switch Rate: {switch_rate_per_min:.2f} switches/min")
    
    if not is_webcam:
        print(f"Output video saved to: {output_path}")

    # 10. Save Prediction Log
    try:
        log_df = pd.DataFrame(prediction_log)
        # Ensure output_path is available or create a default log file name
        if is_webcam:
            log_filename = f"webcam_predictions_{int(time.time())}.csv"
        else:
            base_name = os.path.splitext(os.path.basename(input_source))[0]
            log_filename = f"{base_name}_predictions.csv"

        log_df.to_csv(log_filename, index=False)
        print(f"Prediction log saved to: {log_filename}")
    except Exception as e:
        print(f"Warning: Failed to save prediction log. {e}")

    # 11. Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    set_seed(42)
    
    # Autodetect model directory
    if MODEL_DIR == "runs_advanced/20251122-090120":
        try:
            runs = [d for d in os.listdir("runs_advanced") if os.path.isdir(os.path.join("runs_advanced", d))]
            runs.sort(reverse=True)
            if runs:
                MODEL_DIR = os.path.join("runs_advanced", runs[0])
                MODEL_PATH = os.path.join(MODEL_DIR, 'best_advanced_model_tf.keras')
                NORMALIZATION_STATS_PATH = os.path.join(MODEL_DIR, 'advanced_normalization_stats.json')
                LABEL_ENCODER_CLASSES_PATH = os.path.join(MODEL_DIR, 'label_encoder_classes.json')
                print(f"Auto-detected latest run directory: {MODEL_DIR}")
            else:
                 print("Could not find any run directory in 'runs_advanced'. Please run the training script first.")
                 exit()
        except FileNotFoundError:
            print("Could not find 'runs_advanced' directory. Please create it and run the training script first.")
            exit()
        except Exception as e:
            print(f"An error occurred during auto-detection: {e}")
            exit()

    # Input Selection Logic
    print("\n--- Input Selection ---")
    
    parser = argparse.ArgumentParser(description="Multi-Level Attention Recognition")
    parser.add_argument('--webcam', action='store_true', help='Use the default webcam (ID 0).')
    parser.add_argument('--video', type=str, default=INPUT_VIDEO_PATH, help=f'Path to the input video file (default: {INPUT_VIDEO_PATH}).')
    args = parser.parse_args()

    input_source = None
    
    # Determine input source based on command-line arguments
    if args.webcam:
        print("Mode: Webcam (Live Feed)")
        input_source = WEBCAM_ID
    elif os.path.exists(args.video):
        print(f"Mode: Video File (Path: {args.video})")
        input_source = args.video
    else:
        print(f"WARNING: Video file '{args.video}' not found.")
        print("Falling back to Webcam mode (ID 0).")
        input_source = WEBCAM_ID

    # Run the full pipeline
    run_end_to_end_system(
        input_source=input_source,
        model_path=MODEL_PATH,
        stats_path=NORMALIZATION_STATS_PATH,
        labels_path=LABEL_ENCODER_CLASSES_PATH,
        max_len=MAX_SEQUENCE_LENGTH,
        output_path=OUTPUT_VIDEO_PATH
    )