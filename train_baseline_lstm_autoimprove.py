import pandas as pd
import numpy as np
import json
import ast
import os
import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import time


# SCRIPT 9
# Description: This script contains the LSTM Baseline Model for Attention State Classification
# - Trains LSTM on DAiSEE + Collected Dataset combined
# - Saves normalization stats, reports, and visualizations
# - Adds run versioning and misclassification tracking


# =================
# Utility Functions
# =================

# Convert string like '[np.float64(0.3), np.float64(0.4)]' -> [0.3, 0.4].
def parse_seq(seq_str):
    if isinstance(seq_str, str):
        seq_str = seq_str.replace('np.float64', '').replace('(', '').replace(')', '')
        try:
            return np.array(ast.literal_eval(seq_str))
        except Exception:
            return np.array([])
    return np.array([])

# Convert string of tuples -> np.array of [yaw, pitch, roll].
def parse_headpose(seq_str):
    if isinstance(seq_str, str):
        try:
            seq = ast.literal_eval(seq_str)
            return np.array(seq)
        except Exception:
            return np.zeros((1, 3))
    return np.zeros((1, 3))

# Load a feature CSV, process and pad sequences for LSTM input.
def load_and_process(csv_path, max_len=90):
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Parse sequences from string representations
    ear = [parse_seq(x) for x in df['EAR_seq']]
    mar = [parse_seq(x) for x in df['MAR_seq']]
    headpose = [parse_headpose(x) for x in df['HeadPose_seq']]

    X, valid_indices = [], []
    # Filter out sequences with empty features
    for i, (e, m, h) in enumerate(zip(ear, mar, headpose)):
        # Skip sequences with any empty feature
        if len(e) == 0 or len(m) == 0 or len(h) == 0:
            continue
        # Determine the minimum sequence length among features
        min_len = min(len(e), len(m), len(h))
        # Stack features into a single sequence array
        seq = np.column_stack([
            e[:min_len],     # EAR
            m[:min_len],     # MAR
            h[:min_len, 0],  # Yaw
            h[:min_len, 1],  # Pitch
            h[:min_len, 2]   # Roll
        ])
        # Append the processed sequence and its index
        X.append(seq)
        valid_indices.append(i)

    # Pad sequences to the maximum length
    X_pad = pad_sequences(X, maxlen=max_len, dtype='float32', padding='post', truncating='post')
    # Extract labels, file IDs, and sources for valid sequences
    labels = df['CustomLabel'].iloc[valid_indices].values
    file_ids = df.iloc[valid_indices].get('ClipID', pd.Series(range(len(valid_indices)))).values
    sources = df['source'].iloc[valid_indices].values if 'source' in df.columns else None
    
    return X_pad, labels, file_ids, sources


# =============
# Load Datasets
# =============
# data_dir = "combined_features" for collected dataset and DAiSEE dataset
data_dir = "collected_features_origframe" # for collected dataset only
train_csv = os.path.join(data_dir, "train_features.csv")
val_csv = os.path.join(data_dir, "val_features.csv")
test_csv = os.path.join(data_dir, "test_features.csv")

print("Loading data from:", data_dir)
max_len = 90   # Maximum sequence length for padding

# Load and process datasets
X_train, y_train, _, _ = load_and_process(train_csv, max_len)
X_val, y_val, _, _ = load_and_process(val_csv, max_len)
X_test, y_test, test_files, test_sources = load_and_process(test_csv, max_len)

print(f"\nData loaded successfully!")
print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")


# ============================
# Normalize Features (Z-score)
# ============================
print("\nNormalizing input features...")

mean = np.mean(X_train, axis=(0, 1), keepdims=True)
std = np.std(X_train, axis=(0, 1), keepdims=True)

# Save normalization statistics in the main folder of the project
norm_stats = {'mean': mean.tolist(), 'std': std.tolist()}
with open('normalization_stats.json', 'w') as f:
    json.dump(norm_stats, f)
print("Saved normalization stats to normalization_stats.json")

# Normalize datasets using the computed mean and std
X_train = (X_train - mean) / (std + 1e-8)
X_val = (X_val - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)


# =============
# Encode Labels
# =============
le = LabelEncoder()
# Encode labels for training, validation, and test sets
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform([y if y in le.classes_ else le.classes_[0] for y in y_val])
y_test_enc = le.transform([y if y in le.classes_ else le.classes_[0] for y in y_test])
num_classes = len(le.classes_)

print("\nClass labels found:", list(le.classes_))


# =====================
# Compute Class Weights
# =====================
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
class_weights = dict(enumerate(class_weights))
print("\nComputed class weights:", class_weights)


# ============
# Define Model
# ============
# The baseline LSTM model architecture
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(max_len, 5)),
    Dropout(0.3),

    LSTM(64, return_sequences=False),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])

# Define optimizer
optimizer = Adam(learning_rate=1e-3)
# Compile the model with the optimizer, loss function, and evaluation metric
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ========
# Training
# ========
# Define run ID and directory for saving models and logs
run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = f"runs/{run_id}"
os.makedirs(run_dir, exist_ok=True)

# Define callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f'{run_dir}/best_lstm_model.keras', monitor='val_accuracy', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
]

# Train the model
history = model.fit(
    X_train, y_train_enc,
    validation_data=(X_val, y_val_enc),     # Validation data for monitoring
    epochs=120,                             # Number of training epochs
    batch_size=16,                          # Number of samples per gradient update
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,                              # Display training progress
    shuffle=True                            # Shuffle training data each epoch   
)


# ==========
# Evaluation
# ==========
print("\nEvaluating on test set...")
# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test_enc)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Predict class probabilities and labels for the test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Save detailed classification report
report = classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f"{run_dir}/classification_report.csv")
print("\nClassification Report:")
print(report_df)

# Weighted summary
precision, recall, f1, _ = precision_recall_fscore_support(y_test_enc, y_pred, average='weighted')
accuracy = accuracy_score(y_test_enc, y_pred)

# Save weighted summary metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
    'Value': [accuracy, precision, recall, f1]
})
metrics_df.to_csv(f"{run_dir}/evaluation_metrics.csv", index=False)
print(f"\nSaved weighted metrics to {run_dir}/evaluation_metrics.csv")
print(metrics_df)


# ================
# Confusion Matrix
# ================
# Plot and save confusion matrix
cm = confusion_matrix(y_test_enc, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Attention State Classification')
plt.savefig(f"{run_dir}/confusion_matrix.png", dpi=300)
plt.show()

# Plot and save normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6,5))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix')
plt.savefig(f"{run_dir}/confusion_matrix_normalized.png", dpi=300)
plt.show()


# ================
# Save Predictions
# ================
# Save detailed predictions including probabilities for each class
pred_df = pd.DataFrame({
    'ClipID': test_files,
    'Source': test_sources,
    'TrueLabel': le.inverse_transform(y_test_enc),
    'PredictedLabel': le.inverse_transform(y_pred),
})
for i, cls in enumerate(le.classes_):
    pred_df[f'Prob_{cls}'] = y_pred_probs[:, i]

pred_df.to_csv(f"{run_dir}/test_predictions.csv", index=False)
print(f"\nSaved test predictions to {run_dir}/test_predictions.csv")

# Save misclassified samples
errors = pred_df[pred_df['TrueLabel'] != pred_df['PredictedLabel']]
errors.to_csv(f"{run_dir}/misclassified_samples.csv", index=False)
print(f"Saved {len(errors)} misclassified samples to {run_dir}/misclassified_samples.csv")


# ====================
# Plot Training Curves
# ====================
# Plot training and validation accuracy
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

# Plot training and validation loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()

# Show plot 
plt.tight_layout()
plt.savefig(f"{run_dir}/training_curves.png", dpi=300)
plt.show()

# ==============================================================
# Performance & Temporal Stability Metrics (Simulated Real-Time)
# ==============================================================
# This section simulates the frame-by-frame inference of the real-time script
# to calculate efficiency and temporal stability metrics on the test set.
print("\n" + "="*60)
print("--- Performance & Temporal Stability Metrics (Simulated Real-Time) ---")
print("="*60)

# 1. Prepare for Frame-by-Frame Inference Simulation
total_frames = len(X_test)
inference_times = []
state_predictions = []
total_time_inference = 0.0

# 2. Simulate Frame-by-Frame Inference and Timing
# Timing predictions individually is crucial for a real-time simulation
for i in range(total_frames):
    # Get a single sample (a sequence/clip) and ensure it has the batch dimension (1, SEQ_LEN, FEATURES)
    sample_X = np.expand_dims(X_test[i], axis=0) 
    
    # Start timer for Model Inference
    start_time = time.time()
    
    # Model Inference (predict class probabilities for the sample sequence)
    # verbose=0 suppresses the Keras progress bar
    y_prob = model.predict(sample_X, verbose=0)[0]
    
    end_time = time.time()
    
    # Calculate time taken for this single inference
    inference_latency = end_time - start_time
    inference_times.append(inference_latency)
    total_time_inference += inference_latency
    
    # Get the predicted class index (state)
    predicted_class_index = np.argmax(y_prob)
    state_predictions.append(predicted_class_index)


# 3. Calculate Efficiency Metrics
total_time = total_time_inference # Use total inference time as the base for all calculations
avg_inference_latency_s = np.mean(inference_times) if inference_times else 0
avg_inference_latency_ms = avg_inference_latency_s * 1000

# The "Average Processing Latency" is the same as "Model Inference Latency" in this simple simulation
avg_latency_ms = avg_inference_latency_ms 

# Overall Throughput (Frames Per Second)
avg_fps = total_frames / total_time if total_time > 0 else 0


# 4. Calculate Temporal Stability (State Switches)
state_switch_count = 0
if total_frames > 0:
    # A switch occurs when the prediction of the current sequence is different from the previous one
    for i in range(1, total_frames):
        if state_predictions[i] != state_predictions[i-1]:
            state_switch_count += 1

# 5. Calculate State Switch Rate (switches per minute)
total_time_min = total_time / 60.0
switch_rate_per_min = state_switch_count / total_time_min if total_time_min > 0 else 0


# 6. Print the Final Summary (Matching Real-Time Output)
print("\n--- System Summary ---")
print(f"Total Frames Processed: {total_frames}")
print(f"Total Processing Time: {total_time:.2f} seconds")
print(f"Overall Throughput (FPS): {avg_fps:.2f} FPS")
print(f"Average Processing Latency (per frame): {avg_latency_ms:.2f} ms")
print(f"Average **Model Inference** Latency: {avg_inference_latency_ms:.2f} ms")
# --- Report Consistency Metric ---
print(f"Total State Switches: {state_switch_count}")
print(f"State Switch Rate: {switch_rate_per_min:.2f} switches/min")


print(f"\nAll run artifacts saved in: {run_dir}")
