#!/usr/bin/env python3

import os
import re
import ast
import json
import datetime
import random
from typing import Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Masking, Conv1D, LayerNormalization, Activation,
    Add, Dropout, Dense, GlobalAveragePooling1D, SpatialDropout1D,
    BatchNormalization, Bidirectional, GRU, LSTM, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import losses
from tensorflow.keras.layers import Layer


# SCRIPT 10
# Description: Advanced TCN + Attention model for Attention State Classification
# Note: Replaced Lambda layers with serializable custom Keras Layers

# =============================
# Reproducibility / Environment
# =============================
def set_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

# ==========================
# Serializable custom layers
# ==========================
from tensorflow import keras

@tf.keras.utils.register_keras_serializable(package="Custom")    # for serialization which is important for deployment and reproducibility
class SliceFeatures(Layer):
    # Slice features along the last axis: returns x[:, :, start:end]
    def __init__(self, start: int, end: int, **kwargs):
        super().__init__(**kwargs)
        self.start = int(start)
        self.end = int(end)

    def call(self, inputs):
        return inputs[:, :, self.start:self.end]

    # Method to support masking
    def compute_mask(self, inputs, mask=None):
        # If there is an input mask, the mask for the sliced output is the same.
        # Slicing features (the last dimension) does not change the sequence length or which time steps are masked.
        return mask

    # Method to support serialization
    def get_config(self):
        config = super().get_config()
        config.update({"start": self.start, "end": self.end})
        return config


@tf.keras.utils.register_keras_serializable(package="Custom")
class MaskFromInput(Layer):
    # Compute mask boolean float: 1.0 if any feature != 0 across feature axis, else 0.0
    def call(self, inputs):
        m = tf.reduce_any(tf.not_equal(inputs, 0.0), axis=-1)
        return tf.cast(m, tf.float32)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="Custom")
class ExpandMaskTo4D(Layer):
    # Expand mask to shape (batch, 1, 1, seq_len) and invert (1.0 - m)
    def call(self, inputs):
        m = tf.cast(inputs, tf.float32)
        m = 1.0 - m
        m = tf.expand_dims(m, axis=1)
        m = tf.expand_dims(m, axis=1)
        return m

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="Custom")
class ScaledResidual(Layer):
    # Scale the input by a constant factor (default 0.5)
    def __init__(self, scale=0.5, **kwargs):
        super().__init__(**kwargs)
        self.scale = float(scale)

    def call(self, inputs):
        return inputs * tf.cast(self.scale, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


# =========================================
# Parsing utilities (vectorized and robust)
# =========================================
# Compile regex pattern for floating point numbers
_num_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Function to convert various inputs to a list of floats
def _to_float_list(obj):
    if isinstance(obj, (list, tuple)):
        return [float(x) for x in obj]
    if isinstance(obj, (int, float)):
        return [float(obj)]
    s = str(obj)
    found = _num_pattern.findall(s)
    if found:
        return [float(x) for x in found]
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
        if isinstance(val, (int, float)):
            return [float(val)]
    except Exception:
        pass
    return []

# Function to parse a sequence string into a numpy array of floats
def parse_seq(seq_str):
    if seq_str is None or (isinstance(seq_str, float) and np.isnan(seq_str)):
        return np.array([], dtype=np.float32)
    try:
        fl = _to_float_list(seq_str)
        return np.array(fl, dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)

# Function to parse headpose sequence string into a numpy array of shape (N, 3)
def parse_headpose(seq_str):
    if seq_str is None or (isinstance(seq_str, float) and np.isnan(seq_str)):
        return np.zeros((0, 3), dtype=np.float32)
    try:
        val = ast.literal_eval(seq_str) if isinstance(seq_str, str) else seq_str
        if isinstance(val, (list, tuple)) and len(val) > 0:
            arr = np.array(val, dtype=np.float32)
            if arr.ndim == 1:
                if arr.size % 3 == 0:
                    arr = arr.reshape(-1, 3)
                else:
                    return np.zeros((0, 3), dtype=np.float32)
            if arr.shape[1] >= 3:
                return arr[:, :3].astype(np.float32)
    except Exception:
        pass
    fl = _to_float_list(seq_str)
    if len(fl) % 3 == 0 and len(fl) > 0:
        return np.array(fl, dtype=np.float32).reshape(-1, 3)
    return np.zeros((0, 3), dtype=np.float32)


# ========================
# Data loader with caching
# ========================
# Load data from CSV, parse sequences, pad, and cache results
def load_and_process(csv_path: str, max_len: int = 90, cache_dir: str = "cache") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    cache_path = os.path.join(cache_dir, f"{base}.npz")

    print("ENTERRRRRRRRRRRRRRRRRRRRRRRRRRRRR")

    # Check if cached data exists and load it
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            return cached['X'], cached['y'], cached['files'], cached['sources']
        except Exception:
            pass

    print(f"Loading and parsing {csv_path} ...")
    df = pd.read_csv(csv_path)

    print("Read CSV!!!!!!!!!!!!!!!!!!!!")

    # Parse sequences from DataFrame columns
    ear = [parse_seq(x) for x in df.get('EAR_seq', [])]
    mar = [parse_seq(x) for x in df.get('MAR_seq', [])]
    headpose = [parse_headpose(x) for x in df.get('HeadPose_seq', [])]

    X = []
    valid_indices = []

    
    # Filter out sequences with empty or invalid data
    for i, (e, m, h) in enumerate(zip(ear, mar, headpose)):
        # Skip sequences with any empty or invalid data
        if e.size == 0 or m.size == 0 or h.size == 0:
            continue
        min_len = min(len(e), len(m), len(h))
        print(f"\nmin_len: {min_len}")
        # Ensure minimum length is positive
        if min_len <= 0:
            continue

        # start of add for velocity and acceleration features
        # --- 1. Align sequences to min_len ---
        h_aligned = h[:min_len, :]

        # --- 2. Calculate Velocity (First Derivative) ---
        # Velocity = diff(position). np.diff reduces sequence length by 1.
        head_v_seq = np.diff(h_aligned, axis=0) 
        
        # Pad with one zero vector at the start to match length min_len
        head_v_seq = np.concatenate([np.zeros((1, 3), dtype=np.float32), head_v_seq], axis=0)
        
        # --- 3. Calculate Acceleration (Second Derivative) ---
        # Acceleration = diff(velocity). np.diff reduces sequence length by 1.
        head_a_seq = np.diff(head_v_seq, axis=0) 

        # Pad with one zero vector at the start to match length min_len
        head_a_seq = np.concatenate([np.zeros((1, 3), dtype=np.float32), head_a_seq], axis=0)

        # --- 4. Combine ALL features (5 original + 6 derivatives = 11 total) ---
        seq = np.column_stack([
            e[:min_len],       # 0: EAR sequence
            m[:min_len],       # 1: MAR sequence
            h_aligned[:, 0],   # 2: HeadPose Yaw
            h_aligned[:, 1],   # 3: HeadPose Pitch
            h_aligned[:, 2],   # 4: HeadPose Roll
            head_v_seq[:, 0],  # 5: Yaw_V
            head_v_seq[:, 1],  # 6: Pitch_V
            head_v_seq[:, 2],  # 7: Roll_V
            head_a_seq[:, 0],  # 8: Yaw_A
            head_a_seq[:, 1],  # 9: Pitch_A
            head_a_seq[:, 2]   # 10: Roll_A
        ])

        #### end of add

        # Combine sequences into a single array with columns for each feature
        # seq = np.column_stack([
        #     e[:min_len],     # EAR sequence
        #     m[:min_len],     # MAR sequence
        #     h[:min_len, 0],  # HeadPose Yaw
        #     h[:min_len, 1],  # HeadPose Pitch
        #     h[:min_len, 2],  # HeadPose Roll
        # ])
        # Append the combined sequence and record the valid index
        X.append(seq)
        valid_indices.append(i)

    # Check if any valid sequences were found
    if len(X) == 0:
        raise ValueError(f"No valid sequences parsed from {csv_path}")

    # Pad sequences to the maximum length
    X_pad = tf.keras.preprocessing.sequence.pad_sequences(
        X, maxlen=max_len, dtype='float32', padding='post', truncating='post', value=0.0
    )
    X_pad = np.asarray(X_pad, dtype=np.float32)
    X_pad = np.nan_to_num(X_pad, nan=0.0)

    # Extract labels, file IDs, and sources for valid sequences
    labels = df['CustomLabel'].iloc[valid_indices].values
    file_ids = df.iloc[valid_indices].get('ClipID', pd.Series(range(len(valid_indices)))).values
    sources = df['source'].iloc[valid_indices].values if 'source' in df.columns else np.array([''] * len(valid_indices))

    # Save processed data to cache
    try:
        np.savez_compressed(cache_path, X=X_pad, y=labels, files=file_ids, sources=sources)
    except Exception:
        pass

    return X_pad, labels, file_ids, sources

# ==================
# TCN residual block
# ==================
def tcn_residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.1, use_batchnorm=False, name=None):
    # First convolutional layer with causal padding and dilation
    conv1 = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(x)
    if use_batchnorm:
        conv1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(conv1)
    dr1 = Dropout(dropout_rate)(act1)

    # Second convolutional layer with causal padding and dilation
    conv2 = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate)(dr1)
    if use_batchnorm:
        conv2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(conv2)
    dr2 = Dropout(dropout_rate)(act2)

    # Shortcut connection to match the input and output dimensions
    if x.shape[-1] != filters:
        proj = Conv1D(filters, 1, padding='same')(x)
    else:
        proj = x

    # Add the shortcut connection to the output of the second convolutional layer
    out = Add()([proj, dr2])
    out = Activation('relu')(out)
    return out

# =====================
# TemporalSelfAttention
# =====================
@tf.keras.utils.register_keras_serializable(package="Custom")
class TemporalSelfAttention(tf.keras.layers.Layer):
    # Temporal Self-Attention Layer
    def __init__(self, d_model, num_heads=4, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.dropout_rate = dropout

    # Build the layer
    def build(self, input_shape):
        # build Dense layers with known shapes
        self.wq = Dense(self.d_model)
        self.wk = Dense(self.d_model)
        self.wv = Dense(self.d_model)
        self.out_dense = Dense(self.d_model)
        self.dropout = Dropout(self.dropout_rate)
        super().build(input_shape)

    # Split the last dimension into (num_heads, depth) and transpose for attention calculation
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # Forward pass
    def call(self, v, k, q, mask=None, training=False):
        batch_size = tf.shape(q)[0]
        # Apply linear layers to queries, keys, and values
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split heads for multi-head attention
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Calculate scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Apply the attention mask (if provided) to the scaled attention logits
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Calculate attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        # Calculate the weighted sum of the values
        scaled_attention = tf.matmul(attention_weights, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.out_dense(concat_attention)
        return output


# ====
# Loss
# ====
def sparse_focal_loss(gamma=2.0, alpha=0.25):
    @tf.function
    def loss_fn(y_true, y_pred):
        # Calculate the number of classes from the predictions
        num_classes = tf.shape(y_pred)[-1]
        # One-hot encode the true labels
        y_true_ohe = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=num_classes)
        # Clip predictions to prevent log(0) error
        y_pred_clipped = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        # Calculate cross-entropy
        cross_entropy = -y_true_ohe * tf.math.log(y_pred_clipped)
        # Calculate the modulating factor for focal loss
        weight = alpha * tf.math.pow(1 - y_pred_clipped, gamma)
        loss = weight * cross_entropy
        loss = tf.reduce_sum(loss, axis=-1)
        return tf.reduce_mean(loss)
    return loss_fn

# ============================================================
# Build advanced model (with custom layers instead of Lambda)
# ============================================================
def build_advanced_tcn_attention(max_len=90, n_features=10, num_classes=3,
                                 tcn_filters=64, tcn_kernel=3, tcn_layers=6,
                                 attention_dim=128, attention_heads=4, dropout=0.2,
                                 use_batchnorm=False):
    # Define input layer
    inp = Input(shape=(max_len, n_features), name='input_features')
    x = Masking(mask_value=0.0)(inp)

    # Dual-Stream Feature Fusion
    # Extract EAR + MAR features
    ear_mar_input = SliceFeatures(0, 2)(x)     # EAR + MAR
    # Extract Head Pose features (yaw, pitch, roll)
    #headpose_input = SliceFeatures(2, 5)(x)    # Head Pose (yaw, pitch, roll)
    # Extract Head Pose features (yaw, pitch, roll) + Derivatives (9 features total)
    headpose_input = SliceFeatures(2, 11)(x)    # UPDATED from SliceFeatures(2, 5)

    # Apply Conv1D layers to each feature stream
    ear_mar_branch = Conv1D(32, 3, padding='same', activation='relu')(ear_mar_input)
    headpose_branch = Conv1D(32, 3, padding='same', activation='relu')(headpose_input)

    # Merge the two feature streams
    merged = Concatenate(axis=-1)([ear_mar_branch, headpose_branch])

    # Apply initial TCN residual block
    x = tcn_residual_block(merged, 64, 3, 1, dropout_rate=0.2)

    # Apply stacked TCN residual blocks with increasing dilation rates
    for i in range(tcn_layers):
        dilation = 2 ** i
        x = tcn_residual_block(x, filters=tcn_filters, kernel_size=tcn_kernel,
                               dilation_rate=dilation, dropout_rate=dropout,
                               use_batchnorm=use_batchnorm, name=f"tcn_block_{i}")

    # Apply spatial dropout and bidirectional LSTM
    x = SpatialDropout1D(rate=min(0.2, dropout))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    # Project to attention dimension
    x = Conv1D(attention_dim, kernel_size=1, padding='same', activation='relu')(x)

    # Build attention mask from Keras mask: True for valid timesteps
    mask_bool = MaskFromInput()(inp)  # shape (batch, seq_len) float 0/1
    attn_mask = ExpandMaskTo4D()(mask_bool)  # shape (batch, 1, 1, seq_len)

    # Apply temporal self-attention
    attn = TemporalSelfAttention(d_model=attention_dim, num_heads=attention_heads, dropout=dropout)(x, x, x, mask=attn_mask)
    # Add residual connection with scaling
    x = Add()([x, ScaledResidual(0.5)(attn)])
    # Apply layer normalization
    x = LayerNormalization()(x)

    # Pooling and final dense layers
    pooled = GlobalAveragePooling1D()(x)
    pooled = Dropout(dropout)(pooled)
    dense = Dense(128, activation='relu')(pooled)
    dense = Dropout(dropout)(dense)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(dense)

    model = Model(inputs=inp, outputs=outputs, name='tcn_attention_model_v2')
    return model

# =======================
# Learning Rate scheduler
# =======================
def lr_warmup_scheduler(warmup_epochs=5, base_lr=1e-3):
    def scheduler(epoch, lr):
        # Linear warmup of learning rate
        if epoch < warmup_epochs:
            return base_lr * (float(epoch + 1) / float(warmup_epochs))
        return lr
    return scheduler

# ===============================
# Utility: safe inverse_transform
# ===============================
def safe_inverse_transform(le: LabelEncoder, arr: np.ndarray):
    try:
        # Attempt standard inverse transform
        return le.inverse_transform(arr)
    except Exception:
        mapping = {i: c for i, c in enumerate(le.classes_)}
        return np.array([mapping.get(int(x), le.classes_[0]) for x in arr])

# ===========================
# Main training/eval pipeline
# ===========================
def main():
    set_seed(42)  # Set random seed for reproducibility

    # Define data directory and CSV file paths
    # data_dir = "combined_features"  # for collected dataset and DAiSEE dataset
    data_dir = "collected_features_3sec"
    train_csv = os.path.join(data_dir, "train_features.csv")
    val_csv = os.path.join(data_dir, "val_features.csv")
    test_csv = os.path.join(data_dir, "test_features.csv")

    # Define model and training parameters
    max_len = 90 # Changed from 60 to accommodate full 10-second clips at 30fps
    batch_size = 10 # Changed from 18 to conserve memory with longer sequences.
    epochs = 120
    learning_rate = 1e-3
    focal_gamma = 1.5
    focal_alpha = 0.35
    label_smoothing = 0.2 # Changed from 0.05 to encourage the model not to be overconfident in its predictions, thus allowing some margin for 'Focused' predictions near the decision boundary.
    use_focal = True
    use_batchnorm = False

    # Create directories for this run
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = f"runs_advanced/{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    tb_logdir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tb_logdir, exist_ok=True)

    # Load and process datasets
    X_train, y_train, _, _ = load_and_process(train_csv, max_len)
    X_val, y_val, _, _ = load_and_process(val_csv, max_len)
    X_test, y_test, test_files, test_sources = load_and_process(test_csv, max_len)

    # Print data shapes for verification
    print(f"\nData shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Compute and save normalization statistics
    #mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    #std = np.std(X_train, axis=(0, 1), keepdims=True)

    # FIX: REMOVE keepdims=True to save a clean (11,) vector
    mean = np.mean(X_train, axis=(0, 1)) # Shape will be (11,) # Compute mean across all samples (0) and time steps (1)
    std = np.std(X_train, axis=(0, 1))   # Shape will be (11,) # Compute std across all samples (0) and time steps (1)                                                         

    norm_stats = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(os.path.join(run_dir, 'advanced_normalization_stats.json'), 'w') as f:
        json.dump(norm_stats, f)

    # Normalize datasets
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # Encode labels with fallback for unseen classes
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    def _encode_with_fallback(arr, fitted_le):
        out = []
        for v in arr:
            if v in fitted_le.classes_:
                out.append(int(np.where(fitted_le.classes_ == v)[0][0]))
            else:
                out.append(0)
        return np.array(out, dtype=np.int32)
    y_val_enc = _encode_with_fallback(y_val, le)
    y_test_enc = _encode_with_fallback(y_test, le)

    # Determine number of classes and print class labels
    num_classes = len(le.classes_)
    print("Class labels:", list(le.classes_))

    # Compute class weights to handle class imbalance
    class_weights_arr = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_train_enc)
    class_weights = dict(enumerate(class_weights_arr))
    print("Computed class weights:", class_weights)
    

    # Build the advanced TCN with attention model
    # tcn_layers parameters was changed from 4 to 6 to significantly increase the TCN's receptive field to better cover the 90-frame sequence.
    model = build_advanced_tcn_attention(max_len=max_len, n_features=X_train.shape[-1], num_classes=num_classes,
                                         tcn_filters=64, tcn_kernel=3, tcn_layers=6,
                                         attention_dim=128, attention_heads=4, dropout=0.2,
                                         use_batchnorm=use_batchnorm)
    model.summary()

    # Estimate model parameters and FLOPs
    param_count = model.count_params()
    flops = None  
    try:
        concrete = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, max_len, X_train.shape[-1]], tf.float32))
        try:
            # Attempt to import TensorFlow profiler modules for FLOPs calculation
            from tensorflow.python.profiler import model_analyzer
            from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
            profiler = model_analyzer.Profile(model=concrete)
        except Exception:
            flops = None
    except Exception:
        flops = None

    print(f"Model parameters: {param_count}")

    # Define loss function based on whether focal loss is used
    if use_focal:
        loss_fn = sparse_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
    else:
        # Define temporal smoothing loss function
        def smooth_loss(y_true, y_pred):
            num_classes_local = tf.shape(y_pred)[-1]
            y_true_ohe = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=num_classes_local)
            # Apply label smoothing if specified
            if label_smoothing > 0:
                y_true_ohe = y_true_ohe * (1.0 - label_smoothing) + label_smoothing / tf.cast(num_classes_local, tf.float32)
            return tf.reduce_mean(losses.categorical_crossentropy(y_true_ohe, y_pred))
        loss_fn = smooth_loss

    # Compile the model with the AdamW optimizer and loss function
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-5)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Set up callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ModelCheckpoint(os.path.join(run_dir, 'best_advanced_model.keras'), monitor='val_accuracy', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.LearningRateScheduler(lr_warmup_scheduler(warmup_epochs=5, base_lr=learning_rate)),
        tf.keras.callbacks.TensorBoard(log_dir=tb_logdir)
    ]

    # Save training configuration
    config = {
        'timestamp': run_id,
        'seed': 42,
        'max_len': max_len,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'use_focal': use_focal,
        'focal_gamma': focal_gamma,
        'focal_alpha': focal_alpha,
        'label_smoothing': label_smoothing,
        'use_batchnorm': use_batchnorm,
        'param_count': int(param_count),
        'flops_estimate': flops
    }
    with open(os.path.join(run_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Save model architecture visualization
    try:
        tf.keras.utils.plot_model(model, to_file=os.path.join(run_dir, 'model_structure.png'), show_shapes=True)
    except Exception:
        pass

    # Train the model
    history = model.fit(
        X_train, y_train_enc,
        validation_data=(X_val, y_val_enc),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # Evaluate the model on the test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_enc, verbose=1)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Generate predictions and classification report
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test_enc, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(run_dir, 'classification_report.csv'))
    print('\nClassification Report:')
    print(report_df)

    # Calculate and save evaluation metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_enc, y_pred, average='weighted')
    accuracy = accuracy_score(y_test_enc, y_pred)
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_df.to_csv(os.path.join(run_dir, 'evaluation_metrics.csv'), index=False)
    print(metrics_df)

    # Generate and save confusion matrix plots
    cm = confusion_matrix(y_test_enc, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Advanced Model')
    plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # Generate and save normalized confusion matrix plot
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix - Advanced Model')
    plt.savefig(os.path.join(run_dir, 'confusion_matrix_normalized.png'), dpi=300)
    plt.close()

    # Generate and save per-class F1 score plot
    per_class = precision_recall_fscore_support(y_test_enc, y_pred, average=None, labels=range(num_classes))
    f1s = per_class[2]
    plt.figure(figsize=(8, 4))
    plt.bar(le.classes_, f1s)
    plt.ylabel('F1')
    plt.title('Per-class F1')
    plt.savefig(os.path.join(run_dir, 'per_class_f1.png'), dpi=300)
    plt.close()

    # Generate and save per-class F1 score plot
    pred_df = pd.DataFrame({
        'ClipID': test_files,
        'Source': test_sources,
        'TrueLabel': safe_inverse_transform(le, y_test_enc),
        'PredictedLabel': safe_inverse_transform(le, y_pred),
    })
    for i, cls in enumerate(le.classes_):
        pred_df[f'Prob_{cls}'] = y_pred_probs[:, i]
    pred_df.to_csv(os.path.join(run_dir, 'test_predictions.csv'), index=False)

    # Save misclassified samples for further analysis
    errors = pred_df[pred_df['TrueLabel'] != pred_df['PredictedLabel']]
    errors.to_csv(os.path.join(run_dir, 'misclassified_samples.csv'), index=False)
    print(f"Saved {len(errors)} misclassified samples.")

    # Plot training and validation accuracy and loss curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history.get('accuracy', []), label='Train')
    plt.plot(history.history.get('val_accuracy', []), label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history.get('loss', []), label='Train')
    plt.plot(history.history.get('val_loss', []), label='Val')
    plt.title('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_curves.png'), dpi=300)
    plt.close()

    # final model save
    model.save(os.path.join(run_dir, 'final_advanced_model.keras'))
    with open(os.path.join(run_dir, 'label_encoder_classes.json'), 'w') as f:
        json.dump(list(le.classes_), f)

    # Convert best model for TF inference compatibility (now safe; no unsafe deserialization required)
    try:
        best_model_path = os.path.join(run_dir, 'best_advanced_model.keras')
        # Re-load properly with all custom layers
        model_best = tf.keras.models.load_model(
            best_model_path,
            compile=False,
            custom_objects={
                "SliceFeatures": SliceFeatures,
                "MaskFromInput": MaskFromInput,
                "ExpandMaskTo4D": ExpandMaskTo4D,
                "TemporalSelfAttention": TemporalSelfAttention,
                "ScaledResidual": ScaledResidual,
            },
        )

        # Force rebuild if any Dense layer is unbuilt
        dummy_input = tf.zeros((1, max_len, X_train.shape[-1]))
        _ = model_best(dummy_input)         # forward pass to ensure build

        # Save the rebuilt model
        model_best.save(
            os.path.join(run_dir, 'best_advanced_model_tf.keras'),
            include_optimizer=False
        )

        print("Successfully converted best_advanced_model.keras → best_advanced_model_tf.keras (portable).")
    except Exception as e:
        print(f"Conversion failed: {e}")

    print(f"\nAll run artifacts saved in: {run_dir}")

if __name__ == '__main__':
    main()
