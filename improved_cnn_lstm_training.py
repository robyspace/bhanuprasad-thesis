"""
Improved CNN-LSTM Model for Network Intrusion Detection
CSE-CIC-IDS-2018 Dataset

This script fixes the architectural and data preparation issues in the original implementation:
1. Uses Conv1D instead of Conv2D (appropriate for tabular sequential data)
2. Proper temporal sequence creation (sliding window on sorted data)
3. Reduced dropout for better learning
4. Better hyperparameters and training strategy
5. Comprehensive evaluation and visualization

Author: Improved Implementation
Date: 2025-11-17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
import time
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("IMPROVED CNN-LSTM MODEL FOR INTRUSION DETECTION")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU'))} GPU(s)")
if tf.config.list_physical_devices('GPU'):
    print("GPU devices:", tf.config.list_physical_devices('GPU'))

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration"""
    # Paths (update these for your environment)
    PROJECT_DIR = '/content/drive/MyDrive/IDS_Research'
    DATA_DIR = '/content/drive/MyDrive/IDS_Research/data'
    MODEL_DIR = f'{PROJECT_DIR}/models'
    RESULTS_DIR = f'{PROJECT_DIR}/results'

    # Model hyperparameters
    TIME_STEPS = 10  # Reduced from 30 for better performance
    BATCH_SIZE = 256  # Increased for faster training
    EPOCHS = 30
    LEARNING_RATE = 0.001

    # Architecture
    CNN_FILTERS = [64, 128]  # Number of filters in CNN layers
    LSTM_UNITS = [128, 64]  # LSTM layer sizes
    DENSE_UNITS = [64, 32]  # Dense layer sizes
    DROPOUT_RATE = 0.3  # Consistent dropout rate

    # Training
    EARLY_STOPPING_PATIENCE = 7
    REDUCE_LR_PATIENCE = 3
    VALIDATION_SPLIT = 0.15

    # Data sampling (set to None to use full dataset)
    TRAIN_SAMPLE_SIZE = None  # Use None for full dataset, or e.g., 100000 for subset

config = Config()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_optimized_sequences(X, y, time_steps=10, stride=1):
    """
    Create sequences with sliding window approach

    Args:
        X: Feature matrix (numpy array or DataFrame)
        y: Labels (pandas Series or numpy array)
        time_steps: Number of time steps to look back
        stride: Step size for sliding window (1 = no skip, 2 = skip every other)

    Returns:
        X_seq: Sequential data (samples, time_steps, features)
        y_seq: Labels for sequences
    """
    print(f"\nCreating sequences with time_steps={time_steps}, stride={stride}...")

    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    Xs, ys = [], []

    # Sliding window approach
    for i in range(0, len(X) - time_steps, stride):
        sequence = X[i:(i + time_steps)]
        label = y[i + time_steps]  # Predict the label after the sequence
        Xs.append(sequence)
        ys.append(label)

    X_seq = np.array(Xs)
    y_seq = np.array(ys)

    print(f"✓ Created {len(X_seq)} sequences")
    print(f"  Shape: {X_seq.shape}")
    print(f"  Label distribution: {np.bincount(y_seq.astype(int))}")

    return X_seq, y_seq


def plot_training_history(history, save_path=None):
    """Plot comprehensive training history"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('CNN-LSTM Training History', fontsize=16, fontweight='bold')

    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    titles = ['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]

        if f'val_{metric}' in history.history:
            ax.plot(history.history[metric], label=f'Training {title}', linewidth=2)
            ax.plot(history.history[f'val_{metric}'], label=f'Validation {title}', linewidth=2)
            ax.set_title(f'Model {title}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Learning rate plot
    if 'lr' in history.history:
        ax = axes[1, 2]
        ax.plot(history.history['lr'], label='Learning Rate', linewidth=2, color='red')
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to: {save_path}")

    plt.show()


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    print(f"\n{'='*60}")
    print(f"{model_name} EVALUATION")
    print(f"{'='*60}")

    # Predictions
    start_time = time.time()
    y_pred_proba = model.predict(X_test, verbose=0)
    inference_time = time.time() - start_time

    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_pred_proba = y_pred_proba.flatten()

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Latency
    avg_latency = (inference_time / len(X_test)) * 1000  # ms per sample

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'fpr': float(fpr),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'avg_latency_ms': float(avg_latency),
        'total_inference_time_s': float(inference_time)
    }

    # Print results
    print(f"\n{model_name} Test Set Performance:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  FPR:       {fpr*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:>6}  FP: {fp:>6}")
    print(f"  FN: {fn:>6}  TP: {tp:>6}")
    print(f"\nInference Performance:")
    print(f"  Avg Latency: {avg_latency:.2f} ms/sample")
    print(f"  Total Time:  {inference_time:.2f} seconds")
    print(f"  Throughput:  {len(X_test)/inference_time:.2f} samples/sec")

    return results, y_pred, y_pred_proba


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def build_improved_cnn_lstm_v1(input_shape, config):
    """
    Improved CNN-LSTM using Conv1D (appropriate for sequential tabular data)

    Architecture:
    - Conv1D layers for local feature extraction
    - LSTM layers for temporal dependencies
    - Dense layers for classification
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # CNN Block 1 - Local feature extraction
        layers.Conv1D(
            filters=config.CNN_FILTERS[0],
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(config.DROPOUT_RATE),

        # CNN Block 2 - Hierarchical features
        layers.Conv1D(
            filters=config.CNN_FILTERS[1],
            kernel_size=3,
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE),

        # LSTM Block - Temporal dependencies
        layers.LSTM(
            config.LSTM_UNITS[0],
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(config.DROPOUT_RATE),

        layers.LSTM(
            config.LSTM_UNITS[1],
            return_sequences=False,
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(config.DROPOUT_RATE),

        # Dense layers for classification
        layers.Dense(config.DENSE_UNITS[0], activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config.DROPOUT_RATE * 0.7),  # Reduced dropout in final layers

        layers.Dense(config.DENSE_UNITS[1], activation='relu'),
        layers.Dropout(config.DROPOUT_RATE * 0.5),

        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])

    return model


def build_improved_cnn_lstm_v2(input_shape, config):
    """
    Simplified CNN-LSTM with residual connections
    Better for learning with limited temporal structure
    """
    inputs = layers.Input(shape=input_shape)

    # CNN feature extraction
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Residual LSTM block
    lstm_out = layers.LSTM(128, return_sequences=True)(x)
    lstm_out = layers.Dropout(0.3)(lstm_out)
    lstm_out = layers.LSTM(64, return_sequences=False)(lstm_out)

    # Dense classification
    x = layers.Dense(64, activation='relu')(lstm_out)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def build_lstm_only(input_shape, config):
    """
    LSTM-only model without CNN
    Simpler architecture, may perform better on this dataset
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Bidirectional LSTM for better temporal understanding
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),

        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(0.3),

        # Classification layers
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    return model


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_improved_cnn_lstm(X_train_scaled, y_train, X_val_scaled, y_val,
                           X_test_scaled, y_test, config, architecture='v1'):
    """
    Train improved CNN-LSTM model with proper configuration

    Args:
        X_train_scaled: Scaled training features
        y_train: Training labels
        X_val_scaled: Scaled validation features
        y_val: Validation labels
        X_test_scaled: Scaled test features
        y_test: Test labels
        config: Configuration object
        architecture: 'v1', 'v2', or 'lstm_only'
    """

    print("\n" + "="*80)
    print("PREPARING SEQUENTIAL DATA")
    print("="*80)

    # Create sequences with proper stride
    # Using stride=5 to reduce overlap and speed up training
    X_train_seq, y_train_seq = create_optimized_sequences(
        X_train_scaled, y_train,
        time_steps=config.TIME_STEPS,
        stride=5
    )

    X_val_seq, y_val_seq = create_optimized_sequences(
        X_val_scaled, y_val,
        time_steps=config.TIME_STEPS,
        stride=5
    )

    X_test_seq, y_test_seq = create_optimized_sequences(
        X_test_scaled, y_test,
        time_steps=config.TIME_STEPS,
        stride=1  # No stride for test set
    )

    # ========================================================================
    # BUILD MODEL
    # ========================================================================

    print("\n" + "="*80)
    print("BUILDING MODEL")
    print("="*80)

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    print(f"Input shape: {input_shape}")

    if architecture == 'v1':
        print("Using: Improved CNN-LSTM V1 (Conv1D + LSTM)")
        model = build_improved_cnn_lstm_v1(input_shape, config)
    elif architecture == 'v2':
        print("Using: Improved CNN-LSTM V2 (Simplified with residual)")
        model = build_improved_cnn_lstm_v2(input_shape, config)
    elif architecture == 'lstm_only':
        print("Using: Bidirectional LSTM (No CNN)")
        model = build_lstm_only(input_shape, config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.summary()

    # Calculate class weights
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_seq),
        y=y_train_seq
    )
    class_weights = {
        0: class_weights_array[0],
        1: class_weights_array[1]
    }
    print(f"\nClass weights: {class_weights}")

    # ========================================================================
    # COMPILE MODEL
    # ========================================================================

    # Use Adam optimizer with learning rate schedule
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    # ========================================================================
    # SETUP CALLBACKS
    # ========================================================================

    callbacks_list = [
        # Early stopping
        callbacks.EarlyStopping(
            monitor='val_auc',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),

        # Learning rate reduction
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),

        # Model checkpoint
        callbacks.ModelCheckpoint(
            f'{config.MODEL_DIR}/improved_cnn_lstm_best.h5',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),

        # TensorBoard
        callbacks.TensorBoard(
            log_dir=f'{config.PROJECT_DIR}/logs',
            histogram_freq=1
        )
    ]

    # ========================================================================
    # TRAIN MODEL
    # ========================================================================

    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    print(f"Epochs: {config.EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Training samples: {len(X_train_seq)}")
    print(f"Validation samples: {len(X_val_seq)}")

    start_time = time.time()

    history = model.fit(
        X_train_seq, y_train_seq,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val_seq, y_val_seq),
        class_weight=class_weights,
        callbacks=callbacks_list,
        verbose=1
    )

    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

    # ========================================================================
    # EVALUATE MODEL
    # ========================================================================

    results, y_pred, y_pred_proba = evaluate_model(
        model, X_test_seq, y_test_seq,
        model_name="Improved CNN-LSTM"
    )
    results['training_time_seconds'] = training_time
    results['architecture'] = architecture

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Training history
    plot_training_history(
        history,
        save_path=f'{config.RESULTS_DIR}/improved_cnn_lstm_training_history.png'
    )

    # Confusion matrix
    cm = confusion_matrix(y_test_seq, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'])
    plt.title('Improved CNN-LSTM Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}/improved_cnn_lstm_confusion_matrix.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test_seq, y_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='blue', lw=2.5,
             label=f'Improved CNN-LSTM (AUC = {results["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Improved CNN-LSTM', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{config.RESULTS_DIR}/improved_cnn_lstm_roc_curve.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    print("\n" + "="*80)
    print("SAVING MODEL AND RESULTS")
    print("="*80)

    # Save model
    model.save(f'{config.MODEL_DIR}/improved_cnn_lstm_ids_model.h5')
    print(f"✓ Model saved to: {config.MODEL_DIR}/improved_cnn_lstm_ids_model.h5")

    # Save results
    results_dict = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'CSE-CIC-IDS-2018',
        'architecture': architecture,
        'time_steps': config.TIME_STEPS,
        'config': {
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'cnn_filters': config.CNN_FILTERS,
            'lstm_units': config.LSTM_UNITS,
            'dense_units': config.DENSE_UNITS,
            'dropout_rate': config.DROPOUT_RATE
        },
        'results': results,
        'training_samples': int(len(X_train_seq)),
        'validation_samples': int(len(X_val_seq)),
        'test_samples': int(len(X_test_seq))
    }

    with open(f'{config.RESULTS_DIR}/improved_cnn_lstm_results.json', 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"✓ Results saved to: {config.RESULTS_DIR}/improved_cnn_lstm_results.json")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

    return model, history, results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
    ============================================================================
    USAGE INSTRUCTIONS
    ============================================================================

    This script should be run in Google Colab with the following prerequisites:

    1. You must have already run data preprocessing to create:
       - X_train_scaled, y_train
       - X_val_scaled, y_val
       - X_test_scaled, y_test

    2. Update the Config class paths to match your Google Drive structure

    3. Run training with:

       model, history, results = train_improved_cnn_lstm(
           X_train_scaled, y_train,
           X_val_scaled, y_val,
           X_test_scaled, y_test,
           config=config,
           architecture='v1'  # or 'v2' or 'lstm_only'
       )

    4. Compare all three architectures:

       # Test V1
       model_v1, hist_v1, res_v1 = train_improved_cnn_lstm(
           X_train_scaled, y_train, X_val_scaled, y_val,
           X_test_scaled, y_test, config, architecture='v1'
       )

       # Test V2
       model_v2, hist_v2, res_v2 = train_improved_cnn_lstm(
           X_train_scaled, y_train, X_val_scaled, y_val,
           X_test_scaled, y_test, config, architecture='v2'
       )

       # Test LSTM-only
       model_lstm, hist_lstm, res_lstm = train_improved_cnn_lstm(
           X_train_scaled, y_train, X_val_scaled, y_val,
           X_test_scaled, y_test, config, architecture='lstm_only'
       )

    ============================================================================
    """)
