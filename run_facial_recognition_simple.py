#!/usr/bin/env python3
"""
Facial Expression Recognition - Simplified version without cv2
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("FACIAL EXPRESSION RECOGNITION PROJECT")
print("=" * 60)

# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from sklearn.model_selection import train_test_split
    print("✓ All required packages imported successfully")
except ImportError as e:
    print(f"❌ ERROR: Missing required package: {e}")
    print("Please install: pip3 install tensorflow pandas matplotlib scikit-learn pillow")
    exit(1)

# Constants
IMG_DIM = 48
DATA_PATH = '.'  # Current directory (local path)
CLASSES = sorted(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

print(f"\nClasses: {CLASSES}")
print(f"Image dimensions: {IMG_DIM}x{IMG_DIM}")

# Function to create dataframe
def create_dataframe(data_path):
    df = []
    for c in os.listdir(data_path):
        class_folder = os.path.join(data_path, c)
        if os.path.isdir(class_folder):
            for f in os.listdir(class_folder):
                f_path = os.path.join(class_folder, f)
                if f_path.endswith('jpg'):
                    df.append([f_path, c])
    return pd.DataFrame(df, columns=('filename', 'class'))

# Check if data directories exist
train_path = os.path.join(DATA_PATH, 'train')
test_path = os.path.join(DATA_PATH, 'test')

if not os.path.exists(train_path):
    print(f"\n❌ ERROR: Training data directory not found: {train_path}")
    print("Please ensure the 'train' directory exists in the current folder.")
    exit(1)

if not os.path.exists(test_path):
    print(f"\n❌ ERROR: Test data directory not found: {test_path}")
    print("Please ensure the 'test' directory exists in the current folder.")
    exit(1)

print(f"\n✓ Found training data: {train_path}")
print(f"✓ Found test data: {test_path}")

# Create dataframes
print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

try:
    df = create_dataframe(train_path)
    df_test = create_dataframe(test_path)
    
    print(f"\nTraining samples: {len(df)}")
    print(f"Test samples: {len(df_test)}")
    
    if len(df) == 0:
        print("❌ ERROR: No training data found!")
        exit(1)
    
    if len(df_test) == 0:
        print("❌ ERROR: No test data found!")
        exit(1)
    
    # Split training data into train and validation
    df_train, df_val = train_test_split(df, test_size=0.30, random_state=0)
    
    print(f"Training set: {len(df_train)}")
    print(f"Validation set: {len(df_val)}")
    
    # Display class distribution
    print("\nClass distribution in training set:")
    class_counts = df_train['class'].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
except Exception as e:
    print(f"\n❌ ERROR loading data: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# Data augmentation
print("\n" + "=" * 60)
print("SETTING UP DATA GENERATORS")
print("=" * 60)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
batch_size = 32

try:
    train_generator = train_datagen.flow_from_dataframe(
        df_train,
        x_col='filename',
        y_col='class',
        target_size=(IMG_DIM, IMG_DIM),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_dataframe(
        df_val,
        x_col='filename',
        y_col='class',
        target_size=(IMG_DIM, IMG_DIM),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_dataframe(
        df_test,
        x_col='filename',
        y_col='class',
        target_size=(IMG_DIM, IMG_DIM),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )

    print(f"\n✓ Data generators created successfully")
    print(f"  Batch size: {batch_size}")
    
except Exception as e:
    print(f"\n❌ ERROR creating data generators: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# Build the CNN model
print("\n" + "=" * 60)
print("BUILDING CNN MODEL")
print("=" * 60)

num_classes = len(train_generator.class_indices)
print(f"\nNumber of classes detected: {num_classes}")
print(f"Class indices: {train_generator.class_indices}")

model = Sequential()

# Add Convolutional layers
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_DIM, IMG_DIM, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu', name='last_conv'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n✓ Model built successfully")
print("\nModel Summary:")
model.summary()

# Train the model
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

epochs = 30
print(f"\nTraining for up to {epochs} epochs (with early stopping)...")
print("This may take a while depending on your hardware.")
print("Training will stop early if validation loss doesn't improve for 5 epochs.\n")

try:
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    print("\n✓ Training completed successfully!")
    
    # Save final model
    model.save('my_cnn_model.h5')
    print("✓ Final model saved as 'my_cnn_model.h5'")
    
except KeyboardInterrupt:
    print("\n⚠ Training interrupted by user")
    print("Saving current model state...")
    model.save('interrupted_model.h5')
    exit(1)
except Exception as e:
    print(f"\n❌ ERROR during training: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# Evaluate on test set
print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

try:
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"\n✓ Test Loss: {test_loss:.4f}")
    print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
except Exception as e:
    print(f"\n❌ ERROR during evaluation: {str(e)}")
    import traceback
    traceback.print_exc()

# Plot training history
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

try:
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\n✓ Training history plot saved as 'training_history.png'")
    
except Exception as e:
    print(f"\n⚠ Warning: Could not generate plots: {str(e)}")

print("\n" + "=" * 60)
print("PROJECT EXECUTION COMPLETED!")
print("=" * 60)
print("\nGenerated files:")
print("  - best_model.h5 (best model during training)")
print("  - my_cnn_model.h5 (final model)")
print("  - training_history.png (training plots)")
print("\nModel Performance:")
print(f"  - Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  - Final Test Loss: {test_loss:.4f}")
print("\n" + "=" * 60)

# Made with Bob
