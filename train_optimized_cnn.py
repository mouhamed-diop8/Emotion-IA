#!/usr/bin/env python3
"""
Entraînement CNN Optimisé pour Reconnaissance d'Expressions Faciales
Architecture améliorée avec techniques d'optimisation avancées
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, 
    Flatten, Dense, Dropout, GlobalAveragePooling2D,
    Activation, Add, Input
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, LearningRateScheduler
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🎓 ENTRAÎNEMENT CNN OPTIMISÉ - RECONNAISSANCE D'EXPRESSIONS FACIALES")
print("=" * 80)

# Configuration
IMG_DIM = 48
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DATA_PATH = '.'
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

print(f"\n📋 Configuration:")
print(f"  - Taille d'image: {IMG_DIM}x{IMG_DIM}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Époques max: {EPOCHS}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Classes: {len(CLASSES)}")

def create_dataframe(data_path):
    """Créer un DataFrame à partir des images"""
    df = []
    for c in os.listdir(data_path):
        class_folder = os.path.join(data_path, c)
        if os.path.isdir(class_folder) and c in CLASSES:
            for f in os.listdir(class_folder):
                f_path = os.path.join(class_folder, f)
                if f_path.endswith('.jpg'):
                    df.append([f_path, c])
    return pd.DataFrame(df, columns=['filename', 'class'])

def build_optimized_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Architecture CNN optimisée avec:
    - Blocs convolutionnels plus profonds
    - Batch Normalization après chaque Conv
    - Dropout pour régularisation
    - Global Average Pooling
    - L2 regularization
    """
    model = Sequential([
        # Bloc 1
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape, 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Bloc 2
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Bloc 3
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Bloc 4
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Global Average Pooling au lieu de Flatten
        GlobalAveragePooling2D(),
        
        # Couches denses
        Dense(512, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        Dense(256, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        
        # Sortie
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_advanced_data_generators():
    """
    Générateurs de données avec augmentation avancée
    """
    # Augmentation pour l'entraînement
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0
    )
    
    # Validation et test: seulement normalisation
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_test_datagen

def lr_schedule(epoch, lr):
    """
    Learning rate scheduler
    Réduit le learning rate progressivement
    """
    if epoch < 10:
        return lr
    elif epoch < 20:
        return lr * 0.5
    elif epoch < 30:
        return lr * 0.1
    else:
        return lr * 0.01

def plot_training_history(history, save_path='training_history_optimized.png'):
    """
    Visualiser l'historique d'entraînement
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], linewidth=2, color='orange')
        axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy Gap (Overfitting indicator)
    train_acc = np.array(history.history['accuracy'])
    val_acc = np.array(history.history['val_accuracy'])
    gap = train_acc - val_acc
    axes[1, 1].plot(gap, linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Gap (Train - Val Accuracy)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Graphiques sauvegardés: {save_path}")
    
    return fig

def main():
    """Fonction principale d'entraînement"""
    
    # Vérifier les données
    print("\n" + "=" * 80)
    print("📂 CHARGEMENT DES DONNÉES")
    print("=" * 80)
    
    train_path = os.path.join(DATA_PATH, 'train')
    test_path = os.path.join(DATA_PATH, 'test')
    
    if not os.path.exists(train_path):
        print(f"❌ Erreur: Dossier d'entraînement introuvable: {train_path}")
        return
    
    if not os.path.exists(test_path):
        print(f"❌ Erreur: Dossier de test introuvable: {test_path}")
        return
    
    # Créer les DataFrames
    df_train = create_dataframe(train_path)
    df_test = create_dataframe(test_path)
    
    print(f"\n✅ Données chargées:")
    print(f"  - Entraînement: {len(df_train)} images")
    print(f"  - Test: {len(df_test)} images")
    
    # Distribution des classes
    print("\n📊 Distribution des classes (entraînement):")
    class_counts = df_train['class'].value_counts()
    for emotion, count in class_counts.items():
        print(f"  {emotion:10s}: {count:5d} images ({count/len(df_train)*100:.1f}%)")
    
    # Split train/validation
    df_train_split, df_val = train_test_split(
        df_train, 
        test_size=0.2, 
        random_state=42,
        stratify=df_train['class']
    )
    
    print(f"\n✅ Split effectué:")
    print(f"  - Entraînement: {len(df_train_split)} images")
    print(f"  - Validation: {len(df_val)} images")
    
    # Créer les générateurs
    print("\n" + "=" * 80)
    print("🔄 CRÉATION DES GÉNÉRATEURS DE DONNÉES")
    print("=" * 80)
    
    train_datagen, val_test_datagen = create_advanced_data_generators()
    
    train_generator = train_datagen.flow_from_dataframe(
        df_train_split,
        x_col='filename',
        y_col='class',
        target_size=(IMG_DIM, IMG_DIM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_dataframe(
        df_val,
        x_col='filename',
        y_col='class',
        target_size=(IMG_DIM, IMG_DIM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        df_test,
        x_col='filename',
        y_col='class',
        target_size=(IMG_DIM, IMG_DIM),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    print("✅ Générateurs créés avec succès")
    
    # Calculer les poids de classe pour gérer le déséquilibre
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(df_train_split['class']),
        y=df_train_split['class']
    )
    class_weights = dict(enumerate(class_weights_array))
    
    print("\n⚖️  Poids de classe calculés (pour gérer le déséquilibre):")
    for idx, (emotion, weight) in enumerate(zip(sorted(CLASSES), class_weights_array)):
        print(f"  {emotion:10s}: {weight:.3f}")
    
    # Construire le modèle
    print("\n" + "=" * 80)
    print("🏗️  CONSTRUCTION DU MODÈLE CNN OPTIMISÉ")
    print("=" * 80)
    
    model = build_optimized_cnn(
        input_shape=(IMG_DIM, IMG_DIM, 1),
        num_classes=len(train_generator.class_indices)
    )
    
    # Compiler
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n✅ Modèle construit et compilé")
    model.summary()
    
    # Callbacks
    print("\n" + "=" * 80)
    print("⚙️  CONFIGURATION DES CALLBACKS")
    print("=" * 80)
    
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            'best_model_optimized.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Learning rate scheduler
        LearningRateScheduler(lr_schedule, verbose=0)
    ]
    
    print("✅ Callbacks configurés:")
    print("  - Early Stopping (patience=10)")
    print("  - Model Checkpoint (meilleur modèle)")
    print("  - Reduce LR on Plateau")
    print("  - Learning Rate Scheduler")
    
    # Entraînement
    print("\n" + "=" * 80)
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT")
    print("=" * 80)
    print(f"\nEntraînement pour {EPOCHS} époques maximum...")
    print("Cela peut prendre du temps selon votre matériel.\n")
    
    try:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        print("\n✅ Entraînement terminé avec succès!")
        
        # Sauvegarder le modèle final
        model.save('final_model_optimized.h5')
        print("✅ Modèle final sauvegardé: final_model_optimized.h5")
        
    except Exception as e:
        print(f"\n❌ Erreur pendant l'entraînement: {e}")
        return
    
    # Évaluation sur le test set
    print("\n" + "=" * 80)
    print("📊 ÉVALUATION SUR LE TEST SET")
    print("=" * 80)
    
    try:
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
        print(f"\n✅ Résultats sur le test set:")
        print(f"  - Loss: {test_loss:.4f}")
        print(f"  - Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
    except Exception as e:
        print(f"❌ Erreur pendant l'évaluation: {e}")
    
    # Visualisation
    print("\n" + "=" * 80)
    print("📈 GÉNÉRATION DES GRAPHIQUES")
    print("=" * 80)
    
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"⚠️  Erreur lors de la génération des graphiques: {e}")
    
    # Résumé final
    print("\n" + "=" * 80)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    print("\n📁 Fichiers générés:")
    print("  - best_model_optimized.h5 (meilleur modèle)")
    print("  - final_model_optimized.h5 (modèle final)")
    print("  - training_history_optimized.png (graphiques)")
    
    print("\n🎯 Prochaines étapes:")
    print("  1. Tester avec: python3 webcam_enhanced_detection.py")
    print("  2. Utiliser best_model_optimized.h5 pour la détection")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

# Made with Bob
