#!/usr/bin/env python3
"""
Script de Réentraînement avec Équilibrage des Classes
Corrige le problème de prédictions limitées
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("RÉENTRAÎNEMENT DU MODÈLE AVEC ÉQUILIBRAGE")
print("=" * 60)

# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    print("✓ Packages importés avec succès")
except ImportError as e:
    print(f"❌ ERREUR: {e}")
    exit(1)

# Configuration
IMG_DIM = 48
DATA_PATH = '.'
CLASSES = sorted(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])

print(f"\nClasses: {CLASSES}")
print(f"Dimensions: {IMG_DIM}x{IMG_DIM}")

# Fonction pour créer le dataframe
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

# Charger les données
print("\n" + "=" * 60)
print("CHARGEMENT DES DONNÉES")
print("=" * 60)

train_path = os.path.join(DATA_PATH, 'train')
test_path = os.path.join(DATA_PATH, 'test')

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("❌ ERREUR: Dossiers train/ ou test/ introuvables")
    exit(1)

df = create_dataframe(train_path)
df_test = create_dataframe(test_path)

print(f"\nÉchantillons d'entraînement: {len(df)}")
print(f"Échantillons de test: {len(df_test)}")

# Analyser la distribution des classes
print("\n📊 Distribution des classes:")
class_counts = df['class'].value_counts().sort_index()
for cls, count in class_counts.items():
    print(f"  {cls:10s}: {count:5d} images")

# Split train/validation
df_train, df_val = train_test_split(df, test_size=0.30, random_state=42, stratify=df['class'])

print(f"\nEntraînement: {len(df_train)}")
print(f"Validation: {len(df_val)}")

# Calculer les poids des classes pour équilibrage
print("\n" + "=" * 60)
print("CALCUL DES POIDS DES CLASSES")
print("=" * 60)

# Mapper les noms de classes aux indices
class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
y_train = df_train['class'].map(class_to_idx).values

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print("\n⚖️  Poids calculés pour équilibrage:")
for idx, weight in class_weight_dict.items():
    print(f"  {CLASSES[idx]:10s}: {weight:.3f}")

# Générateurs de données avec augmentation
print("\n" + "=" * 60)
print("CONFIGURATION DES GÉNÉRATEURS")
print("=" * 60)

# Augmentation forte pour l'entraînement
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

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

print("✓ Générateurs créés")

# Construction du modèle amélioré
print("\n" + "=" * 60)
print("CONSTRUCTION DU MODÈLE AMÉLIORÉ")
print("=" * 60)

num_classes = len(CLASSES)
print(f"\nNombre de classes: {num_classes}")

model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), padding='same', input_shape=(IMG_DIM, IMG_DIM, 1), activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Block 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3), padding='same', activation='relu', name='last_conv'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Dense layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compiler
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Modèle construit")
model.summary()

# Callbacks
print("\n" + "=" * 60)
print("ENTRAÎNEMENT DU MODÈLE")
print("=" * 60)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model_balanced.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

epochs = 50
print(f"\nEntraînement pour {epochs} époques maximum...")
print("Avec équilibrage des classes et early stopping\n")

try:
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weight_dict,  # ← ÉQUILIBRAGE
        verbose=1
    )
    
    print("\n✅ Entraînement terminé!")
    
    # Sauvegarder le modèle final
    model.save('my_cnn_model_balanced.h5')
    print("✓ Modèle sauvegardé: my_cnn_model_balanced.h5")
    
except KeyboardInterrupt:
    print("\n⚠️  Entraînement interrompu")
    model.save('interrupted_model_balanced.h5')
    exit(1)
except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Évaluation
print("\n" + "=" * 60)
print("ÉVALUATION DU MODÈLE")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n✓ Loss sur test: {test_loss:.4f}")
print(f"✓ Précision sur test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Analyse détaillée des prédictions
print("\n" + "=" * 60)
print("ANALYSE DES PRÉDICTIONS")
print("=" * 60)

predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Distribution des prédictions
unique, counts = np.unique(predicted_classes, return_counts=True)
print("\n📊 Distribution des prédictions:")
for idx, count in zip(unique, counts):
    percentage = (count / len(predicted_classes)) * 100
    print(f"  {CLASSES[idx]:10s}: {count:5d} ({percentage:5.2f}%)")

# Rapport de classification
from sklearn.metrics import classification_report
print("\n📈 Rapport de classification:")
print(classification_report(true_classes, predicted_classes, 
                          target_names=CLASSES, 
                          zero_division=0))

# Graphiques
print("\n" + "=" * 60)
print("GÉNÉRATION DES GRAPHIQUES")
print("=" * 60)

plt.figure(figsize=(15, 5))

# Accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
plt.title('Précision du Modèle', fontsize=14, fontweight='bold')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
plt.title('Loss du Modèle', fontsize=14, fontweight='bold')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Distribution des prédictions
plt.subplot(1, 3, 3)
pred_counts = [np.sum(predicted_classes == i) for i in range(num_classes)]
plt.bar(CLASSES, pred_counts, color='steelblue')
plt.title('Distribution des Prédictions', fontsize=14, fontweight='bold')
plt.xlabel('Émotion')
plt.ylabel('Nombre')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('training_history_balanced.png', dpi=150, bbox_inches='tight')
print("✓ Graphiques sauvegardés: training_history_balanced.png")

print("\n" + "=" * 60)
print("RÉENTRAÎNEMENT TERMINÉ!")
print("=" * 60)
print("\n📁 Fichiers générés:")
print("  - best_model_balanced.h5 (meilleur modèle)")
print("  - my_cnn_model_balanced.h5 (modèle final)")
print("  - training_history_balanced.png (graphiques)")
print("\n🎯 Prochaines étapes:")
print("  1. Vérifiez que toutes les classes sont prédites")
print("  2. Testez avec: python3 diagnose_model.py")
print("  3. Utilisez le nouveau modèle dans vos applications")
print("\n" + "=" * 60)

# Made with Bob
