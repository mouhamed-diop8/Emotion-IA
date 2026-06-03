#!/usr/bin/env python3
"""
Script de diagnostic pour analyser les prédictions du modèle
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

print("=" * 60)
print("DIAGNOSTIC DU MODÈLE - ANALYSE DES PRÉDICTIONS")
print("=" * 60)

# Charger le modèle
print("\n🔄 Chargement du modèle...")
try:
    model = load_model('my_cnn_model.h5')
    print("✅ Modèle chargé avec succès!")
    print(f"   Forme de sortie: {model.output_shape}")
    print(f"   Nombre de classes: {model.output_shape[-1]}")
except Exception as e:
    print(f"❌ Erreur: {e}")
    exit(1)

# Configuration
IMG_DIM = 48
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Créer un générateur pour les données de test
print("\n🔄 Chargement des données de test...")

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

df_test = create_dataframe('test')
print(f"✅ {len(df_test)} images de test chargées")

# Créer le générateur
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    df_test,
    x_col='filename',
    y_col='class',
    target_size=(IMG_DIM, IMG_DIM),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

print(f"\nClasses détectées: {test_generator.class_indices}")

# Faire des prédictions sur toutes les données de test
print("\n🔄 Analyse des prédictions...")
predictions = model.predict(test_generator, verbose=1)

# Analyser les prédictions
print("\n" + "=" * 60)
print("ANALYSE DES PRÉDICTIONS")
print("=" * 60)

# Compter combien de fois chaque classe est prédite
predicted_classes = np.argmax(predictions, axis=1)
unique, counts = np.unique(predicted_classes, return_counts=True)

print("\nNombre de prédictions par classe:")
for idx, count in zip(unique, counts):
    emotion = emotions[idx]
    percentage = (count / len(predicted_classes)) * 100
    print(f"  {emotion:10s}: {count:5d} ({percentage:5.2f}%)")

# Analyser les probabilités moyennes par classe
print("\nProbabilités moyennes par classe:")
mean_probs = np.mean(predictions, axis=0)
for idx, prob in enumerate(mean_probs):
    emotion = emotions[idx]
    print(f"  {emotion:10s}: {prob:.4f}")

# Analyser les probabilités maximales par classe
print("\nProbabilités maximales par classe:")
max_probs = np.max(predictions, axis=0)
for idx, prob in enumerate(max_probs):
    emotion = emotions[idx]
    print(f"  {emotion:10s}: {prob:.4f}")

# Vérifier s'il y a des classes qui ne sont jamais prédites
never_predicted = []
for idx in range(len(emotions)):
    if idx not in unique:
        never_predicted.append(emotions[idx])

if never_predicted:
    print(f"\n⚠️  ATTENTION: Classes JAMAIS prédites: {', '.join(never_predicted)}")
else:
    print("\n✅ Toutes les classes sont prédites au moins une fois")

# Analyser la distribution des confiances
print("\n" + "=" * 60)
print("ANALYSE DES CONFIANCES")
print("=" * 60)

max_confidences = np.max(predictions, axis=1)
print(f"\nConfiance moyenne: {np.mean(max_confidences):.4f}")
print(f"Confiance médiane: {np.median(max_confidences):.4f}")
print(f"Confiance min: {np.min(max_confidences):.4f}")
print(f"Confiance max: {np.max(max_confidences):.4f}")

# Compter les prédictions avec haute confiance (>0.9)
high_conf = np.sum(max_confidences > 0.9)
print(f"\nPrédictions avec confiance > 90%: {high_conf} ({high_conf/len(max_confidences)*100:.2f}%)")

# Analyser les vraies étiquettes vs prédictions
print("\n" + "=" * 60)
print("MATRICE DE CONFUSION SIMPLIFIÉE")
print("=" * 60)

true_classes = test_generator.classes
from sklearn.metrics import classification_report, confusion_matrix

print("\nRapport de classification:")
print(classification_report(true_classes, predicted_classes, 
                          target_names=emotions, 
                          zero_division=0))

print("\n" + "=" * 60)
print("DIAGNOSTIC TERMINÉ")
print("=" * 60)

# Recommandations
print("\n📋 RECOMMANDATIONS:")
if len(never_predicted) > 0:
    print(f"❌ Le modèle ne prédit JAMAIS: {', '.join(never_predicted)}")
    print("   → Le modèle doit être réentraîné!")

# Vérifier le déséquilibre
pred_percentages = (counts / len(predicted_classes)) * 100
if np.max(pred_percentages) > 50:
    dominant_class = emotions[unique[np.argmax(pred_percentages)]]
    print(f"⚠️  Le modèle prédit trop souvent '{dominant_class}' ({np.max(pred_percentages):.1f}%)")
    print("   → Problème de déséquilibre des classes")

if np.mean(max_confidences) < 0.5:
    print("⚠️  Confiance moyenne faible (<50%)")
    print("   → Le modèle n'est pas sûr de ses prédictions")

print("\n" + "=" * 60)

# Made with Bob
