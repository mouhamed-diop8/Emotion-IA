#!/usr/bin/env python3
"""
Interface Web Streamlit OPTIMISÉE - Reconnaissance d'Expressions Faciales
Version haute performance avec cache et traitement optimisé
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os
import hashlib

# Configuration de la page
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS optimisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# Cache global pour le détecteur de visages
@st.cache_resource
def get_face_cascade():
    """Charger le détecteur de visages une seule fois"""
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )


@st.cache_resource
def load_models(classifier_path):
    """Charger le modèle avec cache - une seule fois"""
    if not os.path.exists(classifier_path):
        return None
    try:
        # Désactiver les logs TensorFlow pour plus de rapidité
        tf.get_logger().setLevel('ERROR')
        model = load_model(classifier_path, compile=False)
        # Optimiser le modèle pour l'inférence
        model.trainable = False
        return model
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        return None


def get_image_hash(image_array):
    """Calculer un hash de l'image pour le cache"""
    return hashlib.md5(image_array.tobytes()).hexdigest()


@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def detect_faces_cached(_face_cascade, gray_image, image_hash):
    """Détection de visages avec cache"""
    # Paramètres optimisés pour la vitesse
    faces = _face_cascade.detectMultiScale(
        gray_image, 
        scaleFactor=1.3,  # Plus rapide (était 1.1)
        minNeighbors=4,   # Plus rapide (était 5)
        minSize=(40, 40), # Légèrement plus grand pour éviter faux positifs
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces


def preprocess_face_fast(face_roi):
    """Prétraitement rapide du visage"""
    # Convertir directement en niveaux de gris
    if len(face_roi.shape) == 3:
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face_roi
    
    # Redimensionner avec interpolation rapide
    face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_LINEAR)
    
    # Normaliser
    face_array = face_resized.astype(np.float32) / 255.0
    face_array = face_array.reshape(1, 48, 48, 1)
    
    return face_array


def process_image_optimized(image, model, emotions, face_cascade):
    """Traiter une image de manière optimisée"""
    start_time = time.time()
    
    # Convertir en numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Réduire la résolution si l'image est trop grande (optimisation)
    max_dimension = 800
    height, width = image_np.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_np = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convertir en BGR pour OpenCV
    if len(image_np.shape) == 2:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Détection de visages avec cache
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Égalisation d'histogramme pour améliorer la détection
    gray = cv2.equalizeHist(gray)
    
    image_hash = get_image_hash(gray)
    faces = detect_faces_cached(face_cascade, gray, image_hash)
    
    results = []
    processing_time = 0
    
    # Traiter tous les visages en batch si possible
    if len(faces) > 0:
        face_arrays = []
        face_coords = []
        
        for (x, y, w, h) in faces:
            # Extraire le visage
            face_roi = image_bgr[y:y+h, x:x+w]
            
            # Prétraitement rapide
            face_array = preprocess_face_fast(face_roi)
            face_arrays.append(face_array)
            face_coords.append((x, y, w, h))
        
        # Prédiction en batch (plus rapide)
        pred_start = time.time()
        if len(face_arrays) > 1:
            batch_input = np.vstack(face_arrays)
            predictions_batch = model.predict(batch_input, verbose=0, batch_size=len(face_arrays))
        else:
            predictions_batch = model.predict(face_arrays[0], verbose=0)
            predictions_batch = [predictions_batch[0]]
        processing_time = time.time() - pred_start
        
        # Traiter les résultats
        for idx, (predictions, (x, y, w, h)) in enumerate(zip(predictions_batch, face_coords)):
            emotion_idx = np.argmax(predictions)
            confidence = predictions[emotion_idx]
            emotion = emotions[emotion_idx]
            
            # Dessiner sur l'image
            color = (0, 255, 0) if confidence > 0.7 else (255, 165, 0)
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), color, 2)
            
            text = f"{emotion}: {confidence:.2f}"
            cv2.putText(image_bgr, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'predictions': predictions
            })
    
    total_time = time.time() - start_time
    
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), results, total_time, processing_time


def main():
    """Application principale"""
    
    # En-tête
    st.markdown('<h1 class="main-header">🎭 Reconnaissance d\'Expressions Faciales (OPTIMISÉE)</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        classifier_path = st.text_input("Modèle Classificateur", "best_model.h5")
        
        st.markdown("---")
        st.markdown("### ⚡ Optimisations Actives")
        st.markdown("""
        - ✅ Cache du modèle
        - ✅ Cache de détection
        - ✅ Traitement batch
        - ✅ Résolution adaptative
        - ✅ Interpolation rapide
        """)
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Chargez une image avec un visage
        2. Le système détectera automatiquement les visages
        3. L'émotion sera affichée avec la confiance
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ Émotions Détectées")
        st.markdown("""
        - 😠 Angry (Colère)
        - 🤢 Disgust (Dégoût)
        - 😨 Fear (Peur)
        - 😊 Happy (Joie)
        - 😢 Sad (Tristesse)
        - 😲 Surprise (Surprise)
        - 😐 Neutral (Neutre)
        """)
    
    # Vérifier le modèle
    if not os.path.exists(classifier_path):
        st.error(f"❌ Modèle introuvable: {classifier_path}")
        st.info("💡 Assurez-vous que le fichier best_model.h5 existe dans le répertoire")
        return
    
    # Charger le modèle (avec cache)
    model = load_models(classifier_path)
    
    if model is None:
        st.error("❌ Impossible de charger le modèle")
        return
    
    # Charger le détecteur de visages (avec cache)
    face_cascade = get_face_cascade()
    
    st.success("✅ Modèle chargé avec succès! (Optimisé)")
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Interface principale
    st.markdown("---")
    st.subheader("📤 Charger une Image")
    
    uploaded_file = st.file_uploader("Choisissez une image...", type=['jpg', 'jpeg', 'png'])
    
    # Option webcam
    use_webcam = st.checkbox("📷 Utiliser la webcam")
    
    if use_webcam:
        st.markdown("### 📸 Capture Webcam")
        camera_image = st.camera_input("Prenez une photo")
        
        if camera_image:
            image = Image.open(camera_image)
            
            # Traitement optimisé
            processed_image, results, total_time, pred_time = process_image_optimized(
                image, model, emotions, face_cascade
            )
            
            # Afficher les métriques de performance
            col_perf1, col_perf2, col_perf3 = st.columns(3)
            with col_perf1:
                st.metric("⏱️ Temps Total", f"{total_time:.3f}s")
            with col_perf2:
                st.metric("🧠 Temps Prédiction", f"{pred_time:.3f}s")
            with col_perf3:
                st.metric("👤 Visages Détectés", len(results))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Image Traitée:**")
                st.image(processed_image, use_column_width=True)
            
            with col2:
                st.markdown("**Résultats:**")
                if results:
                    for i, result in enumerate(results):
                        st.markdown(f"**Visage {i+1}:**")
                        st.markdown(f"- Émotion: **{result['emotion'].upper()}**")
                        st.markdown(f"- Confiance: **{result['confidence']:.2%}**")
                        
                        # Barre de progression
                        st.progress(float(result['confidence']))
                        
                        # Toutes les prédictions
                        with st.expander("Voir toutes les probabilités"):
                            for emotion, prob in zip(emotions, result['predictions']):
                                st.write(f"{emotion}: {prob:.3f}")
                        st.markdown("---")
                else:
                    st.warning("⚠️ Aucun visage détecté")
    
    elif uploaded_file:
        image = Image.open(uploaded_file)
        
        st.markdown("### 🖼️ Image Originale")
        st.image(image, use_column_width=True)
        
        # Traitement optimisé
        processed_image, results, total_time, pred_time = process_image_optimized(
            image, model, emotions, face_cascade
        )
        
        st.markdown("---")
        st.markdown("### 🎯 Résultats de l'Analyse")
        
        # Afficher les métriques de performance
        col_perf1, col_perf2, col_perf3 = st.columns(3)
        with col_perf1:
            st.metric("⏱️ Temps Total", f"{total_time:.3f}s")
        with col_perf2:
            st.metric("🧠 Temps Prédiction", f"{pred_time:.3f}s")
        with col_perf3:
            st.metric("👤 Visages Détectés", len(results))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Image avec Détections:**")
            st.image(processed_image, use_column_width=True)
        
        with col2:
            st.markdown("**Détails:**")
            if results:
                for i, result in enumerate(results):
                    st.markdown(f"#### Visage {i+1}")
                    st.markdown(f"**Émotion:** {result['emotion'].upper()}")
                    st.markdown(f"**Confiance:** {result['confidence']:.2%}")
                    st.progress(float(result['confidence']))
                    
                    with st.expander("📊 Toutes les probabilités"):
                        for emotion, prob in zip(emotions, result['predictions']):
                            st.write(f"**{emotion}:** {prob:.3f}")
                    
                    st.markdown("---")
            else:
                st.warning("⚠️ Aucun visage détecté dans l'image")
                st.info("💡 Essayez avec une image contenant un visage visible")
    
    else:
        # Message d'accueil
        st.markdown("""
        <div class="info-box">
        <h3>👋 Bienvenue!</h3>
        <p>Version optimisée pour des performances maximales</p>
        <p>Pour commencer:</p>
        <ul>
            <li>Chargez une image avec le bouton ci-dessus</li>
            <li>Ou activez la webcam pour une capture en direct</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Exemple d'utilisation
        st.markdown("### 📝 Exemple")
        st.markdown("""
        Le système peut détecter 7 émotions différentes:
        - Colère (Angry)
        - Dégoût (Disgust)
        - Peur (Fear)
        - Joie (Happy)
        - Tristesse (Sad)
        - Surprise (Surprise)
        - Neutre (Neutral)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p><b>Système de Reconnaissance d'Expressions Faciales (OPTIMISÉ)</b></p>
    <p>Deep Learning avec TensorFlow & Keras | Version Haute Performance</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# Made with Bob - Optimized Version