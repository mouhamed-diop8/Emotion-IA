#!/usr/bin/env python3
"""
Interface Web Streamlit Simplifiée - Reconnaissance d'Expressions Faciales
Version corrigée qui affiche immédiatement du contenu
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os

# Configuration de la page
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models(classifier_path):
    """Charger le modèle avec cache"""
    if not os.path.exists(classifier_path):
        return None
    try:
        model = load_model(classifier_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Erreur de chargement: {str(e)}")
        return None


def process_image(image, model, emotions):
    """Traiter une image et détecter les émotions"""
    # Convertir en numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Convertir en BGR pour OpenCV
    if len(image_np.shape) == 2:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Détection de visages
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    results = []
    
    for (x, y, w, h) in faces:
        # Extraire le visage
        face_roi = image_bgr[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        
        # Préparer pour classification
        face_gray = face_pil.convert('L').resize((48, 48))
        face_array = np.array(face_gray) / 255.0
        face_array = face_array.reshape(1, 48, 48, 1)
        
        # Prédiction
        predictions = model.predict(face_array, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
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
            'predictions': predictions[0]
        })
    
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), results


def main():
    """Application principale"""
    
    # En-tête
    st.markdown('<h1 class="main-header">🎭 Reconnaissance d\'Expressions Faciales</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        classifier_path = st.text_input("Modèle Classificateur", "best_model.h5")
        
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
    
    # Charger le modèle
    with st.spinner("🔄 Chargement du modèle..."):
        model = load_models(classifier_path)
    
    if model is None:
        st.error("❌ Impossible de charger le modèle")
        return
    
    st.success("✅ Modèle chargé avec succès!")
    
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
            
            with st.spinner("🔍 Analyse en cours..."):
                processed_image, results = process_image(image, model, emotions)
            
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
        
        with st.spinner("🔍 Analyse en cours..."):
            processed_image, results = process_image(image, model, emotions)
        
        st.markdown("---")
        st.markdown("### 🎯 Résultats de l'Analyse")
        
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
    <p><b>Système de Reconnaissance d'Expressions Faciales</b></p>
    <p>Deep Learning avec TensorFlow & Keras</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# Made with Bob
