#!/usr/bin/env python3
"""
Interface Web Streamlit RAPIDE - Reconnaissance d'Expressions Faciales
Version ultra-rapide avec chargement progressif
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
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
        margin-bottom: 1rem;
    }
    .loading-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_face_cascade():
    """Charger le détecteur de visages"""
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )


@st.cache_resource
def load_model_lazy(classifier_path):
    """Charger le modèle TensorFlow de manière paresseuse"""
    if not os.path.exists(classifier_path):
        return None
    
    try:
        # Afficher un message de chargement
        with st.spinner('🔄 Chargement du modèle TensorFlow (cela peut prendre 30-60 secondes)...'):
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            
            # Désactiver les logs
            tf.get_logger().setLevel('ERROR')
            
            # Charger le modèle
            model = load_model(classifier_path, compile=False)
            model.trainable = False
            
            return model
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {str(e)}")
        return None


def detect_faces(image, face_cascade):
    """Détecter les visages dans une image"""
    # Convertir en numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Convertir en BGR
    if len(image_np.shape) == 2:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Détecter les visages
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(40, 40)
    )
    
    # Dessiner les rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_bgr, "Face detected", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), len(faces)


def process_with_model(image, model, emotions, face_cascade):
    """Traiter l'image avec le modèle"""
    # Convertir en numpy array
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Convertir en BGR
    if len(image_np.shape) == 2:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Détecter les visages
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(40, 40))
    
    results = []
    
    for (x, y, w, h) in faces:
        # Extraire et prétraiter le visage
        face_roi = image_bgr[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_array = face_resized.astype(np.float32) / 255.0
        face_array = face_array.reshape(1, 48, 48, 1)
        
        # Prédiction
        predictions = model.predict(face_array, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        confidence = predictions[emotion_idx]
        emotion = emotions[emotion_idx]
        
        # Dessiner
        color = (0, 255, 0) if confidence > 0.7 else (255, 165, 0)
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), color, 2)
        text = f"{emotion}: {confidence:.2f}"
        cv2.putText(image_bgr, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'predictions': predictions
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
        
        use_model = st.checkbox("🧠 Activer la reconnaissance d'émotions", value=True)
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Chargez une image avec un visage
        2. Le système détectera les visages
        3. Si activé, l'émotion sera reconnue
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ Émotions")
        st.markdown("""
        - 😠 Angry
        - 🤢 Disgust
        - 😨 Fear
        - 😊 Happy
        - 😢 Sad
        - 😲 Surprise
        - 😐 Neutral
        """)
    
    # Charger le détecteur de visages (rapide)
    face_cascade = get_face_cascade()
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Interface principale
    st.markdown("---")
    st.subheader("📤 Charger une Image")
    
    uploaded_file = st.file_uploader("Choisissez une image...", type=['jpg', 'jpeg', 'png'])
    
    # Option webcam
    use_webcam = st.checkbox("📷 Utiliser la webcam")
    
    if use_webcam:
        st.markdown("### 📸 Capture Webcam")
        
        # Instructions pour macOS
        st.info("""
        **📋 Instructions pour macOS:**
        1. Autorisez l'accès à la caméra dans votre navigateur (popup en haut)
        2. Si la caméra ne s'affiche pas, vérifiez:
           - Préférences Système → Sécurité et confidentialité → Caméra
           - Assurez-vous que votre navigateur (Chrome/Safari/Firefox) a l'autorisation
        3. Redémarrez le navigateur si nécessaire
        """)
        
        camera_image = st.camera_input("Prenez une photo")
        
        if camera_image:
            image = Image.open(camera_image)
            
            if use_model and os.path.exists(classifier_path):
                # Charger le modèle (avec message de chargement)
                model = load_model_lazy(classifier_path)
                
                if model:
                    processed_image, results = process_with_model(image, model, emotions, face_cascade)
                    
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
                                st.progress(float(result['confidence']))
                                
                                with st.expander("Voir toutes les probabilités"):
                                    for emotion, prob in zip(emotions, result['predictions']):
                                        st.write(f"{emotion}: {prob:.3f}")
                                st.markdown("---")
                        else:
                            st.warning("⚠️ Aucun visage détecté")
            else:
                # Détection simple sans modèle
                processed_image, num_faces = detect_faces(image, face_cascade)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Image Traitée:**")
                    st.image(processed_image, use_column_width=True)
                
                with col2:
                    st.markdown("**Résultats:**")
                    st.metric("👤 Visages Détectés", num_faces)
                    if num_faces == 0:
                        st.warning("⚠️ Aucun visage détecté")
    
    elif uploaded_file:
        image = Image.open(uploaded_file)
        
        st.markdown("### 🖼️ Image Originale")
        st.image(image, use_column_width=True)
        
        if use_model and os.path.exists(classifier_path):
            # Charger le modèle (avec message de chargement)
            model = load_model_lazy(classifier_path)
            
            if model:
                st.markdown("---")
                st.markdown("### 🎯 Résultats de l'Analyse")
                
                processed_image, results = process_with_model(image, model, emotions, face_cascade)
                
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
                        st.warning("⚠️ Aucun visage détecté")
        else:
            # Détection simple sans modèle
            st.markdown("---")
            st.markdown("### 🎯 Détection de Visages")
            
            processed_image, num_faces = detect_faces(image, face_cascade)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Image avec Détections:**")
                st.image(processed_image, use_column_width=True)
            
            with col2:
                st.markdown("**Résultats:**")
                st.metric("👤 Visages Détectés", num_faces)
                if num_faces == 0:
                    st.warning("⚠️ Aucun visage détecté")
                else:
                    st.info("💡 Activez la reconnaissance d'émotions dans la barre latérale")
    
    else:
        # Message d'accueil
        st.markdown("""
        <div class="success-box">
        <h3>👋 Bienvenue!</h3>
        <p>✅ Interface chargée avec succès!</p>
        <p>Pour commencer:</p>
        <ul>
            <li>Chargez une image avec le bouton ci-dessus</li>
            <li>Ou activez la webcam pour une capture en direct</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if not os.path.exists(classifier_path):
            st.warning(f"⚠️ Modèle non trouvé: {classifier_path}")
            st.info("💡 La détection de visages fonctionnera, mais pas la reconnaissance d'émotions")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p><b>Système de Reconnaissance d'Expressions Faciales</b></p>
    <p>Version Rapide - Chargement Progressif</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# Made with Bob
