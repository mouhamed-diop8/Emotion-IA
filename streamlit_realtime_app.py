#!/usr/bin/env python3
"""
Interface Streamlit - Reconnaissance d'Expressions Faciales
Version compatible avec upload d'images et affichage en temps réel
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import time

# Configuration
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="🎭",
    layout="wide"
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
    .emotion-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.2rem;
        padding: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Charger le modèle
@st.cache_resource
def load_emotion_model():
    """Charger le modèle de classification"""
    model = load_model("best_model.h5", compile=False)
    return model

# Initialiser le détecteur de visages
@st.cache_resource
def load_face_detector():
    """Charger le détecteur de visages"""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variables
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_colors = {
    'angry': '#ff4444',
    'disgust': '#aa44ff',
    'fear': '#4444ff',
    'happy': '#44ff44',
    'sad': '#4444aa',
    'surprise': '#ffaa44',
    'neutral': '#888888'
}

emotion_emojis = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
    'neutral': '😐'
}

# En-tête
st.markdown('<h1 class="main-header">🎭 Reconnaissance d\'Expressions Faciales</h1>', 
            unsafe_allow_html=True)

# Charger les modèles
try:
    model = load_emotion_model()
    face_cascade = load_face_detector()
    st.success("✅ Système prêt ! Chargez une image pour commencer.")
except Exception as e:
    st.error(f"❌ Erreur: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode
    st.subheader("📸 Source")
    source_mode = st.radio(
        "Choisir la source",
        ["📤 Charger une image", "📷 Utiliser la webcam (externe)"],
        help="La webcam nécessite une application externe sur macOS"
    )
    
    # Paramètres
    st.subheader("🔧 Paramètres")
    confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.3, 0.05)
    show_all_probs = st.checkbox("Afficher toutes les probabilités", value=True)
    show_face_box = st.checkbox("Afficher le cadre du visage", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Informations Système")
    st.info(f"""
    **TensorFlow:** {tf.__version__}  
    **OpenCV:** {cv2.__version__}  
    **Modèle:** best_model.h5  
    **Émotions:** 7 classes
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Conseils")
    st.info("""
    - Utilisez une photo claire
    - Visage bien éclairé
    - Expression visible
    - Regardez la caméra
    """)

def process_image(image):
    """Traiter une image et détecter l'émotion"""
    # Convertir en array numpy
    img_array = np.array(image)
    
    # Convertir en BGR pour OpenCV
    if len(img_array.shape) == 2:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Détecter les visages
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    results = []
    
    for (x, y, w, h) in faces:
        # Extraire le visage
        face_roi = gray[y:y+h, x:x+w]
        
        # Préparer pour le modèle
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)
        
        # Prédiction
        predictions = model.predict(face_input, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = emotions[emotion_idx]
        confidence = predictions[emotion_idx]
        
        # Dessiner le rectangle
        if show_face_box:
            color = (0, 255, 0) if confidence > confidence_threshold else (255, 165, 0)
            cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 3)
            
            # Ajouter le texte
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(img_bgr, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'predictions': predictions,
            'bbox': (x, y, w, h),
            'face_img': face_roi
        })
    
    # Convertir BGR vers RGB pour affichage
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb, results

# Layout principal
if source_mode == "📤 Charger une image":
    uploaded_file = st.file_uploader(
        "📤 Chargez une image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Sélectionnez une image contenant un visage"
    )
    
    if uploaded_file is not None:
        # Charger l'image
        image = Image.open(uploaded_file)
        
        # Créer deux colonnes
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📥 Image Originale")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Analyse")
            
            with st.spinner("🔄 Analyse en cours..."):
                # Traiter l'image
                processed_img, results = process_image(image)
                
                if results:
                    result = results[0]  # Premier visage
                    
                    # Afficher l'émotion détectée
                    emotion = result['emotion']
                    confidence = result['confidence']
                    emoji = emotion_emojis.get(emotion, '🎭')
                    color = emotion_colors.get(emotion, '#888888')
                    
                    st.markdown(f"""
                    <div class="emotion-display" style="background-color: {color}20; border: 3px solid {color};">
                        <div style="font-size: 4rem;">{emoji}</div>
                        <div style="color: {color};">{emotion.upper()}</div>
                        <div style="font-size: 1.5rem; color: #666;">Confiance: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Barre de progression
                    st.progress(float(confidence))
                    
                    # Graphique des probabilités
                    if show_all_probs:
                        st.subheader("📊 Probabilités par Émotion")
                        fig = go.Figure(data=[
                            go.Bar(
                                x=emotions,
                                y=result['predictions'],
                                marker_color=[emotion_colors.get(e, '#888888') for e in emotions],
                                text=[f"{p:.1%}" for p in result['predictions']],
                                textposition='auto',
                            )
                        ])
                        fig.update_layout(
                            height=300,
                            showlegend=False,
                            yaxis_title="Probabilité",
                            xaxis_title="Émotion"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Afficher le visage détecté
                    with st.expander("👤 Visage Détecté"):
                        st.image(result['face_img'], caption="ROI du visage", width=200)
                
                else:
                    st.warning("⚠️ Aucun visage détecté dans l'image")
                    st.info("""
                    **Suggestions:**
                    - Assurez-vous que le visage est clairement visible
                    - Utilisez une image avec un bon éclairage
                    - Le visage doit être de face
                    """)
        
        # Afficher l'image traitée
        st.subheader("🖼️ Image Traitée")
        st.image(processed_img, use_container_width=True)
        
    else:
        # Message d'accueil
        st.info("👆 **Chargez une image pour commencer l'analyse**")
        
        # Exemples
        st.markdown("### 📸 Exemples d'utilisation")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**✅ Bonne image**")
            st.markdown("- Visage de face")
            st.markdown("- Bien éclairé")
            st.markdown("- Expression claire")
        
        with col2:
            st.markdown("**⚠️ Image moyenne**")
            st.markdown("- Visage de profil")
            st.markdown("- Éclairage faible")
            st.markdown("- Expression neutre")
        
        with col3:
            st.markdown("**❌ Mauvaise image**")
            st.markdown("- Pas de visage")
            st.markdown("- Trop sombre")
            st.markdown("- Floue")

else:
    # Mode webcam externe
    st.info("""
    ### 📷 Utilisation de la Webcam sur macOS
    
    En raison des restrictions de sécurité de macOS, la webcam ne peut pas être utilisée directement dans le navigateur.
    
    **Solution recommandée:**
    
    1. Prenez une photo avec votre webcam (Photo Booth, etc.)
    2. Sauvegardez l'image
    3. Chargez-la via l'option "📤 Charger une image"
    
    **Alternative:**
    
    Utilisez le script Python standalone:
    ```bash
    python3 run_facial_recognition_simple.py
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p style="font-size: 1.2rem;"><b>🎭 Système de Reconnaissance d'Expressions Faciales</b></p>
    <p>Propulsé par Deep Learning | TensorFlow & OpenCV</p>
    <p style="font-size: 0.9rem; color: #999;">
        7 émotions détectées : Colère, Dégoût, Peur, Joie, Tristesse, Surprise, Neutre
    </p>
</div>
""", unsafe_allow_html=True)

# Made with Bob
