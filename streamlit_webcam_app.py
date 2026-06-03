#!/usr/bin/env python3
"""
Interface Streamlit avec Webcam en Temps Réel
Reconnaissance d'Expressions Faciales
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from collections import deque
import plotly.graph_objects as go

# Configuration
st.set_page_config(
    page_title="Facial Expression Recognition - Webcam",
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
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
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

# Variables globales
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

# En-tête
st.markdown('<h1 class="main-header">🎭 Reconnaissance d\'Expressions Faciales en Temps Réel</h1>', 
            unsafe_allow_html=True)

# Charger les modèles
try:
    model = load_emotion_model()
    face_cascade = load_face_detector()
    st.success("✅ Modèles chargés avec succès!")
except Exception as e:
    st.error(f"❌ Erreur de chargement: {str(e)}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Seuil de confiance
    confidence_threshold = st.slider(
        "Seuil de confiance",
        0.0, 1.0, 0.5, 0.05,
        help="Afficher uniquement les prédictions avec une confiance supérieure à ce seuil"
    )
    
    # Paramètres de détection
    st.subheader("🔍 Détection de Visages")
    scale_factor = st.slider("Scale Factor", 1.05, 1.5, 1.1, 0.05)
    min_neighbors = st.slider("Min Neighbors", 3, 10, 5)
    
    # Affichage
    st.subheader("📊 Affichage")
    show_confidence = st.checkbox("Afficher confiance", value=True)
    show_bbox = st.checkbox("Afficher cadre", value=True)
    show_all_emotions = st.checkbox("Afficher toutes les émotions", value=False)
    
    st.markdown("---")
    st.info("""
    **Instructions:**
    1. Cliquez sur "Démarrer la Webcam"
    2. Autorisez l'accès à la caméra
    3. Positionnez votre visage face à la caméra
    4. L'émotion sera détectée en temps réel
    """)

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Flux Vidéo")
    
    # Placeholder pour la vidéo
    video_placeholder = st.empty()
    
    # Contrôles
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        start_button = st.button("▶️ Démarrer la Webcam", use_container_width=True)
    
    with control_col2:
        stop_button = st.button("⏸️ Arrêter", use_container_width=True)
    
    with control_col3:
        snapshot_button = st.button("📸 Capture", use_container_width=True)

with col2:
    st.subheader("📊 Détection Actuelle")
    
    # Placeholders pour les informations
    emotion_display = st.empty()
    confidence_display = st.empty()
    chart_placeholder = st.empty()
    
    # Statistiques
    st.subheader("📈 Statistiques")
    stats_placeholder = st.empty()

# Session state pour la webcam
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = deque(maxlen=100)
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

def process_frame(frame):
    """Traiter une frame et détecter les émotions"""
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Détecter les visages
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=scale_factor, 
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    results = []
    
    for (x, y, w, h) in faces:
        # Extraire le visage
        face_roi = gray[y:y+h, x:x+w]
        
        # Redimensionner pour le modèle
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)
        
        # Prédiction
        predictions = model.predict(face_input, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = emotions[emotion_idx]
        confidence = predictions[emotion_idx]
        
        # Dessiner le rectangle si activé
        if show_bbox:
            color = (0, 255, 0) if confidence > confidence_threshold else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Afficher l'émotion
        if confidence > confidence_threshold:
            text = f"{emotion}: {confidence:.2f}" if show_confidence else emotion
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        results.append({
            'emotion': emotion,
            'confidence': confidence,
            'predictions': predictions,
            'bbox': (x, y, w, h)
        })
    
    return frame, results

# Gestion de la webcam
if start_button:
    st.session_state.webcam_running = True

if stop_button:
    st.session_state.webcam_running = False

if st.session_state.webcam_running:
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Impossible d'ouvrir la webcam. Vérifiez qu'elle n'est pas utilisée par une autre application.")
        st.session_state.webcam_running = False
    else:
        st.info("🎥 Webcam active - Appuyez sur 'Arrêter' pour terminer")
        
        # Boucle de capture
        while st.session_state.webcam_running:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ Erreur de lecture de la webcam")
                break
            
            # Traiter la frame
            processed_frame, results = process_frame(frame)
            
            # Afficher la frame
            video_placeholder.image(
                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_column_width=True
            )
            
            # Mettre à jour les informations
            if results:
                result = results[0]  # Premier visage détecté
                
                # Ajouter à l'historique
                st.session_state.emotion_history.append(result['emotion'])
                st.session_state.frame_count += 1
                
                # Afficher l'émotion
                emotion_color = emotion_colors.get(result['emotion'], '#888888')
                emotion_display.markdown(
                    f"<h2 style='color: {emotion_color}; text-align: center;'>{result['emotion'].upper()}</h2>",
                    unsafe_allow_html=True
                )
                
                # Barre de confiance
                confidence_display.progress(float(result['confidence']))
                confidence_display.write(f"Confiance: {result['confidence']:.1%}")
                
                # Graphique des probabilités
                if show_all_emotions:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=emotions,
                            y=result['predictions'],
                            marker_color=[emotion_colors.get(e, '#888888') for e in emotions]
                        )
                    ])
                    fig.update_layout(
                        height=250,
                        margin=dict(l=0, r=0, t=20, b=0),
                        showlegend=False,
                        yaxis_title="Probabilité"
                    )
                    chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Statistiques
                if len(st.session_state.emotion_history) > 0:
                    emotion_counts = {}
                    for e in st.session_state.emotion_history:
                        emotion_counts[e] = emotion_counts.get(e, 0) + 1
                    
                    most_common = max(emotion_counts, key=emotion_counts.get)
                    
                    stats_placeholder.markdown(f"""
                    <div class="metric-box">
                    <b>Frames analysées:</b> {st.session_state.frame_count}<br>
                    <b>Émotion dominante:</b> {most_common}<br>
                    <b>Fréquence:</b> {emotion_counts[most_common]}/{len(st.session_state.emotion_history)}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Capture d'écran
            if snapshot_button:
                snapshot_path = f"snapshot_{int(time.time())}.jpg"
                cv2.imwrite(snapshot_path, processed_frame)
                st.success(f"📸 Capture sauvegardée: {snapshot_path}")
            
            # Petit délai pour ne pas surcharger
            time.sleep(0.03)  # ~30 FPS
            
            # Vérifier si on doit arrêter
            if stop_button:
                break
        
        cap.release()
        st.info("⏸️ Webcam arrêtée")
else:
    video_placeholder.info("👆 Cliquez sur 'Démarrer la Webcam' pour commencer la détection en temps réel")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
<p><b>Système de Reconnaissance d'Expressions Faciales</b></p>
<p>Détection en temps réel avec Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Made with Bob
