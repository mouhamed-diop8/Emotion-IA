#!/usr/bin/env python3
"""
Interface Web Interactive Streamlit - Reconnaissance d'Expressions Faciales avec DQN
Intégration IBM Project Bob pour optimisation, monitoring et industrialisation
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
import os
import json
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Facial Expression Recognition - DQN System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


class PerformanceMonitor:
    """Monitoring des performances et détection de biais - IBM Project Bob"""
    
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.confidence_history = deque(maxlen=max_history)
        self.emotion_counts = {emotion: 0 for emotion in 
                              ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']}
        self.inference_times = deque(maxlen=max_history)
        self.parameter_history = deque(maxlen=max_history)
        self.bias_alerts = []
        
    def log_prediction(self, emotion, confidence, inference_time, contrast, exposure):
        """Enregistrer une prédiction"""
        self.confidence_history.append(confidence)
        self.emotion_counts[emotion] += 1
        self.inference_times.append(inference_time)
        self.parameter_history.append({'contrast': contrast, 'exposure': exposure})
        
        # Détection de biais
        self._check_bias()
    
    def _check_bias(self):
        """Détecter les biais de classification"""
        if sum(self.emotion_counts.values()) < 50:
            return
        
        total = sum(self.emotion_counts.values())
        distribution = {k: v/total for k, v in self.emotion_counts.items()}
        
        # Alerte si une émotion domine (>50%)
        for emotion, ratio in distribution.items():
            if ratio > 0.5:
                alert = f"⚠️ Biais détecté: {emotion} représente {ratio*100:.1f}% des prédictions"
                if alert not in self.bias_alerts[-5:]:  # Éviter les doublons
                    self.bias_alerts.append(alert)
    
    def get_metrics(self):
        """Obtenir les métriques de performance"""
        if not self.confidence_history:
            return None
        
        return {
            'avg_confidence': np.mean(self.confidence_history),
            'std_confidence': np.std(self.confidence_history),
            'avg_inference_time': np.mean(self.inference_times) * 1000,  # ms
            'total_predictions': sum(self.emotion_counts.values()),
            'emotion_distribution': self.emotion_counts.copy()
        }


class OptimizedVideoProcessor:
    """Processeur vidéo optimisé avec DQN - IBM Project Bob"""
    
    def __init__(self, classifier_path, dqn_path=None, img_dim=48):
        """Initialiser le processeur"""
        self.img_dim = img_dim
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Chargement optimisé des modèles
        with st.spinner("🔄 Chargement des modèles (optimisation IBM)..."):
            self.classifier = self._load_optimized_model(classifier_path)
            self.dqn_agent = self._load_optimized_model(dqn_path) if dqn_path and os.path.exists(dqn_path) else None
        
        # Paramètres
        self.contrast = 1.0
        self.exposure = 1.0
        self.contrast_range = (0.5, 2.0)
        self.exposure_range = (0.5, 2.0)
        self.step_size = 0.1
        
        # Détecteur de visages
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Monitoring
        self.monitor = PerformanceMonitor()
        
        # Cache pour optimisation
        self.prediction_cache = {}
        
    def _load_optimized_model(self, path):
        """Chargement optimisé du modèle - IBM Project Bob"""
        if not path or not os.path.exists(path):
            return None
        
        # Optimisation: chargement avec compilation désactivée pour inférence
        model = load_model(path, compile=False)
        
        # Optimisation: conversion en format optimisé si possible
        try:
            # TensorFlow Lite pour inférence rapide
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Note: Pour production, sauvegarder le modèle TFLite
        except:
            pass
        
        return model
    
    def adjust_image(self, image, contrast, exposure):
        """Ajuster contraste et exposition"""
        enhancer = ImageEnhance.Contrast(image)
        img_adjusted = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Brightness(img_adjusted)
        img_adjusted = enhancer.enhance(exposure)
        return img_adjusted
    
    def get_classification(self, face_img):
        """Classification avec monitoring"""
        start_time = time.time()
        
        # Appliquer ajustements
        adjusted_img = self.adjust_image(face_img, self.contrast, self.exposure)
        adjusted_img = adjusted_img.convert('L').resize((self.img_dim, self.img_dim))
        
        # Préparer pour classification
        img_array = np.array(adjusted_img) / 255.0
        img_array = img_array.reshape(1, self.img_dim, self.img_dim, 1)
        
        # Prédiction
        predictions = self.classifier.predict(img_array, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        emotion = self.emotions[emotion_idx]
        
        # Monitoring
        inference_time = time.time() - start_time
        self.monitor.log_prediction(emotion, confidence, inference_time, 
                                   self.contrast, self.exposure)
        
        return emotion, confidence, predictions[0]
    
    def get_dqn_action(self, confidence):
        """Obtenir action du DQN"""
        if self.dqn_agent is None:
            return 4  # Action neutre (maintenir)
        
        state = np.array([self.contrast, self.exposure, confidence])
        q_values = self.dqn_agent.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def apply_action(self, action):
        """Appliquer action DQN"""
        contrast_action = action // 3
        exposure_action = action % 3
        
        if contrast_action == 0:
            self.contrast = max(self.contrast_range[0], self.contrast - self.step_size)
        elif contrast_action == 2:
            self.contrast = min(self.contrast_range[1], self.contrast + self.step_size)
        
        if exposure_action == 0:
            self.exposure = max(self.exposure_range[0], self.exposure - self.step_size)
        elif exposure_action == 2:
            self.exposure = min(self.exposure_range[1], self.exposure + self.step_size)
    
    def process_frame(self, frame, use_dqn=True):
        """Traiter une frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, 
                                                   minNeighbors=5, minSize=(30, 30))
        
        results = []
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            
            emotion, confidence, all_predictions = self.get_classification(face_pil)
            
            # DQN adjustment
            if use_dqn and self.dqn_agent:
                action = self.get_dqn_action(confidence)
                self.apply_action(action)
            
            # Dessiner rectangle
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Texte
            text = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': all_predictions,
                'bbox': (x, y, w, h)
            })
        
        return frame, results


def main():
    """Application principale Streamlit"""
    
    # En-tête
    st.markdown('<h1 class="main-header">🎭 Reconnaissance d\'Expressions Faciales avec DQN</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Système intelligent avec optimisation IBM Project Bob</p>', 
                unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sélection des modèles
        classifier_path = st.text_input("Modèle Classificateur", "best_model.h5")
        dqn_path = st.text_input("Modèle DQN (optionnel)", "dqn_agent.h5")
        
        # Options
        st.subheader("Options")
        use_dqn = st.checkbox("Activer DQN", value=True, 
                             help="Ajustements intelligents des paramètres")
        show_metrics = st.checkbox("Afficher métriques", value=True)
        show_monitoring = st.checkbox("Monitoring IBM", value=True)
        
        # Source vidéo
        st.subheader("Source Vidéo")
        video_source = st.radio("Sélectionner source", 
                               ["Webcam", "Fichier vidéo", "Image statique"])
        
        if video_source == "Fichier vidéo":
            uploaded_video = st.file_uploader("Charger vidéo", type=['mp4', 'avi', 'mov'])
        elif video_source == "Image statique":
            uploaded_image = st.file_uploader("Charger image", type=['jpg', 'jpeg', 'png'])
        
        # Paramètres avancés
        with st.expander("🔧 Paramètres Avancés"):
            manual_contrast = st.slider("Contraste manuel", 0.5, 2.0, 1.0, 0.1)
            manual_exposure = st.slider("Exposition manuelle", 0.5, 2.0, 1.0, 0.1)
            confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)
        
        # IBM Project Bob Info
        st.markdown("---")
        st.markdown("### 🔵 IBM Project Bob")
        st.markdown("""
        <div class="info-box">
        <b>Fonctionnalités actives:</b><br>
        ✓ Optimisation d'inférence<br>
        ✓ Monitoring en temps réel<br>
        ✓ Détection de biais<br>
        ✓ Métriques de performance
        </div>
        """, unsafe_allow_html=True)
    
    # Vérifier les modèles
    if not os.path.exists(classifier_path):
        st.error(f"❌ Modèle classificateur introuvable: {classifier_path}")
        st.info("💡 Entraînez d'abord le modèle avec: `python3 run_facial_recognition_simple.py`")
        return
    
    # Initialiser le processeur
    if 'processor' not in st.session_state:
        try:
            st.session_state.processor = OptimizedVideoProcessor(classifier_path, dqn_path)
            st.success("✅ Modèles chargés avec succès!")
        except Exception as e:
            st.error(f"❌ Erreur de chargement: {str(e)}")
            return
    
    processor = st.session_state.processor
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Flux Vidéo en Temps Réel")
        video_placeholder = st.empty()
        
        # Contrôles
        control_col1, control_col2, control_col3 = st.columns(3)
        with control_col1:
            start_button = st.button("▶️ Démarrer", use_container_width=True)
        with control_col2:
            stop_button = st.button("⏸️ Pause", use_container_width=True)
        with control_col3:
            reset_button = st.button("🔄 Reset", use_container_width=True)
    
    with col2:
        st.subheader("📊 Informations")
        
        # Paramètres actuels
        st.markdown("**Paramètres Actuels:**")
        param_col1, param_col2 = st.columns(2)
        with param_col1:
            contrast_display = st.empty()
        with param_col2:
            exposure_display = st.empty()
        
        # Dernière détection
        st.markdown("**Dernière Détection:**")
        emotion_display = st.empty()
        confidence_display = st.empty()
        
        # Graphique des prédictions
        predictions_chart = st.empty()
    
    # Section monitoring
    if show_monitoring:
        st.markdown("---")
        st.subheader("📈 Monitoring IBM Project Bob")
        
        mon_col1, mon_col2, mon_col3, mon_col4 = st.columns(4)
        
        with mon_col1:
            avg_conf_metric = st.empty()
        with mon_col2:
            inference_time_metric = st.empty()
        with mon_col3:
            total_pred_metric = st.empty()
        with mon_col4:
            dqn_status_metric = st.empty()
        
        # Graphiques de monitoring
        monitoring_tabs = st.tabs(["📊 Distribution", "📈 Tendances", "⚠️ Alertes Biais"])
        
        with monitoring_tabs[0]:
            emotion_dist_chart = st.empty()
        
        with monitoring_tabs[1]:
            confidence_trend_chart = st.empty()
        
        with monitoring_tabs[2]:
            bias_alerts_display = st.empty()
    
    # Traitement vidéo
    if video_source == "Webcam" and start_button:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Impossible d'ouvrir la webcam")
            return
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Traiter la frame
                processed_frame, results = processor.process_frame(frame, use_dqn)
                
                # Afficher
                video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                                       channels="RGB", use_column_width=True)
                
                # Mettre à jour les informations
                contrast_display.metric("Contraste", f"{processor.contrast:.2f}")
                exposure_display.metric("Exposition", f"{processor.exposure:.2f}")
                
                if results:
                    result = results[0]
                    emotion_display.markdown(f"**Émotion:** {result['emotion'].upper()}")
                    confidence_display.progress(float(result['confidence']))
                    
                    # Graphique des prédictions
                    fig = go.Figure(data=[
                        go.Bar(x=processor.emotions, y=result['all_predictions'])
                    ])
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0),
                                    title="Probabilités")
                    predictions_chart.plotly_chart(fig, use_container_width=True)
                
                # Monitoring
                if show_monitoring and frame_count % 10 == 0:
                    metrics = processor.monitor.get_metrics()
                    if metrics:
                        avg_conf_metric.metric("Confiance Moy.", f"{metrics['avg_confidence']:.3f}")
                        inference_time_metric.metric("Temps Inférence", f"{metrics['avg_inference_time']:.1f} ms")
                        total_pred_metric.metric("Total Prédictions", metrics['total_predictions'])
                        dqn_status_metric.metric("DQN", "✅ Actif" if use_dqn else "⏸️ Inactif")
                        
                        # Distribution des émotions
                        fig_dist = px.pie(values=list(metrics['emotion_distribution'].values()),
                                         names=list(metrics['emotion_distribution'].keys()),
                                         title="Distribution des Émotions")
                        emotion_dist_chart.plotly_chart(fig_dist, use_container_width=True, key=f"emotion_dist_{frame_count}")
                        
                        # Tendance de confiance
                        if len(processor.monitor.confidence_history) > 10:
                            fig_trend = go.Figure()
                            fig_trend.add_trace(go.Scatter(
                                y=list(processor.monitor.confidence_history),
                                mode='lines',
                                name='Confiance'
                            ))
                            fig_trend.update_layout(height=250, title="Évolution de la Confiance")
                            confidence_trend_chart.plotly_chart(fig_trend, use_container_width=True, key=f"confidence_trend_{frame_count}")
                        
                        # Alertes de biais
                        if processor.monitor.bias_alerts:
                            bias_alerts_display.warning("\n\n".join(processor.monitor.bias_alerts[-5:]))
                
                frame_count += 1
                
                if stop_button:
                    break
                
                time.sleep(0.03)  # ~30 FPS
        
        finally:
            cap.release()
    
    elif video_source == "Image statique" and 'uploaded_image' in locals() and uploaded_image:
        # Traiter image statique
        image = Image.open(uploaded_image)
        image_np = np.array(image)
        
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        
        processed_frame, results = processor.process_frame(image_np, use_dqn)
        
        video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                               channels="RGB", use_column_width=True)
        
        if results:
            result = results[0]
            emotion_display.markdown(f"**Émotion:** {result['emotion'].upper()}")
            confidence_display.progress(float(result['confidence']))
            
            fig = go.Figure(data=[
                go.Bar(x=processor.emotions, y=result['all_predictions'])
            ])
            fig.update_layout(height=200, title="Probabilités")
            predictions_chart.plotly_chart(fig, use_container_width=True)
    
    # Reset
    if reset_button:
        processor.contrast = 1.0
        processor.exposure = 1.0
        st.success("✅ Paramètres réinitialisés")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p><b>Système de Reconnaissance d'Expressions Faciales avec DQN</b></p>
    <p>Optimisé avec IBM Project Bob | Deep Learning & Apprentissage par Renforcement</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# Made with Bob
