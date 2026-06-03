#!/usr/bin/env python3
"""
Interface Web Streamlit Simplifiée - Reconnaissance d'Expressions Faciales
Version de débogage avec gestion d'erreurs améliorée
"""

import streamlit as st
import sys
import traceback

# Configuration de la page
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Vérification des dépendances
def check_dependencies():
    """Vérifier et afficher l'état des dépendances"""
    deps_status = {}
    
    # Liste des dépendances requises
    required_deps = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'PIL': 'pillow',
        'tensorflow': 'tensorflow',
        'pandas': 'pandas',
        'plotly': 'plotly'
    }
    
    for module, package in required_deps.items():
        try:
            __import__(module)
            deps_status[package] = "✅ Installé"
        except ImportError:
            deps_status[package] = "❌ Manquant"
    
    return deps_status

# En-tête
st.markdown('<h1 style="text-align: center; color: #1f77b4;">🎭 Reconnaissance d\'Expressions Faciales</h1>', 
            unsafe_allow_html=True)

# Vérification des dépendances
st.subheader("📦 État des Dépendances")
deps_status = check_dependencies()

col1, col2 = st.columns(2)
for i, (package, status) in enumerate(deps_status.items()):
    if i % 2 == 0:
        col1.write(f"**{package}:** {status}")
    else:
        col2.write(f"**{package}:** {status}")

# Vérifier si toutes les dépendances sont installées
all_installed = all("✅" in status for status in deps_status.values())

if not all_installed:
    st.error("⚠️ Certaines dépendances sont manquantes!")
    st.info("""
    **Pour installer les dépendances manquantes, exécutez:**
    ```bash
    python3 -m pip install opencv-python numpy pillow tensorflow pandas plotly --user
    ```
    
    **Note:** L'installation d'OpenCV peut prendre plusieurs minutes.
    """)
    st.stop()

# Si toutes les dépendances sont présentes, charger l'application complète
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import os
    import time
    
    st.success("✅ Toutes les dépendances sont installées!")
    
    # Vérifier le modèle
    st.subheader("🤖 État du Modèle")
    
    model_path = "best_model.h5"
    if os.path.exists(model_path):
        st.success(f"✅ Modèle trouvé: {model_path}")
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        st.info(f"📊 Taille du modèle: {model_size:.2f} MB")
        
        # Charger le modèle
        with st.spinner("🔄 Chargement du modèle..."):
            try:
                model = load_model(model_path, compile=False)
                st.success("✅ Modèle chargé avec succès!")
                
                # Afficher l'architecture
                with st.expander("🔍 Architecture du Modèle"):
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary[:20]))  # Afficher les 20 premières lignes
                
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
                st.code(traceback.format_exc())
    else:
        st.error(f"❌ Modèle introuvable: {model_path}")
        st.info("""
        **Pour entraîner le modèle, exécutez:**
        ```bash
        python3 run_facial_recognition_simple.py
        ```
        """)
        st.stop()
    
    # Interface principale
    st.markdown("---")
    st.subheader("📸 Test de Reconnaissance")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        test_mode = st.radio(
            "Mode de test",
            ["Image statique", "Webcam (à venir)"]
        )
        
        if test_mode == "Image statique":
            uploaded_file = st.file_uploader(
                "Charger une image",
                type=['jpg', 'jpeg', 'png']
            )
        
        st.markdown("---")
        st.markdown("### 📊 Informations")
        st.info(f"""
        **Version:** 1.0 (Simplifiée)
        **TensorFlow:** {tf.__version__}
        **OpenCV:** {cv2.__version__}
        """)
    
    # Zone de test
    if test_mode == "Image statique" and uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Image Originale")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("🎯 Résultat")
            
            with st.spinner("🔄 Analyse en cours..."):
                try:
                    # Convertir en array numpy
                    img_array = np.array(image)
                    
                    # Détection de visage
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                    
                    if len(faces) > 0:
                        st.success(f"✅ {len(faces)} visage(s) détecté(s)")
                        
                        # Traiter le premier visage
                        x, y, w, h = faces[0]
                        face_roi = img_array[y:y+h, x:x+w]
                        
                        # Préparer pour le modèle
                        face_pil = Image.fromarray(face_roi).convert('L').resize((48, 48))
                        face_array = np.array(face_pil) / 255.0
                        face_array = face_array.reshape(1, 48, 48, 1)
                        
                        # Prédiction
                        predictions = model.predict(face_array, verbose=0)
                        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                        
                        emotion_idx = np.argmax(predictions[0])
                        emotion = emotions[emotion_idx]
                        confidence = predictions[0][emotion_idx]
                        
                        # Afficher le résultat
                        st.markdown(f"### Émotion: **{emotion.upper()}**")
                        st.progress(float(confidence))
                        st.write(f"Confiance: {confidence:.2%}")
                        
                        # Graphique des probabilités
                        fig = go.Figure(data=[
                            go.Bar(x=emotions, y=predictions[0])
                        ])
                        fig.update_layout(
                            title="Probabilités par Émotion",
                            xaxis_title="Émotion",
                            yaxis_title="Probabilité",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Afficher le visage détecté
                        st.image(face_roi, caption="Visage détecté", use_column_width=True)
                        
                    else:
                        st.warning("⚠️ Aucun visage détecté dans l'image")
                        st.info("Essayez avec une image contenant un visage clairement visible")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                    st.code(traceback.format_exc())
    
    elif test_mode == "Image statique":
        st.info("👆 Chargez une image dans la barre latérale pour commencer")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    <p><b>Système de Reconnaissance d'Expressions Faciales</b></p>
    <p>Version simplifiée pour débogage</p>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error("❌ Erreur critique lors du chargement de l'application")
    st.code(traceback.format_exc())
    st.info("""
    **Suggestions:**
    1. Vérifiez que toutes les dépendances sont installées
    2. Redémarrez l'application Streamlit
    3. Consultez le fichier TROUBLESHOOTING_STREAMLIT.md
    """)

# Made with Bob
