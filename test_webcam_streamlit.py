#!/usr/bin/env python3
"""
Script de diagnostic pour la webcam dans Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

st.set_page_config(page_title="Test Webcam", page_icon="📷", layout="wide")

st.title("🔍 Diagnostic Webcam pour Streamlit")

st.markdown("---")

# Test 1: Vérifier OpenCV
st.header("1️⃣ Test OpenCV")
with st.spinner("Test de la caméra avec OpenCV..."):
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            st.success(f"✅ OpenCV peut accéder à la caméra! Résolution: {frame.shape[1]}x{frame.shape[0]}")
            # Afficher une frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Frame capturée avec OpenCV", use_column_width=True)
        else:
            st.error("❌ OpenCV peut ouvrir la caméra mais ne peut pas capturer d'image")
        cap.release()
    else:
        st.error("❌ OpenCV ne peut pas accéder à la caméra")

st.markdown("---")

# Test 2: Streamlit camera_input
st.header("2️⃣ Test Streamlit camera_input")

st.info("""
**Instructions importantes pour macOS:**

1. **Autorisations du navigateur:** Quand vous cliquez sur "Prenez une photo", votre navigateur va demander l'autorisation d'accéder à la caméra. Cliquez sur "Autoriser".

2. **Préférences Système:** Si ça ne fonctionne pas:
   - Allez dans **Préférences Système** → **Sécurité et confidentialité** → **Caméra**
   - Cochez votre navigateur (Chrome, Safari, Firefox)
   - **Redémarrez le navigateur**

3. **Problèmes courants:**
   - Si vous voyez un message d'erreur dans le navigateur, vérifiez la console (F12)
   - Fermez les autres applications utilisant la caméra (Zoom, Skype, etc.)
   - Essayez un autre navigateur
""")

camera_image = st.camera_input("📸 Prenez une photo")

if camera_image:
    st.success("✅ Streamlit camera_input fonctionne!")
    image = Image.open(camera_image)
    st.image(image, caption="Photo capturée avec Streamlit", use_column_width=True)
else:
    st.warning("⚠️ En attente de capture...")

st.markdown("---")

# Test 3: Alternative avec bouton
st.header("3️⃣ Alternative: Capture manuelle avec OpenCV")

if st.button("📷 Capturer une image maintenant"):
    with st.spinner("Capture en cours..."):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            # Attendre que la caméra se stabilise
            time.sleep(0.5)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.success("✅ Image capturée avec succès!")
                st.image(frame_rgb, caption="Capture OpenCV", use_column_width=True)
                
                # Sauvegarder
                timestamp = int(time.time())
                filename = f"webcam_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                st.info(f"💾 Image sauvegardée: {filename}")
            else:
                st.error("❌ Erreur de capture")
            cap.release()
        else:
            st.error("❌ Impossible d'ouvrir la caméra")

st.markdown("---")

# Informations système
st.header("ℹ️ Informations Système")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Versions")
    import sys
    st.write(f"**Python:** {sys.version.split()[0]}")
    st.write(f"**Streamlit:** {st.__version__}")
    st.write(f"**OpenCV:** {cv2.__version__}")

with col2:
    st.subheader("Caméra")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        st.write(f"**Résolution:** {width}x{height}")
        st.write(f"**FPS:** {fps}")
        st.write(f"**Backend:** {cap.getBackendName()}")
        cap.release()
    else:
        st.write("❌ Caméra non accessible")

st.markdown("---")

st.markdown("""
### 🆘 Si la webcam ne fonctionne toujours pas:

1. **Vérifiez les permissions macOS:**
   ```bash
   open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
   ```

2. **Testez dans le terminal:**
   ```bash
   python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
   ```

3. **Utilisez une application alternative:**
   - `python3 webcam_universal_app.py` (application standalone)
   - `python3 webcam_macos_app.py` (optimisé pour macOS)

4. **Consultez le guide complet:**
   - Voir `FIX_WEBCAM_STREAMLIT.md`
""")

# Made with Bob
