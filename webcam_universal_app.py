#!/usr/bin/env python3
"""
Application Universelle de Reconnaissance d'Expressions Faciales
Compatible: macOS, Windows, Linux
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import platform
import sys
import os

# Configuration selon le système d'exploitation
SYSTEM = platform.system()
print(f"🖥️  Système détecté: {SYSTEM}")

# Émotions
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_FR = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Neutre', 'Tristesse', 'Surprise']

# Couleurs pour chaque émotion
COLORS = {
    'angry': (0, 0, 255),      # Rouge
    'disgust': (0, 255, 0),    # Vert
    'fear': (128, 0, 128),     # Violet
    'happy': (0, 255, 255),    # Jaune
    'neutral': (255, 255, 255), # Blanc
    'sad': (255, 0, 0),        # Bleu
    'surprise': (255, 165, 0)  # Orange
}

def load_face_detector():
    """Charge le détecteur de visages Haar Cascade"""
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("❌ Erreur: Impossible de charger le détecteur de visages")
        sys.exit(1)
    
    return face_cascade

def preprocess_face(face_img):
    """Prétraite l'image du visage pour le modèle"""
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img

def get_camera_index():
    """Détecte automatiquement l'index de la caméra disponible"""
    print("\n🔍 Recherche de caméras disponibles...")
    
    # Essayer les indices de 0 à 5
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"✅ Caméra trouvée à l'index {i}")
                return i
    
    print("❌ Aucune caméra détectée")
    return None

def configure_camera(cap):
    """Configure la caméra selon le système d'exploitation"""
    if SYSTEM == "Darwin":  # macOS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    elif SYSTEM == "Windows":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:  # Linux
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

def draw_interface(frame, emotion, confidence, fps):
    """Dessine l'interface utilisateur sur la frame"""
    height, width = frame.shape[:2]
    
    # Panneau supérieur
    cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
    
    # Titre
    cv2.putText(frame, "Reconnaissance d'Expressions Faciales", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Système d'exploitation
    cv2.putText(frame, f"Systeme: {SYSTEM}", 
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", 
                (width - 120, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Panneau inférieur avec l'émotion détectée
    if emotion:
        emotion_fr = EMOTIONS_FR[EMOTIONS.index(emotion)]
        color = COLORS[emotion]
        
        # Barre d'émotion
        cv2.rectangle(frame, (0, height - 100), (width, height), (0, 0, 0), -1)
        
        # Nom de l'émotion
        cv2.putText(frame, f"Emotion: {emotion_fr}", 
                    (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Barre de confiance
        bar_width = int((width - 20) * confidence)
        cv2.rectangle(frame, (10, height - 35), (10 + bar_width, height - 15), color, -1)
        cv2.rectangle(frame, (10, height - 35), (width - 10, height - 15), (100, 100, 100), 2)
        
        # Pourcentage
        cv2.putText(frame, f"{confidence*100:.1f}%", 
                    (width - 100, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Q: Quitter | S: Capture | ESPACE: Pause", 
                (10, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

def main():
    print("=" * 70)
    print("🎭 RECONNAISSANCE D'EXPRESSIONS FACIALES - UNIVERSEL")
    print("=" * 70)
    print(f"🖥️  Système: {SYSTEM} ({platform.platform()})")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📹 OpenCV: {cv2.__version__}")
    
    # Charger le modèle
    print("\n🔄 Chargement du modèle...")
    try:
        model = load_model('best_model.h5')
        print("✅ Modèle chargé avec succès!")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        print("💡 Assurez-vous que 'best_model.h5' existe dans le répertoire")
        sys.exit(1)
    
    # Charger le détecteur de visages
    print("🔄 Chargement du détecteur de visages...")
    face_cascade = load_face_detector()
    print("✅ Détecteur de visages chargé!")
    
    # Détecter la caméra
    camera_index = get_camera_index()
    if camera_index is None:
        print("\n❌ Aucune caméra disponible")
        print("💡 Vérifiez que votre caméra est connectée et autorisée")
        sys.exit(1)
    
    # Ouvrir la caméra
    print(f"\n📷 Ouverture de la caméra {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la caméra")
        sys.exit(1)
    
    # Configurer la caméra
    configure_camera(cap)
    print("✅ Caméra configurée!")
    
    print("\n" + "=" * 70)
    print("🎬 DÉMARRAGE DE LA DÉTECTION")
    print("=" * 70)
    print("📋 Commandes:")
    print("   Q ou ESC  : Quitter")
    print("   S         : Prendre une capture d'écran")
    print("   ESPACE    : Pause/Reprendre")
    print("=" * 70)
    
    # Variables
    snapshot_count = 0
    paused = False
    fps = 0
    frame_count = 0
    import time
    start_time = time.time()
    
    current_emotion = None
    current_confidence = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur de lecture de la caméra")
                break
            
            # Calculer FPS
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Convertir en niveaux de gris pour la détection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Détecter les visages
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Traiter chaque visage détecté
            for (x, y, w, h) in faces:
                # Extraire le visage
                face_roi = frame[y:y+h, x:x+w]
                
                # Prétraiter pour le modèle
                processed_face = preprocess_face(face_roi)
                
                # Prédire l'émotion
                predictions = model.predict(processed_face, verbose=0)
                emotion_idx = np.argmax(predictions[0])
                confidence = predictions[0][emotion_idx]
                emotion = EMOTIONS[emotion_idx]
                
                current_emotion = emotion
                current_confidence = confidence
                
                # Dessiner le rectangle autour du visage
                color = COLORS[emotion]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Afficher l'émotion au-dessus du visage
                emotion_fr = EMOTIONS_FR[emotion_idx]
                label = f"{emotion_fr}: {confidence*100:.1f}%"
                
                # Fond pour le texte
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(frame, (x, y-30), (x+text_width+10, y), color, -1)
                
                # Texte
                cv2.putText(frame, label, (x+5, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Dessiner l'interface
            draw_interface(frame, current_emotion, current_confidence, fps)
        
        # Afficher la frame
        cv2.imshow('Reconnaissance Faciale - Universel', frame)
        
        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q ou ESC
            print("\n👋 Fermeture de l'application...")
            break
        elif key == ord('s'):  # S pour snapshot
            snapshot_count += 1
            filename = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Capture sauvegardée: {filename}")
        elif key == ord(' '):  # ESPACE pour pause
            paused = not paused
            status = "⏸️  PAUSE" if paused else "▶️  REPRISE"
            print(status)
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("✅ Application terminée avec succès")
    print(f"📊 Statistiques:")
    print(f"   - Frames traitées: {frame_count}")
    print(f"   - Captures: {snapshot_count}")
    print(f"   - FPS moyen: {fps:.1f}")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()

# Made with Bob
