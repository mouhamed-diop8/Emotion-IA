#!/usr/bin/env python3
"""
Application Webcam Simple - Classification en Temps Réel
Compatible macOS avec détection automatique de la caméra
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import time

# Configuration
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_FR = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Neutre', 'Tristesse', 'Surprise']

COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 255, 0),
    'fear': (128, 0, 128),
    'happy': (0, 255, 255),
    'neutral': (255, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (255, 165, 0)
}

def find_camera():
    """Trouve automatiquement une caméra disponible"""
    print("🔍 Recherche de caméras...")
    
    # Essayer différentes méthodes pour macOS
    methods = [
        (0, None),  # Index 0 par défaut
        (0, cv2.CAP_AVFOUNDATION),  # AVFoundation pour macOS
        (1, None),  # Index 1
        (1, cv2.CAP_AVFOUNDATION),
    ]
    
    for index, backend in methods:
        try:
            if backend:
                cap = cv2.VideoCapture(index, backend)
                method_name = f"index {index} avec AVFoundation"
            else:
                cap = cv2.VideoCapture(index)
                method_name = f"index {index}"
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ Caméra trouvée: {method_name}")
                    return cap
                cap.release()
        except:
            continue
    
    return None

def main():
    print("=" * 70)
    print("🎭 RECONNAISSANCE D'EXPRESSIONS FACIALES - TEMPS RÉEL")
    print("=" * 70)
    
    # 1. Charger le modèle
    print("\n📦 Chargement du modèle...")
    try:
        model = load_model('best_model.h5', compile=False)
        print("✅ Modèle chargé!")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("💡 Assurez-vous que 'best_model.h5' existe")
        return
    
    # 2. Charger le détecteur de visages
    print("📦 Chargement du détecteur de visages...")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    if face_cascade.empty():
        print("❌ Erreur: Détecteur de visages non chargé")
        return
    print("✅ Détecteur chargé!")
    
    # 3. Ouvrir la caméra
    print("\n📹 Ouverture de la caméra...")
    cap = find_camera()
    
    if cap is None:
        print("\n❌ AUCUNE CAMÉRA DÉTECTÉE")
        print("\n💡 Solutions:")
        print("1. Vérifiez que la caméra n'est pas utilisée (Photo Booth, Zoom, etc.)")
        print("2. Autorisez l'accès dans: Préférences Système > Sécurité > Caméra")
        print("3. Redémarrez votre Mac")
        return
    
    # Configuration optimale
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\n" + "=" * 70)
    print("✅ SYSTÈME PRÊT - DÉTECTION EN COURS")
    print("=" * 70)
    print("Commandes:")
    print("  Q ou ESC : Quitter")
    print("  S        : Capture d'écran")
    print("  ESPACE   : Pause")
    print("=" * 70 + "\n")
    
    # Variables
    paused = False
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("⚠️ Erreur de lecture")
                    break
                
                frame_count += 1
                
                # Calculer FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Convertir en niveaux de gris
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Détecter les visages
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(48, 48)
                )
                
                # Traiter chaque visage
                for (x, y, w, h) in faces:
                    # Extraire et prétraiter le visage
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, (48, 48))
                    face_normalized = face_resized.astype('float32') / 255.0
                    face_input = face_normalized.reshape(1, 48, 48, 1)
                    
                    # Prédiction
                    predictions = model.predict(face_input, verbose=0)[0]
                    emotion_idx = np.argmax(predictions)
                    confidence = predictions[emotion_idx]
                    emotion = EMOTIONS[emotion_idx]
                    emotion_fr = EMOTIONS_FR[emotion_idx]
                    
                    # Couleur selon l'émotion
                    color = COLORS[emotion]
                    
                    # Dessiner le rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                    
                    # Préparer le texte
                    label = f"{emotion_fr}: {confidence*100:.1f}%"
                    
                    # Fond pour le texte
                    (text_w, text_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(frame, (x, y-35), (x+text_w+10, y), color, -1)
                    
                    # Texte
                    cv2.putText(frame, label, (x+5, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Afficher les infos
                info_text = f"FPS: {fps:.1f} | Visages: {len(faces)} | Frame: {frame_count}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Instructions
                cv2.putText(frame, "Q:Quitter | S:Capture | ESPACE:Pause", 
                           (10, frame.shape[0]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Afficher
            cv2.imshow('Classification en Temps Reel', frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q ou ESC
                print("\n👋 Fermeture...")
                break
            elif key == ord('s'):  # S
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Sauvegardé: {filename}")
            elif key == ord(' '):  # ESPACE
                paused = not paused
                print("⏸️  PAUSE" if paused else "▶️  REPRISE")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interruption")
    
    finally:
        # Libérer les ressources
        cap.release()
        cv2.destroyAllWindows()
        
        # Statistiques finales
        print("\n" + "=" * 70)
        print("📊 STATISTIQUES")
        print("=" * 70)
        print(f"Frames traitées: {frame_count}")
        print(f"Durée: {elapsed:.1f}s")
        print(f"FPS moyen: {fps:.1f}")
        print("=" * 70)
        print("✅ Terminé avec succès!")

if __name__ == "__main__":
    main()

# Made with Bob
