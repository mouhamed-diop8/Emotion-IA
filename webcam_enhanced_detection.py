#!/usr/bin/env python3
"""
Système Amélioré de Détection Faciale et Reconnaissance d'Émotions
Utilise plusieurs méthodes de détection pour une meilleure précision
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import sys
import os

# Configuration
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # Rouge
    'disgust': (170, 0, 255),  # Violet
    'fear': (255, 0, 0),       # Bleu
    'happy': (0, 255, 0),      # Vert
    'sad': (128, 0, 128),      # Violet foncé
    'surprise': (0, 165, 255), # Orange
    'neutral': (128, 128, 128) # Gris
}

EMOTION_EMOJIS = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
    'sad': '😢', 'surprise': '😲', 'neutral': '😐'
}

class EnhancedFaceDetector:
    """Détecteur de visage amélioré avec plusieurs méthodes"""
    
    def __init__(self, model_path="best_model.h5"):
        print("=" * 70)
        print("🎭 SYSTÈME DE RECONNAISSANCE D'ÉMOTIONS AMÉLIORÉ")
        print("=" * 70)
        
        # Charger le modèle CNN
        print("\n📦 Chargement du modèle CNN...")
        try:
            self.emotion_model = load_model(model_path, compile=False)
            print("✅ Modèle CNN chargé avec succès!")
        except Exception as e:
            print(f"❌ Erreur de chargement du modèle: {e}")
            sys.exit(1)
        
        # Charger les détecteurs de visages
        print("\n📦 Chargement des détecteurs de visages...")
        
        # 1. Haar Cascade (rapide, bon pour frontal)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ Haar Cascade chargé")
        
        # 2. Détecteur alternatif pour profil
        self.face_cascade_profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        print("✅ Détecteur de profil chargé")
        
        # 3. Détecteur d'yeux pour validation
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        print("✅ Détecteur d'yeux chargé")
        
        # Statistiques
        self.frame_count = 0
        self.detection_count = 0
        self.emotion_history = []
        self.confidence_history = []
        self.fps_history = []
        self.last_time = time.time()
        
        print("\n✅ Système initialisé avec succès!")
    
    def detect_faces_multi_method(self, gray_frame):
        """
        Détection de visages avec plusieurs méthodes pour meilleure précision
        """
        faces = []
        
        # Méthode 1: Haar Cascade frontal (principal)
        faces_frontal = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Méthode 2: Avec paramètres plus sensibles si aucun visage détecté
        if len(faces_frontal) == 0:
            faces_frontal = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # Méthode 3: Détection de profil si toujours rien
        if len(faces_frontal) == 0:
            faces_profile = self.face_cascade_profile.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48)
            )
            if len(faces_profile) > 0:
                faces_frontal = faces_profile
        
        # Valider les détections avec les yeux
        validated_faces = []
        for (x, y, w, h) in faces_frontal:
            face_roi = gray_frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
            
            # Si au moins un œil détecté, c'est probablement un vrai visage
            if len(eyes) >= 1 or w * h > 5000:  # Ou si le visage est assez grand
                validated_faces.append((x, y, w, h))
        
        # Si validation trop stricte, utiliser détections originales
        if len(validated_faces) == 0 and len(faces_frontal) > 0:
            validated_faces = faces_frontal
        
        return validated_faces
    
    def preprocess_face(self, face_roi):
        """
        Prétraitement amélioré du visage pour meilleure reconnaissance
        """
        # 1. Égalisation d'histogramme pour améliorer le contraste
        face_equalized = cv2.equalizeHist(face_roi)
        
        # 2. Réduction du bruit
        face_denoised = cv2.fastNlMeansDenoising(face_equalized, h=10)
        
        # 3. Redimensionner à 48x48
        face_resized = cv2.resize(face_denoised, (48, 48), interpolation=cv2.INTER_CUBIC)
        
        # 4. Normalisation
        face_normalized = face_resized / 255.0
        
        # 5. Reshape pour le modèle
        face_input = face_normalized.reshape(1, 48, 48, 1)
        
        return face_input
    
    def predict_emotion(self, face_roi):
        """
        Prédiction d'émotion avec prétraitement amélioré
        """
        # Prétraiter le visage
        face_input = self.preprocess_face(face_roi)
        
        # Prédiction
        predictions = self.emotion_model.predict(face_input, verbose=0)[0]
        
        # Obtenir l'émotion avec la plus haute probabilité
        emotion_idx = np.argmax(predictions)
        emotion = EMOTIONS[emotion_idx]
        confidence = predictions[emotion_idx]
        
        return emotion, confidence, predictions
    
    def draw_enhanced_ui(self, frame, faces_info, fps):
        """
        Interface utilisateur améliorée avec plus d'informations
        """
        h, w = frame.shape[:2]
        
        # Panneau supérieur semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Titre
        cv2.putText(frame, "Reconnaissance d'Emotions - Detection Amelioree", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Statistiques
        cv2.putText(frame, f"FPS: {fps:.1f} | Visages: {len(faces_info)} | Frames: {self.frame_count}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Taux de détection
        detection_rate = (self.detection_count / max(self.frame_count, 1)) * 100
        cv2.putText(frame, f"Taux detection: {detection_rate:.1f}%", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Confiance moyenne
        if len(self.confidence_history) > 0:
            avg_conf = np.mean(self.confidence_history[-30:])
            cv2.putText(frame, f"Confiance moy: {avg_conf:.1%}", 
                       (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Q:Quitter | S:Capture | R:Reset | P:Pause", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_face_info(self, frame, x, y, w, h, emotion, confidence, predictions):
        """
        Dessiner les informations sur le visage détecté
        """
        color = EMOTION_COLORS.get(emotion, (128, 128, 128))
        
        # Rectangle autour du visage (épaisseur selon confiance)
        thickness = 3 if confidence > 0.6 else 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        
        # Coins arrondis pour un meilleur look
        corner_length = 20
        cv2.line(frame, (x, y), (x + corner_length, y), color, 4)
        cv2.line(frame, (x, y), (x, y + corner_length), color, 4)
        cv2.line(frame, (x+w, y), (x+w - corner_length, y), color, 4)
        cv2.line(frame, (x+w, y), (x+w, y + corner_length), color, 4)
        cv2.line(frame, (x, y+h), (x + corner_length, y+h), color, 4)
        cv2.line(frame, (x, y+h), (x, y+h - corner_length), color, 4)
        cv2.line(frame, (x+w, y+h), (x+w - corner_length, y+h), color, 4)
        cv2.line(frame, (x+w, y+h), (x+w, y+h - corner_length), color, 4)
        
        # Étiquette avec émotion
        emoji = EMOTION_EMOJIS.get(emotion, '🎭')
        label = f"{emotion.upper()} {emoji}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Fond de l'étiquette
        cv2.rectangle(frame, (x, y - label_size[1] - 15), 
                     (x + label_size[0] + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Barre de confiance
        conf_text = f"{confidence:.0%}"
        cv2.putText(frame, conf_text, (x, y + h + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Mini barre de confiance
        bar_width = int(w * confidence)
        cv2.rectangle(frame, (x, y + h + 5), (x + w, y + h + 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y + h + 5), (x + bar_width, y + h + 10), color, -1)
        
        return frame
    
    def run(self, camera_index=0):
        """
        Lancer la détection en temps réel avec système amélioré
        """
        print("\n" + "=" * 70)
        print("🎥 DÉMARRAGE DE LA CAMÉRA")
        print("=" * 70)
        
        # Ouvrir la caméra
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra!")
            print("\n💡 Vérifiez:")
            print("  1. Permissions caméra dans Préférences Système")
            print("  2. Aucune autre app n'utilise la caméra")
            print("  3. Essayez un autre index (0, 1, 2...)")
            return
        
        print("✅ Caméra ouverte avec succès!")
        
        # Configuration optimale
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Réduire le buffer pour moins de latence
        
        window_name = "Detection Faciale Amelioree - CNN"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        paused = False
        
        print("\n" + "=" * 70)
        print("✅ SYSTÈME PRÊT - DÉTECTION EN COURS")
        print("=" * 70)
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("⚠️ Erreur de lecture")
                        break
                    
                    self.frame_count += 1
                    
                    # Calculer FPS
                    current_time = time.time()
                    fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
                    self.last_time = current_time
                    self.fps_history.append(fps)
                    if len(self.fps_history) > 30:
                        self.fps_history.pop(0)
                    avg_fps = np.mean(self.fps_history)
                    
                    # Convertir en niveaux de gris
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Détecter les visages avec méthode améliorée
                    faces = self.detect_faces_multi_method(gray)
                    
                    faces_info = []
                    
                    if len(faces) > 0:
                        self.detection_count += 1
                    
                    # Traiter chaque visage
                    for (x, y, w, h) in faces:
                        # Extraire le ROI du visage
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Prédire l'émotion
                        emotion, confidence, predictions = self.predict_emotion(face_roi)
                        
                        # Ajouter à l'historique
                        self.emotion_history.append(emotion)
                        self.confidence_history.append(confidence)
                        if len(self.emotion_history) > 100:
                            self.emotion_history.pop(0)
                            self.confidence_history.pop(0)
                        
                        # Dessiner les informations
                        frame = self.draw_face_info(frame, x, y, w, h, emotion, confidence, predictions)
                        
                        faces_info.append({
                            'emotion': emotion,
                            'confidence': confidence,
                            'bbox': (x, y, w, h)
                        })
                    
                    # Dessiner l'interface
                    frame = self.draw_enhanced_ui(frame, faces_info, avg_fps)
                    
                    # Afficher
                    cv2.imshow(window_name, frame)
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q ou ESC
                    print("\n👋 Arrêt demandé...")
                    break
                elif key == ord('s'):
                    filename = f"capture_enhanced_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Capture sauvegardée: {filename}")
                elif key == ord('r'):
                    self.emotion_history = []
                    self.confidence_history = []
                    self.frame_count = 0
                    self.detection_count = 0
                    print("🔄 Statistiques réinitialisées")
                elif key == ord('p'):
                    paused = not paused
                    print(f"⏸️  {'PAUSE' if paused else 'REPRISE'}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interruption par l'utilisateur")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Statistiques finales
            print("\n" + "=" * 70)
            print("📊 STATISTIQUES FINALES")
            print("=" * 70)
            print(f"Frames traitées: {self.frame_count}")
            print(f"Visages détectés: {self.detection_count}")
            print(f"Taux de détection: {(self.detection_count/max(self.frame_count,1))*100:.1f}%")
            
            if len(self.fps_history) > 0:
                print(f"FPS moyen: {np.mean(self.fps_history):.1f}")
            
            if len(self.confidence_history) > 0:
                print(f"Confiance moyenne: {np.mean(self.confidence_history):.1%}")
                print(f"Confiance min/max: {np.min(self.confidence_history):.1%} / {np.max(self.confidence_history):.1%}")
            
            if len(self.emotion_history) > 0:
                from collections import Counter
                emotion_counts = Counter(self.emotion_history)
                print("\n📈 Distribution des émotions:")
                for emotion, count in emotion_counts.most_common():
                    percentage = (count / len(self.emotion_history)) * 100
                    emoji = EMOTION_EMOJIS[emotion]
                    print(f"  {emoji} {emotion:10s}: {count:4d} ({percentage:5.1f}%)")
            
            print("\n✅ Application fermée avec succès!")

def main():
    """Fonction principale"""
    print("\n")
    
    # Vérifier le modèle
    if not os.path.exists("best_model.h5"):
        print("❌ Erreur: Le fichier 'best_model.h5' est introuvable!")
        print("\n💡 Entraînez d'abord le modèle avec:")
        print("  python3 run_facial_recognition.py")
        return
    
    # Créer le détecteur
    detector = EnhancedFaceDetector()
    
    # Demander l'index de la caméra
    print("\n📷 Configuration de la caméra:")
    print("  0 = Caméra intégrée (défaut)")
    print("  1 = Caméra externe")
    
    try:
        camera_input = input("\nIndex de la caméra [0]: ").strip()
        camera_index = int(camera_input) if camera_input else 0
    except ValueError:
        camera_index = 0
    
    # Lancer
    detector.run(camera_index=camera_index)

if __name__ == "__main__":
    main()

# Made with Bob
