#!/usr/bin/env python3
"""
Application Webcam Optimisée pour macOS
Reconnaissance d'Expressions Faciales en Temps Réel
Compatible avec les restrictions de sécurité macOS
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import sys
import os

# Configuration
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_colors = {
    'angry': (0, 0, 255),      # Rouge
    'disgust': (170, 0, 255),  # Violet
    'fear': (255, 0, 0),       # Bleu
    'happy': (0, 255, 0),      # Vert
    'sad': (128, 0, 128),      # Violet foncé
    'surprise': (0, 165, 255), # Orange
    'neutral': (128, 128, 128) # Gris
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

class WebcamEmotionDetector:
    def __init__(self, model_path="best_model.h5"):
        """Initialiser le détecteur"""
        print("🔄 Chargement du modèle...")
        try:
            self.model = load_model(model_path, compile=False)
            print("✅ Modèle chargé avec succès!")
        except Exception as e:
            print(f"❌ Erreur de chargement du modèle: {e}")
            sys.exit(1)
        
        print("🔄 Chargement du détecteur de visages...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✅ Détecteur de visages chargé!")
        
        # Statistiques
        self.frame_count = 0
        self.emotion_history = []
        self.fps_history = []
        self.last_time = time.time()
        
    def detect_emotion(self, face_roi):
        """Détecter l'émotion d'un visage"""
        # Redimensionner et normaliser
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)
        
        # Prédiction
        predictions = self.model.predict(face_input, verbose=0)[0]
        emotion_idx = np.argmax(predictions)
        emotion = emotions[emotion_idx]
        confidence = predictions[emotion_idx]
        
        return emotion, confidence, predictions
    
    def draw_info_panel(self, frame, emotion, confidence, predictions, fps):
        """Dessiner le panneau d'informations"""
        h, w = frame.shape[:2]
        
        # Panneau semi-transparent
        overlay = frame.copy()
        panel_height = 200
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Titre
        cv2.putText(frame, "Reconnaissance d'Expressions Faciales", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Émotion détectée
        emoji = emotion_emojis.get(emotion, '🎭')
        color = emotion_colors.get(emotion, (128, 128, 128))
        cv2.putText(frame, f"Emotion: {emotion.upper()} {emoji}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Confiance
        cv2.putText(frame, f"Confiance: {confidence:.1%}", 
                   (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Barre de confiance
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (20, 130), (320, 150), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 130), (20 + bar_width, 150), color, -1)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Statistiques
        if len(self.emotion_history) > 0:
            from collections import Counter
            most_common = Counter(self.emotion_history).most_common(1)[0]
            cv2.putText(frame, f"Emotion dominante: {most_common[0]} ({most_common[1]})", 
                       (w - 400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Appuyez sur 'q' pour quitter | 's' pour capture | 'r' pour reset stats", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def draw_emotion_bars(self, frame, predictions):
        """Dessiner les barres de probabilité pour toutes les émotions"""
        h, w = frame.shape[:2]
        bar_height = 20
        bar_spacing = 5
        start_y = h - 200
        max_bar_width = 200
        
        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 250, start_y - 30), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Titre
        cv2.putText(frame, "Probabilites:", 
                   (w - 240, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Barres pour chaque émotion
        for i, (emotion, prob) in enumerate(zip(emotions, predictions)):
            y = start_y + i * (bar_height + bar_spacing)
            bar_width = int(max_bar_width * prob)
            color = emotion_colors.get(emotion, (128, 128, 128))
            
            # Barre de fond
            cv2.rectangle(frame, (w - 240, y), (w - 40, y + bar_height), (50, 50, 50), -1)
            # Barre de probabilité
            cv2.rectangle(frame, (w - 240, y), (w - 240 + bar_width, y + bar_height), color, -1)
            
            # Texte
            text = f"{emotion[:3]}: {prob:.0%}"
            cv2.putText(frame, text, (w - 235, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_index=0, show_probabilities=True):
        """Lancer la détection en temps réel"""
        print(f"\n🎥 Tentative d'ouverture de la webcam (index {camera_index})...")
        
        # Essayer d'ouvrir la webcam avec différentes méthodes
        cap = None
        
        # Méthode 1: Index direct
        cap = cv2.VideoCapture(camera_index)
        
        # Si échec, essayer avec AVFoundation (macOS)
        if not cap.isOpened():
            print("⚠️ Échec avec l'index direct, essai avec AVFoundation...")
            cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        
        # Vérifier si la webcam est ouverte
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la webcam!")
            print("\n💡 Solutions possibles:")
            print("1. Vérifiez que la webcam n'est pas utilisée par une autre application")
            print("2. Autorisez l'accès à la caméra dans Préférences Système > Sécurité")
            print("3. Essayez avec un autre index de caméra (0, 1, 2...)")
            print("4. Redémarrez votre Mac")
            return
        
        print("✅ Webcam ouverte avec succès!")
        print("\n📹 Démarrage de la détection en temps réel...")
        print("=" * 60)
        
        # Configuration de la webcam
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Fenêtre
        window_name = "Reconnaissance d'Expressions Faciales - macOS"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Lire la frame
                ret, frame = cap.read()
                
                if not ret:
                    print("⚠️ Erreur de lecture de la webcam")
                    break
                
                self.frame_count += 1
                
                # Calculer FPS
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time) if (current_time - self.last_time) > 0 else 0
                self.last_time = current_time
                self.fps_history.append(fps)
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
                avg_fps = sum(self.fps_history) / len(self.fps_history)
                
                # Convertir en niveaux de gris
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Détecter les visages
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Traiter chaque visage
                emotion = "neutral"
                confidence = 0.0
                predictions = np.zeros(7)
                
                for (x, y, w, h) in faces:
                    # Extraire le visage
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # Détecter l'émotion
                    emotion, confidence, predictions = self.detect_emotion(face_roi)
                    
                    # Ajouter à l'historique
                    self.emotion_history.append(emotion)
                    if len(self.emotion_history) > 100:
                        self.emotion_history.pop(0)
                    
                    # Dessiner le rectangle
                    color = emotion_colors.get(emotion, (128, 128, 128))
                    thickness = 3 if confidence > 0.5 else 2
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    
                    # Étiquette
                    label = f"{emotion}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                                (x + label_size[0], y), color, -1)
                    cv2.putText(frame, label, (x, y - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Dessiner le panneau d'informations
                frame = self.draw_info_panel(frame, emotion, confidence, predictions, avg_fps)
                
                # Dessiner les barres de probabilité
                if show_probabilities and len(faces) > 0:
                    frame = self.draw_emotion_bars(frame, predictions)
                
                # Afficher la frame
                cv2.imshow(window_name, frame)
                
                # Gestion des touches
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n👋 Arrêt de la détection...")
                    break
                elif key == ord('s'):
                    # Capture d'écran
                    filename = f"capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Capture sauvegardée: {filename}")
                elif key == ord('r'):
                    # Reset des statistiques
                    self.emotion_history = []
                    self.frame_count = 0
                    print("🔄 Statistiques réinitialisées")
                elif key == ord('p'):
                    # Toggle probabilités
                    show_probabilities = not show_probabilities
                    print(f"📊 Probabilités: {'ON' if show_probabilities else 'OFF'}")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interruption par l'utilisateur")
        
        finally:
            # Libérer les ressources
            cap.release()
            cv2.destroyAllWindows()
            
            # Afficher les statistiques finales
            print("\n" + "=" * 60)
            print("📊 STATISTIQUES FINALES")
            print("=" * 60)
            print(f"Frames traitées: {self.frame_count}")
            print(f"FPS moyen: {sum(self.fps_history) / len(self.fps_history):.1f}" if self.fps_history else "N/A")
            
            if self.emotion_history:
                from collections import Counter
                emotion_counts = Counter(self.emotion_history)
                print("\nDistribution des émotions:")
                for emotion, count in emotion_counts.most_common():
                    percentage = (count / len(self.emotion_history)) * 100
                    print(f"  {emotion_emojis[emotion]} {emotion}: {count} ({percentage:.1f}%)")
            
            print("\n✅ Application fermée avec succès!")

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🎭 RECONNAISSANCE D'EXPRESSIONS FACIALES - macOS")
    print("=" * 60)
    print()
    
    # Vérifier le modèle
    if not os.path.exists("best_model.h5"):
        print("❌ Erreur: Le fichier 'best_model.h5' est introuvable!")
        print("Veuillez d'abord entraîner le modèle avec:")
        print("  python3 run_facial_recognition_simple.py")
        return
    
    # Créer le détecteur
    detector = WebcamEmotionDetector()
    
    # Demander l'index de la caméra
    print("\n📷 Configuration de la caméra:")
    print("  0 = Caméra intégrée (par défaut)")
    print("  1 = Caméra externe (si connectée)")
    
    try:
        camera_input = input("\nIndex de la caméra (appuyez sur Entrée pour 0): ").strip()
        camera_index = int(camera_input) if camera_input else 0
    except ValueError:
        camera_index = 0
    
    # Lancer la détection
    detector.run(camera_index=camera_index, show_probabilities=True)

if __name__ == "__main__":
    main()

# Made with Bob
