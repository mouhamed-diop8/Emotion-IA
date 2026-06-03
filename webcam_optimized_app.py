#!/usr/bin/env python3
"""
Application Optimisée de Reconnaissance d'Expressions Faciales
Optimisations: Threading, cache, réduction de la résolution, prédictions asynchrones
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import platform
import sys
import time
from collections import deque
import threading
from queue import Queue

# Configuration
SYSTEM = platform.system()
print(f"🖥️  Système détecté: {SYSTEM}")

# Émotions
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EMOTIONS_FR = ['Colère', 'Dégoût', 'Peur', 'Joie', 'Neutre', 'Tristesse', 'Surprise']

# Couleurs pour chaque émotion
COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 255, 0),
    'fear': (128, 0, 128),
    'happy': (0, 255, 255),
    'neutral': (255, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (255, 165, 0)
}

# Paramètres d'optimisation
SKIP_FRAMES = 2  # Traiter 1 frame sur 3 pour la détection
DETECTION_SCALE = 0.5  # Réduire la résolution pour la détection
SMOOTH_PREDICTIONS = 5  # Lissage des prédictions

class OptimizedFaceRecognition:
    def __init__(self):
        self.model = None
        self.face_cascade = None
        self.prediction_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.running = False
        self.prediction_history = deque(maxlen=SMOOTH_PREDICTIONS)
        
    def load_model(self):
        """Charge le modèle avec optimisations"""
        print("🔄 Chargement du modèle...")
        try:
            self.model = load_model('best_model.h5', compile=False)
            # Compiler avec optimisations
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
            print("✅ Modèle chargé et optimisé!")
            return True
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def load_face_detector(self):
        """Charge le détecteur de visages"""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        return not self.face_cascade.empty()
    
    def preprocess_face(self, face_img):
        """Prétraite l'image du visage (optimisé)"""
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (48, 48), interpolation=cv2.INTER_LINEAR)
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        return face_img
    
    def prediction_worker(self):
        """Thread worker pour les prédictions asynchrones"""
        while self.running:
            try:
                if not self.prediction_queue.empty():
                    face_data = self.prediction_queue.get(timeout=0.01)
                    if face_data is not None:
                        processed_face = self.preprocess_face(face_data)
                        predictions = self.model.predict(processed_face, verbose=0)
                        
                        # Ajouter à l'historique pour lissage
                        self.prediction_history.append(predictions[0])
                        
                        # Moyenne des prédictions pour plus de stabilité
                        if len(self.prediction_history) > 0:
                            smoothed = np.mean(self.prediction_history, axis=0)
                            emotion_idx = np.argmax(smoothed)
                            confidence = smoothed[emotion_idx]
                            
                            if not self.result_queue.full():
                                self.result_queue.put((EMOTIONS[emotion_idx], confidence))
            except:
                pass
    
    def get_camera_index(self):
        """Détecte la caméra disponible"""
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return i
        return None
    
    def configure_camera(self, cap):
        """Configure la caméra pour performance optimale"""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Réduire le buffer
        
        if SYSTEM == "Darwin":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    def draw_interface(self, frame, emotion, confidence, fps, faces_count):
        """Dessine l'interface (optimisé)"""
        height, width = frame.shape[:2]
        
        # Panneau supérieur simplifié
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(frame, "Reconnaissance Faciale OPTIMISEE", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Visages: {faces_count}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Panneau inférieur avec émotion
        if emotion:
            emotion_fr = EMOTIONS_FR[EMOTIONS.index(emotion)]
            color = COLORS[emotion]
            
            cv2.rectangle(frame, (0, height - 80), (width, height), (0, 0, 0), -1)
            cv2.putText(frame, f"{emotion_fr}", 
                        (10, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Barre de confiance
            bar_width = int((width - 20) * confidence)
            cv2.rectangle(frame, (10, height - 25), (10 + bar_width, height - 10), color, -1)
            cv2.putText(frame, f"{confidence*100:.0f}%", 
                        (width - 80, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Instructions simplifiées
        cv2.putText(frame, "Q:Quitter | S:Capture", 
                    (10, height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def run(self):
        """Boucle principale optimisée"""
        print("=" * 70)
        print("🚀 RECONNAISSANCE FACIALE - VERSION OPTIMISÉE")
        print("=" * 70)
        
        if not self.load_model():
            return
        
        if not self.load_face_detector():
            print("❌ Erreur: Détecteur de visages")
            return
        
        camera_index = self.get_camera_index()
        if camera_index is None:
            print("❌ Aucune caméra détectée")
            return
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra")
            return
        
        self.configure_camera(cap)
        
        print("\n✅ Initialisation terminée!")
        print("📋 Optimisations actives:")
        print(f"   - Skip frames: 1/{SKIP_FRAMES+1}")
        print(f"   - Résolution détection: {DETECTION_SCALE*100:.0f}%")
        print(f"   - Lissage prédictions: {SMOOTH_PREDICTIONS} frames")
        print(f"   - Prédictions asynchrones: OUI")
        print("=" * 70)
        
        # Démarrer le thread de prédiction
        self.running = True
        prediction_thread = threading.Thread(target=self.prediction_worker, daemon=True)
        prediction_thread.start()
        
        # Variables
        frame_count = 0
        fps_counter = deque(maxlen=30)
        last_time = time.time()
        current_emotion = None
        current_confidence = 0
        snapshot_count = 0
        last_faces = []
        
        print("🎬 Démarrage...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculer FPS
            current_time = time.time()
            fps_counter.append(1.0 / (current_time - last_time + 0.001))
            last_time = current_time
            fps = np.mean(fps_counter)
            
            # Détection de visages (seulement tous les N frames)
            if frame_count % (SKIP_FRAMES + 1) == 0:
                # Réduire la résolution pour la détection
                small_frame = cv2.resize(frame, None, fx=DETECTION_SCALE, fy=DETECTION_SCALE)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Détecter les visages
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=4,
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Rescaler les coordonnées
                scale_factor = 1.0 / DETECTION_SCALE
                last_faces = [(int(x*scale_factor), int(y*scale_factor), 
                              int(w*scale_factor), int(h*scale_factor)) 
                             for (x, y, w, h) in faces]
                
                # Envoyer le premier visage pour prédiction (asynchrone)
                if len(last_faces) > 0 and self.prediction_queue.empty():
                    x, y, w, h = last_faces[0]
                    face_roi = frame[y:y+h, x:x+w]
                    try:
                        self.prediction_queue.put_nowait(face_roi)
                    except:
                        pass
            
            # Récupérer les résultats de prédiction
            if not self.result_queue.empty():
                try:
                    current_emotion, current_confidence = self.result_queue.get_nowait()
                except:
                    pass
            
            # Dessiner les rectangles sur tous les visages
            for (x, y, w, h) in last_faces:
                color = COLORS.get(current_emotion, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                if current_emotion:
                    emotion_fr = EMOTIONS_FR[EMOTIONS.index(current_emotion)]
                    label = f"{emotion_fr}: {current_confidence*100:.0f}%"
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Dessiner l'interface
            self.draw_interface(frame, current_emotion, current_confidence, fps, len(last_faces))
            
            # Afficher
            cv2.imshow('Reconnaissance Faciale - Optimisée', frame)
            
            # Gestion des touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                snapshot_count += 1
                filename = f"snapshot_opt_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Capture: {filename}")
            
            frame_count += 1
        
        # Nettoyage
        self.running = False
        prediction_thread.join(timeout=1)
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("✅ Application terminée")
        print(f"📊 FPS moyen: {fps:.1f}")
        print(f"📸 Captures: {snapshot_count}")
        print("=" * 70)

if __name__ == "__main__":
    try:
        app = OptimizedFaceRecognition()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interruption")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()

# Made with Bob - Optimized Version