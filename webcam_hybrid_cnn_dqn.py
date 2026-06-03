#!/usr/bin/env python3
"""
Application Hybride: CNN + DQN pour Reconnaissance Faciale Optimisée
Combine Deep Learning (CNN) et Apprentissage par Renforcement (DQN)
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from PIL import Image, ImageEnhance
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

# Couleurs
COLORS = {
    'angry': (0, 0, 255),
    'disgust': (0, 255, 0),
    'fear': (128, 0, 128),
    'happy': (0, 255, 255),
    'neutral': (255, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (255, 165, 0)
}

class SimpleDQNAgent:
    """Agent DQN simplifié pour ajuster contraste et exposition"""
    
    def __init__(self, state_size=3, action_size=9):
        self.state_size = state_size  # [contrast, exposure, confidence]
        self.action_size = action_size  # 3x3 = 9 actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_confidence = 0.0
        
    def _build_model(self):
        """Construit le réseau de neurones DQN"""
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Mémorise une expérience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choisit une action (exploration vs exploitation)"""
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Entraîne le modèle sur un batch d'expériences"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state.reshape(1, -1), verbose=0)[0]
                )
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class HybridFaceRecognition:
    """Système hybride CNN + DQN"""
    
    def __init__(self):
        self.cnn_model = None
        self.dqn_agent = SimpleDQNAgent()
        self.face_cascade = None
        
        # Paramètres vidéo ajustables par DQN
        self.contrast = 1.0
        self.exposure = 1.0
        self.contrast_min = 0.5
        self.contrast_max = 2.0
        self.exposure_min = 0.5
        self.exposure_max = 2.0
        self.step_size = 0.1
        
        # Statistiques
        self.frame_count = 0
        self.dqn_adjustments = 0
        self.confidence_history = deque(maxlen=100)
        
    def load_models(self):
        """Charge le modèle CNN"""
        print("🔄 Chargement du modèle CNN...")
        try:
            self.cnn_model = load_model('best_model.h5', compile=False)
            self.cnn_model.compile(optimizer='adam', loss='categorical_crossentropy')
            print("✅ Modèle CNN chargé!")
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("❌ Erreur: Détecteur de visages")
            return False
        
        print("✅ Agent DQN initialisé!")
        return True
    
    def adjust_image(self, image_pil):
        """Applique les ajustements de contraste et exposition"""
        enhancer = ImageEnhance.Contrast(image_pil)
        img_adjusted = enhancer.enhance(self.contrast)
        
        enhancer = ImageEnhance.Brightness(img_adjusted)
        img_adjusted = enhancer.enhance(self.exposure)
        
        return img_adjusted
    
    def preprocess_face(self, face_roi):
        """Prétraite le visage pour le CNN avec ajustements DQN"""
        # Convertir en PIL pour ajustements
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        
        # Appliquer ajustements DQN
        face_adjusted = self.adjust_image(face_pil)
        
        # Convertir pour CNN
        face_gray = face_adjusted.convert('L')
        face_resized = face_gray.resize((48, 48))
        face_array = np.array(face_resized).astype('float32') / 255.0
        face_array = np.expand_dims(face_array, axis=0)
        face_array = np.expand_dims(face_array, axis=-1)
        
        return face_array
    
    def apply_dqn_action(self, action):
        """Applique l'action DQN pour ajuster les paramètres"""
        contrast_action = action // 3
        exposure_action = action % 3
        
        old_contrast = self.contrast
        old_exposure = self.exposure
        
        # Ajuster contraste
        if contrast_action == 0:
            self.contrast = max(self.contrast_min, self.contrast - self.step_size)
        elif contrast_action == 2:
            self.contrast = min(self.contrast_max, self.contrast + self.step_size)
        
        # Ajuster exposition
        if exposure_action == 0:
            self.exposure = max(self.exposure_min, self.exposure - self.step_size)
        elif exposure_action == 2:
            self.exposure = min(self.exposure_max, self.exposure + self.step_size)
        
        if old_contrast != self.contrast or old_exposure != self.exposure:
            self.dqn_adjustments += 1
    
    def get_camera_index(self):
        """Détecte la caméra"""
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    return i
        return None
    
    def draw_interface(self, frame, emotion, confidence, fps):
        """Dessine l'interface avec infos CNN + DQN"""
        height, width = frame.shape[:2]
        
        # Panneau supérieur
        cv2.rectangle(frame, (0, 0), (width, 90), (0, 0, 0), -1)
        cv2.putText(frame, "SYSTEME HYBRIDE: CNN + DQN", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f} | Ajustements DQN: {self.dqn_adjustments}", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Contraste: {self.contrast:.2f} | Exposition: {self.exposure:.2f}", 
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Panneau inférieur
        if emotion:
            emotion_fr = EMOTIONS_FR[EMOTIONS.index(emotion)]
            color = COLORS[emotion]
            
            cv2.rectangle(frame, (0, height - 100), (width, height), (0, 0, 0), -1)
            cv2.putText(frame, f"CNN: {emotion_fr}", 
                        (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Barre de confiance
            bar_width = int((width - 20) * confidence)
            cv2.rectangle(frame, (10, height - 35), (10 + bar_width, height - 15), color, -1)
            cv2.putText(frame, f"{confidence*100:.0f}%", 
                        (width - 80, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Moyenne de confiance
            if len(self.confidence_history) > 0:
                avg_conf = np.mean(self.confidence_history)
                cv2.putText(frame, f"Moy: {avg_conf*100:.0f}%", 
                            (width - 180, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.putText(frame, "Q:Quitter | R:Reset DQN", 
                    (10, height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def run(self):
        """Boucle principale"""
        print("=" * 70)
        print("🚀 SYSTÈME HYBRIDE: CNN + DQN")
        print("=" * 70)
        
        if not self.load_models():
            return
        
        camera_index = self.get_camera_index()
        if camera_index is None:
            print("❌ Aucune caméra détectée")
            return
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Impossible d'ouvrir la caméra")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("\n✅ Système initialisé!")
        print("📋 Fonctionnement:")
        print("   - CNN: Reconnaissance d'émotions")
        print("   - DQN: Optimisation contraste/exposition")
        print("   - Apprentissage en temps réel")
        print("=" * 70)
        
        fps_counter = deque(maxlen=30)
        last_time = time.time()
        current_emotion = None
        current_confidence = 0
        last_faces = []
        previous_confidence = 0
        
        print("🎬 Démarrage...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculer FPS
            current_time = time.time()
            fps_counter.append(1.0 / (current_time - last_time + 0.001))
            last_time = current_time
            fps = np.mean(fps_counter)
            
            # Détection de visages (tous les 3 frames)
            if self.frame_count % 3 == 0:
                small_frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20)
                )
                
                last_faces = [(int(x*2), int(y*2), int(w*2), int(h*2)) 
                             for (x, y, w, h) in faces]
                
                # Traiter le premier visage
                if len(last_faces) > 0:
                    x, y, w, h = last_faces[0]
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Prédiction CNN avec ajustements DQN
                    processed_face = self.preprocess_face(face_roi)
                    predictions = self.cnn_model.predict(processed_face, verbose=0)
                    emotion_idx = np.argmax(predictions[0])
                    current_confidence = predictions[0][emotion_idx]
                    current_emotion = EMOTIONS[emotion_idx]
                    
                    self.confidence_history.append(current_confidence)
                    
                    # DQN: Ajuster paramètres tous les 10 frames
                    if self.frame_count % 10 == 0 and self.frame_count > 0:
                        state = np.array([self.contrast, self.exposure, previous_confidence])
                        action = self.dqn_agent.act(state)
                        self.apply_dqn_action(action)
                        
                        # Calculer récompense (amélioration de confiance)
                        reward = current_confidence - previous_confidence
                        next_state = np.array([self.contrast, self.exposure, current_confidence])
                        
                        self.dqn_agent.remember(state, action, reward, next_state, False)
                        self.dqn_agent.replay(batch_size=16)
                    
                    previous_confidence = current_confidence
            
            # Dessiner rectangles
            for (x, y, w, h) in last_faces:
                color = COLORS.get(current_emotion, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                if current_emotion:
                    emotion_fr = EMOTIONS_FR[EMOTIONS.index(current_emotion)]
                    label = f"{emotion_fr}: {current_confidence*100:.0f}%"
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Interface
            self.draw_interface(frame, current_emotion, current_confidence, fps)
            
            # Affichage
            cv2.imshow('Systeme Hybride CNN + DQN', frame)
            
            # Touches
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                self.contrast = 1.0
                self.exposure = 1.0
                self.dqn_adjustments = 0
                print("🔄 Paramètres DQN réinitialisés")
            
            self.frame_count += 1
        
        # Nettoyage
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("✅ Application terminée")
        print(f"📊 Statistiques:")
        print(f"   - Frames traitées: {self.frame_count}")
        print(f"   - Ajustements DQN: {self.dqn_adjustments}")
        print(f"   - FPS moyen: {fps:.1f}")
        if len(self.confidence_history) > 0:
            print(f"   - Confiance moyenne: {np.mean(self.confidence_history)*100:.1f}%")
        print(f"   - Epsilon DQN: {self.dqn_agent.epsilon:.3f}")
        print("=" * 70)

if __name__ == "__main__":
    try:
        app = HybridFaceRecognition()
        app.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interruption")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()

# Made with Bob - Hybrid CNN + DQN System