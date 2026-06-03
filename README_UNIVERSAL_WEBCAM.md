# 🎭 Application Universelle de Reconnaissance d'Expressions Faciales

## 📋 Description

Application de reconnaissance d'expressions faciales en temps réel compatible avec **macOS**, **Windows** et **Linux**.

## ✨ Fonctionnalités

- ✅ **Multi-plateforme**: Fonctionne sur macOS, Windows et Linux
- 🎥 **Détection automatique de caméra**: Trouve automatiquement votre webcam
- 🎨 **Interface graphique intuitive**: Affichage en temps réel avec statistiques
- 📊 **7 émotions détectées**: Colère, Dégoût, Peur, Joie, Neutre, Tristesse, Surprise
- 📸 **Captures d'écran**: Sauvegardez vos moments
- ⏸️ **Pause/Reprise**: Contrôlez la détection
- 📈 **Statistiques FPS**: Performances en temps réel

## 🚀 Installation

### Prérequis

```bash
# Python 3.7 ou supérieur
python3 --version

# Installer les dépendances
pip3 install opencv-python tensorflow numpy
```

### Vérification du modèle

Assurez-vous que le fichier `best_model.h5` est présent dans le répertoire du projet.

## 💻 Utilisation

### Lancement de l'application

```bash
python3 webcam_universal_app.py
```

### Commandes clavier

| Touche | Action |
|--------|--------|
| **Q** ou **ESC** | Quitter l'application |
| **S** | Prendre une capture d'écran |
| **ESPACE** | Mettre en pause / Reprendre |

## 🖥️ Compatibilité par système

### macOS
- ✅ Caméra FaceTime intégrée
- ✅ Caméras USB externes
- ✅ Résolution: 640x480
- 📝 Note: Autorisez l'accès à la caméra dans les Préférences Système

### Windows
- ✅ Webcam intégrée
- ✅ Caméras USB externes
- ✅ Résolution: 640x480, 30 FPS
- 📝 Note: Autorisez l'accès à la caméra dans les Paramètres Windows

### Linux
- ✅ Caméras V4L2
- ✅ Caméras USB
- ✅ Format: MJPG
- 📝 Note: Vérifiez les permissions avec `ls -l /dev/video*`

## 🎨 Interface utilisateur

### Panneau supérieur
- Titre de l'application
- Système d'exploitation détecté
- FPS en temps réel

### Zone centrale
- Flux vidéo en direct
- Rectangles colorés autour des visages détectés
- Étiquettes d'émotions avec pourcentages de confiance

### Panneau inférieur
- Émotion principale détectée
- Barre de progression de confiance
- Instructions des commandes

## 🎯 Émotions détectées

| Émotion | Couleur | Description |
|---------|---------|-------------|
| 😠 Colère | Rouge | Visage en colère |
| 🤢 Dégoût | Vert | Expression de dégoût |
| 😨 Peur | Violet | Visage effrayé |
| 😊 Joie | Jaune | Sourire, bonheur |
| 😐 Neutre | Blanc | Expression neutre |
| 😢 Tristesse | Bleu | Visage triste |
| 😲 Surprise | Orange | Expression surprise |

## 📸 Captures d'écran

Les captures sont automatiquement sauvegardées avec un timestamp:
```
snapshot_1234567890.jpg
```

## 🔧 Dépannage

### Problème: Caméra non détectée

**macOS:**
```bash
# Vérifier les autorisations
# Préférences Système > Sécurité et confidentialité > Caméra
```

**Windows:**
```bash
# Vérifier les autorisations
# Paramètres > Confidentialité > Caméra
```

**Linux:**
```bash
# Vérifier les périphériques vidéo
ls -l /dev/video*

# Ajouter l'utilisateur au groupe video
sudo usermod -a -G video $USER
```

### Problème: Erreur de chargement du modèle

```bash
# Vérifier que best_model.h5 existe
ls -l best_model.h5

# Si absent, entraîner le modèle d'abord
python3 run_facial_recognition_simple.py
```

### Problème: Performance lente

- Fermez les autres applications utilisant la caméra
- Réduisez la résolution dans le code si nécessaire
- Vérifiez que TensorFlow utilise le GPU si disponible

### Problème: OpenCV ne s'installe pas

```bash
# Essayer avec pip
pip3 install opencv-python --upgrade

# Ou avec conda
conda install -c conda-forge opencv

# Sur Linux, installer les dépendances système
sudo apt-get install python3-opencv  # Ubuntu/Debian
sudo yum install opencv-python        # Fedora/CentOS
```

## 📊 Statistiques affichées

- **FPS**: Images par seconde traitées
- **Frames traitées**: Nombre total d'images analysées
- **Captures**: Nombre de screenshots pris
- **Confiance**: Pourcentage de certitude de la prédiction

## 🔒 Confidentialité

- ✅ Traitement local uniquement
- ✅ Aucune donnée envoyée sur Internet
- ✅ Aucun stockage automatique (sauf captures manuelles)
- ✅ Contrôle total de l'utilisateur

## 🎓 Utilisation avancée

### Modifier la résolution

Éditez `webcam_universal_app.py`:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Largeur
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Hauteur
```

### Changer l'index de caméra

Si vous avez plusieurs caméras:
```python
camera_index = 1  # Essayez 0, 1, 2, etc.
cap = cv2.VideoCapture(camera_index)
```

### Ajuster la sensibilité de détection

```python
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,      # Réduire pour plus de détections
    minNeighbors=5,       # Augmenter pour moins de faux positifs
    minSize=(30, 30)      # Taille minimale du visage
)
```

## 🆘 Support

### Logs de débogage

L'application affiche des informations détaillées:
- Système d'exploitation détecté
- Version Python et OpenCV
- État du chargement du modèle
- Détection de caméra
- Statistiques en temps réel

### Problèmes courants

1. **"Aucune caméra disponible"**
   - Vérifiez que la caméra est connectée
   - Vérifiez les autorisations système
   - Fermez les autres applications utilisant la caméra

2. **"Erreur lors du chargement du modèle"**
   - Vérifiez que `best_model.h5` existe
   - Entraînez le modèle si nécessaire

3. **Performance lente**
   - Fermez les applications en arrière-plan
   - Réduisez la résolution
   - Vérifiez l'utilisation CPU/GPU

## 📝 Notes techniques

### Architecture
- **Détection de visages**: Haar Cascade Classifier
- **Reconnaissance d'émotions**: CNN (Convolutional Neural Network)
- **Résolution d'entrée**: 48x48 pixels (niveaux de gris)
- **Framework**: TensorFlow/Keras

### Performance
- **FPS typique**: 15-30 FPS (selon le matériel)
- **Latence**: < 100ms par frame
- **Précision**: ~56% (selon le modèle entraîné)

## 🌟 Fonctionnalités futures

- [ ] Support de plusieurs visages simultanés
- [ ] Enregistrement vidéo
- [ ] Export des statistiques
- [ ] Mode sombre/clair
- [ ] Personnalisation des couleurs
- [ ] Support de modèles personnalisés

## 📄 Licence

Ce projet est fourni à des fins éducatives.

## 👥 Contribution

Pour contribuer:
1. Testez sur votre système d'exploitation
2. Signalez les bugs
3. Proposez des améliorations
4. Partagez vos résultats

---

**Développé avec ❤️ pour la reconnaissance d'émotions multi-plateforme**