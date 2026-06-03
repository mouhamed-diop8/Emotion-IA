# 🎥 Guide Webcam pour macOS

## 📋 Vue d'ensemble

Ce guide explique comment utiliser la webcam sur macOS pour la reconnaissance d'expressions faciales en temps réel.

## 🚀 Solution Recommandée : Application Standalone

### Pourquoi une application standalone ?

Sur macOS, les navigateurs web (Safari, Chrome, Firefox) ont des restrictions de sécurité strictes qui empêchent Streamlit d'accéder directement à la webcam. La solution est d'utiliser une application Python standalone qui utilise OpenCV directement.

### ✅ Avantages

- ✨ Accès direct à la webcam via OpenCV
- 🚀 Performances optimales (pas de latence navigateur)
- 📊 Interface riche avec statistiques en temps réel
- 💾 Capture d'écran intégrée
- 🎯 Compatible avec AVFoundation (framework macOS)

## 🛠️ Installation

### 1. Vérifier les dépendances

```bash
python3 -c "import cv2, tensorflow; print('✅ Tout est installé!')"
```

Si erreur, installer :

```bash
python3 -m pip install opencv-python tensorflow --user
```

### 2. Vérifier le modèle

```bash
ls -lh best_model.h5
```

Si absent, entraîner le modèle :

```bash
python3 run_facial_recognition_simple.py
```

## 🎬 Utilisation

### Lancer l'application

```bash
python3 webcam_macos_app.py
```

### Interface Interactive

L'application vous demandera :
```
Index de la caméra (appuyez sur Entrée pour 0):
```

- **0** = Caméra intégrée (FaceTime HD)
- **1** = Caméra externe USB (si connectée)

### Contrôles Clavier

| Touche | Action |
|--------|--------|
| `q` | Quitter l'application |
| `s` | Prendre une capture d'écran |
| `r` | Réinitialiser les statistiques |
| `p` | Activer/désactiver les probabilités |

## 📊 Fonctionnalités

### Panneau Principal

- 🎭 **Émotion détectée** avec emoji
- 📈 **Barre de confiance** visuelle
- ⚡ **FPS en temps réel**
- 📊 **Statistiques cumulées**

### Panneau Latéral

- 📊 **Barres de probabilité** pour les 7 émotions
- 🎨 **Code couleur** par émotion :
  - 😠 Rouge = Colère
  - 🤢 Violet = Dégoût
  - 😨 Bleu = Peur
  - 😊 Vert = Joie
  - 😢 Violet foncé = Tristesse
  - 😲 Orange = Surprise
  - 😐 Gris = Neutre

### Détection de Visage

- ✅ Rectangle coloré autour du visage
- 🏷️ Étiquette avec émotion et confiance
- 🎯 Épaisseur variable selon la confiance

## 🔧 Résolution de Problèmes

### ❌ "Impossible d'ouvrir la webcam"

**Solutions :**

1. **Vérifier les autorisations macOS**
   ```
   Préférences Système > Sécurité et confidentialité > Confidentialité > Caméra
   ```
   Assurez-vous que Terminal/Python a accès à la caméra.

2. **Fermer les autres applications**
   - Photo Booth
   - FaceTime
   - Zoom, Teams, etc.

3. **Essayer un autre index de caméra**
   ```bash
   python3 webcam_macos_app.py
   # Puis entrer 1 au lieu de 0
   ```

4. **Redémarrer le Mac**
   Parfois nécessaire après l'installation d'OpenCV.

### ⚠️ "Erreur de lecture de la webcam"

**Causes possibles :**
- Webcam déconnectée (USB)
- Pilotes obsolètes
- Conflit avec une autre application

**Solution :**
```bash
# Vérifier les caméras disponibles
system_profiler SPCameraDataType
```

### 🐌 FPS faible (< 15)

**Optimisations :**

1. **Réduire la résolution**
   Modifier dans `webcam_macos_app.py` :
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Au lieu de 1280
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Au lieu de 720
   ```

2. **Désactiver les probabilités**
   Appuyer sur `p` pendant l'exécution

3. **Fermer les applications gourmandes**

### 🔴 Erreur "AVFoundation"

Si vous voyez des erreurs AVFoundation, c'est normal sur macOS. L'application essaie automatiquement différentes méthodes d'accès à la caméra.

## 🆚 Comparaison des Solutions

### Application Standalone (Recommandé ✅)

**Avantages :**
- ✅ Fonctionne directement sur macOS
- ✅ Performances optimales
- ✅ Interface riche
- ✅ Pas de configuration navigateur

**Inconvénients :**
- ❌ Nécessite un terminal
- ❌ Pas d'interface web

### Streamlit (Alternative)

**Avantages :**
- ✅ Interface web élégante
- ✅ Facile à partager

**Inconvénients :**
- ❌ Ne fonctionne pas avec la webcam sur macOS
- ❌ Nécessite de charger des images

**Utilisation :**
```bash
streamlit run streamlit_realtime_app.py
```
Puis charger des images via l'interface.

## 💡 Conseils d'Utilisation

### Pour de meilleurs résultats :

1. **Éclairage**
   - Lumière naturelle de face
   - Éviter le contre-jour
   - Pas d'ombres sur le visage

2. **Position**
   - Visage de face
   - Distance : 50-100 cm de la caméra
   - Centré dans le cadre

3. **Expression**
   - Expression claire et marquée
   - Maintenir 2-3 secondes
   - Éviter les mouvements brusques

4. **Environnement**
   - Fond neutre si possible
   - Pas de visages multiples
   - Webcam stable

## 📸 Captures d'Écran

Les captures sont sauvegardées automatiquement :
```
capture_1780240274.jpg
capture_1780240275.jpg
...
```

Format : `capture_<timestamp>.jpg`

## 📈 Statistiques

À la fin de chaque session, l'application affiche :

- 📊 Nombre total de frames traitées
- ⚡ FPS moyen
- 🎭 Distribution des émotions détectées
- 📈 Pourcentages par émotion

Exemple :
```
📊 STATISTIQUES FINALES
Frames traitées: 1523
FPS moyen: 28.4

Distribution des émotions:
  😊 happy: 856 (56.2%)
  😐 neutral: 432 (28.4%)
  😲 surprise: 235 (15.4%)
```

## 🔄 Mise à Jour

Pour mettre à jour l'application :

```bash
# Sauvegarder vos modifications
cp webcam_macos_app.py webcam_macos_app.py.backup

# Télécharger la nouvelle version
# (si disponible)

# Restaurer si nécessaire
cp webcam_macos_app.py.backup webcam_macos_app.py
```

## 🆘 Support

### Problèmes courants

1. **Modèle non trouvé**
   ```bash
   python3 run_facial_recognition_simple.py
   ```

2. **OpenCV non installé**
   ```bash
   python3 -m pip install opencv-python --user
   ```

3. **TensorFlow non installé**
   ```bash
   python3 -m pip install tensorflow --user
   ```

### Logs de débogage

Pour plus d'informations sur les erreurs :
```bash
python3 webcam_macos_app.py 2>&1 | tee webcam_debug.log
```

## 🎯 Performances Attendues

### Configuration Minimale
- **Mac** : 2015 ou plus récent
- **RAM** : 4 GB
- **FPS** : 15-20

### Configuration Recommandée
- **Mac** : 2018 ou plus récent
- **RAM** : 8 GB+
- **FPS** : 25-30

### Configuration Optimale
- **Mac** : M1/M2 ou Intel i7+
- **RAM** : 16 GB+
- **FPS** : 30+

## 📚 Ressources

- [Documentation OpenCV](https://docs.opencv.org/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [macOS Camera Access](https://support.apple.com/guide/mac-help/control-access-to-your-camera-mchlf6d108da/mac)

## ✅ Checklist de Démarrage

- [ ] Python 3.7+ installé
- [ ] OpenCV installé
- [ ] TensorFlow installé
- [ ] Modèle `best_model.h5` présent
- [ ] Autorisations caméra accordées
- [ ] Webcam fonctionnelle
- [ ] Aucune autre app n'utilise la caméra

## 🎉 Prêt à Commencer !

```bash
python3 webcam_macos_app.py
```

Profitez de la reconnaissance d'expressions faciales en temps réel ! 🎭✨