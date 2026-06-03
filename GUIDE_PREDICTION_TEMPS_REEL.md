# 🎭 Guide - Prédiction d'Expressions Faciales en Temps Réel

## 📋 Table des Matières
1. [Vue d'ensemble](#vue-densemble)
2. [Installation](#installation)
3. [Utilisation](#utilisation)
4. [Fonctionnalités](#fonctionnalités)
5. [Optimisations](#optimisations)
6. [Dépannage](#dépannage)
7. [Conseils](#conseils)

---

## 🎯 Vue d'ensemble

Cette application utilise votre **webcam** pour détecter et prédire les expressions faciales **en temps réel**.

### Caractéristiques principales :
- ✅ **Détection en temps réel** avec la webcam
- ✅ **7 émotions** détectées : Colère, Dégoût, Peur, Joie, Neutre, Tristesse, Surprise
- ✅ **Interface visuelle** avec graphiques et statistiques
- ✅ **Performances optimisées** (30+ FPS)
- ✅ **Validation des visages** avec détection des yeux
- ✅ **Lissage des prédictions** pour plus de stabilité
- ✅ **Compatible macOS** avec optimisations spécifiques

---

## 🚀 Installation

### Prérequis
```bash
# Python 3.8 ou supérieur
python3 --version

# Installer les dépendances
pip3 install opencv-python tensorflow numpy
```

### Vérifier l'installation
```bash
# Tester OpenCV
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"

# Tester TensorFlow
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

### Modèle requis
L'application cherche automatiquement un modèle dans cet ordre :
1. `best_model_optimized.h5` (recommandé)
2. `best_model.h5`
3. `my_cnn_model.h5`

---

## 🎬 Utilisation

### Démarrage rapide
```bash
# Lancer l'application
python3 webcam_realtime_prediction.py
```

### Commandes pendant l'exécution
| Touche | Action |
|--------|--------|
| **Q** ou **ESC** | Quitter l'application |
| **S** | Prendre une capture d'écran |
| **R** | Réinitialiser l'historique des prédictions |

### Exemple de session
```
🖥️  Système: Darwin
🔄 Chargement du modèle...
✅ Modèle chargé: best_model.h5
✅ Détecteurs chargés
🔍 Recherche de la caméra...
✅ Caméra trouvée: index 0

✅ Initialisation terminée!

📋 Optimisations actives:
   • Skip frames: 1/3
   • Résolution détection: 50%
   • Lissage prédictions: 5 frames
   • Prédictions asynchrones: OUI
   • Validation visages: OUI

🎬 Démarrage de la capture vidéo...
```

---

## 🎨 Fonctionnalités

### 1. Interface visuelle

#### En-tête (haut de l'écran)
- Titre de l'application
- **FPS** en temps réel
- Nombre de **visages détectés**

#### Instructions (gauche)
- Raccourcis clavier
- Commandes disponibles

#### Détection de visage
- **Rectangle coloré** autour du visage
- **Label** avec émotion et confiance
- Couleur selon l'émotion détectée

#### Panneau d'émotion (bas de l'écran)
- **Emoji** de l'émotion
- **Nom** de l'émotion en français
- **Barre de confiance** (0-100%)
- **Mini-graphique** des probabilités

### 2. Émotions détectées

| Émotion | Emoji | Couleur | Description |
|---------|-------|---------|-------------|
| **Colère** | 😠 | Rouge | Visage tendu, sourcils froncés |
| **Dégoût** | 🤢 | Vert | Nez plissé, lèvre supérieure relevée |
| **Peur** | 😨 | Violet | Yeux écarquillés, bouche ouverte |
| **Joie** | 😊 | Jaune | Sourire, yeux plissés |
| **Neutre** | 😐 | Blanc | Expression neutre, détendue |
| **Tristesse** | 😢 | Bleu | Coins de la bouche baissés |
| **Surprise** | 😲 | Orange | Yeux et bouche grands ouverts |

### 3. Captures d'écran

Les captures sont sauvegardées avec un nom unique :
```
capture_realtime_1780240274.jpg
capture_realtime_1780240275.jpg
...
```

---

## ⚡ Optimisations

### 1. Traitement asynchrone
- **Thread séparé** pour les prédictions
- Pas de blocage de l'affichage vidéo
- Performances fluides

### 2. Skip frames
- Traite **1 frame sur 3** pour la détection
- Réduit la charge CPU
- Maintient un FPS élevé

### 3. Réduction de résolution
- Détection sur image **50% plus petite**
- Affichage en résolution complète
- Gain de performance significatif

### 4. Lissage des prédictions
- Moyenne sur **5 frames**
- Réduit les fluctuations
- Prédictions plus stables

### 5. Validation des visages
- Détection des **yeux** pour confirmer
- Réduit les faux positifs
- Meilleure précision

### 6. Optimisations macOS
- Configuration spécifique de la caméra
- Codec MJPG pour meilleures performances
- Buffer réduit pour latence minimale

---

## 🔧 Dépannage

### Problème : Caméra non détectée

**Solution 1 : Vérifier les permissions macOS**
```bash
# Ouvrir les préférences de confidentialité
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
```
- Cochez **Terminal** ou votre IDE
- Redémarrez l'application

**Solution 2 : Tester la caméra**
```bash
# Test simple
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

**Solution 3 : Fermer les autres applications**
- Fermez Zoom, Skype, FaceTime, etc.
- Une seule application peut utiliser la caméra

### Problème : Modèle non trouvé

**Solution : Vérifier les fichiers**
```bash
# Lister les modèles disponibles
ls -lh *.h5
```

Si aucun modèle n'existe :
```bash
# Entraîner un modèle
python3 train_optimized_cnn.py
```

### Problème : FPS faible

**Solution 1 : Ajuster les paramètres**
Modifiez dans le fichier `webcam_realtime_prediction.py` :
```python
SKIP_FRAMES = 3          # Augmenter (traiter moins de frames)
DETECTION_SCALE = 0.4    # Réduire (résolution plus basse)
SMOOTH_PREDICTIONS = 3   # Réduire (moins de lissage)
```

**Solution 2 : Fermer les applications**
- Fermez les applications gourmandes
- Libérez de la RAM

### Problème : Détection instable

**Solution : Augmenter le lissage**
```python
SMOOTH_PREDICTIONS = 7   # Plus de lissage
MIN_CONFIDENCE = 0.4     # Seuil plus élevé
```

### Problème : Pas de visage détecté

**Vérifications :**
- ✅ Visage bien éclairé
- ✅ Visage de face (pas de profil)
- ✅ Distance appropriée (50cm - 1m)
- ✅ Pas d'obstruction (lunettes OK, masque NON)

---

## 💡 Conseils pour de meilleurs résultats

### 1. Éclairage
- **Lumière frontale** (pas de contre-jour)
- Éclairage **uniforme** sur le visage
- Éviter les ombres fortes

### 2. Position
- **Face à la caméra** (angle < 30°)
- Distance : **50cm à 1 mètre**
- Visage **centré** dans l'image

### 3. Expression
- **Exagérer légèrement** l'expression
- Maintenir l'expression **2-3 secondes**
- Éviter les expressions mixtes

### 4. Environnement
- Fond **simple et uni** (si possible)
- Pas de mouvement brusque
- Caméra **stable**

### 5. Performance
- Fermer les **applications inutiles**
- Utiliser un **bon éclairage** (réduit le bruit)
- Nettoyer la **lentille** de la caméra

---

## 📊 Statistiques affichées

### FPS (Frames Per Second)
- **30+ FPS** : Excellent
- **20-30 FPS** : Bon
- **15-20 FPS** : Acceptable
- **< 15 FPS** : Optimisation nécessaire

### Confiance
- **> 70%** : Très fiable
- **50-70%** : Fiable
- **30-50%** : Incertain
- **< 30%** : Non affiché (seuil minimum)

---

## 🎓 Comprendre les résultats

### Graphique des probabilités
Le mini-graphique en bas montre la probabilité de chaque émotion :
- **Barre haute** = Probabilité élevée
- **Couleur** = Émotion correspondante
- **Somme** = 100% (toutes les barres)

### Lissage temporel
Les prédictions sont lissées sur 5 frames :
- Réduit les **fluctuations**
- Améliore la **stabilité**
- Peut créer un **léger délai** (< 0.2s)

---

## 🔬 Paramètres avancés

### Configuration dans le code

```python
# Fréquence de traitement
SKIP_FRAMES = 2              # 0 = toutes les frames, 2 = 1/3

# Résolution de détection
DETECTION_SCALE = 0.5        # 0.5 = 50%, 0.3 = 30%

# Stabilité des prédictions
SMOOTH_PREDICTIONS = 5       # Nombre de frames pour moyenne

# Seuil d'affichage
MIN_CONFIDENCE = 0.3         # 0.0 à 1.0
```

### Résolution de la caméra

```python
def configure_camera(self, cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Largeur
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Hauteur
    cap.set(cv2.CAP_PROP_FPS, 30)            # FPS cible
```

---

## 📝 Exemples d'utilisation

### Cas d'usage 1 : Test rapide
```bash
# Lancer et tester quelques expressions
python3 webcam_realtime_prediction.py

# Tester : sourire, froncer les sourcils, surprise
# Appuyer sur Q pour quitter
```

### Cas d'usage 2 : Capture d'expressions
```bash
# Lancer l'application
python3 webcam_realtime_prediction.py

# Pour chaque expression :
# 1. Faire l'expression
# 2. Attendre stabilisation (2-3s)
# 3. Appuyer sur S pour capturer
# 4. Répéter pour autres expressions
```

### Cas d'usage 3 : Analyse de performance
```bash
# Lancer avec monitoring
python3 webcam_realtime_prediction.py

# Observer :
# - FPS moyen
# - Temps de réponse
# - Stabilité des prédictions

# Ajuster les paramètres si nécessaire
```

---

## 🆘 Support

### Problèmes courants

**Q : L'application se ferme immédiatement**
- Vérifiez que la caméra est accessible
- Vérifiez qu'un modèle .h5 existe
- Regardez les messages d'erreur

**Q : Les prédictions sont incorrectes**
- Vérifiez l'éclairage
- Exagérez l'expression
- Entraînez un meilleur modèle

**Q : L'application est lente**
- Augmentez SKIP_FRAMES
- Réduisez DETECTION_SCALE
- Fermez les autres applications

### Logs et débogage

L'application affiche des informations détaillées :
```
✅ Initialisation terminée!
📋 Optimisations actives: ...
🎬 Démarrage de la capture vidéo...
📸 Capture sauvegardée: ...
✅ Application terminée
📊 FPS moyen: 28.5
```

---

## 🎯 Résumé

### Pour commencer
1. **Installer** les dépendances
2. **Vérifier** qu'un modèle existe
3. **Lancer** `python3 webcam_realtime_prediction.py`
4. **Tester** différentes expressions

### Pour de meilleurs résultats
- Bon **éclairage**
- Visage de **face**
- Expressions **claires**
- Distance **appropriée**

### Commandes essentielles
- **Q** : Quitter
- **S** : Capturer
- **R** : Reset

---

## 📚 Ressources supplémentaires

- **Guide d'optimisation** : `GUIDE_OPTIMISATION.md`
- **Système amélioré** : `GUIDE_SYSTEME_AMELIORE.md`
- **Webcam macOS** : `GUIDE_WEBCAM_MACOS.md`
- **Dépannage Streamlit** : `TROUBLESHOOTING_STREAMLIT.md`

---

## ✨ Fonctionnalités futures

- [ ] Enregistrement vidéo avec annotations
- [ ] Export des statistiques en CSV
- [ ] Mode multi-visages simultanés
- [ ] Graphiques de tendance temporelle
- [ ] Interface web avec Streamlit
- [ ] Support GPU pour accélération

---

**Créé avec ❤️ par Bob**

*Dernière mise à jour : 2026-05-31*