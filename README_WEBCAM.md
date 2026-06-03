# 🎥 Webcam - Démarrage Rapide

## 🚀 Lancement Rapide

```bash
python3 webcam_macos_app.py
```

Appuyez sur **Entrée** pour utiliser la caméra par défaut (0).

## ⌨️ Contrôles

- **q** : Quitter
- **s** : Capture d'écran
- **r** : Reset statistiques
- **p** : Toggle probabilités

## 📋 Prérequis

```bash
# Vérifier les dépendances
python3 -c "import cv2, tensorflow; print('✅ OK')"

# Si erreur, installer
python3 -m pip install opencv-python tensorflow --user
```

## 🔧 Problèmes ?

### Webcam ne s'ouvre pas

1. **Autorisations macOS**
   - Préférences Système → Sécurité → Caméra
   - Autoriser Terminal/Python

2. **Fermer les autres apps**
   - Photo Booth, FaceTime, Zoom, etc.

3. **Essayer caméra externe**
   ```bash
   python3 webcam_macos_app.py
   # Entrer 1 au lieu de 0
   ```

### Modèle manquant

```bash
python3 run_facial_recognition_simple.py
```

## 📚 Documentation Complète

Voir **GUIDE_WEBCAM_MACOS.md** pour :
- Guide détaillé
- Résolution de problèmes
- Optimisations
- Conseils d'utilisation

## ✨ Fonctionnalités

- ✅ Détection en temps réel
- 📊 7 émotions reconnues
- 📈 Statistiques live
- 🎨 Interface colorée
- 📸 Captures d'écran
- ⚡ Optimisé pour macOS

## 🎯 Émotions Détectées

😠 Colère | 🤢 Dégoût | 😨 Peur | 😊 Joie  
😢 Tristesse | 😲 Surprise | 😐 Neutre

---

**Prêt ?** → `python3 webcam_macos_app.py` 🚀