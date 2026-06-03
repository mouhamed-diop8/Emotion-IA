# 🔧 Guide de Dépannage - Webcam Streamlit sur macOS

## Problème: La webcam ne fonctionne pas dans Streamlit

### Solutions par ordre de priorité:

## 1️⃣ Vérifier les Permissions macOS

### Étape A: Préférences Système
```bash
# Ouvrir les préférences de sécurité
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
```

**Actions:**
1. Allez dans **Préférences Système** → **Sécurité et confidentialité** → **Caméra**
2. Vérifiez que votre navigateur (Chrome, Safari, Firefox) est coché ✅
3. Si non coché, cochez-le
4. **Redémarrez votre navigateur**

### Étape B: Permissions du Terminal (si vous utilisez Terminal)
```bash
# Vérifier si Terminal a accès à la caméra
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
```
- Cochez aussi **Terminal** ou **iTerm2** si vous les utilisez

## 2️⃣ Vérifier les Permissions du Navigateur

### Chrome:
1. Ouvrez Chrome
2. Allez sur `chrome://settings/content/camera`
3. Vérifiez que la caméra n'est pas bloquée
4. Ajoutez `http://localhost:8501` aux sites autorisés

### Safari:
1. Safari → Préférences → Sites web → Caméra
2. Autorisez l'accès pour localhost

### Firefox:
1. about:preferences#privacy
2. Permissions → Caméra → Paramètres
3. Autorisez localhost

## 3️⃣ Tester la Caméra

### Test rapide avec Python:
```bash
python3 << 'EOF'
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Caméra accessible!")
    ret, frame = cap.read()
    if ret:
        print("✅ Capture d'image réussie!")
    else:
        print("❌ Impossible de capturer une image")
    cap.release()
else:
    print("❌ Impossible d'ouvrir la caméra")
EOF
```

## 4️⃣ Redémarrer Streamlit

### Arrêter tous les processus Streamlit:
```bash
pkill -9 -f streamlit
```

### Relancer l'application:
```bash
python3 -m streamlit run streamlit_app_fast.py --server.port 8501
```

## 5️⃣ Utiliser une Application Alternative

Si Streamlit ne fonctionne toujours pas, utilisez une des applications webcam natives:

### Option A: Application Webcam Universelle
```bash
python3 webcam_universal_app.py
```

### Option B: Application macOS Optimisée
```bash
python3 webcam_macos_app.py
```

### Option C: Application Temps Réel
```bash
python3 streamlit_realtime_app.py
```

## 6️⃣ Problèmes Courants

### Erreur: "Camera not found"
**Solution:** Vérifiez que votre caméra n'est pas utilisée par une autre application
```bash
# Voir les processus utilisant la caméra
lsof | grep -i camera
```

### Erreur: "Permission denied"
**Solution:** Réinitialisez les permissions
```bash
# Réinitialiser les permissions de la caméra (nécessite un redémarrage)
tccutil reset Camera
```

### La webcam s'affiche mais l'image est noire
**Solution:** 
1. Fermez toutes les applications utilisant la caméra (Zoom, Skype, etc.)
2. Redémarrez le navigateur
3. Relancez Streamlit

### Le bouton "Prenez une photo" ne répond pas
**Solution:**
1. Vérifiez la console du navigateur (F12) pour les erreurs JavaScript
2. Essayez un autre navigateur
3. Videz le cache du navigateur

## 7️⃣ Configuration Streamlit Avancée

Créez un fichier `.streamlit/config.toml`:
```bash
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
serverAddress = "localhost"
EOF
```

Puis relancez:
```bash
pkill -9 -f streamlit
python3 -m streamlit run streamlit_app_fast.py
```

## 8️⃣ Vérification Finale

### Checklist:
- [ ] Permissions macOS activées pour le navigateur
- [ ] Permissions macOS activées pour Terminal (si applicable)
- [ ] Navigateur autorise l'accès à la caméra pour localhost
- [ ] Aucune autre application n'utilise la caméra
- [ ] Streamlit est bien lancé sur le port 8501
- [ ] Le navigateur affiche bien http://localhost:8501

## 9️⃣ Alternative: Utiliser OpenCV Direct

Si Streamlit ne fonctionne vraiment pas, utilisez cette application simple:

```bash
python3 << 'EOF'
import cv2
import numpy as np

print("🎥 Démarrage de la webcam...")
print("Appuyez sur 'q' pour quitter")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Impossible d'ouvrir la caméra!")
    exit(1)

print("✅ Caméra ouverte avec succès!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Erreur de lecture")
        break
    
    cv2.imshow('Webcam Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Test terminé")
EOF
```

## 🆘 Support Supplémentaire

Si aucune solution ne fonctionne:
1. Vérifiez que votre caméra fonctionne dans d'autres applications (Photo Booth, FaceTime)
2. Redémarrez votre Mac
3. Mettez à jour macOS et votre navigateur
4. Consultez les logs Streamlit pour plus de détails

## 📝 Logs Utiles

```bash
# Voir les logs Streamlit en temps réel
tail -f ~/.streamlit/logs/*.log
```

---
**Note:** Sur macOS, les permissions de caméra sont strictes pour des raisons de sécurité. Assurez-vous de bien autoriser tous les accès nécessaires.