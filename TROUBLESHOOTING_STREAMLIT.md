# 🔧 Guide de Dépannage - Interface Streamlit

## ✅ Vérifications Rapides

### 1. L'application est-elle en cours d'exécution ?
```bash
ps aux | grep streamlit
```
Si vous voyez des processus streamlit, l'application fonctionne.

### 2. Le port 8501 est-il accessible ?
```bash
curl http://localhost:8501
```
Si vous obtenez du HTML, le serveur fonctionne correctement.

### 3. Ouvrir l'interface dans le navigateur
**URL à utiliser:** http://localhost:8501

**Navigateurs recommandés:**
- Google Chrome
- Mozilla Firefox
- Safari

## 🚨 Problèmes Courants et Solutions

### Problème 1: Page blanche ou vide
**Causes possibles:**
- Dépendances manquantes
- Erreurs JavaScript dans le navigateur
- Cache du navigateur

**Solutions:**
1. Vider le cache du navigateur (Cmd+Shift+R sur Mac)
2. Ouvrir la console développeur (F12) pour voir les erreurs
3. Installer les dépendances manquantes:
```bash
python3 -m pip install plotly streamlit-webrtc --user
```

### Problème 2: Erreur "Model not found"
**Solution:**
Assurez-vous que le modèle `best_model.h5` existe:
```bash
ls -lh best_model.h5
```

Si absent, entraînez le modèle:
```bash
python3 run_facial_recognition_simple.py
```

### Problème 3: Webcam ne fonctionne pas
**Solutions:**
1. Autoriser l'accès à la webcam dans les paramètres système
2. Vérifier que la webcam n'est pas utilisée par une autre application
3. Essayer avec une image statique d'abord

### Problème 4: Erreurs d'import
**Vérifier les dépendances:**
```bash
python3 -c "import streamlit, cv2, tensorflow, plotly; print('✅ Toutes les dépendances sont installées')"
```

**Installer les dépendances manquantes:**
```bash
python3 -m pip install streamlit opencv-python tensorflow plotly pandas pillow --user
```

## 🔄 Redémarrage Complet

Si rien ne fonctionne, redémarrez complètement:

```bash
# 1. Arrêter tous les processus Streamlit
pkill -f streamlit

# 2. Attendre 2 secondes
sleep 2

# 3. Relancer l'application
streamlit run streamlit_app.py
```

## 📊 Vérification de l'État du Système

### Vérifier les logs en temps réel
Dans le terminal où Streamlit s'exécute, vous devriez voir:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Tester l'API Streamlit
```bash
curl -s http://localhost:8501/_stcore/health
```
Devrait retourner: `{"status":"ok"}`

## 🌐 Accès depuis un autre appareil

Si vous voulez accéder à l'interface depuis un autre appareil sur le même réseau:

1. Trouvez votre adresse IP locale:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

2. Utilisez l'URL réseau affichée par Streamlit
3. Assurez-vous que le pare-feu autorise les connexions sur le port 8501

## 📝 Logs de Débogage

Pour obtenir plus d'informations sur les erreurs:

```bash
streamlit run streamlit_app.py --logger.level=debug
```

## 🆘 Support Supplémentaire

Si le problème persiste:

1. **Vérifier la console du navigateur** (F12 → Console)
2. **Vérifier les logs du terminal** où Streamlit s'exécute
3. **Tester avec un navigateur différent**
4. **Redémarrer votre ordinateur** (en dernier recours)

## ✨ Fonctionnalités de l'Interface

Une fois l'interface chargée, vous devriez voir:

- **En-tête:** "🎭 Reconnaissance d'Expressions Faciales avec DQN"
- **Sidebar (gauche):** Configuration et options
- **Zone principale:** Flux vidéo et contrôles
- **Panneau droit:** Informations et métriques
- **Section monitoring:** Graphiques et statistiques

Si vous ne voyez pas ces éléments, consultez les solutions ci-dessus.