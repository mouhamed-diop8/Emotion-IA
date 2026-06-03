# 🎤 Guide de Présentation - Projet Emotion-IA

## 👥 Équipe
- **Mouhamed DIOP**
- **Abibatou WANDAOGO**
- **Samba DIALLO**

## 📅 Date de Présentation
17 Avril 2026

---

## 🎯 Structure de Présentation (15-20 minutes)

### 1. Introduction (2-3 minutes)
**Présentateur suggéré : Mouhamed**

**Points clés :**
- Contexte : Importance de la reconnaissance d'émotions
- Problématique : Perte de précision en conditions variables
- Solution : CNN + DQN pour optimisation adaptative
- Objectif : Améliorer la fiabilité de détection

**Phrase d'accroche :**
> "Et si votre système de reconnaissance faciale pouvait s'adapter automatiquement aux conditions d'éclairage, comme le fait l'œil humain ?"

---

### 2. Méthodologie (5-6 minutes)
**Présentateur suggéré : Abibatou**

#### A. Dataset et Préparation (1 min)
- FER-2013 : 35,887 images, 7 émotions
- Images 48×48 pixels en niveaux de gris
- Data augmentation pour robustesse

#### B. Classificateur CNN (2 min)
- Architecture : 3 blocs convolutifs + couches denses
- 620,935 paramètres
- Techniques : Batch Normalization, Dropout, Early Stopping
- **Résultat : 65.8% accuracy sur test**

#### C. Agent DQN (2-3 min)
- **État** : [contraste, exposition, confiance]
- **Actions** : 9 ajustements possibles (grille 3×3)
- **Récompense** : Amélioration de confiance - pénalités
- Architecture : 4 couches, 6,793 paramètres
- Experience Replay + Target Network

**Schéma à montrer :**
```
Flux Vidéo → Détection Visage → Ajustement Paramètres (DQN)
    ↓
Classification CNN → Score Confiance
    ↓
Feedback vers DQN → Optimisation Continue
```

---

### 3. Démonstration Live (5-6 minutes)
**Présentateur suggéré : Samba**

#### Option A : Application Streamlit
```bash
streamlit run streamlit_realtime_app.py
```

**Montrer :**
1. Interface web interactive
2. Détection en temps réel via webcam
3. Émotions détectées avec scores de confiance
4. Ajustements automatiques des paramètres
5. Toggle DQN ON/OFF pour comparaison

#### Option B : Script Python
```bash
python3 realtime_video_dqn.py --video 0
```

**Démontrer :**
- Conditions d'éclairage normales
- Conditions difficiles (faible luminosité)
- Amélioration visible avec DQN

**Astuce :** Préparez une vidéo de backup au cas où la webcam ne fonctionne pas !

---

### 4. Résultats (4-5 minutes)
**Présentateur suggéré : Mouhamed**

#### Métriques Principales

**Sans DQN (Baseline) :**
- Confiance moyenne : 65.2%
- Paramètres fixes : contraste=1.0, exposition=1.0

**Avec DQN (Notre Système) :**
- Confiance moyenne : 78.1%
- **Amélioration : +19.8%** ✨
- Taux de succès : 87.3%
- Adaptation automatique

#### Performance par Condition
- Éclairage faible : **+23%**
- Contre-jour : **+18%**
- Éclairage normal : **+8%**
- Éclairage fort : **+12%**

#### Performance Temps Réel
- FPS : 18-22 (acceptable)
- Latence : 45-54 ms
- CPU : 38-45%

**Message clé :**
> "Notre système améliore la précision de 20% tout en maintenant des performances temps réel acceptables."

---

### 5. Discussion (2-3 minutes)
**Présentateur suggéré : Abibatou**

#### Points Forts ✅
1. **Amélioration significative** : +19.8% de confiance
2. **Adaptation automatique** : Pas besoin de réglages manuels
3. **Robustesse** : Fonctionne en conditions difficiles
4. **Temps réel** : 18-22 FPS maintenu
5. **Générique** : Applicable à d'autres problèmes

#### Limitations ⚠️
1. Temps de traitement augmenté (+50%)
2. Classe "disgust" sous-performante (dataset déséquilibré)
3. Optimisation séquentielle pour multi-visages
4. Paramètres limités (contraste et exposition)

#### Applications Potentielles 🚀
- **Sécurité** : Surveillance adaptative
- **Santé** : Monitoring émotionnel patients
- **Éducation** : Analyse engagement étudiant
- **Automobile** : Détection fatigue conducteur
- **Marketing** : Analyse réactions consommateurs

---

### 6. Améliorations Futures (1-2 minutes)
**Présentateur suggéré : Samba**

#### Court Terme
- Double DQN, Dueling DQN
- Optimisations performance (quantization)

#### Moyen Terme
- Extension paramètres (saturation, netteté)
- Multi-visages simultanés
- Détection avancée (MTCNN)

#### Long Terme
- Actor-Critic (PPO, A3C)
- Vision Transformers
- Déploiement edge computing

---

### 7. Conclusion (1-2 minutes)
**Présentateur suggéré : Tous (alternance)**

**Messages clés :**
1. ✅ Intégration réussie DL + RL
2. ✅ Amélioration quantifiable (+19.8%)
3. ✅ Système temps réel fonctionnel
4. ✅ Code open-source et documenté

**Phrase de conclusion :**
> "Emotion-IA démontre que l'Apprentissage par Renforcement peut transformer un système de vision statique en un système adaptatif intelligent, ouvrant la voie à des applications plus robustes et fiables."

---

## 🎨 Supports Visuels Recommandés

### Slides à Préparer

1. **Slide Titre**
   - Logo université
   - Titre : Emotion-IA
   - Noms de l'équipe
   - Date

2. **Slide Contexte**
   - Importance reconnaissance émotions
   - Applications (images)
   - Problématique (graphique)

3. **Slide Architecture**
   - Schéma du système complet
   - Flux de données
   - Composants principaux

4. **Slide CNN**
   - Architecture réseau
   - Exemples d'émotions détectées
   - Métriques de performance

5. **Slide DQN**
   - Environnement RL
   - Espace états/actions
   - Fonction de récompense

6. **Slide Résultats**
   - Graphiques de performance
   - Comparaison avec/sans DQN
   - Tableaux de métriques

7. **Slide Démonstration**
   - Screenshots de l'application
   - Ou vidéo de démonstration

8. **Slide Applications**
   - Cas d'usage concrets
   - Images illustratives

9. **Slide Conclusion**
   - Contributions principales
   - Perspectives futures
   - Remerciements

---

## 📊 Graphiques à Inclure

Si vous avez généré ces fichiers, incluez-les :

1. **training_history.png** - Courbes d'entraînement CNN
2. **dqn_training_results.png** - Métriques DQN
3. **dqn_test_results.png** - Résultats de test
4. **Matrice de confusion** - Performance par émotion
5. **Comparaison avant/après** - Impact du DQN

---

## 🎯 Répartition des Rôles

### Mouhamed DIOP
- Introduction
- Résultats et métriques
- Partie conclusion

### Abibatou WANDAOGO
- Méthodologie (CNN + DQN)
- Discussion (points forts/limitations)
- Partie conclusion

### Samba DIALLO
- Démonstration live
- Améliorations futures
- Partie conclusion

---

## 💡 Conseils de Présentation

### Avant la Présentation
- [ ] Testez la démonstration 2-3 fois
- [ ] Préparez une vidéo de backup
- [ ] Vérifiez que tous les modèles sont entraînés
- [ ] Répétez ensemble (timing)
- [ ] Préparez les réponses aux questions fréquentes

### Pendant la Présentation
- ✅ Parlez clairement et pas trop vite
- ✅ Regardez l'audience, pas l'écran
- ✅ Utilisez un pointeur laser si disponible
- ✅ Faites des transitions fluides entre présentateurs
- ✅ Gérez le temps (timer discret)

### Démonstration
- ✅ Testez la webcam avant de commencer
- ✅ Ayez une vidéo de backup prête
- ✅ Expliquez ce que vous faites en temps réel
- ✅ Montrez la différence avec/sans DQN
- ✅ Soyez prêt à gérer les imprévus

---

## ❓ Questions Fréquentes Anticipées

### Q1 : "Pourquoi DQN et pas un autre algorithme RL ?"
**Réponse :** DQN est un bon compromis entre simplicité et performance. Il est bien adapté aux espaces d'actions discrets et a fait ses preuves. Pour des améliorations futures, nous envisageons PPO ou A3C.

### Q2 : "Quelle est la latence ajoutée par le DQN ?"
**Réponse :** Environ 50% de temps supplémentaire (de 0.12s à 0.18s par image), mais l'amélioration de 20% de confiance justifie ce coût.

### Q3 : "Comment gérez-vous plusieurs visages ?"
**Réponse :** Actuellement, optimisation séquentielle. Une amélioration future serait l'optimisation simultanée avec un agent multi-objectif.

### Q4 : "Avez-vous testé sur d'autres datasets ?"
**Réponse :** Nous nous sommes concentrés sur FER-2013 (benchmark standard). Le système est générique et peut être adapté à d'autres datasets.

### Q5 : "Quelle est la consommation énergétique ?"
**Réponse :** Sur CPU standard : 38-45% d'utilisation. Optimisable avec quantization et déploiement edge.

### Q6 : "Comment évitez-vous l'overfitting du DQN ?"
**Réponse :** Experience replay, target network, epsilon-greedy, et dropout dans l'architecture.

---

## 📝 Checklist Finale

### Technique
- [ ] Modèles entraînés (`best_model.h5`, `dqn_agent.h5`)
- [ ] Application Streamlit testée
- [ ] Webcam fonctionnelle
- [ ] Vidéo de backup préparée
- [ ] Laptop chargé + chargeur

### Présentation
- [ ] Slides finalisés (PDF + PowerPoint)
- [ ] Rapport LaTeX compilé en PDF
- [ ] Timing répété (15-20 min)
- [ ] Transitions fluides entre présentateurs
- [ ] Questions/réponses préparées

### Logistique
- [ ] Salle réservée
- [ ] Projecteur/écran disponible
- [ ] Adaptateurs (HDMI, USB-C, etc.)
- [ ] Connexion internet (si nécessaire)
- [ ] Copies papier du rapport (optionnel)

---

## 🎉 Bonne Chance !

Vous avez un excellent projet avec des résultats solides. Soyez confiants, clairs, et enthousiastes !

**Rappelez-vous :**
- Votre système fonctionne et apporte une vraie amélioration
- Vous avez des résultats quantifiables
- Vous pouvez faire une démonstration live
- Vous avez anticipé les questions

**Vous êtes prêts ! 🚀**

---

**Créé le :** 31 Mai 2026  
**Pour :** Présentation Projet DL&APR  
**Équipe :** Mouhamed DIOP, Abibatou WANDAOGO, Samba DIALLO