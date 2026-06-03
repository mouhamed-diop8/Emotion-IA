# Guide de Compilation du Rapport LaTeX

## 📄 Fichier Créé

Le rapport complet du projet a été créé : **`rapport_projet.tex`**

## 🎯 Options pour Compiler le Rapport

### Option 1 : Overleaf (Recommandé - Plus Simple)

1. **Aller sur Overleaf** : https://www.overleaf.com
2. **Créer un compte gratuit** (si vous n'en avez pas)
3. **Créer un nouveau projet** : "New Project" → "Blank Project"
4. **Copier le contenu** de `rapport_projet.tex` dans le fichier principal
5. **Compiler** : Cliquez sur "Recompile" (le bouton vert)
6. **Télécharger le PDF** : Menu → "Download PDF"

**Avantages :**
- ✅ Aucune installation nécessaire
- ✅ Compilation en ligne
- ✅ Collaboration facile avec vos collègues
- ✅ Toutes les packages LaTeX disponibles

### Option 2 : Installation LaTeX sur macOS

#### A. Installer MacTeX (Distribution complète)

```bash
# Télécharger MacTeX depuis : http://www.tug.org/mactex/
# Ou installer via Homebrew :
brew install --cask mactex

# Après installation, compiler :
cd "/Users/user/Desktop/pROJET B"
pdflatex rapport_projet.tex
pdflatex rapport_projet.tex  # Deux fois pour les références
```

**Taille :** ~4 GB (distribution complète)

#### B. Installer BasicTeX (Distribution légère)

```bash
# Installation via Homebrew
brew install --cask basictex

# Ajouter au PATH
export PATH="/Library/TeX/texbin:$PATH"

# Installer les packages nécessaires
sudo tlmgr update --self
sudo tlmgr install babel-french
sudo tlmgr install algorithm2e
sudo tlmgr install algorithmicx

# Compiler
cd "/Users/user/Desktop/pROJET B"
pdflatex rapport_projet.tex
pdflatex rapport_projet.tex
```

**Taille :** ~100 MB (plus léger)

### Option 3 : Éditeurs LaTeX avec Interface Graphique

#### TeXShop (macOS)
- Inclus avec MacTeX
- Interface simple et intuitive
- Ouvrir `rapport_projet.tex` et cliquer sur "Typeset"

#### TeXstudio (Multi-plateforme)
```bash
brew install --cask texstudio
```
- Interface complète avec auto-complétion
- Ouvrir le fichier et appuyer sur F5 pour compiler

#### Visual Studio Code avec Extension LaTeX
```bash
# Installer l'extension LaTeX Workshop dans VS Code
# Ouvrir rapport_projet.tex
# Ctrl+Alt+B pour compiler
```

### Option 4 : Services en Ligne (Alternatives à Overleaf)

1. **Papeeria** : https://papeeria.com
2. **CoCalc** : https://cocalc.com
3. **Authorea** : https://www.authorea.com

## 📋 Structure du Rapport

Le rapport contient les sections suivantes :

1. **Page de titre** avec noms de l'équipe
2. **Table des matières**
3. **Résumé (Abstract)**
4. **Introduction** - Contexte, problématique, objectifs
5. **Méthodologie** - Architecture, CNN, DQN
6. **Résultats** - Performance, métriques, comparaisons
7. **Discussion** - Points forts, limitations, applications
8. **Améliorations Futures** - Court, moyen et long terme
9. **Conclusion** - Synthèse et perspectives
10. **Références bibliographiques**
11. **Annexes** - Commandes, structure projet

**Total :** ~20 pages

## 🎨 Personnalisation

### Ajouter le Logo de votre Université

Dans le fichier `rapport_projet.tex`, ligne 60 :
```latex
\includegraphics[width=0.3\textwidth]{logo_universite.png}
```

Remplacez par le chemin de votre logo ou commentez cette ligne si vous n'avez pas de logo.

### Modifier les Informations

- **Encadrant** : Ligne 71
- **Institution** : Ligne 72
- **Date** : Ligne 75

### Ajouter des Graphiques

Pour inclure vos résultats visuels :

```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{dqn_training_results.png}
\caption{Résultats d'entraînement du DQN}
\label{fig:dqn_results}
\end{figure}
```

## 📊 Fichiers Graphiques à Inclure (Optionnel)

Si vous avez généré ces fichiers, vous pouvez les ajouter au rapport :

- `training_history.png` - Historique d'entraînement CNN
- `dqn_training_results.png` - Métriques DQN
- `dqn_test_results.png` - Résultats de test
- `confusion_matrix.png` - Matrice de confusion

## 🔧 Dépannage

### Erreur : "Package not found"
```bash
sudo tlmgr install <nom_du_package>
```

### Erreur : "Font not found"
Utilisez XeLaTeX au lieu de pdflatex :
```bash
xelatex rapport_projet.tex
```

### Erreur de compilation
1. Vérifiez la syntaxe LaTeX
2. Compilez deux fois (pour les références)
3. Supprimez les fichiers auxiliaires (.aux, .log) et recompilez

### Caractères spéciaux
Le fichier utilise UTF-8 et le package `babel-french` pour le français.

## 📤 Partage avec vos Collègues

### Via Overleaf
1. Créez le projet sur Overleaf
2. Cliquez sur "Share" en haut à droite
3. Invitez vos collègues par email
4. Ils pourront éditer en temps réel

### Via Git
```bash
git add rapport_projet.tex
git commit -m "Ajout du rapport LaTeX"
git push
```

### Via Email
Envoyez simplement le fichier `rapport_projet.tex`

## 🎓 Présentation

Pour la présentation avec vos collègues :

1. **Compilez le PDF** via une des méthodes ci-dessus
2. **Imprimez** ou **projetez** le rapport
3. **Préparez des slides** basés sur les sections principales
4. **Démonstration live** : Utilisez `streamlit_realtime_app.py`

### Suggestion de Structure de Présentation (15-20 min)

1. **Introduction** (2 min) - Contexte et problématique
2. **Méthodologie** (5 min) - Architecture CNN + DQN
3. **Démonstration** (5 min) - Application temps réel
4. **Résultats** (5 min) - Métriques et comparaisons
5. **Conclusion** (3 min) - Contributions et perspectives

## ✅ Checklist Avant Présentation

- [ ] Rapport compilé en PDF
- [ ] Modèles entraînés (`best_model.h5`, `dqn_agent.h5`)
- [ ] Application Streamlit testée
- [ ] Slides de présentation préparés
- [ ] Démonstration webcam fonctionnelle
- [ ] Questions/réponses anticipées

## 📞 Support

Si vous rencontrez des problèmes :

1. **Overleaf** : Support intégré dans la plateforme
2. **LaTeX** : https://tex.stackexchange.com
3. **GitHub du projet** : https://github.com/mouhamed-diop8/PROJECT-IA-APPRENTISSAGE-PAR-RENFORCEMENT

## 🎉 Félicitations !

Vous avez maintenant un rapport professionnel complet pour votre projet Emotion-IA !

---

**Créé le :** 31 Mai 2026  
**Équipe :** Mouhamed DIOP, Abibatou WANDAOGO, Samba DIALLO