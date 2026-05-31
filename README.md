# Emotion-IA
Reconnaissance faciale d'émotions en temps réel et optimisation adaptative via Reinforcement Learning

_**Contexte :** _
La compréhension de l'état émotionnel humain est un enjeu majeur pour l'évolution des interfaces numériques. Ce projet propose une approche intelligente de la vision par ordinateur pour une interaction homme-machine plus fluide.
Problématique : Les modèles de vision classiques perdent souvent en précision face à des environnements changeants (éclairage instable). L'objectif est de créer un système auto- adaptatif capable de maintenir une haute fiabilité de détection.

_**Solution proposée :**_
- Deep Learning : Implémentation d'un réseau de neurones convolutifs (CNN)
performant, entraîné sur le dataset FER-2013 pour classer 7 émotions faciales.
- Apprentissage par Renforcement : Un agent (DQN) est chargé de réguler dynamiquement les paramètres du flux vidéo (contraste et exposition) pour maximiser le score de confiance de la classification.
