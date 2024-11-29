# Guide d'utilisation : Simulation de voitures autonomes
## Description générale

Ce projet simule des voitures autonomes qui apprennent à naviguer sur une piste grâce à un algorithme génétique. Les voitures utilisent un réseau neuronal pour évaluer leur environnement et ajuster leurs mouvements. Le projet est développé avec Python et utilise la bibliothèque Pygame pour la simulation visuelle.

## Prérequis
1. Python 3.7 ou supérieur.

2. Les bibliothèques nécessaires :
  - pygame
  - numpy
  - pandas
  - matplotlib
  - seaborn

## Fichiers du projet
1. main.py
  - Usage principal : Exécute une série d'expériences avec différentes tailles de population et taux de mutation pour analyser leurs impacts sur les performances des voitures.
  - Sortie : Résultats enregistrés dans un fichier CSV (experiment_results.csv).
  - Commandes :
    - Lancez le script avec :
      ```
      python main.py
      ```
    - Observez les performances avec **analyze_results.py**
    - Génère des graphiques pour visualiser les tendances de fitness en fonction des tailles de population et des taux de mutation.
      ```
      python analyze_results.py
      ```

2. SingleExperiment.py
  - Usage : Démarre une simulation unique pour visualiser les performances des voitures sur une seule piste.
  - Commandes :
    - Lancez la simulation avec :
      ```
      python SingleExperiment.py
      ```

3. GeneticAlgorithm.py
  - Responsable de :
    - La sélection des meilleurs candidats.
    - Le croisement pour produire de nouveaux individus.
    - La mutation des poids des réseaux neuronaux.

4. Car.py
  - Définit les voitures :
    - Comprend leur logique de mouvement.
    - Calcule leur fitness (aptitude).
    - Suit leur progression sur la piste.

5. Track.py
  - Définit la piste :
    - Contient les limites intérieures et extérieures.
    - Inclut une ligne de course calculée pour optimiser le chemin.


## Exécution de simulations
  1. Simulation unique
    - Lancez SingleExperiment.py pour une simulation en temps réel.
    - Observez les voitures apprendre et évoluer au fil des générations.

  2. Expériences multiples
    - Lancez main.py pour effectuer plusieurs simulations avec des paramètres variés.
    - Les résultats sont enregistrés dans experiment_results.csv.

## Visualisation et analyse
  - Après avoir exécuté main.py, utilisez analyze_results.py pour :
    - Visualiser les tendances de fitness sur les générations.
    - Analyser l'impact des tailles de population et des taux de mutation.

## Structure du code

#### Modules principaux
- Algorithme génétique :
    - Sélection des meilleurs candidats.
    - Croisement et mutation pour générer une nouvelle population.

- Piste :
    - Suivi des voitures sur la piste.
    - Calcul de la progression et des directions.

- Voitures :
    - Contrôle basé sur des réseaux neuronaux.
    - Évaluation de l'aptitude avec des métriques comme le temps au tour et l'adhérence à la piste.
