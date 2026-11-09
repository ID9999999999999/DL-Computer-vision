> Historical preparation note. Training completed on 25 October 2026. See README.md and results/Final_Report.md for the current submission status.

# Lab 06 — prêt à exécuter

Le code et le mémo préalable sont préparés. Les entraînements ne sont pas encore exécutés : les images sont sur ton ordinateur. Aucun résultat expérimental n'est inventé.

## Démarrage sous Windows

1. Décompresse ce ZIP dans un nouveau dossier.
2. Double-clique sur **RUN_WINDOWS.bat**. Python 3.11 doit être installé.
3. Dans la fenêtre, sélectionne **demo_quality_dataset** : le dossier qui contient directement **metadata.jsonl**, **good** et **bad**.

Le programme installe les bibliothèques dans son propre environnement, télécharge les poids préentraînés et entraîne les quatre modèles. Une connexion Internet est nécessaire pour ces téléchargements. Sur CPU, les entraînements peuvent être longs ; aucune durée précise n'a été mesurée sur ton ordinateur.

Ne ferme pas la fenêtre pendant l'exécution. Une interruption après un modèle terminé peut être reprise en relançant le même fichier : les modèles terminés sont conservés. Si une étape échoue, garde le message d'erreur ; le programme ne déclare pas le travail terminé.

## Résultats générés après une exécution réussie

- **results/Final_Report.md** : tableau complet, comparaison, analyse et décision.
- **results/comparison.csv** : accuracy, précision, rappel, F1, latence, taille, paramètres et durée.
- **results/accuracy_vs_speed.png** et **results/confusion_matrices.png**.
- **results/mobilenetv2/curves.png**, **results/efficientnetb0/curves.png** et les courbes YOLO.
- Une galerie d'erreurs par modèle et **results/defect_analysis.csv**.
- **results/Presentation_5min.md** : déroulé de présentation fondé sur tes résultats.
- **results/run.log**, **protocol.json**, **manifest.json**, **environment.txt** : traces de reproductibilité.

Les documents scientifiques et le code sont en anglais. Le mémo préalable est dans **docs/Model_Selection_Memo.md**. Le notebook **Lab06.ipynb** permet aussi une exécution cellule par cellule.

## Répartition des données

Le précédent Lab05 utilise 1230 / 228 / 342 images. Ce programme prépare un **nouveau manifeste** avec 1260 / 270 / 270 images, soit 70 / 15 / 15 %, en regroupant toutes les variantes d'un même document source. Il ne réécrit pas les métadonnées originales. Les liens matériels vers les images, lorsqu'ils sont possibles, évitent de les copier ; ne modifie pas les images pendant l'expérience.

Le nouveau split est justifié ici parce que les fichiers fournis montrent un audit et un chargement de données, sans résultat d'entraînement. Si ces images ont déjà servi à sélectionner un modèle ou des hyperparamètres à partir de l'ancien test, cette redistribution ne crée pas un test indépendant : il faudra réserver de nouvelles sources pour une évaluation finale valide.

## Commande manuelle, si nécessaire

Dans un environnement Python 3.11 avec les dépendances installées :

```bash
python -m pip install -r requirements.txt
python lab06.py run --dataset "C:/chemin/vers/demo_quality_dataset"
```

Pour une nouvelle expérience, définis ses paramètres **avant** de lancer et utilise un nouveau dossier avec `--out results_new`. Le programme fige le protocole ; changer les paramètres après préparation est refusé.

## État de livraison

Préparé : code, notebook, comparaison théorique sourcée, protocole et génération des rapports. Vérifié ici : compilation Python, tests de séparation par source, calcul des métriques, décision indépendante des valeurs du test, génération de rapport sur des données de test artificielles.

À exécuter chez toi : installation réelle des bibliothèques de deep learning, téléchargement des poids et entraînements sur tes images. L'intégration TensorFlow/Ultralytics n'a pas pu être exécutée ici. Les versions exactes installées chez toi seront enregistrées ; le fichier requirements donne des plages compatibles, pas un environnement verrouillé et déjà validé.

Avant remise : examine les résultats, comprends les choix, ajoute les fichiers dans ton dépôt et fais le commit demandé. L'exposé et la défense en classe restent à réaliser. Le document fourni est un barème, pas l'énoncé complet ; les seuils numériques proposés dans config.json sont des objectifs de travail, pas des exigences attribuées à l'enseignant.
