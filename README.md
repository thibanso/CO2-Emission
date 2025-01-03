# Prédiction des émissions de CO2 des véhicules

## 📋 Contexte
Les émissions de dioxyde de carbone (CO2) sont au cœur des préoccupations environnementales et économiques. En 2022, les émissions mondiales ont atteint un record de 36,8 gigatonnes, avec le secteur des transports contribuant à environ 15 % de ces émissions. Ce projet vise à développer un modèle capable de prédire les émissions de CO2 des véhicules en fonction de leurs caractéristiques techniques, afin de mieux cibler les innovations nécessaires pour réduire ces émissions.

## 🚀 Objectif
Le projet a pour but de :
- Identifier les caractéristiques techniques des véhicules influençant le plus les émissions de CO2.
- Construire un modèle prédictif pour estimer les émissions de CO2 des nouveaux véhicules.
- Fournir un outil interactif, à l'aide de Streamlit, pour simuler les émissions en fonction des paramètres fournis par l'utilisateur.

## 🛠️ Technologies utilisées
- **Python** : Langage principal pour le traitement des données et la modélisation.
- **Bibliothèques principales** : 
  - `Pandas`, `NumPy` : Manipulation et traitement des données.
  - `Matplotlib`, `Seaborn` : Visualisation des données.
  - `scikit-learn` : Prétraitement, modélisation et évaluation.
  - `Streamlit` : Interface utilisateur interactive.
  - `joblib` : Sauvegarde et chargement du modèle entraîné.
- **Données** :
  - data_cleaned.csv est un jeu de données créé à partir de Data.csv (lien ci-dessous) qui a été nettoyé et transformé. 
  - Jeu de données initial [2023 European Environment Agency Dataset](https://www.eea.europa.eu/en/datahub/datahubitem-view/fa8b1229-3db6-495d-b18e-9c9b3267c02b).

## 📊 Données utilisées
Deux jeux de données ont été exploités pour cette étude :
1. **Dataset français de 2014** : Informations sur les émissions de CO2 et caractéristiques des véhicules commercialisés en France.
2. **Dataset européen de 2023** : Données récentes et étendues à plusieurs pays européens, avec des mesures conformes aux normes WLTP (Worldwide Harmonized Light Vehicles Test Procedure).

Le second dataset a été retenu pour sa récence et sa pertinence par rapport à la norme WLTP.

## ⚙️ Étapes du projet
1. **Exploration et analyse des données** :
   - Compréhension des variables.
   - Identification des valeurs manquantes et anomalies.
2. **Nettoyage des données** :
   - Suppression des colonnes inutiles et des doublons.
   - Gestion des valeurs manquantes.
3. **Prétraitement des données** :
   - Encodage des variables catégorielles.
   - Standardisation des variables numériques.
4. **Modélisation** :
   - Modèle utilisé : Random Forest Regressor.
   - Évaluation des performances : Score R² et MAE.
5. **Application interactive** :
   - Développement d'une interface utilisateur avec Streamlit pour prédire les émissions de CO2.

## 🧩 Fonctionnalités de l'application
L'application Streamlit permet de :
- Sélectionner la catégorie du véhicule (Tourisme, Utilitaire).
- Entrer des paramètres techniques : poids, motorisation, capacité du moteur, puissance, consommation, etc.
- Obtenir une prédiction des émissions de CO2 (g/km), avec une indication visuelle de la performance écologique du véhicule.

## 📂 Structure du projet

## 👥 Membres de l'équipe
- **Antoine BARBIER**  
  - [GitHub](https://github.com/Antoine-DA)  
  - [LinkedIn](https://www.linkedin.com/in/antoine-barbier-83654415b/)
- **Flora BREN**  
  - [GitHub](https://github.com/bobmartin)  
  - [LinkedIn](https://www.linkedin.com/in/flora-b-68a80013a/)
- **Thibault EL MANSOURI**  
  - [GitHub](https://github.com/thibanso)  
  - [LinkedIn](https://www.linkedin.com/in/el-mansouri-299932130/)

## 📦 Installation et utilisation
- Télécharger streamlit.py, model.zip et tableau.xlsx
Executer sur VS Code streamlit.py en local
