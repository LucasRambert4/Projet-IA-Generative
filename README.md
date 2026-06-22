# Projet Data Science - Analyse des retards de vols

Projet de data science sur l'analyse et la prediction des retards de vols aux Etats-Unis. Le projet combine nettoyage de donnees, analyse exploratoire, dashboard interactif Streamlit, modelisation machine learning et rapport de synthese.

## Equipe

- Julian GABRY
- Joseph RAMBERT

## Objectif

L'objectif est d'aider un responsable des operations aeriennes a comprendre les facteurs associes aux retards et a anticiper les vols les plus a risque.

Un vol est considere comme en retard si son retard a l'arrivee est superieur ou egal a 15 minutes.

Le projet cherche a repondre aux questions suivantes :

- Quels sont les niveaux de retard observes ?
- Quelles compagnies, aeroports, routes ou periodes sont les plus associes aux retards ?
- Peut-on identifier les situations operationnelles les plus critiques ?
- Peut-on construire un premier modele de prediction du risque de retard ?
- Comment restituer les resultats dans un dashboard utile pour la decision ?

## Structure du depot

```text
.
|-- README.md
|-- rapport_projet.md
|-- requirements.txt
|-- flights_delay.ipynb
|-- streamlit_app.py
|-- dashboard_metrics.py
|-- propagation_analysis.py
|-- flights.csv
|-- airlines.csv
|-- airports.csv
`-- dashboard_data/
    |-- flights_dashboard.csv
    |-- kpi_global.csv
    |-- airline_delay.csv
    |-- airport_delay.csv
    |-- route_delay.csv
    |-- model_metrics.csv
    |-- model_confusion_matrix.csv
    |-- model_feature_importance.csv
    `-- autres fichiers agreges pour le dashboard
```

## Donnees

Le projet utilise trois fichiers principaux :

- `flights.csv` : table principale des vols ;
- `airlines.csv` : table de correspondance des compagnies aeriennes ;
- `airports.csv` : table de correspondance des aeroports.

Le fichier `flights.csv` est volumineux. Les calculs globaux sont faits dans le notebook, puis des fichiers agreges sont exportes dans `dashboard_data/` afin que le dashboard soit plus rapide a charger.

## Installation

Depuis la racine du depot :

```bash
python -m venv .venv
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Sous macOS / Linux :

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer le dashboard

```bash
streamlit run streamlit_app.py
```

Puis ouvrir l'URL affichee par Streamlit, generalement :

```text
http://localhost:8501
```

Si `localhost` ne repond pas sur certaines configurations Windows, utiliser :

```text
http://127.0.0.1:8501
```

## Lancer le notebook

Ouvrir `flights_delay.ipynb` dans Jupyter, VS Code ou un environnement compatible.

Le notebook contient :

- chargement des donnees ;
- comprehension du dataset ;
- selection des colonnes utiles ;
- nettoyage ;
- creation de la cible binaire ;
- feature engineering ;
- analyse exploratoire ;
- preparation des exports dashboard ;
- modelisation ;
- comparaison des modeles ;
- conclusion.

Le notebook peut etre long a executer car `flights.csv` contient plusieurs millions de lignes.

## Methodologie

### 1. Preparation des donnees

Les donnees sont nettoyees et enrichies dans le notebook. La cible `IS_DELAYED` est creee a partir du retard a l'arrivee avec un seuil de 15 minutes.

Les colonnes connues uniquement apres le vol sont exclues de la modelisation afin de limiter le data leakage.

### 2. Analyse exploratoire

L'EDA analyse les retards selon plusieurs axes :

- repartition globale des retards ;
- mois, jour de semaine et heure de depart ;
- compagnie aerienne ;
- aeroport de depart et d'arrivee ;
- route ;
- situations a risque ;
- propagation des retards par avion.

### 3. Dashboard Streamlit

Le dashboard permet d'explorer les resultats avec des filtres et plusieurs onglets :

- vue globale ;
- tendances temporelles ;
- aeroports ;
- compagnies ;
- routes ;
- propagation ;
- modele ;
- recommandations.

Les fichiers du dossier `dashboard_data/` servent de couche de donnees optimisee pour l'application.

### 4. Modelisation

Le projet traite la prediction comme une classification binaire :

- `0` : pas de retard ;
- `1` : retard a l'arrivee superieur ou egal a 15 minutes.

Deux modeles sont compares :

- Regression logistique, comme baseline simple et interpretable ;
- Random Forest, comme modele principal.

Les metriques suivies sont :

- accuracy ;
- precision sur la classe retard ;
- recall sur la classe retard ;
- F1-score ;
- ROC-AUC ;
- balanced accuracy ;
- matrice de confusion.

## Resultats principaux

Indicateurs globaux :

- nombre total de vols : environ 5,8 millions ;
- taux de retard : environ 18 % ;
- retard moyen a l'arrivee : environ 4,3 minutes ;
- retard median a l'arrivee : environ -5 minutes.

Resultats modeles :

| Modele | Split | Accuracy | Precision retard | Recall retard | F1 retard | ROC-AUC | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | test | 0.650 | 0.242 | 0.426 | 0.308 | 0.601 | 0.563 |
| Random Forest | validation | 0.651 | 0.169 | 0.451 | 0.246 | 0.596 | 0.565 |
| Logistic Regression | validation | 0.669 | 0.169 | 0.413 | 0.240 | 0.592 | 0.559 |

Le Random Forest est retenu comme modele principal. Le recall sur la classe retard est privilegie, car manquer un vrai retard est plus couteux qu'envoyer une alerte supplementaire.

Variables importantes du modele :

- heure de depart prevue ;
- compagnie aerienne ;
- distance ;
- periode de la journee ;
- duree prevue ;
- aeroport de depart ;
- aeroport d'arrivee ;
- mois ;
- jour de semaine ;
- week-end.

## Rapport

Le rapport detaille du projet est disponible ici :

[rapport_projet.md](rapport_projet.md)

Il contient :

- contexte et objectif ;
- preparation des donnees ;
- analyse exploratoire ;
- dashboard ;
- modelisation ;
- evaluation ;
- recommandations metier ;
- strategie d'integration IA ;
- gouvernance ;
- feuille de route ;
- ecoresponsabilite ;
- limites et ameliorations.

## Strategie IA

Le cas d'usage IA cible est un score de risque de retard calcule avant le depart.

Ce score peut etre integre dans un processus operationnel :

1. recuperer les informations disponibles avant le vol ;
2. calculer le risque de retard ;
3. afficher le score dans le dashboard ;
4. declencher une alerte si le risque est eleve ;
5. laisser la decision finale a un responsable metier.

Le modele doit etre suivi dans le temps avec une gouvernance claire : responsable metier, responsable technique, monitoring des performances, reentrainement regulier et validation par les parties prenantes.

## Ecoresponsabilite

Le projet limite les calculs inutiles en utilisant :

- une baseline simple ;
- un echantillonnage controle pour la modelisation ;
- des fichiers agreges pour le dashboard ;
- un modele Random Forest raisonnable plutot qu'un modele tres lourd ;
- une separation entre calculs lourds et visualisation interactive.

Cette approche rend le projet plus fluide a utiliser et plus raisonnable en ressources.

## Recommandations metier

- Surveiller les combinaisons compagnie / aeroport / heure avec fort volume et fort taux de retard.
- Distinguer les situations a fort taux des situations a fort impact operationnel.
- Suivre les routes sensibles et les rotations d'avions.
- Utiliser le modele comme outil d'aide a la decision, pas comme decision automatique.
- Revalider regulierement les resultats avec des donnees recentes.

## Limites

- Les donnees sont historiques.
- Certaines variables externes importantes, comme la meteo detaillee, ne sont pas disponibles.
- Les performances du modele restent moderees.
- Le dashboard depend des fichiers exportes dans `dashboard_data/`.
- Une validation metier serait necessaire avant tout deploiement operationnel.

## Commandes utiles

Installer les dependances :

```bash
pip install -r requirements.txt
```

Lancer le dashboard :

```bash
streamlit run streamlit_app.py
```

Verifier rapidement les scripts Python :

```bash
python -m py_compile dashboard_metrics.py propagation_analysis.py streamlit_app.py
```

## Etat du projet

Le projet contient une chaine complete :

- notebook d'analyse ;
- dashboard interactif ;
- donnees agregees ;
- comparaison de modeles ;
- rapport projet ;
- README de prise en main.

Il peut etre ameliore avec des donnees temps reel, des donnees meteo, un suivi de performance automatise et une validation metier plus poussee.
