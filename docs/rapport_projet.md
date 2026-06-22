# Rapport projet - Analyse et prediction des retards de vols

## 1. Contexte et objectif

Ce projet de data science analyse les retards de vols a partir d'un jeu de donnees aerien compose principalement de `data/raw/flights.csv`, complete par les tables de correspondance `data/raw/airlines.csv` et `data/raw/airports.csv`.

L'objectif metier est d'aider un responsable des operations aeriennes a comprendre les facteurs associes aux retards, a suivre les situations a risque dans un dashboard interactif, puis a utiliser un premier modele predictif pour anticiper les vols susceptibles d'arriver avec au moins 15 minutes de retard.

Le projet repond aux attendus suivants :

- preparation et nettoyage des donnees ;
- analyse exploratoire documentee ;
- tableau de bord interactif ;
- modelisation predictive ;
- comparaison de modeles ;
- strategie d'integration de l'IA dans un processus metier.

## 2. Donnees utilisees

Trois fichiers principaux sont utilises :

- `data/raw/flights.csv` : table principale des vols, horaires, aeroports, compagnies et retards ;
- `data/raw/airlines.csv` : correspondance entre codes compagnies et noms complets ;
- `data/raw/airports.csv` : informations sur les aeroports, villes, etats et coordonnees geographiques.

Le fichier `flights.csv` contient plus de 5,8 millions de lignes. Le projet s'appuie donc sur des aggregations pour les analyses globales et sur un echantillon controle pour certaines parties interactives ou de modelisation.

La cible principale est binaire :

- `0` : vol non retarde ;
- `1` : vol retarde, si le retard a l'arrivee est superieur ou egal a 15 minutes.

## 3. Preparation et nettoyage

Les principales etapes de preparation sont documentees dans le notebook `flights_delay.ipynb`.

Les choix importants sont les suivants :

- conservation des vols exploitables pour l'analyse des retards ;
- traitement des valeurs manquantes sur les retards ;
- creation d'une cible binaire `IS_DELAYED` ;
- creation de variables temporelles comme l'heure de depart prevue, la periode de la journee et l'indicateur week-end ;
- ajout de variables utiles pour l'analyse par compagnie, aeroport, route et periode ;
- retrait des variables connues uniquement apres le vol pour limiter le data leakage.

Le data leakage a ete traite comme un point central du projet. Les colonnes directement liees au resultat final ou connues apres l'arrivee ne sont pas utilisees pour predire le retard avant le vol.

## 4. Analyse exploratoire

L'analyse exploratoire met en evidence plusieurs dimensions importantes :

- le taux de retard global est d'environ 18 % selon le seuil de 15 minutes ;
- les retards varient selon l'heure de depart prevue ;
- certaines compagnies presentent des taux de retard plus eleves que la moyenne ;
- certains aeroports et certaines routes concentrent davantage de vols a risque ;
- les situations les plus importantes a surveiller combinent volume eleve et taux de retard eleve ;
- les retards peuvent se propager dans une meme journee de rotation d'avion.

Cette analyse permet de passer d'une lecture descriptive generale a une lecture operationnelle : il ne suffit pas d'identifier les taux les plus hauts, il faut aussi tenir compte du volume de vols concernes.

## 5. Dashboard interactif

Le dashboard Streamlit est disponible dans `src/streamlit_app.py`.

Il s'appuie sur les fichiers prepares dans `data/dashboard/` et permet de suivre :

- les indicateurs globaux ;
- les tendances temporelles ;
- les retards par aeroport ;
- les retards par compagnie ;
- les routes les plus touchees ;
- les situations a risque ;
- la propagation des retards par avion ;
- les resultats du modele predictif ;
- les recommandations metier.

Les filtres permettent une lecture par compagnie, mois, aeroport de depart, aeroport d'arrivee et jour de semaine.

Le dashboard a ete pense comme un outil de decision pour prioriser les zones de vigilance, comparer les compagnies et aeroports, et detecter les combinaisons operationnelles les plus sensibles.

## 6. Modelisation predictive

Le probleme est traite comme une classification binaire : predire si un vol aura au moins 15 minutes de retard a l'arrivee.

Deux modeles sont compares :

- Regression logistique : baseline simple, rapide et interpretable ;
- Random Forest : modele principal, capable de capter des relations non lineaires entre les variables.

Les variables explicatives retenues sont disponibles avant le vol :

- mois ;
- jour de semaine ;
- heure de depart prevue ;
- duree prevue ;
- distance ;
- indicateur week-end ;
- compagnie aerienne ;
- aeroport de depart ;
- aeroport d'arrivee ;
- periode de la journee.

Le decoupage train / validation / test est chronologique afin de simuler une prediction sur des vols futurs.

## 7. Evaluation et comparaison des modeles

Les modeles sont evalues avec plusieurs metriques :

- accuracy ;
- precision sur la classe retard ;
- recall sur la classe retard ;
- F1-score ;
- ROC-AUC ;
- balanced accuracy ;
- matrice de confusion.

Les resultats exportes dans `data/dashboard/model_metrics.csv` sont les suivants :

| Modele | Split | Accuracy | Precision retard | Recall retard | F1 retard | ROC-AUC | Balanced accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | test | 0.650 | 0.242 | 0.426 | 0.308 | 0.601 | 0.563 |
| Random Forest | validation | 0.651 | 0.169 | 0.451 | 0.246 | 0.596 | 0.565 |
| Logistic Regression | validation | 0.669 | 0.169 | 0.413 | 0.240 | 0.592 | 0.559 |

Le modele retenu est le Random Forest. Il donne une meilleure lecture des facteurs importants et obtient les meilleurs resultats globaux sur les metriques prioritaires pour ce cas d'usage.

Dans ce contexte, le recall est une metrique importante : manquer un vrai retard peut etre plus couteux qu'envoyer une alerte supplementaire. La balanced accuracy et le ROC-AUC completent cette lecture car le dataset est desequilibre.

Les variables les plus importantes du modele sont notamment :

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

## 8. Recommandations metier

Les principales recommandations sont les suivantes :

- surveiller en priorite les combinaisons compagnie / aeroport / heure avec volume important et taux de retard eleve ;
- distinguer les situations a fort taux de retard des situations a fort impact operationnel ;
- suivre les routes les plus sensibles pour anticiper les retards recurrents ;
- renforcer la vigilance sur les vols de fin de journee et les rotations d'avions ;
- utiliser le modele comme un outil d'aide a la decision, pas comme une decision automatique.

## 9. Strategie d'integration de l'IA

### Cas d'usage cible

Le cas d'usage principal est la mise en place d'un score de risque de retard avant depart. Ce score peut aider les equipes operations a prioriser les vols a surveiller.

### Integration operationnelle

Une integration realiste pourrait suivre ce processus :

1. recuperer les informations disponibles avant le vol ;
2. calculer un score de risque avec le modele ;
3. afficher ce score dans un dashboard operationnel ;
4. declencher une alerte si le score depasse un seuil defini ;
5. laisser la decision finale aux equipes metier.

### Gouvernance

Pour une utilisation reelle, il faudrait definir :

- un responsable metier du modele ;
- un responsable technique du pipeline ;
- une frequence de reentrainement ;
- un suivi des performances dans le temps ;
- une documentation des variables utilisees ;
- une validation reguliere par les parties prenantes.

### Feuille de route

Court terme :

- consolider les donnees et automatiser les exports ;
- utiliser le dashboard comme outil d'analyse et de pilotage.

Moyen terme :

- ajouter des donnees meteo, congestion aeroportuaire et informations temps reel ;
- tester d'autres modeles et optimiser le seuil d'alerte ;
- suivre les performances par compagnie, aeroport et periode.

Long terme :

- integrer le score dans un outil operationnel ;
- mettre en place un monitoring automatique ;
- reentrainer le modele avec des donnees recentes ;
- formaliser un processus de validation humaine.

## 10. Ecoresponsabilite

Le projet adopte une approche raisonnable en ressources :

- utilisation d'une regression logistique comme baseline simple ;
- echantillonnage controle pour la modelisation ;
- choix d'un Random Forest limite plutot qu'un modele beaucoup plus lourd ;
- reutilisation de fichiers agreges pour le dashboard au lieu de recalculer toute la base a chaque interaction ;
- separation entre calculs lourds et visualisation interactive.

Cette approche permet de reduire les temps de calcul, de rendre le dashboard plus fluide et de limiter les traitements inutiles.

## 11. Limites et ameliorations possibles

Les principales limites sont les suivantes :

- les donnees sont historiques et doivent etre revalidees avec des donnees recentes ;
- certaines causes externes comme la meteo detaillee ne sont pas disponibles ;
- les performances du modele restent moderees ;
- le modele doit etre utilise comme aide a la decision, pas comme outil automatique ;
- une validation metier serait necessaire avant tout deploiement.

Les ameliorations possibles sont :

- ajouter des variables temps reel ;
- tester des seuils de classification differents ;
- comparer davantage de modeles ;
- mesurer le cout metier des faux positifs et faux negatifs ;
- documenter les retours des utilisateurs du dashboard.

## 12. Conclusion

Le projet fournit une chaine complete : preparation des donnees, analyse exploratoire, dashboard interactif, modelisation predictive, comparaison de modeles et recommandations metier.

Le Random Forest est retenu comme premier modele de scoring du risque de retard. Les resultats montrent que le modele peut aider a prioriser les vols a surveiller, mais qu'il doit rester encadre par une validation humaine et par un suivi regulier des performances.

Le dashboard constitue le livrable operationnel principal : il permet de transformer les resultats analytiques en outil de lecture et de decision pour les operations aeriennes.
