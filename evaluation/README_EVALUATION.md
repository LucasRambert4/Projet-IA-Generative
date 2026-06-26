# Pipeline d’évaluation IA générative

Ce dossier contient le POC d’évaluation de l’assistant IA du dashboard vocal.

## Objectif

L’objectif est d’évaluer la qualité des réponses générées par l’assistant IA.

Le pipeline suit trois étapes :

1. Simuler des inputs.
2. Générer des outputs.
3. Évaluer les outputs.

## Scénarios évalués

Le dataset contient trois types de scénarios :

- scénario idéal ;
- scénario réaliste ;
- scénario adverse.

## Métriques utilisées

Le pipeline combine plusieurs types de métriques :

- métriques déterministes ;
- string matching ;
- contrôles métier ;
- LLM as a judge ;
- mesure de latence.

## Lancement

Depuis la racine du projet :

```powershell
python evaluation/run_evaluation.py