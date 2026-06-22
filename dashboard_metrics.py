"""Agrégats dashboard — métriques compagnies équitables (taux / nb_vols)."""

from __future__ import annotations

import pandas as pd

# Libellés affichés (UI, graphiques, tableaux) — clés = noms techniques des colonnes
COLUMN_LABELS: dict[str, str] = {
    "taux_retard_percent": "Taux de retard (%)",
    "taux_retard_arrivee_percent": "Retard à l'arrivée (%)",
    "taux_retard_depart_percent": "Retard au départ (%)",
    "part_vols_reseau_percent": "Part des vols (%)",
    "part_retards_reseau_percent": "Part des retards (%)",
    "indice_vs_taille_flotte": "Indice vs taille de flotte",
    "ecart_vs_reseau_pts": "Écart vs moyenne du groupe (pts)",
    "nb_vols": "Nombre de vols",
    "nb_retards": "Nombre de retards",
    "retard_moyen_arrivee": "Retard moyen à l'arrivée (min)",
    "retard_moyen_depart": "Retard moyen au départ (min)",
    "distance_moyenne": "Distance moyenne (mi)",
    "part_percent": "Part (%)",
    "baseline_arrivee_percent": "Référence réseau arrivée (%)",
    "baseline_depart_percent": "Référence réseau départ (%)",
    "baseline_percent": "Référence réseau (%)",
    "prob_percent": "Probabilité (%)",
    "lift": "Multiplicateur de risque",
    "risk_score": "Score de risque",
    "impact_score": "Score d'impact",
    "flight_seq": "Rang dans la journée",
    "first_flight_delayed_label": "Premier vol de la journée",
    "condition": "Condition",
    "condition_met": "Condition remplie",
    "target": "Cible",
    "MONTH": "Mois",
    "DAY_OF_WEEK": "Jour de semaine",
    "SCHEDULED_DEP_HOUR": "Heure de départ prévue",
    "DAY_PERIOD": "Période",
    "DELAY_LEVEL": "Niveau de retard",
    "AIRLINE": "Code compagnie",
    "AIRLINE_LABEL": "Compagnie",
    "AIRLINE_NAME": "Compagnie",
    "ORIGIN_AIRPORT": "Aéroport de départ",
    "DESTINATION_AIRPORT": "Aéroport d'arrivée",
    "AIRPORT_LABEL": "Aéroport",
    "AIRPORT_NAME": "Nom de l'aéroport",
    "CITY": "Ville",
    "STATE": "État",
    "route": "Route",
    "ROUTE_LABEL": "Route",
    "distance_tranche": "Distance de la route",
    "taux_moyen": "Taux de retard (%)",
    "nb_routes": "Nombre de routes",
    "modele": "Modèle",
    "split": "Jeu de données",
    "feature": "Variable",
    "importance": "Importance",
    "accuracy": "Exactitude",
    "precision_retard": "Précision (retard)",
    "recall_retard": "Rappel (retard)",
    "f1_retard": "F1 (retard)",
    "roc_auc": "AUC ROC",
    "balanced_accuracy": "Exactitude équilibrée",
    "reel": "Réel",
    "predit": "Prédit",
    "nombre": "Nombre",
    "indicateur": "Indicateur",
    "valeur": "Valeur",
    "prev_arrival_delayed_label": "Vol précédent",
    "Situation": "Situation",
    "Ecart vs habituel (pts)": "Écart vs habituel (pts)",
    "LATITUDE": "Latitude",
    "LONGITUDE": "Longitude",
    "hover": "Aéroport",
    "prev_arr_delayed": "Vol précédent en retard",
    "first_flight_delayed": "Premier vol en retard",
    "first_flight_delayed_label": "Premier vol de la journée",
}


def label_for(column: str) -> str:
    """Libellé lisible pour une colonne technique."""
    return COLUMN_LABELS.get(column, column.replace("_", " ").strip().capitalize())


def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Renomme les colonnes pour l'affichage Streamlit / rapport."""
    return df.rename(
        columns={col: label_for(col) for col in df.columns},
        errors="ignore",
    )


def plotly_axis_labels(**extra: str) -> dict[str, str]:
    """Dictionnaire `labels=` pour Plotly Express."""
    return {**COLUMN_LABELS, **extra}


def build_airline_delay(
    source_df: pd.DataFrame,
    airlines_df: pd.DataFrame | None = None,
    min_vols: int = 1,
) -> pd.DataFrame:
    """
    Statistiques par compagnie.

    - taux_retard_percent = nb_retards / nb_vols de la compagnie (pas de biais volume seul)
    - part_vols_reseau_percent = part des vols parmi les compagnies retenues (>= min_vols)
    - part_retards_reseau_percent = part des retards parmi les compagnies retenues
    - indice_vs_taille_flotte = part_retards / part_vols (>1 : plus de retards que sa taille ne le suggère)
    """
    grouped = (
        source_df.groupby("AIRLINE", observed=True)
        .agg(
            nb_vols=("IS_DELAYED", "size"),
            nb_retards=("IS_DELAYED", "sum"),
            retard_moyen_arrivee=("ARRIVAL_DELAY_FILLED", "mean"),
            retard_moyen_depart=("DEPARTURE_DELAY_FILLED", "mean"),
            distance_moyenne=("DISTANCE", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["nb_vols"] >= min_vols].copy()
    if grouped.empty:
        return grouped

    total_vols = grouped["nb_vols"].sum()
    total_retards = grouped["nb_retards"].sum()
    taux_global = total_retards / total_vols * 100 if total_vols else 0.0

    grouped["taux_retard_percent"] = (grouped["nb_retards"] / grouped["nb_vols"] * 100).round(2)
    grouped["part_vols_reseau_percent"] = (grouped["nb_vols"] / total_vols * 100).round(2)
    grouped["part_retards_reseau_percent"] = (
        grouped["nb_retards"] / total_retards * 100
    ).round(2)
    grouped["indice_vs_taille_flotte"] = (
        grouped["part_retards_reseau_percent"] / grouped["part_vols_reseau_percent"]
    ).round(2)
    grouped["ecart_vs_reseau_pts"] = (grouped["taux_retard_percent"] - taux_global).round(2)
    grouped["retard_moyen_arrivee"] = grouped["retard_moyen_arrivee"].round(2)
    grouped["retard_moyen_depart"] = grouped["retard_moyen_depart"].round(2)
    grouped["distance_moyenne"] = grouped["distance_moyenne"].round(2)

    if airlines_df is not None and {"IATA_CODE", "AIRLINE"}.issubset(airlines_df.columns):
        names = airlines_df.rename(columns={"IATA_CODE": "AIRLINE", "AIRLINE": "AIRLINE_NAME"})
        grouped = grouped.merge(names[["AIRLINE", "AIRLINE_NAME"]], on="AIRLINE", how="left")
    else:
        grouped["AIRLINE_NAME"] = grouped["AIRLINE"]

    grouped["AIRLINE_LABEL"] = grouped["AIRLINE_NAME"].fillna(grouped["AIRLINE"])
    return recalculate_airline_shares(grouped)


def build_route_delay(source_df: pd.DataFrame, min_vols: int = 1) -> pd.DataFrame:
    """
    Statistiques par paire départ → arrivée.

    - taux_retard_percent : part de vols en retard sur la route
    - ecart_vs_reseau_pts : écart au taux moyen des routes retenues
    - impact_score : nb_retards × retard moyen à l'arrivée (minutes « perdues »)
    """
    if source_df.empty:
        return pd.DataFrame()

    grouped = (
        source_df.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], observed=True)
        .agg(
            nb_vols=("IS_DELAYED", "size"),
            nb_retards=("IS_DELAYED", "sum"),
            retard_moyen_arrivee=("ARRIVAL_DELAY_FILLED", "mean"),
            retard_moyen_depart=("DEPARTURE_DELAY_FILLED", "mean"),
            distance_moyenne=("DISTANCE", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["nb_vols"] >= min_vols].copy()
    if grouped.empty:
        return grouped

    total_vols = grouped["nb_vols"].sum()
    total_retards = grouped["nb_retards"].sum()
    taux_global = total_retards / total_vols * 100 if total_vols else 0.0

    grouped["taux_retard_percent"] = (grouped["nb_retards"] / grouped["nb_vols"] * 100).round(2)
    grouped["ecart_vs_reseau_pts"] = (grouped["taux_retard_percent"] - taux_global).round(2)
    grouped["retard_moyen_arrivee"] = grouped["retard_moyen_arrivee"].round(2)
    grouped["retard_moyen_depart"] = grouped["retard_moyen_depart"].round(2)
    grouped["distance_moyenne"] = grouped["distance_moyenne"].round(2)
    grouped["impact_score"] = (
        grouped["nb_retards"] * grouped["retard_moyen_arrivee"]
    ).round(1)
    grouped["ROUTE_LABEL"] = (
        grouped["ORIGIN_AIRPORT"].astype(str)
        + " → "
        + grouped["DESTINATION_AIRPORT"].astype(str)
    )
    return grouped.sort_values("taux_retard_percent", ascending=False).reset_index(drop=True)


def recalculate_airline_shares(grouped: pd.DataFrame) -> pd.DataFrame:
    """Recalcule parts reseau, indice et ecart a partir de nb_vols / nb_retards deja agreges."""
    out = grouped.copy()
    if out.empty:
        return out

    total_vols = out["nb_vols"].sum()
    total_retards = out["nb_retards"].sum()
    taux_global = total_retards / total_vols * 100 if total_vols else 0.0

    if "taux_retard_percent" not in out.columns:
        out["taux_retard_percent"] = (out["nb_retards"] / out["nb_vols"] * 100).round(2)

    out["part_vols_reseau_percent"] = (out["nb_vols"] / total_vols * 100).round(2)
    out["part_retards_reseau_percent"] = (
        out["nb_retards"] / total_retards * 100
    ).round(2)
    out["indice_vs_taille_flotte"] = (
        out["part_retards_reseau_percent"] / out["part_vols_reseau_percent"]
    ).round(2)
    out["ecart_vs_reseau_pts"] = (out["taux_retard_percent"] - taux_global).round(2)
    return out.sort_values("taux_retard_percent", ascending=False).reset_index(drop=True)
