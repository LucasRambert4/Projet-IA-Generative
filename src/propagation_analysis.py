"""Analyse de propagation des retards par avion (TAIL_NUMBER) et par escale."""

from __future__ import annotations

import pandas as pd

DELAY_THRESHOLD_MIN = 15
DAY_KEY = ["TAIL_NUMBER", "YEAR", "MONTH", "DAY"]


def _prepare_flights(source_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "TAIL_NUMBER",
        "YEAR",
        "MONTH",
        "DAY",
        "SCHEDULED_DEPARTURE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "CANCELLED",
        "DIVERTED",
    }
    missing = required - set(source_df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour l'analyse propagation : {sorted(missing)}")

    work = source_df.copy()
    work = work[(work["CANCELLED"] == 0) & (work["DIVERTED"] == 0)]
    work = work[work["TAIL_NUMBER"].notna()].copy()

    if "ARRIVAL_DELAY_FILLED" not in work.columns:
        work["ARRIVAL_DELAY_FILLED"] = work["ARRIVAL_DELAY"].fillna(0)
    if "DEPARTURE_DELAY_FILLED" not in work.columns:
        work["DEPARTURE_DELAY_FILLED"] = work["DEPARTURE_DELAY"].fillna(0)

    work["IS_DELAYED"] = (work["ARRIVAL_DELAY_FILLED"] >= DELAY_THRESHOLD_MIN).astype(int)
    work["IS_DEP_DELAYED"] = (work["DEPARTURE_DELAY_FILLED"] >= DELAY_THRESHOLD_MIN).astype(int)

    work = work.sort_values(DAY_KEY + ["SCHEDULED_DEPARTURE"])
    work["flight_seq"] = work.groupby(DAY_KEY, observed=True).cumcount() + 1
    return work


def _rate_table(grouped: pd.DataFrame, arr_col: str, dep_col: str) -> pd.DataFrame:
    out = grouped.copy()
    out["taux_retard_arrivee_percent"] = (
        out[arr_col] / out["nb_vols"] * 100
    ).round(2)
    out["taux_retard_depart_percent"] = (
        out[dep_col] / out["nb_vols"] * 100
    ).round(2)
    return out


def build_propagation_by_sequence(source_df: pd.DataFrame) -> pd.DataFrame:
    """Taux de retard selon la position du vol dans la journée (1er, 2e, ...)."""
    work = _prepare_flights(source_df)
    multi = work.groupby(DAY_KEY, observed=True).filter(lambda g: len(g) >= 2)

    global_arr = work["IS_DELAYED"].mean() * 100
    global_dep = work["IS_DEP_DELAYED"].mean() * 100

    rows = []
    for first_delayed in [0, 1]:
        first_days = multi.loc[multi["flight_seq"] == 1, DAY_KEY + ["IS_DELAYED"]]
        first_days = first_days.rename(columns={"IS_DELAYED": "first_arr_delayed"})
        subset = multi.merge(first_days, on=DAY_KEY, how="inner")
        subset = subset[subset["first_arr_delayed"] == first_delayed]

        grouped = (
            subset.groupby("flight_seq", observed=True)
            .agg(
                nb_vols=("IS_DELAYED", "size"),
                nb_retards_arrivee=("IS_DELAYED", "sum"),
                nb_retards_depart=("IS_DEP_DELAYED", "sum"),
            )
            .reset_index()
        )
        grouped["first_flight_delayed"] = first_delayed
        grouped["first_flight_delayed_label"] = (
            "Premier vol en retard" if first_delayed else "Premier vol a l'heure"
        )
        grouped = _rate_table(grouped, "nb_retards_arrivee", "nb_retards_depart")
        grouped["baseline_arrivee_percent"] = round(global_arr, 2)
        grouped["baseline_depart_percent"] = round(global_dep, 2)
        rows.append(grouped)

    result = pd.concat(rows, ignore_index=True)
    result = result[result["flight_seq"] <= 8]
    return result.sort_values(["first_flight_delayed", "flight_seq"]).reset_index(drop=True)


def build_propagation_conditional(source_df: pd.DataFrame) -> pd.DataFrame:
    """Probabilités conditionnelles : 1er vol retardé -> vols suivants retardés."""
    work = _prepare_flights(source_df)
    multi = work.groupby(DAY_KEY, observed=True).filter(lambda g: len(g) >= 2)

    first = multi.loc[multi["flight_seq"] == 1, DAY_KEY + ["IS_DELAYED", "IS_DEP_DELAYED"]]
    first = first.rename(
        columns={"IS_DELAYED": "first_arr_delayed", "IS_DEP_DELAYED": "first_dep_delayed"}
    )
    chain = multi.loc[multi["flight_seq"] > 1].merge(first, on=DAY_KEY, how="inner")

    baseline = {
        ("arrivee", "vol_suivant"): chain["IS_DELAYED"].mean() * 100,
        ("depart", "vol_suivant"): chain["IS_DEP_DELAYED"].mean() * 100,
    }

    scenarios = [
        ("first_arr_delayed", "Retard arrivee 1er vol", "IS_DELAYED", "Retard arrivee vol suivant", "arrivee"),
        ("first_arr_delayed", "Retard arrivee 1er vol", "IS_DEP_DELAYED", "Retard depart vol suivant", "depart"),
        ("first_dep_delayed", "Retard depart 1er vol", "IS_DELAYED", "Retard arrivee vol suivant", "arrivee"),
        ("first_dep_delayed", "Retard depart 1er vol", "IS_DEP_DELAYED", "Retard depart vol suivant", "depart"),
    ]

    rows = []
    for cond_col, cond_label, target_col, target_label, baseline_key in scenarios:
        for cond_val in [0, 1]:
            subset = chain[chain[cond_col] == cond_val]
            if subset.empty:
                continue
            prob = subset[target_col].mean() * 100
            base = baseline[(baseline_key, "vol_suivant")]
            rows.append(
                {
                    "condition": cond_label,
                    "condition_met": "oui" if cond_val else "non",
                    "target": target_label,
                    "nb_vols": len(subset),
                    "prob_percent": round(prob, 2),
                    "baseline_percent": round(base, 2),
                    "lift": round(prob / base, 2) if base else None,
                }
            )

    return pd.DataFrame(rows)


def build_propagation_turnaround(source_df: pd.DataFrame, min_volume: int = 200) -> pd.DataFrame:
    """Propagation au prochain vol quand l'avion revient sur le même aéroport (escale)."""
    work = _prepare_flights(source_df)
    multi = work.groupby(DAY_KEY, observed=True).filter(lambda g: len(g) >= 2)
    ordered = multi.sort_values(DAY_KEY + ["SCHEDULED_DEPARTURE"])

    ordered["prev_arr_delayed"] = ordered.groupby(DAY_KEY, observed=True)["IS_DELAYED"].shift(1)
    ordered["prev_dest"] = ordered.groupby(DAY_KEY, observed=True)["DESTINATION_AIRPORT"].shift(1)
    turn = ordered.dropna(subset=["prev_arr_delayed", "prev_dest"]).copy()
    turn["same_hub_turnaround"] = turn["prev_dest"] == turn["ORIGIN_AIRPORT"]
    turn = turn[turn["same_hub_turnaround"]]

    airport_baseline = (
        work.groupby("ORIGIN_AIRPORT", observed=True)["IS_DEP_DELAYED"]
        .mean()
        .mul(100)
        .round(2)
        .rename("baseline_depart_percent")
        .reset_index()
    )

    grouped = (
        turn.groupby(["ORIGIN_AIRPORT", "prev_arr_delayed"], observed=True)
        .agg(nb_vols=("IS_DEP_DELAYED", "size"), nb_retards_depart=("IS_DEP_DELAYED", "sum"))
        .reset_index()
    )
    grouped["taux_retard_depart_percent"] = (
        grouped["nb_retards_depart"] / grouped["nb_vols"] * 100
    ).round(2)
    grouped["prev_arrival_delayed_label"] = grouped["prev_arr_delayed"].map(
        {0: "Vol precedent a l'heure", 1: "Vol precedent en retard"}
    )
    grouped = grouped.merge(airport_baseline, on="ORIGIN_AIRPORT", how="left")
    grouped["lift"] = (
        grouped["taux_retard_depart_percent"] / grouped["baseline_depart_percent"]
    ).round(2)
    grouped = grouped[grouped["nb_vols"] >= min_volume]
    return grouped.sort_values("lift", ascending=False).reset_index(drop=True)


def build_propagation_kpi(
    by_seq: pd.DataFrame,
    conditional: pd.DataFrame,
    turnaround: pd.DataFrame,
) -> pd.DataFrame:
    """KPIs synthétiques pour le dashboard."""
    rows = []

    if not by_seq.empty:
        seq_all = by_seq[by_seq["first_flight_delayed"] == 0]
        if not seq_all.empty:
            rows.append(
                {
                    "indicateur": "Taux retard vol 1 (reference)",
                    "valeur": float(seq_all.loc[seq_all["flight_seq"] == 1, "taux_retard_arrivee_percent"].iloc[0]),
                }
            )
        seq_late = by_seq[by_seq["first_flight_delayed"] == 1]
        if len(seq_late) >= 2:
            first_late = float(
                seq_late.loc[seq_late["flight_seq"] == 1, "taux_retard_arrivee_percent"].iloc[0]
            )
            second_after_late = float(
                seq_late.loc[seq_late["flight_seq"] == 2, "taux_retard_arrivee_percent"].iloc[0]
            )
            rows.extend(
                [
                    {"indicateur": "Taux retard vol 2 si 1er vol en retard (%)", "valeur": second_after_late},
                    {
                        "indicateur": "Ecart vol 2 vs vol 1 si 1er retard (pts)",
                        "valeur": round(second_after_late - first_late, 2),
                    },
                ]
            )

    if not conditional.empty:
        mask = (
            (conditional["condition"] == "Retard arrivee 1er vol")
            & (conditional["condition_met"] == "oui")
            & (conditional["target"] == "Retard arrivee vol suivant")
        )
        if mask.any():
            row = conditional.loc[mask].iloc[0]
            rows.append(
                {
                    "indicateur": "P(retard arrivee suivant | 1er vol retard) (%)",
                    "valeur": float(row["prob_percent"]),
                }
            )
            rows.append(
                {
                    "indicateur": "Lift propagation 1er vol -> suivant",
                    "valeur": float(row["lift"]) if pd.notna(row["lift"]) else 0.0,
                }
            )

    if not turnaround.empty:
        top = turnaround.iloc[0]
        rows.append(
            {
                "indicateur": "Hub max lift turnaround",
                "valeur": str(top["ORIGIN_AIRPORT"]),
            }
        )
        rows.append(
            {
                "indicateur": "Lift max turnaround",
                "valeur": float(top["lift"]),
            }
        )

    return pd.DataFrame(rows)


def build_propagation_tables(source_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Construit toutes les tables d'export propagation."""
    by_seq = build_propagation_by_sequence(source_df)
    conditional = build_propagation_conditional(source_df)
    turnaround = build_propagation_turnaround(source_df)
    kpi = build_propagation_kpi(by_seq, conditional, turnaround)
    return {
        "propagation_by_seq": by_seq,
        "propagation_conditional": conditional,
        "propagation_turnaround": turnaround,
        "propagation_kpi": kpi,
    }


def export_propagation_tables(source_df: pd.DataFrame, output_dir) -> dict[str, pd.DataFrame]:
    """Calcule et exporte les CSV propagation dans output_dir."""
    from pathlib import Path

    tables = build_propagation_tables(source_df)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    return tables
