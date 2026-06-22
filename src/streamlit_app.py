from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_metrics import (
    build_airline_delay,
    build_route_delay,
    label_for,
    plotly_axis_labels,
    recalculate_airline_shares,
    rename_for_display,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "dashboard"

# Palette bleue du dashboard
COLOR_PRIMARY = "#2563eb"
COLOR_LIGHT = "#93c5fd"
COLOR_DARK = "#1e40af"
COLOR_MUTED = "#64748b"
COLOR_SCALE = "Blues"


st.set_page_config(page_title="Dashboard retards de vols", layout="wide")


REQUIRED_FILES = {
    "sample": "flights_dashboard.csv",
    "kpi": "kpi_global.csv",
    "monthly": "monthly_delay.csv",
    "day_period": "day_period_delay.csv",
    "weekday_hour": "weekday_hour_delay.csv",
    "month_hour": "month_hour_delay.csv",
    "delay_level": "delay_level_distribution.csv",
    "airline": "airline_delay.csv",
    "airport": "airport_delay.csv",
    "airport_map": "airport_map.csv",
    "destination_airport": "destination_airport_delay.csv",
    "route": "route_delay.csv",
    "risk": "risk_situations.csv",
    "impact": "impact_situations.csv",
    "model_metrics": "model_metrics.csv",
    "model_confusion": "model_confusion_matrix.csv",
    "feature_importance": "model_feature_importance.csv",
    "propagation_by_seq": "propagation_by_seq.csv",
    "propagation_conditional": "propagation_conditional.csv",
    "propagation_turnaround": "propagation_turnaround.csv",
}


def enrich_airline_delay(airline_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les colonnes equitable si l'export CSV est ancien."""
    airline = airline_df.copy()
    required = {
        "part_vols_reseau_percent",
        "part_retards_reseau_percent",
        "indice_vs_taille_flotte",
        "ecart_vs_reseau_pts",
    }
    if required.issubset(airline.columns):
        return airline

    if "taux_retard_percent" not in airline.columns:
        airline["taux_retard_percent"] = (
            airline["nb_retards"] / airline["nb_vols"] * 100
        ).round(2)

    return recalculate_airline_shares(airline)


@st.cache_data
def load_dashboard_data(_cache_version: int = 3):
    missing = [
        file_name for file_name in REQUIRED_FILES.values()
        if not (DATA_DIR / file_name).exists()
    ]
    if missing:
        st.error(
            "Fichiers dashboard manquants : "
            + ", ".join(missing)
            + ". Regenerez les exports du dossier data/dashboard."
        )
        st.stop()

    loaded = {
        name: pd.read_csv(DATA_DIR / file_name)
        for name, file_name in REQUIRED_FILES.items()
    }
    loaded["airline"] = enrich_airline_delay(loaded["airline"])
    return loaded


def selected_filter(label, values):
    values = sorted(pd.Series(values).dropna().astype(str).unique())
    return st.sidebar.multiselect(label, values)


def filter_dashboard(df):
    st.sidebar.header("Filtres")
    st.sidebar.caption("Laisser vide pour conserver toutes les valeurs.")

    selected_airlines = selected_filter("Compagnies", df["AIRLINE"])
    selected_months = st.sidebar.multiselect(
        "Mois", sorted(df["MONTH"].dropna().unique())
    )
    selected_origins = selected_filter("Aeroports de depart", df["ORIGIN_AIRPORT"])
    selected_destinations = selected_filter(
        "Aeroports d'arrivee", df["DESTINATION_AIRPORT"]
    )
    selected_weekdays = st.sidebar.multiselect(
        "Jours de semaine", sorted(df["DAY_OF_WEEK"].dropna().unique())
    )
    selected_periods = selected_filter("Periodes de journee", df["DAY_PERIOD"])

    filtered = df.copy()
    if selected_airlines:
        filtered = filtered[filtered["AIRLINE"].astype(str).isin(selected_airlines)]
    if selected_months:
        filtered = filtered[filtered["MONTH"].isin(selected_months)]
    if selected_origins:
        filtered = filtered[
            filtered["ORIGIN_AIRPORT"].astype(str).isin(selected_origins)
        ]
    if selected_destinations:
        filtered = filtered[
            filtered["DESTINATION_AIRPORT"].astype(str).isin(selected_destinations)
        ]
    if selected_weekdays:
        filtered = filtered[filtered["DAY_OF_WEEK"].isin(selected_weekdays)]
    if selected_periods:
        filtered = filtered[filtered["DAY_PERIOD"].astype(str).isin(selected_periods)]

    return filtered


def compute_kpis(df):
    if df.empty:
        return {
            "total": 0,
            "delay_rate": 0.0,
            "arrival_delay": 0.0,
            "departure_delay": 0.0,
        }
    return {
        "total": len(df),
        "delay_rate": df["IS_DELAYED"].mean() * 100,
        "arrival_delay": df["ARRIVAL_DELAY_FILLED"].mean(),
        "departure_delay": df["DEPARTURE_DELAY_FILLED"].mean(),
    }


def metric_from_kpi(kpi_df, indicator):
    row = kpi_df.loc[kpi_df["indicateur"] == indicator, "valeur"]
    return float(row.iloc[0]) if not row.empty else 0.0


def heatmap_chart(df, index, columns, values, title, x_title, y_title):
    pivot = df.pivot(index=index, columns=columns, values=values).sort_index()
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=COLOR_SCALE,
        labels={
            "x": x_title,
            "y": y_title,
            "color": label_for(values),
        },
        title=title,
    )
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return fig


def show_table(df: pd.DataFrame, **kwargs) -> None:
    """Tableau avec libellés de colonnes lisibles."""
    st.dataframe(rename_for_display(df), **kwargs)


def _cond_row(conditional, condition, met, target):
    mask = (
        (conditional["condition"] == condition)
        & (conditional["condition_met"] == met)
        & (conditional["target"] == target)
    )
    if not mask.any():
        return None
    return conditional.loc[mask].iloc[0]


def render_propagation_page(propagation_data):
    """Onglet 5 : lecture simple de la propagation des retards par avion."""
    by_seq = propagation_data["propagation_by_seq"]
    conditional = propagation_data["propagation_conditional"]
    turnaround = propagation_data["propagation_turnaround"]

    st.markdown(
        """
        ### L'idée en une phrase
        Un **même avion** enchaîne plusieurs vols dans la journée. On vérifie si un retard
        sur le **premier vol** ou sur le **vol précédent** augmente le risque pour la suite.
        """
    )
    st.info(
        "**Retard** = arrivée ou départ avec au moins **15 minutes** de retard · "
        "**Même avion** = code d'immatriculation (`TAIL_NUMBER`) · Données **US 2015**"
    )

    # --- 1. Question principale ---
    st.markdown("---")
    st.markdown("#### 1. Si le premier vol du matin est en retard, et le suivant ?")

    row_on_time = _cond_row(
        conditional,
        "Retard arrivee 1er vol",
        "non",
        "Retard arrivee vol suivant",
    )
    row_late = _cond_row(
        conditional,
        "Retard arrivee 1er vol",
        "oui",
        "Retard arrivee vol suivant",
    )

    if row_on_time is not None and row_late is not None:
        p_ok = float(row_on_time["prob_percent"])
        p_late = float(row_late["prob_percent"])
        baseline = float(row_on_time["baseline_percent"])
        ecart = round(p_late - p_ok, 2)

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Premier vol à l'heure",
            f"{p_ok:.1f} %",
            help=f"Sur {int(row_on_time['nb_vols']):,} vols suivants observés",
        )
        c2.metric(
            "Premier vol en retard",
            f"{p_late:.1f} %",
            delta=f"+{ecart:.1f} pts vs à l'heure",
            delta_color="inverse",
            help=f"Sur {int(row_late['nb_vols']):,} vols suivants observés",
        )
        c3.metric(
            "Réseau (référence)",
            f"{baseline:.1f} %",
            help="Taux moyen de retard sur tous les vols suivants dans la journée",
        )

        fig_compare = px.bar(
            pd.DataFrame(
                {
                    "Situation du 1er vol": [
                        "1er vol à l'heure",
                        "1er vol en retard",
                        "Moyenne réseau",
                    ],
                    "Risque de retard au vol suivant (%)": [p_ok, p_late, baseline],
                }
            ),
            x="Situation du 1er vol",
            y="Risque de retard au vol suivant (%)",
            color="Situation du 1er vol",
            text="Risque de retard au vol suivant (%)",
            color_discrete_map={
                "1er vol à l'heure": COLOR_LIGHT,
                "1er vol en retard": COLOR_DARK,
                "Moyenne réseau": COLOR_MUTED,
            },
        )
        fig_compare.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig_compare.update_layout(showlegend=False, yaxis_title="Part des vols en retard (%)")
        st.plotly_chart(fig_compare)

        if ecart >= 2:
            st.warning(
                f"Écart notable : **+{ecart:.1f} points** de risque sur le vol suivant "
                "quand le premier vol est déjà en retard."
            )
        else:
            st.success(
                f"L'écart reste modéré (**{ecart:+.1f} points**). La propagation existe "
                "mais d'autres facteurs (météo, hub saturé) jouent aussi un rôle."
            )

    # --- 2. Au fil de la journée ---
    st.markdown("---")
    st.markdown("#### 2. Comment évolue le risque au fil de la journée ?")

    st.markdown(
        """
        On suit le **même avion** vol après vol. Le **1er vol** n'est pas affiché ici :
        s'il était « à l'heure », son taux est mécaniquement 0 % dans ce sous-ensemble ;
        s'il était « en retard », il est à 100 %. L'intérêt est surtout sur le **2e vol et suivants**.
        """
    )

    rank_labels = {
        1: "1er vol",
        2: "2e vol",
        3: "3e vol",
        4: "4e vol",
        5: "5e vol",
        6: "6e vol",
    }
    rank_order = [rank_labels[i] for i in range(2, 7)]

    seq_evolution = by_seq[by_seq["flight_seq"].between(2, 6)].copy()
    seq_evolution["Rang dans la journée"] = seq_evolution["flight_seq"].map(rank_labels)
    seq_evolution["Rang dans la journée"] = pd.Categorical(
        seq_evolution["Rang dans la journée"],
        categories=rank_order,
        ordered=True,
    )
    baseline_arr = float(seq_evolution["baseline_arrivee_percent"].iloc[0])

    vol2 = seq_evolution[seq_evolution["flight_seq"] == 2]
    if len(vol2) == 2:
        v2_ok = float(
            vol2.loc[vol2["first_flight_delayed"] == 0, "taux_retard_arrivee_percent"].iloc[0]
        )
        v2_late = float(
            vol2.loc[vol2["first_flight_delayed"] == 1, "taux_retard_arrivee_percent"].iloc[0]
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("2e vol — 1er était à l'heure", f"{v2_ok:.1f} %")
        c2.metric(
            "2e vol — 1er était en retard",
            f"{v2_late:.1f} %",
            delta=f"{v2_late - v2_ok:+.1f} pts",
            delta_color="inverse",
        )
        c3.metric("Référence réseau", f"{baseline_arr:.1f} %")

    view_mode = st.radio(
        "Type de graphique",
        ["Courbes (évolution)", "Barres groupées (comparaison)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "Courbes (évolution)":
        fig_day = px.line(
            seq_evolution,
            x="Rang dans la journée",
            y="taux_retard_arrivee_percent",
            color="first_flight_delayed_label",
            markers=True,
            line_shape="spline",
            title="Évolution du taux de retard à l'arrivée (2e → 6e vol)",
            labels=plotly_axis_labels(),
            color_discrete_map={
                "Premier vol a l'heure": COLOR_LIGHT,
                "Premier vol en retard": COLOR_DARK,
            },
        )
    else:
        fig_day = px.bar(
            seq_evolution,
            x="Rang dans la journée",
            y="taux_retard_arrivee_percent",
            color="first_flight_delayed_label",
            barmode="group",
            text="taux_retard_arrivee_percent",
            title="Comparaison par rang dans la journée (2e → 6e vol)",
            labels=plotly_axis_labels(),
            color_discrete_map={
                "Premier vol a l'heure": COLOR_LIGHT,
                "Premier vol en retard": COLOR_DARK,
            },
        )
        fig_day.update_traces(texttemplate="%{y:.1f}%", textposition="outside")

    fig_day.add_hline(
        y=baseline_arr,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Réseau : {baseline_arr:.1f} %",
    )
    fig_day.update_layout(
        legend=dict(title=""),
        xaxis_title="Rang du vol dans la journée (même avion)",
        yaxis_title="Part des vols en retard à l'arrivée (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_day)

    # Lecture automatique sur le 2e vol
    if len(vol2) == 2:
        ecart_vol2 = round(v2_late - v2_ok, 2)
        if ecart_vol2 > 1:
            st.warning(
                f"**2e vol** : après un 1er vol en retard, le taux monte à **{v2_late:.1f} %** "
                f"(vs **{v2_ok:.1f} %** si le 1er était à l'heure). "
                f"Les vols **3e à 6e** restent proches du réseau (~{baseline_arr:.0f} %)."
            )
        else:
            st.info(
                f"**2e vol** : écart faible (**{ecart_vol2:+.1f} pts**). "
                "Après le 2e vol, les deux courbes suivent surtout la **moyenne du réseau**."
            )

    with st.expander("Voir aussi le 1er vol (effet de définition)"):
        seq_first = by_seq[by_seq["flight_seq"] == 1].copy()
        seq_first["Rang"] = "1er vol"
        show_table(
            seq_first[
                [
                    "first_flight_delayed_label",
                    "taux_retard_arrivee_percent",
                    "nb_vols",
                ]
            ],
            hide_index=True,
        )
        st.caption(
            "100 % ou 0 % ici : ce n'est pas une surprise opérationnelle, "
            "c'est la façon dont on a découpé les journées."
        )

    # --- 3. À l'escale ---
    st.markdown("---")
    st.markdown("#### 3. Au prochain départ, sur le même aéroport (escale)")

    st.markdown(
        """
        Quand l'avion **atterrit en retard** sur un aéroport et **repart du même lieu**
        (escale / rotation), le **départ du vol suivant** est-il plus souvent en retard
        qu'à l'habitude sur cet aéroport ?
        """
    )

    min_turn_vol = st.slider(
        "Fiabilité minimale (nombre de vols observés par aéroport)",
        min_value=50,
        max_value=2000,
        value=200,
        step=50,
    )
    turn = turnaround[turnaround["nb_vols"] >= min_turn_vol].copy()

    if turn.empty:
        st.warning("Pas assez de données pour ce seuil. Baissez le curseur.")
    else:
        turn["Situation"] = turn["prev_arr_delayed"].map(
            {
                0: "Vol précédent à l'heure",
                1: "Vol précédent en retard",
            }
        )
        airports_with_both = turn.groupby("ORIGIN_AIRPORT")["prev_arr_delayed"].nunique()
        airports_with_both = airports_with_both[airports_with_both >= 2].index.tolist()

        if airports_with_both:
            turn_compare = turn[turn["ORIGIN_AIRPORT"].isin(airports_with_both)]
            turn_compare["Ecart vs habituel (pts)"] = (
                turn_compare["taux_retard_depart_percent"]
                - turn_compare["baseline_depart_percent"]
            ).round(2)
            top_hubs = (
                turn_compare[turn_compare["prev_arr_delayed"] == 1]
                .sort_values("Ecart vs habituel (pts)", ascending=False)
                .head(8)["ORIGIN_AIRPORT"]
            )
            turn_top = turn_compare[turn_compare["ORIGIN_AIRPORT"].isin(top_hubs)]

            fig_turn = px.bar(
                turn_top,
                x="ORIGIN_AIRPORT",
                y="taux_retard_depart_percent",
                color="Situation",
                barmode="group",
                text="taux_retard_depart_percent",
                title="Retard au départ : après escale à l'heure vs en retard",
                labels=plotly_axis_labels(),
                color_discrete_map={
                    "Vol précédent à l'heure": COLOR_LIGHT,
                    "Vol précédent en retard": COLOR_DARK,
                },
            )
            fig_turn.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            st.plotly_chart(fig_turn)

            best = turn_compare[turn_compare["prev_arr_delayed"] == 1].sort_values(
                "Ecart vs habituel (pts)", ascending=False
            ).iloc[0]
            st.info(
                f"Exemple : à **{best['ORIGIN_AIRPORT']}**, après un vol en retard, "
                f"**{best['taux_retard_depart_percent']:.1f} %** des départs suivants sont en retard "
                f"(habituellement **{best['baseline_depart_percent']:.1f} %**, "
                f"soit **+{best['Ecart vs habituel (pts)']:.1f} points**)."
            )
        else:
            fig_simple = px.bar(
                turn.sort_values("lift", ascending=False).head(10),
                x="ORIGIN_AIRPORT",
                y="taux_retard_depart_percent",
                color="Situation",
                title="Taux de retard au départ après escale",
                labels=plotly_axis_labels(),
            )
            st.plotly_chart(fig_simple)
            st.caption(
                "Peu d'aéroports ont les deux situations dans l'échantillon : "
                "régénérez les exports avec le notebook sur le fichier complet."
            )

    # --- Synthèse ---
    st.markdown("---")
    st.markdown("#### À retenir pour les opérations")
    st.markdown(
        """
        | Signal | Action possible |
        |--------|-----------------|
        | 1er vol de l'avion déjà en retard | Surveiller les **2e et 3e vols** du même appareil |
        | Hub avec fort écart après escale | Renforcer les créneaux de **turnaround** sur cet aéroport |
        | Taux proche de la moyenne réseau | Le retard peut venir du **réseau global**, pas seulement de l'avion |
        """
    )

    with st.expander("Voir les tableaux détaillés (données brutes)"):
        st.markdown("**Probabilités conditionnelles**")
        cond_display = conditional.copy()
        cond_display["condition_met"] = cond_display["condition_met"].map(
            {"non": "Non", "oui": "Oui"}
        )
        show_table(cond_display, hide_index=True)
        st.markdown("**Par position dans la journée**")
        show_table(by_seq, hide_index=True)


def _enrich_route_table(route_df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute ROUTE_LABEL et ecart si l'export CSV est minimal."""
    out = route_df.copy()
    if "ROUTE_LABEL" not in out.columns:
        out["ROUTE_LABEL"] = (
            out["ORIGIN_AIRPORT"].astype(str)
            + " → "
            + out["DESTINATION_AIRPORT"].astype(str)
        )
    if "ecart_vs_reseau_pts" not in out.columns and "taux_retard_percent" in out.columns:
        taux_global = out["nb_retards"].sum() / out["nb_vols"].sum() * 100
        out["ecart_vs_reseau_pts"] = (out["taux_retard_percent"] - taux_global).round(2)
    return out


ROUTES_DISTANCE_BINS = [0, 500, 1000, 1500, 20_000]
ROUTES_DISTANCE_LABELS = [
    "Court (< 500 mi)",
    "Moyen (500–999 mi)",
    "Long (1 000–1 499 mi)",
    "Très long (≥ 1 500 mi)",
]


def _routes_by_distance_bin(routes: pd.DataFrame) -> pd.DataFrame:
    """Taux de retard pondéré par volume, par tranche de distance."""
    binned = routes.copy()
    binned["distance_tranche"] = pd.cut(
        binned["distance_moyenne"],
        bins=ROUTES_DISTANCE_BINS,
        labels=ROUTES_DISTANCE_LABELS,
        right=False,
    )
    agg = (
        binned.groupby("distance_tranche", observed=True)
        .agg(
            nb_retards=("nb_retards", "sum"),
            nb_vols=("nb_vols", "sum"),
            nb_routes=("ROUTE_LABEL", "count"),
        )
        .reset_index()
    )
    agg["taux_moyen"] = (agg["nb_retards"] / agg["nb_vols"] * 100).round(1)
    return agg


def _route_slider_bounds(filtered_df: pd.DataFrame, static_routes: pd.DataFrame) -> tuple[int, int]:
    """Min / max du curseur volume, arrondis par tranche de 20 vols."""
    step = 20
    if not filtered_df.empty:
        counts = filtered_df.groupby(
            ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], observed=True
        ).size()
        raw_max = int(counts.max()) if len(counts) else step
    elif not static_routes.empty:
        raw_max = int(static_routes["nb_vols"].max())
    else:
        raw_max = step
    max_value = max(step, ((raw_max + step - 1) // step) * step)
    return step, max_value


def render_routes_page(filtered_df: pd.DataFrame, static_routes: pd.DataFrame) -> None:
    """Analyse des liaisons depart → arrivee et tendance au retard."""
    st.subheader("Routes et tendance au retard")
    st.markdown(
        """
        Une **route** = couple **aéroport de départ → aéroport d'arrivée** (le sens compte :
        *LAX → SFO* n'est pas *SFO → LAX*).

        Sur l'échantillon, les taux varient fortement (**~7 % à ~40 %** selon la liaison),
        ce qui justifie une lecture par route et pas seulement par aéroport isolé.
        """
    )

    slider_min, slider_max = _route_slider_bounds(filtered_df, static_routes)
    default_min = min(500, slider_max)
    default_min = max(slider_min, (default_min // 20) * 20)

    min_vols = st.slider(
        "Volume minimum sur la route (fiabilité du taux)",
        min_value=slider_min,
        max_value=slider_max,
        value=default_min,
        step=20,
        help="Palier de 20 vols. Maximum = route la plus chargée dans les données affichées.",
    )

    if filtered_df.empty:
        st.warning("Aucun vol ne correspond aux filtres actuels.")
        routes = _enrich_route_table(static_routes)
        routes = routes[routes["nb_vols"] >= min_vols]
        st.caption("Données issues de l'export global (filtres sidebar ignorés).")
    else:
        routes = build_route_delay(filtered_df, min_vols=min_vols)
        st.caption("Taux recalculés sur l'échantillon filtré (sidebar).")

    if routes.empty:
        st.warning("Aucune route ne dépasse ce seuil de volume. Baissez le curseur.")
        return

    taux_global = routes["nb_retards"].sum() / routes["nb_vols"].sum() * 100
    spread = routes["taux_retard_percent"].max() - routes["taux_retard_percent"].min()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Routes retenues", f"{len(routes):,}")
    c2.metric("Taux moyen (routes)", f"{taux_global:.2f} %")
    c3.metric("Écart min / max", f"{spread:.1f} pts")
    c4.metric(
        "Route la plus exposée",
        routes.iloc[0]["ROUTE_LABEL"],
        help=f"{routes.iloc[0]['taux_retard_percent']:.1f} % de retards",
    )

    top_risk = routes.iloc[0]
    best = routes.sort_values("taux_retard_percent").iloc[0]
    top_impact = routes.sort_values("impact_score", ascending=False).iloc[0]

    if top_risk["ecart_vs_reseau_pts"] >= 5:
        st.warning(
            f"**{top_risk['ROUTE_LABEL']}** : **{top_risk['taux_retard_percent']:.1f} %** "
            f"de retards (**+{top_risk['ecart_vs_reseau_pts']:.1f} pts** vs la moyenne des routes affichées)."
        )
    else:
        st.info(
            f"Écart modéré sur la route la plus exposée (**{top_risk['ecart_vs_reseau_pts']:+.1f} pts**). "
            "Les hubs saturés et la météo locale expliquent aussi une partie des retards."
        )

    focus_origin = st.selectbox(
        "Filtrer les graphiques par aéroport de départ",
        ["Tous"] + sorted(routes["ORIGIN_AIRPORT"].astype(str).unique()),
    )
    plot_routes = routes
    if focus_origin != "Tous":
        plot_routes = routes[routes["ORIGIN_AIRPORT"].astype(str) == focus_origin]

    col_left, col_right = st.columns(2)
    with col_left:
        top_delay = plot_routes.sort_values("taux_retard_percent", ascending=False).head(12)
        fig_high = px.bar(
            top_delay,
            y="ROUTE_LABEL",
            x="taux_retard_percent",
            orientation="h",
            text="taux_retard_percent",
            title="Routes les plus touchées par les retards",
            labels=plotly_axis_labels(),
            color_discrete_sequence=[COLOR_DARK],
        )
        fig_high.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
        fig_high.update_layout(margin=dict(r=40))
        st.plotly_chart(fig_high)

    with col_right:
        top_impact_plot = plot_routes.sort_values("impact_score", ascending=False).head(12)
        fig_impact = px.bar(
            top_impact_plot,
            y="ROUTE_LABEL",
            x="impact_score",
            orientation="h",
            title="Impact opérationnel (retards × durée moyenne)",
            labels=plotly_axis_labels(),
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        st.plotly_chart(fig_impact)
        st.caption(
            f"Ex. volume : **{top_impact['ROUTE_LABEL']}** cumule le plus de minutes de retard "
            f"({top_impact['nb_retards']:,} retards, {top_impact['retard_moyen_arrivee']:.1f} min en moyenne)."
        )

    st.markdown("#### Distance et taux de retard")
    dist_agg = _routes_by_distance_bin(plot_routes)
    if dist_agg.empty:
        st.info("Pas assez de routes pour résumer par distance.")
    else:
        y_max = max(dist_agg["taux_moyen"].max() * 1.2, 22)
        fig_distance = px.bar(
            dist_agg,
            x="distance_tranche",
            y="taux_moyen",
            text="taux_moyen",
            custom_data=["nb_routes", "nb_vols"],
            title="Taux de retard selon la distance (pondéré par le volume de vols)",
            labels=plotly_axis_labels(),
            color="taux_moyen",
            color_continuous_scale=COLOR_SCALE,
            range_color=[dist_agg["taux_moyen"].min(), dist_agg["taux_moyen"].max()],
        )
        fig_distance.update_traces(
            texttemplate="%{y:.1f} %",
            textposition="outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Taux : %{y:.1f} %<br>"
                "Routes : %{customdata[0]}<br>"
                "Vols : %{customdata[1]:,}<extra></extra>"
            ),
        )
        fig_distance.update_layout(
            showlegend=False,
            yaxis_range=[0, y_max],
            xaxis_title="Tranche de distance",
            yaxis_title=label_for("taux_moyen"),
            coloraxis_showscale=False,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig_distance)
        st.caption(
            "Taux pondéré par le nombre de vols dans chaque tranche — plus lisible "
            "qu'un nuage de centaines de routes."
        )

    with st.expander("Routes les plus fiables (faible taux de retard)"):
        low = plot_routes.sort_values("taux_retard_percent").head(10)
        show_table(
            low[
                [
                    "ROUTE_LABEL",
                    "nb_vols",
                    "taux_retard_percent",
                    "ecart_vs_reseau_pts",
                    "distance_moyenne",
                ]
            ],
            hide_index=True,
        )
        st.caption(
            f"Meilleure liaison affichée : **{best['ROUTE_LABEL']}** "
            f"({best['taux_retard_percent']:.1f} %)."
        )

    st.markdown("#### Tableau détaillé")
    display_cols = [
        "ROUTE_LABEL",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "nb_vols",
        "nb_retards",
        "taux_retard_percent",
        "ecart_vs_reseau_pts",
        "retard_moyen_arrivee",
        "distance_moyenne",
        "impact_score",
    ]
    show_table(
        routes[display_cols],
        hide_index=True,
    )


data = load_dashboard_data()
dashboard_df = data["sample"]
filtered_df = filter_dashboard(dashboard_df)

st.title("Dashboard des retards de vols")
st.caption(
    "Vue globale exacte, exploration filtree sur echantillon et lecture decisionnelle."
)

tabs = st.tabs(
    [
        "1. Vue globale",
        "2. Temporalite",
        "3. Aeroports",
        "4. Compagnies",
        "5. Même avion",
        "6. Routes",
        "7. Decisions",
        "8. Modelisation",
        "9. Conclusion",
    ]
)

with tabs[0]:
    st.subheader("Vue globale exacte")
    kpi = data["kpi"]

    total = metric_from_kpi(kpi, "Nombre total de vols")
    delayed = metric_from_kpi(kpi, "Vols en retard")
    delay_rate = metric_from_kpi(kpi, "Taux de retard (%)")
    arrival_delay = metric_from_kpi(kpi, "Retard moyen arrivee (min)")
    departure_delay = metric_from_kpi(kpi, "Retard moyen depart (min)")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Vols", f"{total:,.0f}")
    col2.metric("Retards", f"{delayed:,.0f}")
    col3.metric("Taux retard", f"{delay_rate:.2f} %")
    col4.metric("Retard arrivee", f"{arrival_delay:.2f} min")
    col5.metric("Retard depart", f"{departure_delay:.2f} min")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                data["delay_level"],
                x="DELAY_LEVEL",
                y="nb_vols",
                text="part_percent",
                title="Repartition des niveaux de retard",
                labels=plotly_axis_labels(),
            ),
        )
    with right:
        st.plotly_chart(
            px.scatter(
                data["monthly"],
                x="nb_vols",
                y="taux_retard_percent",
                size="nb_retards",
                text="MONTH",
                title="Mois : volume de vols vs taux de retard",
                labels=plotly_axis_labels(),
            ),
        )

    st.subheader("Vue filtree sur l'echantillon interactif")
    sample_kpis = compute_kpis(filtered_df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vols filtres", f"{sample_kpis['total']:,}")
    col2.metric("Taux retard", f"{sample_kpis['delay_rate']:.2f} %")
    col3.metric("Retard arrivee", f"{sample_kpis['arrival_delay']:.2f} min")
    col4.metric("Retard depart", f"{sample_kpis['departure_delay']:.2f} min")

with tabs[1]:
    st.subheader("Temporalite des retards")
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            px.line(
                data["monthly"],
                x="MONTH",
                y=["taux_retard_percent", "retard_moyen_arrivee"],
                markers=True,
                title="Evolution mensuelle : taux et retard moyen",
                labels=plotly_axis_labels(),
            ),
        )
    with col_right:
        st.plotly_chart(
            px.bar(
                data["day_period"].sort_values("taux_retard_percent"),
                x="DAY_PERIOD",
                y="taux_retard_percent",
                text="nb_vols",
                title="Taux de retard par periode de journee",
                labels=plotly_axis_labels(),
            ),
        )

    st.plotly_chart(
        heatmap_chart(
            data["weekday_hour"],
            "DAY_OF_WEEK",
            "SCHEDULED_DEP_HOUR",
            "taux_retard_percent",
            "Heatmap jour de semaine x heure",
            "Heure de depart",
            "Jour de semaine",
        ),
    )
    st.plotly_chart(
        heatmap_chart(
            data["month_hour"],
            "MONTH",
            "SCHEDULED_DEP_HOUR",
            "taux_retard_percent",
            "Heatmap mois x heure",
            "Heure de depart",
            "Mois",
        ),
    )

with tabs[2]:
    st.subheader("Lecture aeroportuaire globale")
    airport_map = data["airport_map"].copy()
    airport_map["hover"] = (
        airport_map["ORIGIN_AIRPORT"].astype(str)
        + " - "
        + airport_map["AIRPORT_LABEL"].astype(str)
    )
    st.plotly_chart(
        px.scatter_map(
            airport_map,
            lat="LATITUDE",
            lon="LONGITUDE",
            size="nb_retards",
            color="taux_retard_percent",
            hover_name="hover",
            labels=plotly_axis_labels(),
            hover_data={
                "nb_vols": ":,",
                "nb_retards": ":,",
                "retard_moyen_arrivee": True,
                "LATITUDE": False,
                "LONGITUDE": False,
            },
            color_continuous_scale=COLOR_SCALE,
            zoom=2.7,
            height=520,
            title="Carte des aeroports de depart : volume et taux de retard",
        ),
    )

    min_volume = st.slider(
        "Volume minimum pour classer les aeroports",
        min_value=1_000,
        max_value=100_000,
        value=10_000,
        step=1_000,
    )
    airport_filtered = data["airport"][
        data["airport"]["nb_vols"] >= min_volume
    ].copy()

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            px.bar(
                airport_filtered.sort_values("nb_retards", ascending=False).head(15),
                y="ORIGIN_AIRPORT",
                x="nb_retards",
                orientation="h",
                title="Top aeroports par volume de retards",
                labels=plotly_axis_labels(),
            ),
        )
    with col_right:
        st.plotly_chart(
            px.bar(
                airport_filtered.sort_values("taux_retard_percent", ascending=False).head(15),
                y="ORIGIN_AIRPORT",
                x="taux_retard_percent",
                orientation="h",
                title="Top aeroports par taux de retard",
                labels=plotly_axis_labels(),
            ),
        )

    focus_airport = st.selectbox(
        "Focus aeroport de depart",
        sorted(data["airport"]["ORIGIN_AIRPORT"].astype(str).unique()),
    )
    focus_rows = filtered_df[
        filtered_df["ORIGIN_AIRPORT"].astype(str) == str(focus_airport)
    ]
    focus_kpis = compute_kpis(focus_rows)
    col1, col2, col3 = st.columns(3)
    col1.metric("Vols dans l'echantillon", f"{focus_kpis['total']:,}")
    col2.metric("Taux retard", f"{focus_kpis['delay_rate']:.2f} %")
    col3.metric("Retard moyen", f"{focus_kpis['arrival_delay']:.2f} min")

    st.subheader("Destinations les plus touchees")
    show_table(
        data["destination_airport"]
        .sort_values("nb_retards", ascending=False)
        .head(20),
    )
    st.caption(
        "Pour comparer les **liaisons depart → arrivee** (taux, ecart, impact), "
        "voir l'onglet **6. Routes**."
    )

with tabs[3]:
    st.subheader("Compagnies aeriennes")
    st.markdown(
        """
        **Taux propre** = retards / vols totaux de la compagnie (equitable, ne depend pas du volume des autres).

        Utilisez le curseur **Top N** : en retirant des compagnies, les **parts** et l'**indice** sont
        **recalcules** entre celles qui restent — les graphiques bougent vraiment.
        """
    )

    if filtered_df.empty:
        st.warning("Aucun vol ne correspond aux filtres actuels.")
    else:
        label_map = data["airline"][["AIRLINE", "AIRLINE_LABEL"]].drop_duplicates()

        def _with_labels(table):
            if table.empty:
                return table
            out = table.drop(columns=["AIRLINE_NAME", "AIRLINE_LABEL"], errors="ignore").merge(
                label_map, on="AIRLINE", how="left"
            )
            out["AIRLINE_LABEL"] = out["AIRLINE_LABEL"].fillna(out["AIRLINE"])
            return out

        airline_all = _with_labels(build_airline_delay(filtered_df, min_vols=1))
        n_companies = len(airline_all)

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            n_top = st.slider(
                "Nombre de compagnies affichees (top par volume)",
                min_value=3,
                max_value=n_companies,
                value=n_companies,
                help="Reduire N retire les petits operateurs et recalcule les parts entre les restants.",
            )
        with col_ctrl2:
            metric_choice = st.selectbox(
                "Metrique des barres (reactif au Top N)",
                [
                    "Part des retards (%)",
                    "Part des vols (%)",
                    "Taux propre (%)",
                    "Indice vs taille",
                    "Ecart vs moyenne du groupe (pts)",
                ],
            )

        metric_col_map = {
            "Part des retards (%)": "part_retards_reseau_percent",
            "Part des vols (%)": "part_vols_reseau_percent",
            "Taux propre (%)": "taux_retard_percent",
            "Indice vs taille": "indice_vs_taille_flotte",
            "Ecart vs moyenne du groupe (pts)": "ecart_vs_reseau_pts",
        }
        metric_col = metric_col_map[metric_choice]

        airline_f = recalculate_airline_shares(
            airline_all.sort_values("nb_vols", ascending=False).head(n_top)
        )

        if airline_f.empty:
            st.warning("Aucune compagnie a afficher.")
        else:
            total_v = int(airline_f["nb_vols"].sum())
            total_r = int(airline_f["nb_retards"].sum())
            taux_global = total_r / total_v * 100 if total_v else 0.0
            n_exclues = n_companies - len(airline_f)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Compagnies", f"{len(airline_f)} / {n_companies}")
            c2.metric("Vols (sous-ensemble)", f"{total_v:,}")
            c3.metric("Taux global", f"{taux_global:.2f} %")
            c4.metric(
                "Max part retards",
                f"{airline_f['part_retards_reseau_percent'].max():.1f} %",
                help="Part des retards du reseau parmi les compagnies affichees — change avec Top N.",
            )

            if n_exclues > 0:
                masquees = airline_all[
                    ~airline_all["AIRLINE"].isin(airline_f["AIRLINE"])
                ]["AIRLINE_LABEL"].tolist()
                st.caption(f"Hors affichage : {', '.join(masquees)}")

            col_left, col_right = st.columns(2)
            with col_left:
                plot_df = airline_f.sort_values(metric_col, ascending=True).tail(12)
                fig_rate = px.bar(
                    plot_df,
                    y="AIRLINE_LABEL",
                    x=metric_col,
                    orientation="h",
                    text=metric_col,
                    title=metric_choice,
                    labels=plotly_axis_labels(),
                    color_discrete_sequence=[COLOR_PRIMARY],
                )
                fig_rate.update_traces(texttemplate="%{x:.2f}", textposition="outside")
                if metric_col == "taux_retard_percent":
                    fig_rate.add_vline(
                        x=taux_global,
                        line_dash="dash",
                        line_color=COLOR_MUTED,
                        annotation_text=f"Moy. {taux_global:.1f}%",
                    )
                if metric_col == "indice_vs_taille_flotte":
                    fig_rate.add_vline(x=1.0, line_dash="dash", line_color=COLOR_MUTED)
                st.plotly_chart(fig_rate)
                if metric_choice == "Taux propre (%)":
                    st.caption(
                        "Le taux propre change peu pour une meme compagnie ; passez a "
                        "**Part des retards** pour voir l'effet du Top N."
                    )

            with col_right:
                fig_scatter = px.scatter(
                    airline_f,
                    x="part_vols_reseau_percent",
                    y="part_retards_reseau_percent",
                    size="nb_vols",
                    color="taux_retard_percent",
                    hover_name="AIRLINE_LABEL",
                    title="Parts vols vs retards (recalculees sur le Top N)",
                    labels=plotly_axis_labels(),
                    color_continuous_scale=COLOR_SCALE,
                )
                max_axis = max(
                    airline_f["part_vols_reseau_percent"].max(),
                    airline_f["part_retards_reseau_percent"].max(),
                    1,
                )
                fig_scatter.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=max_axis,
                    y1=max_axis,
                    line=dict(color=COLOR_MUTED, dash="dash"),
                )
                st.plotly_chart(fig_scatter)

            display_cols = [
                "AIRLINE_LABEL",
                "nb_vols",
                "nb_retards",
                "taux_retard_percent",
                "part_vols_reseau_percent",
                "part_retards_reseau_percent",
                "indice_vs_taille_flotte",
                "ecart_vs_reseau_pts",
            ]
            show_table(
                airline_f[display_cols],
                hide_index=True,
            )

with tabs[4]:
    st.subheader("Propagation des retards — même avion")
    render_propagation_page(data)


with tabs[5]:
    render_routes_page(filtered_df, data["route"])

with tabs[6]:
    st.subheader("Situations a risque et impact operationnel")
    col_left, col_right = st.columns(2)
    with col_left:
        st.write("Risque eleve : taux de retard fort avec volume suffisant.")
        show_table(data["risk"].head(25))
    with col_right:
        st.write("Impact eleve : volume de retards et minutes perdues.")
        show_table(data["impact"].head(25))

    st.plotly_chart(
        px.bar(
            data["risk"].head(15),
            y="ORIGIN_AIRPORT",
            x="risk_score",
            color="AIRLINE",
            orientation="h",
            title="Top situations par score de risque",
            labels=plotly_axis_labels(),
        ),
    )

    st.subheader("Recommandations automatiques")
    top_risk = data["risk"].iloc[0]
    top_impact = data["impact"].iloc[0]
    st.markdown(
        f"""
        - Surveiller en priorite les departs autour de **{int(top_risk['SCHEDULED_DEP_HOUR'])}h**
          pour la compagnie **{top_risk['AIRLINE']}** depuis **{top_risk['ORIGIN_AIRPORT']}**.
        - Prioriser les plans d'action sur **{top_impact['ORIGIN_AIRPORT']}**, car l'impact operationnel y est le plus fort.
        - Piloter les decisions avec deux angles : le **taux** pour reperer le risque, et le **volume** pour mesurer l'impact.
        """
    )

with tabs[7]:
    st.subheader("Modelisation predictive")
    show_table(data["model_metrics"])

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            px.bar(
                data["model_metrics"],
                x="modele",
                y=["recall_retard", "balanced_accuracy", "roc_auc"],
                barmode="group",
                title="Comparaison des metriques modele",
                labels=plotly_axis_labels(),
            ),
        )
    with col_right:
        confusion_pivot = data["model_confusion"].pivot(
            index="reel", columns="predit", values="nombre"
        )
        st.plotly_chart(
            px.imshow(
                confusion_pivot,
                text_auto=True,
                color_continuous_scale="Blues",
                title="Matrice de confusion du modele retenu",
                labels={
                    "x": label_for("predit"),
                    "y": label_for("reel"),
                    "color": label_for("nombre"),
                },
            ),
        )

    st.plotly_chart(
        px.bar(
            data["feature_importance"].sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title="Facteurs les plus importants",
            labels=plotly_axis_labels(),
        ),
    )

    st.markdown(
        """
        Le modele sert surtout de systeme d'alerte avant depart. Dans ce contexte,
        manquer un vrai retard est souvent plus problematique que declencher une alerte en trop.
        Le recall et la balanced accuracy sont donc plus utiles que l'accuracy seule.
        """
    )

with tabs[8]:
    st.subheader("Conclusion generale")
    st.markdown(
        """
        Le projet montre que les retards de vols ne sont pas repartis au hasard.
        Ils dependent fortement du moment du vol, de l'aeroport, de la compagnie
        et du volume d'activite. L'analyse de propagation par avion (`TAIL_NUMBER`)
        confirme qu'un retard sur le premier vol de la journee augmente le risque
        sur les rotations suivantes et sur le depart au prochain aeroport.
        """
    )

    st.subheader("Insights principaux")
    st.markdown(
        """
        - Le taux global de retard est de **18,28 %**.
        - Le mois de **juin** est le plus expose, avec environ **22,97 %** de retards.
        - Les departs autour de **20h** sont les plus sensibles.
        - **Southwest Airlines Co.** concentre le plus grand nombre de retards en volume.
        - **ORD** est l'aeroport qui concentre le plus de retards en volume.
        - La propagation avion : surveiller les appareils deja en retard le matin.
        - Certaines **routes** (ex. hubs vers lies eloignes) depassent **30 %** de retards quand d'autres restent sous **10 %**.
        """
    )

    st.subheader("Limites")
    st.markdown(
        """
        - Les lignes sans `ARRIVAL_DELAY` ont ete reintegrees comme des vols sans retard.
        - Certaines variables utiles manquent (meteo, congestion temps reel).
        - Le modele reste une premiere approche interpretable avant depart.
        - Les vues interactives utilisent un echantillon ; les agregats sont exacts.
        """
    )

    st.subheader("Recommandations")
    st.markdown(
        """
        - Surveiller les vols de fin de journee et les avions en retard des la premiere rotation.
        - Prioriser les aeroports et compagnies a fort taux **et** fort volume.
        - Utiliser les heatmaps, l'onglet **Routes** et la propagation pour anticiper l'effet domino.
        - Utiliser le modele comme aide a la decision, pas comme decision automatique.
        """
    )

    st.subheader("Recommandation modele")
    st.markdown(
        """
        Le modele retenu est le **Random Forest**, avec un focus sur le **recall**
        et la **balanced accuracy** pour limiter les vrais retards non detectes.
        """
    )
