import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from supabase import create_client

RACINE = Path(__file__).resolve().parent.parent
CAMPAGNES = RACINE / "data" / "campagnes"
TABLE = "journal_predictions"
TAILLE_PAGE = 1000
BORDS_SCORE = np.linspace(0, 1, 21)

load_dotenv(RACINE / ".env")
st.set_page_config(page_title="Monitoring — scoring crédit", layout="wide")


# Streamlit relance tout le script a chaque interaction : sans cache, chaque clic
# relirait la table entiere.
@st.cache_data(ttl=300)
def lire_journal() -> pd.DataFrame:
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    lignes, depart = [], 0
    while True:
        page = (
            client.table(TABLE)
            .select("*")
            .order("id")
            .range(depart, depart + TAILLE_PAGE - 1)
            .execute()
            .data
        )
        lignes += page
        if len(page) < TAILLE_PAGE:
            break
        depart += TAILLE_PAGE

    journal = pd.DataFrame(lignes)
    if not journal.empty:
        # PostgreSQL omet les microsecondes nulles : le format varie d'une ligne a l'autre.
        journal["horodatage"] = pd.to_datetime(journal["horodatage"], utc=True, format="ISO8601")
    return journal


def lister_periodes(journal: pd.DataFrame) -> dict:
    periodes = {"Toute la table": (journal["horodatage"].min(), journal["horodatage"].max())}
    # Les manifestes delimitent les campagnes simulees. Une heure sans fuseau est lue
    # en UTC, comme le fait Supabase quand le notebook filtre avec ces bornes.
    for fichier in sorted(CAMPAGNES.glob("*.json")):
        manifeste = json.loads(fichier.read_text())
        periodes[f"Campagne {manifeste['campagne']}"] = (
            pd.to_datetime(manifeste["debut"], utc=True),
            pd.to_datetime(manifeste["fin"], utc=True),
        )
    return periodes


def filtrer(journal: pd.DataFrame, bornes: tuple) -> pd.DataFrame:
    debut, fin = bornes
    return journal[journal["horodatage"].between(debut, fin)]


def proportions_par_classe(scores: pd.Series) -> np.ndarray:
    # Des proportions et non des effectifs : les deux periodes n'ont pas le meme volume.
    effectifs, _ = np.histogram(scores.dropna(), bins=BORDS_SCORE)
    return effectifs / max(effectifs.sum(), 1)


def mesurer_derive_score(courant: pd.DataFrame, reference: pd.DataFrame) -> dict:
    schema = DataDefinition(numerical_columns=["score_risque"])
    # Ordre impose par Evidently 0.7 : courant d'abord, reference ensuite.
    resultat = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(courant[["score_risque"]].dropna().reset_index(drop=True), data_definition=schema),
        Dataset.from_pandas(reference[["score_risque"]].dropna().reset_index(drop=True), data_definition=schema),
    )
    for m in resultat.dict()["metrics"]:
        if not m["config"]["type"].endswith("ValueDrift"):
            continue
        methode, score, seuil = m["config"]["method"], m["value"], m["config"]["threshold"]
        # Une p-value signale une derive quand elle est petite, une distance quand elle est grande.
        derive = score <= seuil if "p_value" in methode else score >= seuil
        return {"methode": methode, "score": score, "seuil": seuil, "derive": derive}


journal = lire_journal()
if journal.empty:
    st.warning("Le journal est vide.")
    st.stop()

periodes = lister_periodes(journal)
noms = list(periodes)


def position(nom: str) -> int:
    return noms.index(nom) if nom in noms else 0


st.sidebar.header("Périodes")
nom_courant = st.sidebar.selectbox("Période surveillée", noms, index=position("Campagne derive"))
nom_reference = st.sidebar.selectbox("Période de référence", noms, index=position("Campagne reference"))

st.sidebar.header("Alerte")
# Valeur provisoire : aucun engagement de temps de reponse n'existe encore.
seuil_latence = st.sidebar.number_input("Latence anormale au-delà de (ms)", min_value=1.0, value=50.0, step=5.0)

if st.sidebar.button("Relire Supabase"):
    lire_journal.clear()
    st.rerun()

courant = filtrer(journal, periodes[nom_courant])
reference = filtrer(journal, periodes[nom_reference])
servis = courant[courant["statut"] == 200]

debut, fin = periodes[nom_courant]
st.title("Monitoring — API de scoring crédit")
st.caption(f"{nom_courant} : du {debut:%d/%m/%Y %H:%M} au {fin:%d/%m/%Y %H:%M} (UTC)")

if courant.empty:
    st.info("Aucun appel sur cette période.")
    st.stop()

colonnes = st.columns(6)
colonnes[0].metric("Appels", len(courant))
colonnes[1].metric("Taux d'erreur", f"{100 * (courant['statut'] != 200).mean():.1f} %")
colonnes[2].metric("Taux de refus", f"{100 * (servis['decision'] == 'REFUSE').mean():.1f} %")
colonnes[3].metric("Latence médiane", f"{servis['duree_totale_ms'].median():.1f} ms")
colonnes[4].metric("Latence p95", f"{servis['duree_totale_ms'].quantile(0.95):.1f} ms")
colonnes[5].metric("Inférence médiane", f"{servis['duree_inference_ms'].median():.1f} ms")

st.subheader("Distribution des scores")
if nom_courant == nom_reference:
    st.info("Choisir deux périodes différentes pour les comparer.")
elif reference["score_risque"].dropna().empty:
    st.info("Aucun score sur la période de référence.")
else:
    st.line_chart(
        pd.DataFrame(
            {
                "référence": proportions_par_classe(reference["score_risque"]),
                "surveillée": proportions_par_classe(courant["score_risque"]),
            },
            index=BORDS_SCORE[:-1].round(2),
        )
    )
    derive = mesurer_derive_score(courant, reference)
    mesures = st.columns(3)
    mesures[0].metric("Dérive des scores", "détectée" if derive["derive"] else "non détectée")
    mesures[1].metric("Score de dérive", f"{derive['score']:.3g}")
    mesures[2].metric("Seuil", derive["seuil"])
    st.caption(f"Méthode choisie par Evidently : {derive['methode']}")

st.subheader("Latence")
# Vega dessine dans le fuseau et le format du navigateur : l'échelle utc et %H:%M
# alignent l'axe sur les heures du journal.
st.altair_chart(
    alt.Chart(servis)
    .mark_line()
    .encode(
        x=alt.X(
            "horodatage:T",
            title="Heure (UTC)",
            scale=alt.Scale(type="utc"),
            axis=alt.Axis(format="%H:%M"),
        ),
        y=alt.Y("duree_totale_ms:Q", title="Durée totale (ms)"),
    )
)
anormaux = servis[servis["duree_totale_ms"] > seuil_latence]
st.write(f"**{len(anormaux)}** appel(s) au-delà de {seuil_latence:g} ms.")
if not anormaux.empty:
    st.dataframe(
        anormaux[["horodatage", "sk_id_curr", "duree_inference_ms", "duree_totale_ms"]],
        hide_index=True,
    )

st.subheader("Statuts HTTP")
st.dataframe(courant["statut"].value_counts().rename("appels"))
