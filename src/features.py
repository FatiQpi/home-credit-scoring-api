"""
Construction du vecteur de features pour l'inference.

Fait le pont entre une demande de credit recue par l'API et le vecteur de
145 colonnes attendu par le modele. Le resultat doit etre identique a ce
qu'aurait produit le pipeline d'entrainement pour ce meme client.

Voir docs/features_explained.md pour le detail du raisonnement.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Repris a l'identique de utils.clean_application() : applique a
# l'entrainement, donc obligatoire a l'inference.
CAPS = {
    "AMT_INCOME_TOTAL": 5_000_000,
    "AMT_REQ_CREDIT_BUREAU_QRT": 10,
    "OBS_30_CNT_SOCIAL_CIRCLE": 50,
    "DEF_30_CNT_SOCIAL_CIRCLE": 50,
    "OBS_60_CNT_SOCIAL_CIRCLE": 50,
    "DEF_60_CNT_SOCIAL_CIRCLE": 50,
}

# Valeur aberrante (positive, isolee, ~1000 ans) parmi des durees negatives.
# Remplacee par NaN a l'entrainement.
DAYS_EMPLOYED_SENTINEL = 365243

# Seuil auquel le modele retenu a ete evalue.
DECISION_THRESHOLD = 0.50


class ClientNotFoundError(Exception):
    """Aucun historique disponible pour cet identifiant client."""


class InvalidClientDataError(Exception):
    """Donnees du dossier invalides ou inexploitables."""


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    """
    Charge le contrat d'entree du modele.

    A appeler une seule fois au demarrage, jamais dans une requete.

    Returns:
        dict avec 'feature_names' (145 noms ordonnes) et 'categories'
        (16 colonnes -> modalites ordonnees).
    """
    artifacts_dir = Path(artifacts_dir)

    with open(artifacts_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(artifacts_dir / "categories.json") as f:
        categories = json.load(f)

    inconnues = set(categories) - set(feature_names)
    if inconnues:
        raise ValueError(
            f"categories.json reference des colonnes absentes de "
            f"feature_names.json : {sorted(inconnues)}"
        )

    return {"feature_names": feature_names, "categories": categories}


def load_store(path) -> pd.DataFrame:
    """Charge le magasin d'historique, indexe sur SK_ID_CURR."""
    store = pd.read_parquet(path)

    if store.index.name != "SK_ID_CURR":
        if "SK_ID_CURR" not in store.columns:
            raise ValueError("Le magasin doit contenir une colonne SK_ID_CURR.")
        store = store.set_index("SK_ID_CURR")

    return store


def build_features(
    payload: dict,
    store: pd.DataFrame,
    artifacts: dict,
) -> pd.DataFrame:
    """
    Transforme une demande de credit en vecteur pret pour le modele.

    Args:
        payload:   champs du dossier client, dont SK_ID_CURR (~120 features)
        store:     magasin d'historique indexe sur SK_ID_CURR (~25 features)
        artifacts: sortie de load_artifacts()

    Returns:
        DataFrame de 1 ligne x 145 colonnes, ordonne et type.

    Raises:
        ClientNotFoundError:    identifiant absent du magasin
        InvalidClientDataError: donnees inexploitables
    """
    feature_names = artifacts["feature_names"]
    categories = artifacts["categories"]

    if "SK_ID_CURR" not in payload:
        raise InvalidClientDataError("Champ obligatoire manquant : SK_ID_CURR")

    try:
        client_id = int(payload["SK_ID_CURR"])
    except (TypeError, ValueError):
        raise InvalidClientDataError(
            f"SK_ID_CURR doit etre un entier, recu : {payload['SK_ID_CURR']!r}"
        )

    if client_id not in store.index:
        raise ClientNotFoundError(
            f"Aucun historique pour le client {client_id}. "
            f"Un nouveau client necessite un traitement specifique."
        )

    historique = store.loc[client_id]

    if isinstance(historique, pd.DataFrame):
        raise InvalidClientDataError(
            f"Le magasin contient {len(historique)} lignes pour le client "
            f"{client_id}. L'index doit etre unique."
        )

    # L'historique ecrase le payload en cas de collision : ces colonnes sont
    # produites par le batch, elles ne doivent pas etre surchargeables par
    # une requete HTTP.
    ligne = {k: v for k, v in payload.items() if k != "SK_ID_CURR"}
    ligne.update(historique.to_dict())

    df = pd.DataFrame([ligne])

    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
            DAYS_EMPLOYED_SENTINEL, np.nan
        )

    for col, plafond in CAPS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(upper=plafond)

    colonnes_ignorees = set(df.columns) - set(feature_names)

    # Garde le connu, complete en NaN, jette l'inconnu, impose l'ordre.
    # LightGBM traite les NaN nativement : un dossier partiel est acceptable.
    df = df.reindex(columns=feature_names)

    # LightGBM encode les categorielles en entiers. Sur une ligne unique,
    # astype('category') deduirait les modalites de cette seule valeur et
    # decalerait les codes -- prediction fausse, sans erreur. D'ou les
    # modalites imposees depuis l'artefact.
    for col, modalites in categories.items():
        valeur = df[col].iloc[0]

        # Modalites ecartees a l'entrainement (CODE_GENDER='XNA',
        # NAME_FAMILY_STATUS='Unknown') : le modele ne les a jamais vues.
        if pd.notna(valeur) and valeur not in modalites:
            raise InvalidClientDataError(
                f"Valeur inconnue pour {col} : {valeur!r}. "
                f"Attendu parmi : {modalites}"
            )

        df[col] = pd.Categorical(df[col], categories=modalites)

    # errors='raise' : un champ texte dans une colonne numerique doit
    # provoquer un rejet, pas un NaN silencieux.
    for col in df.columns:
        if col in categories:
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors="raise").astype("float64")
        except (ValueError, TypeError):
            raise InvalidClientDataError(
                f"Valeur non numerique pour {col} : {df[col].iloc[0]!r}"
            )

    assert df.shape == (1, len(feature_names)), (
        f"Attendu (1, {len(feature_names)}), obtenu {df.shape}"
    )
    assert list(df.columns) == feature_names, "Ordre des colonnes incorrect"

    df.attrs["colonnes_ignorees"] = sorted(colonnes_ignorees)

    return df


def predict(model, features: pd.DataFrame, seuil: float = DECISION_THRESHOLD) -> dict:
    """
    Applique le modele au vecteur construit.

    Le modele a ete entraine avec is_unbalance=True : les probabilites ne
    sont pas calibrees. 'probabilite' vaut ici score de risque, pas
    frequence attendue de defaut.
    """
    proba = float(model.predict_proba(features)[0, 1])

    return {
        "probabilite_defaut": round(proba, 4),
        "decision": "REFUSE" if proba >= seuil else "ACCORDE",
        "seuil": seuil,
    }
