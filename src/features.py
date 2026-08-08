"""
Acces au modele et au magasin de features.

Le magasin contient les 145 features deja construites pour chaque client.
L'inference se reduit donc a une lecture par identifiant : aucune
transformation n'est rejouee ici, ce qui supprime tout risque de divergence
avec l'entrainement.
"""

import json
import pickle
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = RACINE / "artifacts"
STORE_PATH = RACINE / "data" / "store.parquet"

# Seuil auquel le modele retenu a ete evalue.
DECISION_THRESHOLD = 0.50


class ClientNotFoundError(Exception):
    """Identifiant absent du magasin."""


def load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    """
    Charge le contrat d'entree du modele.

    A appeler une seule fois au demarrage, jamais dans une requete.

    Returns:
        dict avec 'feature_names' (145 noms ordonnes) et 'categories'
        (16 colonnes -> modalites ordonnees).
    """
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


def load_model(artifacts_dir: Path = ARTIFACTS_DIR):
    """Charge le modele serialise. A appeler une seule fois au demarrage."""
    with open(artifacts_dir / "model.pkl", "rb") as f:
        return pickle.load(f)


def load_store(feature_names: list[str], path: Path = STORE_PATH) -> pd.DataFrame:
    """
    Charge le magasin de features, indexe sur SK_ID_CURR.

    Args:
        feature_names: colonnes attendues, dans l'ordre du contrat
        path:          chemin du fichier Parquet

    Raises:
        ValueError: magasin non conforme au contrat du modele
    """
    store = pd.read_parquet(path)

    # Un magasin construit avec un autre feature_names.json produirait des
    # predictions fausses sans lever d'erreur. Echouer au demarrage plutot
    # que servir un score errone.
    if list(store.columns) != feature_names:
        manquantes = sorted(set(feature_names) - set(store.columns))
        surplus = sorted(set(store.columns) - set(feature_names))
        raise ValueError(
            f"Magasin non conforme au contrat du modele. "
            f"Manquantes : {manquantes or 'aucune'}. "
            f"En trop : {surplus or 'aucune'}. "
            f"Sinon, l'ordre des colonnes differe. "
            f"Reconstruire avec scripts/build_store.py."
        )

    if store.index.name != "SK_ID_CURR":
        raise ValueError("Le magasin doit etre indexe sur SK_ID_CURR.")

    return store


def get_features(client_id: int, store: pd.DataFrame) -> pd.DataFrame:
    """
    Lit le vecteur de features d'un client.

    Returns:
        DataFrame de 1 ligne x 145 colonnes, ordonne et type.

    Raises:
        ClientNotFoundError: identifiant absent du magasin
    """
    if client_id not in store.index:
        raise ClientNotFoundError(f"Aucune donnee pour le client {client_id}.")

    # Crochets doubles : predict_proba attend un DataFrame, pas une Series.
    return store.loc[[client_id]]


def predict(model, features: pd.DataFrame, seuil: float = DECISION_THRESHOLD) -> dict:
    """
    Applique le modele au vecteur lu.

    Le modele a ete entraine avec is_unbalance=True : sa sortie est exprimee
    sur une population reponderee, pas sur la population reelle. Elle ordonne
    correctement les clients mais ne s'interprete pas comme une frequence de
    defaut, d'ou 'score_risque' plutot que 'probabilite'.
    """
    proba = float(model.predict_proba(features)[0, 1])

    return {
        "score_risque": round(proba, 4),
        "decision": "REFUSE" if proba >= seuil else "ACCORDE",
        "seuil": seuil,
    }
