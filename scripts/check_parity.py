"""
Test de parite entrainement / inference.

Verifie que le vecteur lu dans le magasin est celui que l'entrainement avait
produit pour le meme client. Egalite stricte des probabilites = pas de
training-serving skew.

Deux passages :
  - global, sur tous les clients : valide l'aller-retour Parquet (valeurs,
    types, codes categoriels, ordre des colonnes) ;
  - par identifiant : valide le chemin reellement emprunte par l'API.

Usage :
    python scripts/check_parity.py                  # les deux passages
    python scripts/check_parity.py --rapide         # sans le passage global
    python scripts/check_parity.py 100002 100024    # identifiants imposes
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from src.features import (  # noqa: E402
    get_features,
    load_artifacts,
    load_model,
    load_store,
)

# Chemin machine-specifique, identique a scripts/build_store.py.
CSV = Path(
    "/Users/fatih/Desktop/AI ENGINEER/"
    "Projet_6_initiez-vous_au_MLOps/home-credit-risk"
) / "data" / "processed" / "train_featured.csv"

N_CLIENTS = 3


def diagnostic(reference: pd.Series, construit: pd.Series) -> None:
    """Affiche les colonnes qui different entre deux vecteurs d'un client."""
    for col in reference.index:
        a, b = reference[col], construit[col]
        if a != b and not (pd.isna(a) and pd.isna(b)):
            print(f"      {col} : {a!r} -> {b!r}")


def main() -> None:
    rapide = "--rapide" in sys.argv
    cibles = [int(v) for v in sys.argv[1:] if not v.startswith("-")]

    artifacts = load_artifacts()
    feature_names = artifacts["feature_names"]
    store = load_store(feature_names)
    model = load_model()

    # Reference : preparation du notebook 03. Aucun nettoyage a rejouer,
    # train_featured.csv sort deja de clean_application. reduce_memory() non
    # plus : le magasin est en float64.
    train = pd.read_csv(CSV)
    X = train.drop(columns=["TARGET", "SK_ID_CURR"])
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category")
    X.index = train["SK_ID_CURR"]
    del train

    # Un decalage de modalites fausse la prediction sans lever d'erreur, et
    # reste invisible chez un client dont la valeur n'est pas concernee. D'ou
    # une verification sur toutes les colonnes plutot que sur les clients
    # testes.
    deduites = {col: list(X[col].cat.categories) for col in cat_cols}
    assert deduites == artifacts["categories"], "modalites divergentes"

    X = X.loc[store.index, feature_names]

    if not rapide:
        p_ref = model.predict_proba(X)[:, 1]
        p_store = model.predict_proba(store)[:, 1]
        ecarts = np.flatnonzero(p_ref != p_store)

        print(f"global : {len(store)} clients, {len(ecarts)} ecart(s)")

        for pos in ecarts[:3]:
            cid = store.index[pos]
            print(f"   {cid} : {p_ref[pos]:.12f} != {p_store[pos]:.12f}")
            diagnostic(X.iloc[pos], store.iloc[pos])

        del p_ref, p_store

    cibles = cibles or [
        int(v) for v in store.index.to_series().sample(N_CLIENTS, random_state=42)
    ]

    for cid in cibles:
        reference = X.loc[[cid]]
        construit = get_features(cid, store)

        p_ref = model.predict_proba(reference)[0, 1]
        p_api = model.predict_proba(construit)[0, 1]
        accord = p_ref == p_api

        print(
            f"{cid} : reference {p_ref:.12f} | api {p_api:.12f} | "
            f"{'OK' if accord else 'ECHEC'}"
        )

        if not accord:
            diagnostic(reference.iloc[0], construit.iloc[0])


if __name__ == "__main__":
    main()
