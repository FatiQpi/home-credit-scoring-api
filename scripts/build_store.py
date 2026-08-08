"""
Construction du magasin de features.

Ecrit une fois pour tous les clients les 145 features attendues par le modele.
L'API lit ensuite une ligne entiere par son identifiant.

Usage :
    python scripts/build_store.py
"""

import json
from pathlib import Path

import pandas as pd

# Chemin machine-specifique : les CSV sources restent hors depot.
SOURCE = Path(
    "/Users/fatih/Desktop/AI ENGINEER/"
    "Projet_6_initiez-vous_au_MLOps/home-credit-risk"
) / "data" / "processed" / "train_featured.csv"

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "data" / "store.parquet"


def main() -> None:
    with open(RACINE / "artifacts" / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(RACINE / "artifacts" / "categories.json") as f:
        categories = json.load(f)

    # usecols echoue si une feature du contrat manque au CSV. Un magasin
    # partiel serait inexploitable : l'API n'a aucune autre source.
    store = pd.read_csv(SOURCE, usecols=["SK_ID_CURR"] + feature_names)

    if store["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR duplique : l'index doit etre unique.")

    # Codes entiers plutot que chaines Python : le magasin reste charge en
    # memoire pendant toute la vie de l'API.
    for col in categories:
        store[col] = store[col].astype("category")

    # read_csv rend les colonnes dans l'ordre du fichier, pas dans celui de
    # usecols : l'ordre du contrat est impose ici, une fois pour toutes.
    store = store.set_index("SK_ID_CURR").sort_index()[feature_names]

    # Parquet plutot que CSV : les types sont ecrits dans le fichier et relus
    # tels quels, dtype categoriel compris.
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(SORTIE, compression="snappy")

    taille = SORTIE.stat().st_size / 1e6
    print(f"-> {SORTIE}")
    print(f"   {store.shape[0]} clients x {store.shape[1]} colonnes, {taille:.1f} Mo")


if __name__ == "__main__":
    main()
