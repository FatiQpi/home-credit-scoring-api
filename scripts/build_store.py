"""
Construction du magasin d'historique (feature store).

Calcule une fois pour tous les clients les 25 features d'historique, trop
couteuses a produire par requete. L'API lit ensuite une ligne par son
identifiant.

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
) / "data" / "processed"

RACINE = Path(__file__).resolve().parent.parent
SORTIE = RACINE / "data" / "store.parquet"


def main() -> None:
    with open(RACINE / "artifacts" / "feature_names.json") as f:
        feature_names = json.load(f)

    # Une feature absente d'application vient forcement de l'agregation des
    # tables annexes : elle n'est pas recalculable a la volee.
    colonnes_app = pd.read_csv(
        SOURCE / "application_train_cleaned.csv", nrows=5
    ).columns

    hist = [c for c in feature_names if c not in colonnes_app]
    print(f"Features d'historique : {len(hist)}")

    # Les 120 features d'application ne sont PAS stockees : elles seraient
    # perimees. Le montant demande aujourd'hui arrive par la requete.
    # usecols evite de charger 198 Mo pour n'en garder qu'une fraction.
    store = pd.read_csv(
        SOURCE / "train_featured.csv",
        usecols=["SK_ID_CURR"] + hist,
    )
    print(f"Charge : {store.shape[0]} clients x {store.shape[1]} colonnes")

    if store["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR duplique : l'index doit etre unique.")

    # Index unique : recherche en O(1) cote API.
    store = store.set_index("SK_ID_CURR").sort_index()

    # Parquet plutot que CSV : les types sont ecrits dans le fichier, donc
    # relus tels quels. Un CSV les ferait redeviner a chaque chargement.
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    store.to_parquet(SORTIE, compression="snappy")

    taille = SORTIE.stat().st_size / 1e6
    print(f"\n-> {SORTIE}")
    print(f"   {taille:.1f} Mo pour {len(store)} clients")


if __name__ == "__main__":
    main()
