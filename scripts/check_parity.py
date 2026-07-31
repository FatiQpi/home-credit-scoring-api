"""
Test de parite entrainement / inference.

Verifie que build_features() reconstruit, depuis un identifiant et les champs
de la demande, le vecteur que l'entrainement avait produit pour ce client.
Egalite des probabilites = pas de training-serving skew sur ce chemin.

Usage :
    python scripts/check_parity.py                  # 3 clients au hasard
    python scripts/check_parity.py 100024 100471    # identifiants imposes
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent

# src/ n'est pas un paquet : ajout au chemin d'import.
sys.path.insert(0, str(RACINE / "src"))
from features import build_features, load_artifacts, load_store  # noqa: E402

# Chemin machine-specifique, identique a scripts/build_store.py.
CSV = Path(
    "/Users/fatih/Desktop/AI ENGINEER/"
    "Projet_6_initiez-vous_au_MLOps/home-credit-risk"
) / "data" / "processed" / "train_featured.csv"

N_CLIENTS = 3

artifacts = load_artifacts()
store = load_store(RACINE / "data" / "store.parquet")
with open(RACINE / "artifacts" / "model.pkl", "rb") as f:
    model = pickle.load(f)

feature_names = artifacts["feature_names"]

# Reference : preparation du notebook 03. Aucun nettoyage a rejouer,
# train_featured.csv sort deja de clean_application. reduce_memory() non plus :
# l'API travaillera en float64.
train = pd.read_csv(CSV)
X = train.drop(columns=["TARGET", "SK_ID_CURR"])
cat_cols = X.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")
X.index = train["SK_ID_CURR"]

# Un decalage de modalites fausse la prediction sans lever d'erreur, et reste
# invisible chez un client dont la valeur n'est pas concernee. D'ou une
# verification sur l'ensemble des colonnes plutot que sur les clients testes.
deduites = {col: list(X[col].cat.categories) for col in cat_cols}
assert deduites == artifacts["categories"], "modalites divergentes"

# Le payload ne porte que les champs de la demande : privees des 25 features
# d'historique, leur seule source possible est le magasin, donc il est teste.
colonnes_app = [c for c in feature_names if c not in store.columns]
print(f"{len(colonnes_app)} champs dans le payload, "
      f"{store.shape[1]} colonnes dans le magasin")

cibles = [int(v) for v in sys.argv[1:]] or list(
    X.index.intersection(store.index).to_series().sample(
        N_CLIENTS, random_state=42
    )
)

for cid in cibles:
    reference = X.loc[[cid], feature_names]

    # .item() ramene les scalaires numpy a des types Python, comme le ferait
    # un corps JSON decode.
    payload = {"SK_ID_CURR": int(cid)}
    payload.update({
        col: (val.item() if hasattr(val, "item") else val)
        for col, val in reference.iloc[0][colonnes_app].items()
    })

    construit = build_features(payload, store, artifacts)

    p_ref = model.predict_proba(reference)[0, 1]
    p_api = model.predict_proba(construit)[0, 1]
    accord = p_ref == p_api

    print(f"{cid} : reference {p_ref:.12f} | api {p_api:.12f} | "
          f"{'OK' if accord else 'ECHEC'}")

    if not accord:
        # Libelles plutot que codes : l'assertion ci-dessus garantit qu'un
        # libelle identique implique un code identique.
        for col in feature_names:
            a, b = reference[col].iloc[0], construit[col].iloc[0]
            if a != b and not (pd.isna(a) and pd.isna(b)):
                origine = "historique" if col in store.columns else "application"
                print(f"   {col} ({origine}) : {a!r} -> {b!r}")
