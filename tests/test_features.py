"""Tests de la couche metier : artefacts, magasin, lecture, decision."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ClientNotFoundError,
    get_features,
    load_store,
    predict,
)


def ecrire_magasin(chemin, colonnes, nom_index="SK_ID_CURR"):
    """Ecrit un magasin minimal pour tester les rejets de load_store."""
    df = pd.DataFrame({col: [0.0] for col in colonnes})
    df.index = pd.Index([1], name=nom_index)
    df.to_parquet(chemin)
    return chemin


class ModeleFixe:
    """Modele de substitution : renvoie une probabilite imposee."""

    def __init__(self, proba):
        self.proba = proba

    def predict_proba(self, features):
        return np.array([[1 - self.proba, self.proba]])


# --- Artefacts ---


def test_artefacts_contrat(artifacts):
    assert len(artifacts["feature_names"]) == 145
    assert len(artifacts["categories"]) == 16
    assert set(artifacts["categories"]) <= set(artifacts["feature_names"])


# --- Magasin ---


def test_magasin_conforme(store, artifacts):
    assert store.shape == (307505, 145)
    assert store.index.name == "SK_ID_CURR"
    assert list(store.columns) == artifacts["feature_names"]
    assert store.index.is_unique


def test_magasin_dtypes_categoriels(store, artifacts):
    for col in artifacts["categories"]:
        assert str(store[col].dtype) == "category"
        assert list(store[col].cat.categories) == artifacts["categories"][col]


def test_load_store_rejette_colonnes_manquantes(tmp_path, artifacts):
    chemin = ecrire_magasin(tmp_path / "partiel.parquet", artifacts["feature_names"][:10])

    with pytest.raises(ValueError, match="Manquantes"):
        load_store(artifacts["feature_names"], path=chemin)


def test_load_store_rejette_mauvais_ordre(tmp_path, artifacts):
    # Memes colonnes, ordre inverse : LightGBM identifie les features par
    # position, ce cas produirait des scores faux sans lever d'erreur.
    permute = list(reversed(artifacts["feature_names"]))
    chemin = ecrire_magasin(tmp_path / "permute.parquet", permute)

    with pytest.raises(ValueError):
        load_store(artifacts["feature_names"], path=chemin)


def test_load_store_rejette_mauvais_index(tmp_path, artifacts):
    chemin = ecrire_magasin(
        tmp_path / "index.parquet", artifacts["feature_names"], nom_index="AUTRE"
    )

    with pytest.raises(ValueError, match="SK_ID_CURR"):
        load_store(artifacts["feature_names"], path=chemin)


# --- Lecture d'un client ---


def test_get_features_forme(store, artifacts, client_id):
    ligne = get_features(client_id, store)

    assert ligne.shape == (1, 145)
    assert list(ligne.columns) == artifacts["feature_names"]
    assert ligne.index[0] == client_id


def test_get_features_valeurs(store, client_id):
    ligne = get_features(client_id, store)

    pd.testing.assert_frame_equal(ligne, store.loc[[client_id]])


def test_get_features_client_absent(store):
    with pytest.raises(ClientNotFoundError):
        get_features(-1, store)


def test_get_features_accepte_les_nan(store, artifacts):
    # 93,7 % des clients ont un historique incomplet. Les NaN sont natifs
    # pour LightGBM, pas une anomalie a corriger.
    ligne = get_features(int(store.index[0]), store)

    assert ligne.shape == (1, 145)


# --- Decision ---


def test_predict_structure(model, store, client_id):
    resultat = predict(model, get_features(client_id, store))

    assert set(resultat) == {"score_risque", "decision", "seuil"}
    assert 0.0 <= resultat["score_risque"] <= 1.0
    assert resultat["decision"] in ("ACCORDE", "REFUSE")


def test_predict_sous_le_seuil():
    resultat = predict(ModeleFixe(0.30), pd.DataFrame([[0]]), seuil=0.50)

    assert resultat["decision"] == "ACCORDE"


def test_predict_au_dessus_du_seuil():
    resultat = predict(ModeleFixe(0.70), pd.DataFrame([[0]]), seuil=0.50)

    assert resultat["decision"] == "REFUSE"


def test_predict_seuil_exact():
    # La comparaison est >=, donc la valeur du seuil elle-meme est refusee.
    resultat = predict(ModeleFixe(0.50), pd.DataFrame([[0]]), seuil=0.50)

    assert resultat["decision"] == "REFUSE"


def test_predict_compare_avant_arrondi():
    # score_risque est arrondi a 4 decimales, mais la comparaison porte sur
    # la valeur non arrondie : 0.50004 s'affiche 0.5 et reste un refus.
    resultat = predict(ModeleFixe(0.500049), pd.DataFrame([[0]]), seuil=0.50)

    assert resultat["score_risque"] == 0.5
    assert resultat["decision"] == "REFUSE"
