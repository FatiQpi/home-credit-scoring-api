"""Tests de la couche HTTP : codes de statut, contrat de reponse, documentation."""

import pytest

from src.features import get_features, predict

CLIENT_INCONNU = 999999999


# --- Cas nominal ---


def test_predict_200(client, client_id):
    reponse = client.post("/predict", json={"SK_ID_CURR": client_id})

    assert reponse.status_code == 200


def test_predict_contrat_de_reponse(client, client_id):
    corps = client.post("/predict", json={"SK_ID_CURR": client_id}).json()

    assert set(corps) == {"SK_ID_CURR", "score_risque", "decision", "seuil"}
    assert corps["SK_ID_CURR"] == client_id
    assert 0.0 <= corps["score_risque"] <= 1.0
    assert corps["decision"] in ("ACCORDE", "REFUSE")


def test_predict_identique_a_l_appel_direct(client, client_id, store, model):
    # L'API ne doit rien ajouter au calcul de la couche metier.
    corps = client.post("/predict", json={"SK_ID_CURR": client_id}).json()
    attendu = predict(model, get_features(client_id, store))

    assert corps["score_risque"] == attendu["score_risque"]
    assert corps["decision"] == attendu["decision"]


# --- Client inconnu : 404 ---


def test_client_inconnu_404(client):
    reponse = client.post("/predict", json={"SK_ID_CURR": CLIENT_INCONNU})

    assert reponse.status_code == 404
    assert "detail" in reponse.json()


# --- Demande malformee : 422 ---


@pytest.mark.parametrize(
    "corps",
    [
        pytest.param({}, id="champ_manquant"),
        pytest.param({"SK_ID_CURR": "abc"}, id="type_incorrect"),
        pytest.param({"SK_ID_CURR": -1}, id="hors_plage_negatif"),
        pytest.param({"SK_ID_CURR": 0}, id="hors_plage_zero"),
        pytest.param({"SK_ID_CURR": 1.5}, id="non_entier"),
    ],
)
def test_demande_malformee_422(client, corps):
    assert client.post("/predict", json=corps).status_code == 422


def test_champ_surnumeraire_refuse(client, client_id):
    # Garde d'architecture : aucune feature ne doit transiter par la requete.
    # Sans extra='forbid', le champ serait ignore en silence et le client
    # croirait qu'il a ete pris en compte.
    reponse = client.post(
        "/predict", json={"SK_ID_CURR": client_id, "AMT_CREDIT": 500000}
    )

    assert reponse.status_code == 422


# --- Sonde ---


def test_health(client):
    reponse = client.get("/health")

    assert reponse.status_code == 200
    assert reponse.json() == {"status": "ok"}


# --- Documentation ---


def test_openapi_declare_les_erreurs(client):
    schema = client.get("/openapi.json").json()
    reponses = schema["paths"]["/predict"]["post"]["responses"]

    assert "404" in reponses
    assert "422" in reponses


def test_openapi_expose_le_contrat_d_entree(client):
    schema = client.get("/openapi.json").json()
    demande = schema["components"]["schemas"]["Demande"]

    assert demande["required"] == ["SK_ID_CURR"]
    assert demande["additionalProperties"] is False
