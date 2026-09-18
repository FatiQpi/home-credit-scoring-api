"""Tests du module de journalisation : coupure, envoi, defaillances."""

import email.message
import io
import json
import urllib.error
import urllib.request

import pytest

from src.journal import enregistrer

URL = "https://exemple.supabase.co"
CLE = "cle-de-test"


@pytest.fixture
def configure(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", URL)
    monkeypatch.setenv("SUPABASE_KEY", CLE)


def _intercepter(monkeypatch) -> list:
    """Remplace urlopen et retourne la liste, vivante, des requetes emises."""
    emises = []

    class _Reponse:
        def close(self):
            pass

    def faux_urlopen(requete, timeout=None):
        emises.append(requete)
        return _Reponse()

    monkeypatch.setattr(urllib.request, "urlopen", faux_urlopen)
    return emises


def _faire_echouer(monkeypatch, erreur: Exception) -> None:
    def faux_urlopen(requete, timeout=None):
        raise erreur

    monkeypatch.setattr(urllib.request, "urlopen", faux_urlopen)


def test_sans_configuration_aucun_envoi(monkeypatch, capsys):
    # Verifie que l'absence des variables coupe l'envoi en amont, et non qu'il
    # echoue : c'est ce qui protege la CI et la base reelle.
    emises = _intercepter(monkeypatch)

    enregistrer(statut=200, duree_totale_ms=1.0)

    assert emises == []
    assert json.loads(capsys.readouterr().out)["statut"] == 200


def test_envoi_url_entetes_et_corps(monkeypatch, configure):
    emises = _intercepter(monkeypatch)

    enregistrer(
        statut=200,
        duree_totale_ms=32.6,
        sk_id_curr=100002,
        score_risque=0.8344,
        decision="REFUSE",
        duree_inference_ms=24.4,
        version_api="1.0.0",
    )

    requete = emises[0]
    entetes = {nom.lower(): valeur for nom, valeur in requete.headers.items()}

    assert requete.full_url == f"{URL}/rest/v1/journal_predictions"
    assert requete.get_method() == "POST"
    assert entetes["apikey"] == CLE
    assert entetes["authorization"] == f"Bearer {CLE}"
    assert entetes["prefer"] == "return=minimal"

    corps = json.loads(requete.data)
    assert corps["sk_id_curr"] == 100002
    assert corps["score_risque"] == 0.8344
    assert corps["duree_inference_ms"] == 24.4
    # id et horodatage sont generes par la base : les envoyer serait une erreur.
    assert "id" not in corps
    assert "horodatage" not in corps


def test_panne_reseau_ne_leve_pas(monkeypatch, configure, capsys):
    _faire_echouer(monkeypatch, urllib.error.URLError("injoignable"))

    enregistrer(statut=200, duree_totale_ms=1.0)

    signal = json.loads(capsys.readouterr().err)
    assert signal["journal"] == "echec_envoi"
    assert "injoignable" in signal["cause"]


def test_erreur_http_remonte_le_message_de_supabase(monkeypatch, configure, capsys):
    # Cas rencontre en developpement : sans le corps de la reponse, un refus de
    # politique de securite et une colonne absente donneraient le meme message.
    corps = b'{"code":"PGRST303","message":"JWT issued at future"}'
    _faire_echouer(
        monkeypatch,
        urllib.error.HTTPError(
            URL, 401, "Unauthorized", email.message.Message(), io.BytesIO(corps)
        ),
    )

    enregistrer(statut=200, duree_totale_ms=1.0)

    cause = json.loads(capsys.readouterr().err)["cause"]
    assert "401" in cause
    assert "PGRST303" in cause
