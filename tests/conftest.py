"""Fixtures partagees par les tests."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.features import load_artifacts, load_model, load_store


# Portee session : le magasin de 40 Mo est lu une fois pour toute la suite.
@pytest.fixture(scope="session")
def artifacts():
    return load_artifacts()


@pytest.fixture(scope="session")
def store(artifacts):
    return load_store(artifacts["feature_names"])


@pytest.fixture(scope="session")
def model():
    return load_model()


@pytest.fixture(scope="session")
def client_id(store):
    """Un identifiant garanti present."""
    return int(store.index[0])


@pytest.fixture(scope="session")
def client():
    # Le gestionnaire de contexte declenche le lifespan. Sans lui, app.state
    # reste vide et toutes les routes echouent.
    with TestClient(app) as c:
        yield c
