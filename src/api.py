"""
API de scoring credit.

Expose le modele derriere deux endpoints HTTP. Seule couche du projet qui
connait le protocole : les exceptions de features.py y sont traduites en
codes de statut.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.features import (
    ClientNotFoundError,
    get_features,
    load_artifacts,
    load_model,
    load_store,
    predict,
)
from src.schemas import Demande, Reponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargements couteux effectues une seule fois. Un echec ici empeche le
    # demarrage, plutot que de servir des requetes vouees a l'erreur.
    app.state.artifacts = load_artifacts()
    app.state.store = load_store(app.state.artifacts["feature_names"])
    app.state.model = load_model()
    yield


app = FastAPI(
    title="API de scoring credit",
    description=(
        "Retourne un score de risque pour un client Home Credit. "
        "La demande ne porte qu'un identifiant : les features sont lues "
        "dans un magasin charge au demarrage."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Identifiant bien forme mais inconnu : la demande est valide, la ressource
# n'existe pas. Distinct du 422 que Pydantic renvoie pour un identifiant
# malforme.
@app.exception_handler(ClientNotFoundError)
async def client_not_found_handler(
    request: Request, exc: ClientNotFoundError
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


# def et non async def : pandas et LightGBM bloquent. FastAPI execute alors la
# route dans un thread auxiliaire et la boucle d'evenements reste libre.
@app.post(
    "/predict",
    response_model=Reponse,
    summary="Score un client par son identifiant",
    responses={404: {"description": "Aucune donnee pour cet identifiant."}},
)
def predire(demande: Demande, request: Request) -> dict:
    features = get_features(demande.SK_ID_CURR, request.app.state.store)
    resultat = predict(request.app.state.model, features)

    return {"SK_ID_CURR": demande.SK_ID_CURR, **resultat}


@app.get("/health", summary="Sonde de vivacite")
def health(request: Request) -> dict:
    # Un echec du lifespan empeche le demarrage, donc ces attributs sont
    # presents des lors que la route repond. La sonde reste utile en cas de
    # chargement partiel.
    etat = request.app.state
    charge = all(hasattr(etat, nom) for nom in ("artifacts", "store", "model"))

    return {"status": "ok" if charge else "degraded"}
