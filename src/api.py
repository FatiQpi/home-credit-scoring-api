"""
API de scoring credit.

Expose le modele derriere deux endpoints HTTP. Seule couche du projet qui
connait le protocole : les exceptions de features.py y sont traduites en
codes de statut.
"""

from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from src.features import (
    ClientNotFoundError,
    get_features,
    load_artifacts,
    load_model,
    load_store,
    predict,
)
from src.journal import enregistrer
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
    # L'objet BackgroundTasks injecte dans la route est perdu des qu'elle leve :
    # FastAPI ne l'attache qu'a un retour normal. Le 404 se journalise donc ici,
    # sur une file construite a la main et portee par la reponse d'erreur.
    debut = getattr(request.state, "debut", None)

    taches = BackgroundTasks()
    taches.add_task(
        enregistrer,
        statut=404,
        # -1.0 marque un chemin theoriquement inatteignable : seule predire leve
        # cette exception, et elle pose debut des sa premiere ligne.
        duree_totale_ms=(perf_counter() - debut) * 1000 if debut else -1.0,
        sk_id_curr=getattr(request.state, "sk_id_curr", None),
        version_api=request.app.version,
    )

    return JSONResponse(
        status_code=404, content={"detail": str(exc)}, background=taches
    )


# Sans cette route, la racine du Space repond 404 au visiteur qui ouvre le
# lien.
@app.get("/", summary="Description du service")
def racine(request: Request) -> dict:
    return {
        "service": request.app.title,
        "version": request.app.version,
        "documentation": "/docs",
        "endpoints": ["/predict", "/health"],
    }


# def et non async def : pandas et LightGBM bloquent. FastAPI execute alors la
# route dans un thread auxiliaire et la boucle d'evenements reste libre.
@app.post(
    "/predict",
    response_model=Reponse,
    summary="Score un client par son identifiant",
    responses={404: {"description": "Aucune donnee pour cet identifiant."}},
)
def predire(demande: Demande, request: Request, taches: BackgroundTasks) -> dict:
    # perf_counter et non time : compteur monotone, insensible aux ajustements
    # d'horloge du systeme, qui produiraient des durees negatives.
    debut = perf_counter()

    # Deposes pour le gestionnaire de 404, qui n'a acces ni aux chronometres ni
    # a la demande une fois l'exception levee.
    request.state.debut = debut
    request.state.sk_id_curr = demande.SK_ID_CURR

    features = get_features(demande.SK_ID_CURR, request.app.state.store)

    avant = perf_counter()
    resultat = predict(request.app.state.model, features)
    apres = perf_counter()

    # Executee apres l'envoi de la reponse : un echec d'ecriture ne peut pas
    # transformer ce 200 en 500, et l'aller-retour Supabase sort du chemin
    # chronometre.
    taches.add_task(
        enregistrer,
        statut=200,
        duree_totale_ms=(apres - debut) * 1000,
        sk_id_curr=demande.SK_ID_CURR,
        score_risque=resultat["score_risque"],
        decision=resultat["decision"],
        duree_inference_ms=(apres - avant) * 1000,
        version_api=request.app.version,
    )

    return {"SK_ID_CURR": demande.SK_ID_CURR, **resultat}


@app.get("/health", summary="Sonde de vivacite")
def health(request: Request) -> dict:
    # Un echec du lifespan empeche le demarrage, donc ces attributs sont
    # presents des lors que la route repond. La sonde reste utile en cas de
    # chargement partiel.
    etat = request.app.state
    charge = all(hasattr(etat, nom) for nom in ("artifacts", "store", "model"))

    return {"status": "ok" if charge else "degraded"}
