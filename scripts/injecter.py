"""Injecte un trafic simule vers l'API deployee pour alimenter le journal.

Deux campagnes :
  reference : tirage uniforme dans le magasin, sans biais.
  derive    : tirage dans le quartile le plus jeune.

Le script n'ecrit rien dans Supabase : c'est l'API qui journalise. Il produit un
manifeste local decrivant la campagne, seule cle permettant au notebook
d'analyse d'isoler ses lignes.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
MAGASIN = RACINE / "data" / "store.parquet"
CAMPAGNES = RACINE / "data" / "campagnes"

# DAYS_BIRTH est un nombre de jours negatif : le quartile SUPERIEUR en valeur est
# le plus jeune. Inverser cette comparaison inverserait tout le scenario.
QUANTILE_JEUNES = 0.75

JOURS_PAR_AN = 365.25
ECHECS_CONSECUTIFS_MAX = 10
DELAI_S = 30.0
MARGE_FINALE_S = 3.0


def tirer(mode: str, nombre: int, graine: int) -> tuple[list[int], pd.Series]:
    """Retourne les identifiants a appeler et leur age, pour controle."""
    # Lecture colonnaire : seule DAYS_BIRTH est lue, pas les 145 features.
    ages = pd.read_parquet(MAGASIN, columns=["DAYS_BIRTH"])["DAYS_BIRTH"]

    if mode == "derive":
        seuil = ages.quantile(QUANTILE_JEUNES)
        candidats = ages[ages >= seuil]
    else:
        candidats = ages

    if nombre > len(candidats):
        raise SystemExit(f"{nombre} demandes pour {len(candidats)} candidats.")

    tires = random.Random(graine).sample(list(candidats.index), nombre)
    return [int(i) for i in tires], candidats.loc[tires]


def appeler(url: str, sk_id: int) -> int:
    """Retourne le code de statut, ou 0 si la requete n'a pas abouti."""
    requete = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps({"SK_ID_CURR": sk_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(requete, timeout=DELAI_S) as reponse:
            return reponse.status
    except urllib.error.HTTPError as erreur:
        # 404 et 422 sont des reponses, pas des pannes : elles comptent.
        return erreur.code
    except Exception:
        return 0


def injecter(url: str, identifiants: list[int], rythme: float) -> dict[str, int]:
    intervalle = 1.0 / rythme
    reponses: dict[str, int] = {}
    echecs = 0

    for rang, sk_id in enumerate(identifiants, start=1):
        depart = time.perf_counter()
        statut = appeler(url, sk_id)
        reponses[str(statut)] = reponses.get(str(statut), 0) + 1

        echecs = echecs + 1 if statut == 0 else 0
        if echecs >= ECHECS_CONSECUTIFS_MAX:
            raise SystemExit(
                f"\n{echecs} echecs consecutifs : service injoignable, arret."
            )

        if rang % 100 == 0:
            print(f"  {rang}/{len(identifiants)}", file=sys.stderr, flush=True)

        reste = intervalle - (time.perf_counter() - depart)
        if reste > 0:
            time.sleep(reste)

    return reponses


def main() -> None:
    analyseur = argparse.ArgumentParser(description=__doc__)
    analyseur.add_argument("mode", choices=["reference", "derive"])
    analyseur.add_argument("--url", required=True, help="Racine de l'API, sans /predict")
    analyseur.add_argument("--nombre", type=int, default=2000)
    analyseur.add_argument("--rythme", type=float, default=2.0, help="requetes/seconde")
    analyseur.add_argument("--graine", type=int, default=42)
    arguments = analyseur.parse_args()

    url = arguments.url.rstrip("/")
    identifiants, ages = tirer(arguments.mode, arguments.nombre, arguments.graine)

    en_annees = -ages / JOURS_PAR_AN
    print(
        f"Campagne {arguments.mode} : {len(identifiants)} clients, "
        f"age median {en_annees.median():.1f} ans "
        f"({en_annees.min():.1f} a {en_annees.max():.1f})",
        file=sys.stderr,
    )

    # Reveille un Space en veille : le premier appel peut sinon depasser le delai
    # et fausser la premiere mesure de latence. Par /health et non /predict :
    # /predict ecrirait une ligne en tache d'arriere-plan, arrivee en base apres
    # la prise de l'heure de debut, donc comptee dans la campagne.
    try:
        urllib.request.urlopen(f"{url}/health", timeout=DELAI_S).close()
    except Exception:
        pass

    debut = datetime.now(timezone.utc)
    reponses = injecter(url, identifiants, arguments.rythme)

    # Les ecritures partent en tache d'arriere-plan : la derniere ligne arrive
    # dans Supabase apres le retour de la derniere reponse.
    time.sleep(MARGE_FINALE_S)
    fin = datetime.now(timezone.utc)

    CAMPAGNES.mkdir(parents=True, exist_ok=True)
    manifeste = CAMPAGNES / f"{arguments.mode}_{debut:%Y%m%d_%H%M%S}.json"
    manifeste.write_text(
        json.dumps(
            {
                "campagne": arguments.mode,
                "url": url,
                "graine": arguments.graine,
                "debut": debut.isoformat(),
                "fin": fin.isoformat(),
                "reponses": reponses,
                "age_median_ans": round(float(en_annees.median()), 1),
                "identifiants": identifiants,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"{reponses}\n{manifeste}", file=sys.stderr)


if __name__ == "__main__":
    main()
