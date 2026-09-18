"""Journalisation des appels de prediction : sortie standard, puis Supabase."""

import json
import os
import sys
import urllib.error
import urllib.request

TABLE = "journal_predictions"
DELAI_ENVOI_S = 5.0


def enregistrer(
    statut: int,
    duree_totale_ms: float,
    sk_id_curr: int | None = None,
    score_risque: float | None = None,
    decision: str | None = None,
    duree_inference_ms: float | None = None,
    version_api: str | None = None,
) -> None:
    """Ecrit un appel sur la sortie standard, puis l'envoie a Supabase.

    Ne leve jamais : appelee depuis une tache d'arriere-plan, dont les exceptions
    ne sont rattrapees par personne.
    """
    enregistrement = {
        "statut": statut,
        "duree_totale_ms": duree_totale_ms,
        "sk_id_curr": sk_id_curr,
        "score_risque": score_risque,
        "decision": decision,
        "duree_inference_ms": duree_inference_ms,
        "version_api": version_api,
    }
    corps = json.dumps(enregistrement)

    # flush : la sortie d'un conteneur est bufferisee par blocs. Sans vidage, les
    # lignes en attente disparaissent au redemarrage du Space.
    print(corps, flush=True)

    # Lecture a chaque appel et non a l'import : permet aux tests de simuler
    # l'absence de configuration.
    url = os.environ.get("SUPABASE_URL")
    cle = os.environ.get("SUPABASE_KEY")
    if not url or not cle:
        return

    requete = urllib.request.Request(
        f"{url.rstrip('/')}/rest/v1/{TABLE}",
        data=corps.encode("utf-8"),
        headers={
            "apikey": cle,
            "Authorization": f"Bearer {cle}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        method="POST",
    )

    try:
        urllib.request.urlopen(requete, timeout=DELAI_ENVOI_S).close()
    except urllib.error.HTTPError as erreur:
        # Le corps porte le message de Supabase : sans lui, un refus de politique
        # de securite et une colonne absente se ressemblent.
        detail = erreur.read().decode("utf-8", "replace")
        _signaler(f"HTTP {erreur.code} {detail}")
    except Exception as erreur:  # noqa: BLE001
        # Large volontairement : voir docstring. Une perte non signalee serait pire.
        _signaler(f"{type(erreur).__name__}: {erreur}")


def _signaler(cause: str) -> None:
    print(
        json.dumps({"journal": "echec_envoi", "cause": cause}),
        file=sys.stderr,
        flush=True,
    )
