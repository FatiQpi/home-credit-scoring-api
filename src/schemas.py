"""
Contrats d'entree et de sortie de l'API.

La demande ne porte qu'un identifiant : toutes les features sont lues dans
le magasin. extra='forbid' fait respecter cette regle plutot que d'ignorer
en silence les champs surnumeraires.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Demande(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"SK_ID_CURR": 100002}},
    )

    # gt=0 : un identifiant negatif est une demande malformee (422), pas un
    # client introuvable (404).
    SK_ID_CURR: int = Field(gt=0, description="Identifiant du client au dossier.")


class Reponse(BaseModel):
    SK_ID_CURR: int
    score_risque: float = Field(
        description=(
            "Score de risque entre 0 et 1. Ordonne les clients mais ne "
            "s'interprete pas comme une frequence de defaut."
        )
    )
    decision: Literal["ACCORDE", "REFUSE"]
    seuil: float = Field(description="Seuil au-dela duquel la demande est refusee.")
