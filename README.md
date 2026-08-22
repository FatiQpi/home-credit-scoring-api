# API de scoring crédit — Home Credit

[![CI](https://github.com/FatiQpi/home-credit-scoring-api/actions/workflows/ci.yml/badge.svg)](https://github.com/FatiQpi/home-credit-scoring-api/actions/workflows/ci.yml)

API REST qui retourne un score de risque de défaut pour un client de l'organisme Home Credit, à partir d'un modèle LightGBM entraîné sur le jeu de données [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).

L'API reçoit **un identifiant client**. Elle lit les 145 features de ce client dans un magasin chargé au démarrage, applique le modèle, et renvoie un score accompagné d'une décision.

> Projet 8 du parcours **AI Engineer** — déploiement et monitoring d'un modèle de scoring crédit. Le modèle lui-même provient du projet 6.

---

## Sommaire

- [Fonctionnement](#fonctionnement)
- [Prérequis](#prérequis)
- [Démarrage rapide avec Docker](#démarrage-rapide-avec-docker)
- [Installation locale](#installation-locale)
- [Référence de l'API](#référence-de-lapi)
- [Tests](#tests)
- [Structure du dépôt](#structure-du-dépôt)
- [Reconstruire le magasin](#reconstruire-le-magasin)
- [Modèle et données](#modèle-et-données)

---

## Fonctionnement

```
                 ┌──────────────────── au démarrage ─────────────────────┐
                 │  artifacts/feature_names.json  →  contrat d'entrée    │
                 │  artifacts/categories.json     →  modalités figées    │
                 │  data/store.parquet            →  307 505 × 145       │
                 │  artifacts/model.pkl           →  LightGBM            │
                 └───────────────────────────────────────────────────────┘

  POST /predict
  {"SK_ID_CURR": 100002}
          │
          ▼
   lecture par index  ──►  145 features  ──►  predict_proba  ──►  seuil 0,50
          │                                                            │
          ▼                                                            ▼
   identifiant absent → 404                          {"score_risque": 0.8344,
                                                      "decision": "REFUSE"}
```

**Aucune feature ne transite par la requête.** Le corps de la demande ne porte qu'un identifiant ; tout champ supplémentaire est rejeté en 422. Ce choix reflète l'usage visé : un conseiller bancaire saisit un numéro de dossier, pas 145 variables.

Les fichiers coûteux — modèle et magasin — sont chargés **une seule fois** au démarrage du service, via le gestionnaire `lifespan` de FastAPI, et réutilisés par toutes les requêtes. Un échec de chargement empêche le démarrage plutôt que de servir des requêtes vouées à l'erreur.

---

## Prérequis

| Pour | Il faut |
|---|---|
| Lancer le service | Docker |
| Développer et tester | Python 3.11 |

Le magasin de features (`data/store.parquet`, 40 Mo) est **versionné dans le dépôt** : un simple clone suffit, aucun téléchargement de données n'est nécessaire.

---

## Démarrage rapide avec Docker

```bash
git clone https://github.com/FatiQpi/home-credit-scoring-api.git
cd home-credit-scoring-api

docker build -t scoring-api .
docker run --rm -p 8000:7860 scoring-api
```

Sur une machine ARM (Mac Apple Silicon) destinée à produire une image pour une cible x86-64 :

```bash
docker build --platform linux/amd64 -t scoring-api .
```

Le conteneur écoute sur le port **7860** ; l'option `-p 8000:7860` le publie sur le port 8000 de la machine hôte. Le port interne est configurable par la variable d'environnement `PORT`.

Le service est prêt lorsque le journal affiche :

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:7860
```

Vérification, dans un second terminal :

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"SK_ID_CURR": 100002}'
# {"SK_ID_CURR":100002,"score_risque":0.8344,"decision":"REFUSE","seuil":0.5}
```

---

## Installation locale

```bash
python3.11 -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate

pip install -r requirements.txt   # service seul
pip install -r requirements-dev.txt   # service + outils de test

python -m uvicorn src.api:app --reload
```

Le service écoute alors sur `http://127.0.0.1:8000`.

> **Utilisez `python -m uvicorn` et `python -m pytest`**, et non les exécutables nus. Sur une machine où coexistent plusieurs environnements — conda notamment — `uvicorn` et `pytest` peuvent se résoudre vers un interpréteur autre que celui du `venv` actif, et échouer sur un `ModuleNotFoundError` déroutant.

---

## Référence de l'API

Documentation interactive générée automatiquement, une fois le service lancé :

| Interface | URL |
|---|---|
| Swagger UI | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| Schéma OpenAPI | `http://localhost:8000/openapi.json` |

### `POST /predict`

Retourne le score de risque d'un client identifié.

**Corps de la demande**

| Champ | Type | Contrainte |
|---|---|---|
| `SK_ID_CURR` | entier | strictement positif, **seul champ accepté** |

```json
{"SK_ID_CURR": 100002}
```

**Réponse — `200 OK`**

| Champ | Type | Description |
|---|---|---|
| `SK_ID_CURR` | entier | l'identifiant repris de la demande |
| `score_risque` | flottant | score entre 0 et 1, arrondi à 4 décimales |
| `decision` | `"ACCORDE"` ou `"REFUSE"` | résultat de la comparaison au seuil |
| `seuil` | flottant | seuil appliqué, `0.5` |

```json
{
  "SK_ID_CURR": 100002,
  "score_risque": 0.8344,
  "decision": "REFUSE",
  "seuil": 0.5
}
```

> **`score_risque` n'est pas une probabilité de défaut.** Le modèle a été entraîné avec `is_unbalance=True`, qui rééquilibre les classes : la sortie ordonne correctement les clients par risque, mais ne s'interprète pas comme une fréquence de défaut attendue. Le nom du champ est choisi pour éviter cette confusion.

**Codes d'erreur**

| Code | Cause | Exemple de corps |
|---|---|---|
| `404` | identifiant bien formé, mais absent du magasin | `{"SK_ID_CURR": 999999999}` |
| `422` | demande malformée | champ manquant, type incorrect, valeur négative ou nulle, champ surnuméraire |

La distinction est délibérée : un `422` signale une demande invalide, un `404` une demande valide portant sur une ressource inexistante.

```bash
# 404
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"SK_ID_CURR": 999999999}'
# {"detail":"Aucune donnee pour le client 999999999."}

# 422 — champ surnuméraire
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"SK_ID_CURR": 100002, "AMT_CREDIT": 500000}'
```

### `GET /health`

Sonde de vivacité, destinée aux plateformes d'hébergement.

```json
{"status": "ok"}
```

Renvoie `"degraded"` si le modèle ou le magasin ne sont pas chargés.

---

## Tests

```bash
source venv/bin/activate
python -m pytest                                  # 28 tests
python -m pytest --cov=src --cov-report=term      # avec la couverture
```

**28 tests, 99 % de couverture sur `src/`.**

| Fichier | Tests | Couvre |
|---|---|---|
| `tests/test_features.py` | 15 | artefacts, gardes du chargement, lecture par index, règle de décision |
| `tests/test_api.py` | 13 | codes 200 / 404 / 422, contrat de réponse, sonde, schéma OpenAPI |

Les fixtures de `tests/conftest.py` sont de portée *session* : le magasin de 40 Mo est lu une seule fois pour l'ensemble de la suite.

### Contrôle de parité

`scripts/check_parity.py` vérifie que le vecteur lu dans le magasin est **identique** à celui produit lors de l'entraînement pour le même client — égalité stricte des probabilités, sur les 307 505 clients. C'est la garantie qu'aucun écart n'existe entre entraînement et inférence.

```bash
python scripts/check_parity.py            # passage complet
python scripts/check_parity.py --rapide   # échantillon
python scripts/check_parity.py 100002     # identifiants imposés
```

> Ce script lit le CSV d'entraînement d'origine, **absent du dépôt**. Il n'est exécutable que sur une machine disposant des données sources du projet 6.

---

## Structure du dépôt

```
home-credit-scoring-api/
├── artifacts/
│   ├── model.pkl                    modèle LightGBM sérialisé
│   ├── feature_names.json           les 145 features, dans l'ordre attendu
│   ├── categories.json              modalités figées des 16 colonnes catégorielles
│   └── permutation_importance.csv
├── data/
│   └── store.parquet                307 505 clients × 145 features, 40 Mo
├── scripts/
│   ├── build_store.py               construit le magasin depuis les CSV sources
│   └── check_parity.py              contrôle de parité entraînement / inférence
├── src/
│   ├── api.py                       routes, gestion des erreurs, cycle de vie
│   ├── features.py                  chargement des artefacts, lecture, prédiction
│   └── schemas.py                   contrats d'entrée et de sortie (Pydantic)
├── tests/
│   ├── conftest.py                  fixtures de portée session
│   ├── test_api.py                  couche HTTP
│   └── test_features.py             couche métier
├── Dockerfile
├── .dockerignore
├── pytest.ini
├── requirements.txt                 8 dépendances directes, versions figées
└── requirements-dev.txt             les précédentes, plus les outils de test
```

L'image Docker ne reçoit que ce qui est nécessaire à l'exécution : `src/`, trois artefacts, le magasin et `requirements.txt`. Tests, scripts hors ligne et fichiers de développement en sont exclus.

---

## Reconstruire le magasin

Le magasin est un **artefact de déploiement**, solidaire de la version du modèle : il est versionné avec le code, et non produit à la volée. Le reconstruire n'est nécessaire que si le modèle ou son contrat d'entrée changent.

```bash
python scripts/build_store.py
```

Le script lit `artifacts/feature_names.json` et `artifacts/categories.json`, extrait les colonnes correspondantes du CSV d'entraînement, applique les types catégoriels, ordonne les colonnes selon le contrat, puis écrit le fichier Parquet.

> **Le CSV source n'est pas versionné.** Le chemin est codé en dur dans le script et pointe vers l'arborescence locale du projet 6. Cette étape n'est donc pas reproductible depuis un simple clone — c'est assumé : le magasin produit, lui, est dans le dépôt.

L'ordre des colonnes est vérifié à chaque démarrage du service. Un magasin dont les colonnes ne correspondent pas exactement au contrat, **ordre compris**, provoque un échec au démarrage : LightGBM identifie les features par position, et une seule colonne permutée fausserait tous les scores sans lever d'erreur.

---

## Modèle et données

| | |
|---|---|
| Algorithme | LightGBM 4.6.0, `LGBMClassifier`, 100 arbres |
| Features | 145, dont 16 catégorielles |
| Seuil de décision | 0,50 |
| Valeurs manquantes | prises en charge nativement par LightGBM, **sans imputation** |
| Jeu de données | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) (Kaggle) |

### Pile technique

Python 3.11 · FastAPI · Uvicorn · Pydantic v2 · pandas · PyArrow · scikit-learn · LightGBM

Les huit dépendances directes sont figées à une version précise dans `requirements.txt`. `scikit-learn` et `pyarrow` n'apparaissent pas dans les imports du code applicatif mais sont indispensables à l'exécution : la première au désérialisage du modèle, la seconde comme moteur de lecture Parquet.

---

## Licence

Projet réalisé dans un cadre pédagogique. Les données sont soumises aux conditions d'utilisation de la compétition Kaggle dont elles proviennent.
