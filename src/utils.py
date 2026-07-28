import pandas as pd
import numpy as np

"""
Fonctions utilitaires pour le projet Home Credit Default Risk.
"""

# =================================================================================
# Fonctions d'exploration et de reduction de mémoire
# =================================================================================

def overview(df, name="DataFrame"):
    """Affiche un résumé rapide d'un DataFrame."""
    print(f"\n{'='*50}")          # Ligne de séparation visuelle
    print(f"  {name}")             # Le nom que tu donnes au df
    print(f"{'='*50}")
    
    # --- Dimensions ---
    print(f"  Shape : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    
    # --- Poids en mémoire ---
    print(f"  Mémoire : {df.memory_usage(deep=True).sum() / 1e6:.1f} Mo")
    
    # --- Lignes dupliquées ---
    print(f"  Doublons : {df.duplicated().sum()}")
    
     # --- Valeurs manquantes triées par % décroissant ---
    missing_df = pd.DataFrame({
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(1),
        'dtype': df.dtypes
    }).sort_values('missing_pct', ascending=False)
    missing_df = missing_df[missing_df['missing_pct'] > 0]

    print(f"  Valeurs manquantes : {len(missing_df)} colonnes")
    if len(missing_df) > 0:
        print(missing_df.to_string())      
    else:
        print("    Aucune !")

    return missing_df


def reduce_memory(df):
    """Réduit l'utilisation mémoire en optimisant les types."""
    start_mem = df.memory_usage(deep=True).sum() / 1e6

    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:  # On ne touche pas aux strings
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Pour les entiers : int64 → int32 → int16 → int8
            if str(col_type).startswith("int"):
                if c_min >= -128 and c_max <= 127:
                    df[col] = df[col].astype(np.int8)       # 1 octet
                elif c_min >= -32768 and c_max <= 32767:
                    df[col] = df[col].astype(np.int16)      # 2 octets
                elif c_min >= -2147483648 and c_max <= 2147483647:
                    df[col] = df[col].astype(np.int32)      # 4 octets
                # sinon reste en int64                        # 8 octets
            
            # Pour les floats : float64 → float32
            elif str(col_type).startswith("float"):
                if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)    # 4 octets au lieu de 8

    end_mem = df.memory_usage(deep=True).sum() / 1e6
    print(f"Mémoire : {start_mem:.1f} Mo → {end_mem:.1f} Mo "
          f"({100 * (start_mem - end_mem) / start_mem:.0f}% de réduction)")
    return df




# =================================================================================
# Fonctions de nettoyage
# =================================================================================

def clean_application(df, caps=None):
    """Nettoyage de application_{train|test}.csv
    
    Args:
        df: le DataFrame à nettoyer
        caps: dict de seuils calculés sur le train. Si None, les calcule.
    Returns:
        df nettoyé, caps utilisés
    """
    df = df.copy()
    
    # DAYS_EMPLOYED : remplacer le code 365243 par NaN
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # CODE_GENDER : supprimer les "XNA"
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # NAME_FAMILY_STATUS : supprimer les "Unknown"
    df = df[df['NAME_FAMILY_STATUS'] != 'Unknown']
    
    # Calculer les caps sur le train ou réutiliser ceux du train
    if caps is None:
        caps = {
            'AMT_INCOME_TOTAL': 5_000_000,
            'AMT_REQ_CREDIT_BUREAU_QRT': 10,
            'OBS_30_CNT_SOCIAL_CIRCLE': 50,
            'DEF_30_CNT_SOCIAL_CIRCLE': 50,
            'OBS_60_CNT_SOCIAL_CIRCLE': 50,
            'DEF_60_CNT_SOCIAL_CIRCLE': 50,
        }
    
    # Appliquer les caps
    for col, cap in caps.items():
        df[col] = df[col].clip(upper=cap)
    
    print(f"Shape après nettoyage : {df.shape}")
    return df, caps

