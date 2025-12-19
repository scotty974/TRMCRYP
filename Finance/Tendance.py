"""
Module pour calculer les tendances basées sur les moyennes mobiles simples.
"""

import pandas as pd
from pathlib import Path
from Finance.MMS import add_mms


def add_tendance(
    df: pd.DataFrame,
    mms_short: int = 20,
    mms_medium: int = 50,
    mms_long: int = 200,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Ajoute des colonnes de tendance au dataframe basées sur les MMS.
    
    La tendance est déterminée ainsi:
    - 'up' (haussière): MMS_20 > MMS_50 > MMS_200
    - 'down' (baissière): MMS_20 < MMS_50 < MMS_200
    - 'neutral' (neutre): autres cas
    
    Args:
        df: DataFrame contenant les données de prix
        mms_short: Fenêtre pour la MMS courte (par défaut 20)
        mms_medium: Fenêtre pour la MMS moyenne (par défaut 50)
        mms_long: Fenêtre pour la MMS longue (par défaut 200)
        price_col: Nom de la colonne de prix à utiliser
    
    Returns:
        DataFrame avec les colonnes Tendance et Tendance_Code ajoutées
    """
    df = df.copy()
    
    # Colonnes des MMS attendues
    col_short = f"MMS_{mms_short}"
    col_medium = f"MMS_{mms_medium}"
    col_long = f"MMS_{mms_long}"
    
    # Vérifier si les colonnes MMS existent, sinon les calculer
    if col_short not in df.columns or col_medium not in df.columns or col_long not in df.columns:
        df = add_mms(df, windows=[mms_short, mms_medium, mms_long], price_col=price_col)
    
    # Déterminer la tendance
    def determine_tendance(row):
        if pd.isna(row[col_short]) or pd.isna(row[col_medium]) or pd.isna(row[col_long]):
            return 'neutral'
        
        if row[col_short] > row[col_medium] > row[col_long]:
            return 'up'
        elif row[col_short] < row[col_medium] < row[col_long]:
            return 'down'
        else:
            return 'neutral'
    
    df['Tendance'] = df.apply(determine_tendance, axis=1)
    
    # Ajouter un encodage numérique
    tendance_map = {'up': 1, 'down': -1, 'neutral': 0}
    df['Tendance_Code'] = df['Tendance'].map(tendance_map)
    
    return df


def process_parquet_tendance(
    input_path: str,
    output_path: str = None,
    mms_short: int = 20,
    mms_medium: int = 50,
    mms_long: int = 200,
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Charge un parquet, ajoute les tendances, et sauvegarde le résultat.
    
    Args:
        input_path: Chemin vers le fichier parquet d'entrée
        output_path: Chemin vers le fichier parquet de sortie (optionnel)
        mms_short: Fenêtre pour la MMS courte
        mms_medium: Fenêtre pour la MMS moyenne
        mms_long: Fenêtre pour la MMS longue
        price_col: Nom de la colonne de prix à utiliser
    
    Returns:
        DataFrame enrichi avec les tendances
    """
    # Charger le parquet
    df = pd.read_parquet(input_path)
    
    # Ajouter les tendances
    df = add_tendance(df, mms_short=mms_short, mms_medium=mms_medium, 
                     mms_long=mms_long, price_col=price_col)
    
    # Déterminer le chemin de sortie si non spécifié
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_enhanced{input_path_obj.suffix}")
    
    # Sauvegarder le résultat
    df.to_parquet(output_path, index=False)
    print(f"Fichier enrichi sauvegardé: {output_path}")
    
    return df


if __name__ == "__main__":
    # Exemple d'utilisation
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        process_parquet_tendance(input_file, output_file)
    else:
        print("Usage: python Tendance.py <input_parquet> [output_parquet]")
