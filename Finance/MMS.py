"""
Module pour calculer les moyennes mobiles simples (MMS) sur un dataframe.
"""

import pandas as pd
from pathlib import Path


def add_mms(df: pd.DataFrame, windows: list = [20, 50, 200], price_col: str = "Close") -> pd.DataFrame:
    """
    Ajoute des colonnes de moyennes mobiles simples au dataframe.
    
    Args:
        df: DataFrame contenant les données de prix
        windows: Liste des fenêtres temporelles pour les moyennes mobiles
        price_col: Nom de la colonne de prix à utiliser (par défaut "Close")
    
    Returns:
        DataFrame avec les colonnes MMS ajoutées
    
    Raises:
        ValueError: Si la colonne price_col n'existe pas dans le dataframe
    """
    if price_col not in df.columns:
        raise ValueError(f"La colonne '{price_col}' n'existe pas dans le dataframe")
    
    df = df.copy()
    
    for window in windows:
        col_name = f"MMS_{window}"
        df[col_name] = df[price_col].rolling(window=window, min_periods=1).mean()
    
    return df


def process_parquet_mms(
    input_path: str,
    output_path: str = None,
    windows: list = [20, 50, 200],
    price_col: str = "Close"
) -> pd.DataFrame:
    """
    Charge un parquet, ajoute les MMS, et sauvegarde le résultat.
    
    Args:
        input_path: Chemin vers le fichier parquet d'entrée
        output_path: Chemin vers le fichier parquet de sortie (optionnel)
        windows: Liste des fenêtres temporelles pour les moyennes mobiles
        price_col: Nom de la colonne de prix à utiliser
    
    Returns:
        DataFrame enrichi avec les MMS
    """
    # Charger le parquet
    df = pd.read_parquet(input_path)
    
    # Ajouter les MMS
    df = add_mms(df, windows=windows, price_col=price_col)
    
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
        process_parquet_mms(input_file, output_file)
    else:
        print("Usage: python MMS.py <input_parquet> [output_parquet]")
