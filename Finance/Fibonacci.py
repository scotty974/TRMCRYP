"""
Module pour calculer les niveaux de retracement de Fibonacci sur un dataframe.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def add_fibonacci_levels(
    df: pd.DataFrame,
    price_col: str = "Close",
    window: int = None,
    add_distance: bool = True
) -> pd.DataFrame:
    """
    Ajoute les niveaux de retracement de Fibonacci au dataframe.
    
    Les niveaux de Fibonacci sont calculés entre le min et max d'une série:
    - 0.236 (23.6%)
    - 0.382 (38.2%)
    - 0.500 (50%)
    - 0.618 (61.8%)
    - 0.786 (78.6%)
    
    Args:
        df: DataFrame contenant les données de prix
        price_col: Nom de la colonne de prix à utiliser (par défaut "Close")
        window: Fenêtre glissante pour le calcul (None = toute la série)
        add_distance: Si True, ajoute une colonne avec la distance au niveau le plus proche
    
    Returns:
        DataFrame avec les colonnes Fibonacci ajoutées
    """
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"La colonne '{price_col}' n'existe pas dans le dataframe")
    
    # Ratios de Fibonacci standard
    fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    
    if window is None:
        # Calcul global sur toute la série
        price_max = df[price_col].max()
        price_min = df[price_col].min()
        
        for ratio in fib_ratios:
            col_name = f"Fib_{str(ratio).replace('.', '_')}"
            # Pour une tendance haussière: niveau = max - (max - min) * ratio
            df[col_name] = price_max - (price_max - price_min) * ratio
    else:
        # Calcul glissant sur une fenêtre
        for ratio in fib_ratios:
            col_name = f"Fib_{str(ratio).replace('.', '_')}"
            
            # Calculer les niveaux sur une fenêtre glissante
            rolling_max = df[price_col].rolling(window=window, min_periods=1).max()
            rolling_min = df[price_col].rolling(window=window, min_periods=1).min()
            
            df[col_name] = rolling_max - (rolling_max - rolling_min) * ratio
    
    # Ajouter la distance au niveau le plus proche (optionnel)
    if add_distance:
        fib_cols = [f"Fib_{str(ratio).replace('.', '_')}" for ratio in fib_ratios]
        
        # Calculer la distance absolue à chaque niveau
        distances = pd.DataFrame()
        for col in fib_cols:
            distances[col] = (df[price_col] - df[col]).abs()
        
        # Trouver le niveau le plus proche et la distance
        df['Fib_Nearest_Distance'] = distances.min(axis=1)
        
        # Identifier quel niveau est le plus proche
        df['Fib_Nearest_Level'] = distances.idxmin(axis=1)
        
        # Calculer la position relative entre les niveaux extrêmes
        fib_max = df[[col for col in fib_cols]].max(axis=1)
        fib_min = df[[col for col in fib_cols]].min(axis=1)
        fib_range = fib_max - fib_min
        
        df['Fib_Position'] = 0.5  # Valeur par défaut
        mask = fib_range > 0
        df.loc[mask, 'Fib_Position'] = (
            (df.loc[mask, price_col] - fib_min[mask]) / fib_range[mask]
        )
        
        # Limiter entre 0 et 1
        df['Fib_Position'] = df['Fib_Position'].clip(0, 1)
    
    return df


def process_parquet_fibonacci(
    input_path: str,
    output_path: str = None,
    price_col: str = "Close",
    window: int = None,
    add_distance: bool = True
) -> pd.DataFrame:
    """
    Charge un parquet, ajoute les niveaux de Fibonacci, et sauvegarde le résultat.
    
    Args:
        input_path: Chemin vers le fichier parquet d'entrée
        output_path: Chemin vers le fichier parquet de sortie (optionnel)
        price_col: Nom de la colonne de prix à utiliser
        window: Fenêtre glissante pour le calcul (None = toute la série)
        add_distance: Si True, ajoute la distance au niveau le plus proche
    
    Returns:
        DataFrame enrichi avec les niveaux de Fibonacci
    """
    # Charger le parquet
    df = pd.read_parquet(input_path)
    
    # Ajouter les niveaux de Fibonacci
    df = add_fibonacci_levels(df, price_col=price_col, window=window, add_distance=add_distance)
    
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
        window = int(sys.argv[3]) if len(sys.argv) > 3 else None
        process_parquet_fibonacci(input_file, output_file, window=window)
    else:
        print("Usage: python Fibonacci.py <input_parquet> [output_parquet] [window]")
