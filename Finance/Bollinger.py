"""
Module pour calculer les bandes de Bollinger sur un dataframe.
"""

import pandas as pd
from pathlib import Path
from Finance.MMS import add_mms
from Finance.ECT import add_ecart_type


def add_bollinger(
    df: pd.DataFrame,
    window: int = 20,
    price_col: str = "Close",
    n_std: float = 2.0,
    add_position: bool = True
) -> pd.DataFrame:
    """
    Ajoute les bandes de Bollinger au dataframe.
    
    Les bandes de Bollinger sont composées de:
    - Bande du milieu: moyenne mobile simple
    - Bande supérieure: moyenne mobile + n_std * écart-type
    - Bande inférieure: moyenne mobile - n_std * écart-type
    
    Args:
        df: DataFrame contenant les données de prix
        window: Fenêtre temporelle pour le calcul (par défaut 20)
        price_col: Nom de la colonne de prix à utiliser (par défaut "Close")
        n_std: Nombre d'écarts-types pour les bandes (par défaut 2.0)
        add_position: Si True, ajoute une colonne de position relative dans les bandes
    
    Returns:
        DataFrame avec les colonnes Bollinger ajoutées
    """
    df = df.copy()
    
    mms_col = f"MMS_{window}"
    ect_col = f"ECT_{window}"
    
    # Vérifier si les colonnes MMS et ECT existent, sinon les calculer
    if mms_col not in df.columns:
        df = add_mms(df, windows=[window], price_col=price_col)
    
    if ect_col not in df.columns:
        df = add_ecart_type(df, windows=[window], price_col=price_col)
    
    # Calculer les bandes de Bollinger
    df['Bollinger_Middle'] = df[mms_col]
    df['Bollinger_Upper'] = df[mms_col] + n_std * df[ect_col]
    df['Bollinger_Lower'] = df[mms_col] - n_std * df[ect_col]
    
    # Ajouter la position relative dans les bandes (optionnel)
    if add_position:
        # Position relative: 0 = à la bande inférieure, 0.5 = au milieu, 1 = à la bande supérieure
        band_width = df['Bollinger_Upper'] - df['Bollinger_Lower']
        
        # Éviter la division par zéro
        df['Bollinger_Position'] = 0.5  # Valeur par défaut
        mask = band_width > 0
        df.loc[mask, 'Bollinger_Position'] = (
            (df.loc[mask, price_col] - df.loc[mask, 'Bollinger_Lower']) / band_width[mask]
        )
        
        # Limiter entre 0 et 1 (le prix peut sortir des bandes)
        df['Bollinger_Position'] = df['Bollinger_Position'].clip(0, 1)
    
    return df


def process_parquet_bollinger(
    input_path: str,
    output_path: str = None,
    window: int = 20,
    price_col: str = "Close",
    n_std: float = 2.0,
    add_position: bool = True
) -> pd.DataFrame:
    """
    Charge un parquet, ajoute les bandes de Bollinger, et sauvegarde le résultat.
    
    Args:
        input_path: Chemin vers le fichier parquet d'entrée
        output_path: Chemin vers le fichier parquet de sortie (optionnel)
        window: Fenêtre temporelle pour le calcul
        price_col: Nom de la colonne de prix à utiliser
        n_std: Nombre d'écarts-types pour les bandes
        add_position: Si True, ajoute la position relative dans les bandes
    
    Returns:
        DataFrame enrichi avec les bandes de Bollinger
    """
    # Charger le parquet
    df = pd.read_parquet(input_path)
    
    # Ajouter les bandes de Bollinger
    df = add_bollinger(df, window=window, price_col=price_col, 
                      n_std=n_std, add_position=add_position)
    
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
        process_parquet_bollinger(input_file, output_file)
    else:
        print("Usage: python Bollinger.py <input_parquet> [output_parquet]")
