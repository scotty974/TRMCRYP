import pandas as pd
import matplotlib.pyplot as plt

def analyze_volume(df, window=20):
    """
    Analyse les volumes :
    1. Calcule la Moyenne Mobile Simple (SMA) des volumes.
    2. Identifie les pics de volume (volume > 2 * moyenne mobile).
    
    Args:
        df (pd.DataFrame): DataFrame contenant une colonne 'Volume'.
        window (int): Fenêtre pour la moyenne mobile des volumes (défaut: 20).
        
    Returns:
        pd.DataFrame: DataFrame enrichi avec 'Volume_SMA_{window}' et 'Volume_Surge'.
    """
    col_sma = f'Volume_SMA_{window}'
    
    # Calcul de la moyenne mobile des volumes
    df[col_sma] = df['Volume'].rolling(window=window).mean()
    
    # Identification des pics de volume (Volume > 2 * Moyenne)
    # True si pic, False sinon
    df['Volume_Surge'] = df['Volume'] > (df[col_sma] * 2)
    
    print(f"\n[VOLUMES] Colonne ajoutée : {col_sma}")
    print(f"[VOLUMES] Nombre de pics de volume détectés : {df['Volume_Surge'].sum()}")
    print(f"[VOLUMES] Aperçu des dernières valeurs :")
    print(df[['Volume', col_sma, 'Volume_Surge']].tail())
    
    return df

