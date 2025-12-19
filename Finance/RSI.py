import pandas as pd
import numpy as np

def calculate_rsi(df, window=14):
    """
    Calcule le RSI (Relative Strength Index).
    
    Args:
        df (pd.DataFrame): DataFrame contenant une colonne 'Close'.
        window (int): Période de calcul (défaut: 14).
        
    Returns:
        pd.DataFrame: DataFrame enrichi avec 'RSI_{window}'.
    """
    col_name = f'RSI_{window}'
    
    # Calcul des variations de prix
    delta = df['Close'].diff()
    
    # Séparation gains et pertes
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Calcul des moyennes lissées (Wilder's Smoothing)
    # Première moyenne simple, puis lissage exponentiel
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    # Pour être précis avec la méthode standard Wilder, on utiliserait ewm
    # avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    # avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
    # Mais rolling mean est souvent utilisé comme approximation simple. 
    # Utilisons ewm pour plus de précision standard trading.
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    
    # Calcul du RS
    rs = avg_gain / avg_loss
    
    # Calcul du RSI
    df[col_name] = 100 - (100 / (1 + rs))
    
    print(f"\n[RSI] Calculé avec fenêtre {window}")
    print(f"[RSI] Aperçu des dernières valeurs :")
    print(df[['Close', col_name]].tail())
    
    return df

