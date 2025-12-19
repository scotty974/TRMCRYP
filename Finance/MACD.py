import pandas as pd

def calculate_macd(df, fast=12, slow=26, signal=9):
    """
    Calcule le MACD (Moving Average Convergence Divergence).
    1. MACD Line = EMA(fast) - EMA(slow)
    2. Signal Line = EMA(signal) de la MACD Line
    3. Histogram = MACD Line - Signal Line
    
    Args:
        df (pd.DataFrame): DataFrame contenant une colonne 'Close'.
        fast (int): Fenêtre rapide (défaut: 12).
        slow (int): Fenêtre lente (défaut: 26).
        signal (int): Fenêtre du signal (défaut: 9).
        
    Returns:
        pd.DataFrame: DataFrame enrichi avec 'MACD_Line', 'Signal_Line', 'MACD_Hist'.
    """
    # Calcul des EMA rapide et lente
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    # Calcul de la ligne MACD
    df['MACD_Line'] = ema_fast - ema_slow
    
    # Calcul de la ligne de Signal
    df['Signal_Line'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    
    # Calcul de l'Histogramme
    df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']
    
    print(f"\n[MACD] Calculé avec paramètres ({fast}, {slow}, {signal})")
    print(f"[MACD] Aperçu des dernières valeurs :")
    print(df[['MACD_Line', 'Signal_Line', 'MACD_Hist']].tail())
    
    return df

