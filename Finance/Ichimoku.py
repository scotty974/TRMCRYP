import pandas as pd

def calculate_ichimoku(df):
    """
    Calcule l'indicateur Ichimoku Kinko Hyo.
    
    Args:
        df (pd.DataFrame): DataFrame contenant 'High', 'Low', 'Close'.
        
    Returns:
        pd.DataFrame: DataFrame enrichi avec:
            - Tenkan_Sen (Conversion Line)
            - Kijun_Sen (Base Line)
            - Senkou_Span_A (Leading Span A)
            - Senkou_Span_B (Leading Span B)
    """
    # Paramètres standards
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    displacement = 26
    
    # 1. Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = df['High'].rolling(window=tenkan_period).max()
    low_9 = df['Low'].rolling(window=tenkan_period).min()
    df['Tenkan_Sen'] = (high_9 + low_9) / 2
    
    # 2. Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = df['High'].rolling(window=kijun_period).max()
    low_26 = df['Low'].rolling(window=kijun_period).min()
    df['Kijun_Sen'] = (high_26 + low_26) / 2
    
    # 3. Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    # Projeté 26 périodes dans le futur -> Shift positif
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(displacement)
    
    # 4. Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    # Projeté 26 périodes dans le futur -> Shift positif
    high_52 = df['High'].rolling(window=senkou_b_period).max()
    low_52 = df['Low'].rolling(window=senkou_b_period).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(displacement)
    
    print(f"\n[ICHIMOKU] Calculé (9, 26, 52, 26)")
    print(f"[ICHIMOKU] Aperçu des dernières valeurs (avec nuage futur) :")
    cols = ['Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B']
    print(df[cols].tail())
    
    return df

