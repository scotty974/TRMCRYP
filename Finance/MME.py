import pandas as pd

def calculate_mme(df, window=20):
    """
    Calcule la Moyenne Mobile Exponentielle (MME / EMA).
    
    Args:
        df (pd.DataFrame): Le DataFrame contenant une colonne 'Close'.
        window (int): La période pour la MME (ex: 20 jours).
        
    Returns:
        pd.DataFrame: Le DataFrame original avec une nouvele colonne 'MME_{window}'.
    """
    col_name = f'MME_{window}'
    
    # Calcul de la MME sur la colonne 'Close'
    # adjust=False correspond au calcul classique de l'EMA (Alpha = 2/(N+1))
    df[col_name] = df['Close'].ewm(span=window, adjust=False).mean()
    
    print(f"\n[MME] Colonne ajoutée : {col_name}")
    print(f"[MME] Aperçu des dernières valeurs :")
    print(df[['Close', col_name]].tail())
    
    return df

