"""
Script de fusion chronologique des données BTC de 2021 à 2024
Les données 2025 sont réservées pour la validation
"""

import pandas as pd
import os

# Mapping des colonnes
column_mapping = {
    '0': 'Open Time', '1': 'Open', '2': 'High', '3': 'Low', '4': 'Close',
    '5': 'Volume', '6': 'Close Time', '7': 'Quote Volume', '8': 'Trades',
    '9': 'Taker Buy Base', '10': 'Taker Buy Quote', '11': 'Ignore'
}

def load_and_rename(filepath: str) -> pd.DataFrame:
    """Charge un fichier parquet et renomme les colonnes"""
    df = pd.read_parquet(filepath)
    
    # Renommer les colonnes si elles sont numériques (en string)
    current_cols = df.columns.tolist()
    rename_dict = {}
    
    for col in current_cols:
        col_str = str(col)
        if col_str in column_mapping:
            rename_dict[col] = column_mapping[col_str]
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
    
    return df

def merge_btc_data():
    """Fusionne les données BTC de 2021 à 2024"""
    
    # Répertoire courant (CryptoDataset)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Fichiers à fusionner (2021-2024, excluant 2025 pour validation)
    years = [2021, 2022, 2023, 2024]
    dataframes = []
    
    print("=" * 50)
    print("FUSION DES DONNÉES BTC (2021-2024)")
    print("=" * 50)
    
    for year in years:
        filepath = os.path.join(base_dir, f"BTC{year}.parquet")
        
        if os.path.exists(filepath):
            df = load_and_rename(filepath)
            print(f"✓ BTC{year}.parquet chargé : {len(df):,} lignes")
            dataframes.append(df)
        else:
            print(f"✗ BTC{year}.parquet non trouvé")
    
    if not dataframes:
        print("Aucune donnée à fusionner!")
        return None
    
    # Concaténation
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nTotal après concaténation : {len(merged_df):,} lignes")
    
    # Tri chronologique par Open Time
    if 'Open Time' in merged_df.columns:
        merged_df = merged_df.sort_values('Open Time').reset_index(drop=True)
        print("✓ Données triées chronologiquement par 'Open Time'")
    
    # Suppression des doublons éventuels
    initial_len = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    if len(merged_df) < initial_len:
        print(f"✓ {initial_len - len(merged_df)} doublons supprimés")
    
    # Sauvegarde
    output_path = os.path.join(base_dir, "BTC_merged_2021_2024.parquet")
    merged_df.to_parquet(output_path, index=False)
    print(f"\n✓ Fichier sauvegardé : BTC_merged_2021_2024.parquet")
    print(f"  Taille finale : {len(merged_df):,} lignes")
    
    # Afficher les colonnes
    print(f"\nColonnes : {list(merged_df.columns)}")
    
    # Aperçu des dates
    if 'Open Time' in merged_df.columns:
        print(f"\nPériode couverte :")
        print(f"  Début : {merged_df['Open Time'].min()}")
        print(f"  Fin   : {merged_df['Open Time'].max()}")
    
    print("=" * 50)
    
    return merged_df

if __name__ == "__main__":
    df = merge_btc_data()

