"""
Script de fusion des données BTC ProcessedDataset
Fusionne tous les fichiers parquet des sous-dossiers (par année)
Les données sont déjà nettoyées et contiennent des features supplémentaires
"""

import pandas as pd
from pathlib import Path
import sys

def merge_btc_processed_data():
    """
    Fusionne tous les fichiers parquet du BTC_ProcessedDataset.
    Les données sont organisées par dossiers d'années, chaque fichier faisant ~76MB.
    """
    # Répertoire courant (BTC_ProcessedDataset)
    current_dir = Path(__file__).parent
    
    print("=" * 70)
    print("FUSION DES DONNÉES BTC PROCESSED (2021-2024)")
    print("=" * 70)
    
    # Parcourir tous les sous-dossiers
    child_dirs = [d for d in current_dir.iterdir() if d.is_dir()]
    
    if not child_dirs:
        print("❌ Aucun sous-dossier trouvé !")
        return None
    
    print(f"\n📁 {len(child_dirs)} dossier(s) trouvé(s): {[d.name for d in child_dirs]}")
    
    dataframes = []
    total_files = 0
    
    # Parcourir chaque dossier et charger tous les fichiers parquet
    for year_dir in sorted(child_dirs):
        print(f"\n📂 Traitement du dossier: {year_dir.name}")
        
        parquet_files = list(year_dir.glob("*.parquet"))
        
        if not parquet_files:
            print(f"   ⚠️  Aucun fichier parquet trouvé dans {year_dir.name}")
            continue
        
        print(f"   → {len(parquet_files)} fichier(s) parquet trouvé(s)")
        
        dir_dataframes = []
        for file in sorted(parquet_files):
            try:
                df = pd.read_parquet(file)
                dir_dataframes.append(df)
                print(f"      ✓ {file.name}: {len(df):,} lignes")
            except Exception as e:
                print(f"      ❌ Erreur lors de la lecture de {file.name}: {e}")
        
        if dir_dataframes:
            # Concaténer les fichiers du dossier
            year_df = pd.concat(dir_dataframes, ignore_index=True)
            print(f"   ✓ Total {year_dir.name}: {len(year_df):,} lignes")
            dataframes.append(year_df)
            total_files += len(dir_dataframes)
    
    if not dataframes:
        print("\n❌ Aucune donnée à fusionner!")
        return None
    
    print(f"\n🔄 Fusion de {total_files} fichiers parquet...")
    
    # Concaténation de tous les dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"   ✓ Total après concaténation: {len(merged_df):,} lignes")
    
    # Tri chronologique
    if 'datetime' in merged_df.columns:
        print("\n🔄 Tri chronologique par 'datetime'...")
        merged_df = merged_df.sort_values('datetime').reset_index(drop=True)
        print("   ✓ Données triées")
    elif 'Open_Time' in merged_df.columns:
        print("\n🔄 Tri chronologique par 'Open_Time'...")
        merged_df = merged_df.sort_values('Open_Time').reset_index(drop=True)
        print("   ✓ Données triées")
    
    # Suppression des doublons éventuels
    initial_len = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    if len(merged_df) < initial_len:
        print(f"   ✓ {initial_len - len(merged_df):,} doublons supprimés")
    
    # Informations sur les colonnes
    print(f"\n📊 Informations sur le dataset:")
    print(f"   Nombre de lignes: {len(merged_df):,}")
    print(f"   Nombre de colonnes: {len(merged_df.columns)}")
    print(f"   Colonnes: {list(merged_df.columns)}")
    
    # Aperçu des dates
    if 'datetime' in merged_df.columns:
        print(f"\n📅 Période couverte:")
        print(f"   Début: {merged_df['datetime'].min()}")
        print(f"   Fin:   {merged_df['datetime'].max()}")
    elif 'Open_Time' in merged_df.columns:
        print(f"\n📅 Période couverte (Open_Time):")
        print(f"   Début: {pd.to_datetime(merged_df['Open_Time'].min(), unit='ms')}")
        print(f"   Fin:   {pd.to_datetime(merged_df['Open_Time'].max(), unit='ms')}")
    
    # Statistiques basiques
    print(f"\n📈 Aperçu des données (prix):")
    if 'Close' in merged_df.columns:
        print(f"   Close - Min: ${merged_df['Close'].min():.2f}")
        print(f"   Close - Max: ${merged_df['Close'].max():.2f}")
        print(f"   Close - Moyenne: ${merged_df['Close'].mean():.2f}")
    
    # Sauvegarde
    output_path = current_dir / "BTC_processed_merged_2017_2024.parquet"
    print(f"\n💾 Sauvegarde vers: {output_path}")
    merged_df.to_parquet(output_path, index=False)
    print(f"   ✓ Fichier sauvegardé: {output_path.name}")
    print(f"   Taille: {len(merged_df):,} lignes, {len(merged_df.columns)} colonnes")
    
    print("\n" + "=" * 70)
    print("✅ FUSION TERMINÉE AVEC SUCCÈS!")
    print("=" * 70)
    
    return merged_df


def explore_data(df: pd.DataFrame):
    """Affiche des informations détaillées sur le dataset"""
    print("\n" + "=" * 70)
    print("EXPLORATION DES DONNÉES")
    print("=" * 70)
    
    print(f"\n🔍 Shape: {df.shape}")
    print(f"\n📋 Info du DataFrame:")
    print(df.info())
    
    print(f"\n📊 Statistiques descriptives:")
    print(df.describe())
    
    print(f"\n👁️  Aperçu des premières lignes:")
    print(df.head())
    
    print(f"\n👁️  Aperçu des dernières lignes:")
    print(df.tail())
    
    # Vérifier les valeurs manquantes
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Valeurs manquantes:")
        print(missing[missing > 0])
    else:
        print(f"\n✅ Aucune valeur manquante")


if __name__ == "__main__":
    # Fusionner les données
    df = merge_btc_processed_data()
    
    # Explorer si l'option est passée
    if df is not None and len(sys.argv) > 1 and sys.argv[1] == '--explore':
        explore_data(df)

