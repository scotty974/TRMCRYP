"""
Script orchestrateur pour enrichir un dataframe avec tous les indicateurs d'Aymeric.

Ce script charge un fichier parquet, applique séquentiellement tous les indicateurs
(MMS, Tendance, ECT, Bollinger, Fibonacci), puis sauvegarde le résultat enrichi.
"""

import pandas as pd
import sys
from pathlib import Path
from Finance.MMS import add_mms
from Finance.Tendance import add_tendance
from Finance.ECT import add_ecart_type
from Finance.Bollinger import add_bollinger
from Finance.Fibonacci import add_fibonacci_levels


def enrich_dataframe(
    df: pd.DataFrame,
    price_col: str = "Close",
    mms_windows: list = [20, 50, 200],
    ect_windows: list = [20, 50, 200],
    bollinger_window: int = 20,
    bollinger_n_std: float = 2.0,
    fibonacci_window: int = None
) -> pd.DataFrame:
    """
    Enrichit un dataframe avec tous les indicateurs d'Aymeric.
    
    Args:
        df: DataFrame à enrichir
        price_col: Colonne de prix à utiliser
        mms_windows: Fenêtres pour les moyennes mobiles simples
        ect_windows: Fenêtres pour les écarts-types
        bollinger_window: Fenêtre pour les bandes de Bollinger
        bollinger_n_std: Nombre d'écarts-types pour Bollinger
        fibonacci_window: Fenêtre pour Fibonacci (None = global)
    
    Returns:
        DataFrame enrichi avec tous les indicateurs
    """
    print(f"Enrichissement du dataframe ({len(df)} lignes)...")
    
    # 1. Ajouter les moyennes mobiles simples
    print("  → Ajout des MMS...")
    df = add_mms(df, windows=mms_windows, price_col=price_col)
    
    # 2. Ajouter les tendances
    print("  → Ajout des tendances...")
    df = add_tendance(df, mms_short=mms_windows[0], mms_medium=mms_windows[1], 
                     mms_long=mms_windows[2], price_col=price_col)
    
    # 3. Ajouter les écarts-types
    print("  → Ajout des écarts-types...")
    df = add_ecart_type(df, windows=ect_windows, price_col=price_col)
    
    # 4. Ajouter les bandes de Bollinger
    print("  → Ajout des bandes de Bollinger...")
    df = add_bollinger(df, window=bollinger_window, price_col=price_col, 
                      n_std=bollinger_n_std, add_position=True)
    
    # 5. Ajouter les niveaux de Fibonacci
    print("  → Ajout des niveaux de Fibonacci...")
    df = add_fibonacci_levels(df, price_col=price_col, window=fibonacci_window, 
                             add_distance=True)
    
    print("✓ Enrichissement terminé !")
    return df


def process_parquet(
    input_path: str,
    output_path: str = None,
    price_col: str = "Close",
    **kwargs
) -> pd.DataFrame:
    """
    Charge un parquet, l'enrichit avec tous les indicateurs, et sauvegarde le résultat.
    
    Args:
        input_path: Chemin vers le fichier parquet d'entrée
        output_path: Chemin vers le fichier parquet de sortie (optionnel)
        price_col: Colonne de prix à utiliser
        **kwargs: Paramètres supplémentaires pour enrich_dataframe
    
    Returns:
        DataFrame enrichi
    """
    print(f"\n📊 Traitement de: {input_path}")
    
    # Charger le parquet
    print("Chargement du fichier parquet...")
    df = pd.read_parquet(input_path)
    print(f"  → {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Enrichir le dataframe
    df = enrich_dataframe(df, price_col=price_col, **kwargs)
    
    # Déterminer le chemin de sortie si non spécifié
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_enhanced{input_path_obj.suffix}")
    
    # Sauvegarder le résultat
    print(f"\n💾 Sauvegarde vers: {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"  → {len(df)} lignes, {len(df.columns)} colonnes")
    
    print("\n✅ Traitement terminé avec succès !")
    return df


def main():
    """Point d'entrée principal du script."""
    if len(sys.argv) < 2:
        print("Usage: python app.py <input_parquet> [output_parquet]")
        print("\nExemple:")
        print("  python app.py ../CryptoDataset/SOL2021.parquet")
        print("  python app.py ../CryptoDataset/SOL2021.parquet ../CryptoDataset/SOL2021_enhanced.parquet")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        process_parquet(input_file, output_file)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
