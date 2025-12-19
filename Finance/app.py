"""
Script orchestrateur pour enrichir un dataframe avec tous les indicateurs d'Aymeric.

Ce script charge un fichier parquet, applique séquentiellement tous les indicateurs
(MMS, Tendance, ECT, Bollinger, Fibonacci), puis sauvegarde le résultat enrichi.

Pour des données à intervalle de 1 minute, utilisez les presets adaptés :
- 'short' : scalping/daytrading (30min, 1h, 4h)
- 'medium' : swing trading (1j, 5j, 10j)
- 'long' : position trading (1sem, 1mois, 2mois)
- 'multi' : multi-timeframe (1h, 1j, 1sem) [PAR DÉFAUT]
"""

import pandas as pd
import sys
from pathlib import Path
from Finance.MMS import add_mms
from Finance.Tendance import add_tendance
from Finance.ECT import add_ecart_type
from Finance.Bollinger import add_bollinger
from Finance.Fibonacci import add_fibonacci_levels
from Finance.MME import calculate_mme
from Finance.MACD import calculate_macd
from Finance.RSI import calculate_rsi
from Finance.Volumes import analyze_volume
from Finance.Ichimoku import calculate_ichimoku
from Finance.config import get_preset, list_presets, minutes_to_human


def enrich_dataframe(
    df: pd.DataFrame,
    price_col: str = "Close",
    preset: str = 'multi',
    mms_windows: list = None,
    ect_windows: list = None,
    bollinger_window: int = None,
    bollinger_n_std: float = 2.0,
    fibonacci_window: int = None
) -> pd.DataFrame:
    """
    Enrichit un dataframe avec tous les indicateurs d'Aymeric.
    
    Args:
        df: DataFrame à enrichir
        price_col: Colonne de prix à utiliser
        preset: Preset de fenêtres ('short', 'medium', 'long', 'multi', 'classic')
        mms_windows: Fenêtres pour les MMS (override preset si fourni)
        ect_windows: Fenêtres pour les écarts-types (override preset si fourni)
        bollinger_window: Fenêtre pour Bollinger (override preset si fourni)
        bollinger_n_std: Nombre d'écarts-types pour Bollinger
        fibonacci_window: Fenêtre pour Fibonacci (override preset si fourni)
    
    Returns:
        DataFrame enrichi avec tous les indicateurs
    """
    # Charger le preset
    config = get_preset(preset)
    
    # Override avec les paramètres fournis
    if mms_windows is None:
        mms_windows = config['mms_windows']
    if ect_windows is None:
        ect_windows = config['ect_windows']
    if bollinger_window is None:
        bollinger_window = config['bollinger_window']
    if fibonacci_window is None:
        fibonacci_window = config['fibonacci_window']
    
    print(f"Enrichissement du dataframe ({len(df)} lignes)...")
    print(f"  Preset: {preset} - {config['description']}")
    windows_str = [minutes_to_human(w) for w in mms_windows]
    print(f"  Fenêtres MMS: {windows_str}")
    
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
    
    # 6. Ajouter la moyenne mobile exponentielle (Tom)
    print("  → Ajout de la MME...")
    df = calculate_mme(df, window=mms_windows[0])
    
    # 7. Ajouter le MACD (Tom)
    print("  → Ajout du MACD...")
    # Adapter les fenêtres au preset
    if preset == 'short':
        fast, slow, signal = 12, 26, 9
    elif preset == 'medium':
        fast, slow, signal = 12*60, 26*60, 9*60
    else:
        fast, slow, signal = 12*60, 26*60, 9*60
    df = calculate_macd(df, fast=fast, slow=slow, signal=signal)
    
    # 8. Ajouter le RSI (Tom)
    print("  → Ajout du RSI...")
    rsi_window = 14 if preset == 'short' else 14*60
    df = calculate_rsi(df, window=rsi_window)
    
    # 9. Ajouter l'analyse des volumes (Tom)
    print("  → Ajout de l'analyse des volumes...")
    df = analyze_volume(df, window=mms_windows[0])
    
    # 10. Ajouter le nuage d'Ichimoku (Tom)
    print("  → Ajout du nuage d'Ichimoku...")
    df = calculate_ichimoku(df)
    
    print("✓ Enrichissement terminé !")
    return df


def process_parquet(
    input_path: str,
    output_path: str = None,
    price_col: str = "Close",
    preset: str = 'multi',
    **kwargs
) -> pd.DataFrame:
    """
    Charge un parquet, l'enrichit avec tous les indicateurs, et sauvegarde le résultat.
    
    Args:
        input_path: Chemin vers le fichier parquet d'entrée
        output_path: Chemin vers le fichier parquet de sortie (optionnel)
        price_col: Colonne de prix à utiliser
        preset: Preset de fenêtres temporelles ('short', 'medium', 'long', 'multi', 'classic')
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
    df = enrich_dataframe(df, price_col=price_col, preset=preset, **kwargs)
    
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
        print("Usage: python app.py <input_parquet> [output_parquet] [preset]")
        print("\nExemples:")
        print("  python app.py ../CryptoDataset/SOL2021.parquet")
        print("  python app.py ../CryptoDataset/SOL2021.parquet output.parquet")
        print("  python app.py ../CryptoDataset/SOL2021.parquet output.parquet medium")
        print("\nOptions spéciales:")
        print("  python app.py --list-presets    # Afficher tous les presets disponibles")
        sys.exit(1)
    
    # Option pour lister les presets
    if sys.argv[1] == '--list-presets':
        list_presets()
        sys.exit(0)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    preset = sys.argv[3] if len(sys.argv) > 3 else 'multi'
    
    try:
        process_parquet(input_file, output_file, preset=preset)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
