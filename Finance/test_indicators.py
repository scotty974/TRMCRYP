"""
Script de test pour vérifier les indicateurs d'Aymeric avec visualisations.
"""

import pandas as pd
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Ajouter le répertoire parent au path pour l'import
sys.path.insert(0, str(Path(__file__).parent.parent))

from Finance.MMS import add_mms
from Finance.Tendance import add_tendance
from Finance.ECT import add_ecart_type
from Finance.Bollinger import add_bollinger
from Finance.Fibonacci import add_fibonacci_levels
from Finance.MACD import calculate_macd
from Finance.MME import calculate_mme
from Finance.RSI import calculate_rsi
from Finance.Volumes import analyze_volume
from Finance.Ichimoku import calculate_ichimoku


def test_indicators():
    """Teste tous les indicateurs sur les données réelles SOL2021."""
    print("=" * 60)
    print("TEST DES INDICATEURS FINANCIERS D'AYMERIC")
    print("=" * 60)
    
    # Charger le fichier parquet réel (essayer le fichier fusionné d'abord, sinon SOL2021)
    print("\n1. Chargement des données SOL...")
    
    # Essayer d'abord le fichier fusionné qui contient plus de données
    parquet_path = Path(__file__).parent.parent / "CryptoDataset" / "SOL_merged_2021_2024.parquet"
    
    if not parquet_path.exists():
        # Fallback sur SOL2021.parquet
        parquet_path = Path(__file__).parent.parent / "CryptoDataset" / "SOL2021.parquet"
        print(f"   -> Utilisation de SOL2021.parquet")
    else:
        print(f"   -> Utilisation de SOL_merged_2021_2024.parquet")
    
    if not parquet_path.exists():
        print(f"   [X] Fichier non trouvé: {parquet_path}")
        print("   Veuillez vérifier le chemin du fichier.")
        sys.exit(1)
    
    df = pd.read_parquet(parquet_path)
    print(f"   [OK] DataFrame chargé: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Prendre un échantillon pour la visualisation (derniers 1000 points)
    sample_size = min(1000, len(df))
    df_sample = df.tail(sample_size).copy()
    print(f"   [OK] Échantillon pour visualisation: {len(df_sample)} lignes")
    
    # Convertir Open Time en datetime pour les plots
    df_sample['DateTime'] = pd.to_datetime(df_sample['Open Time'], unit='ms')
    print(f"   Colonnes disponibles: {list(df_sample.columns)}")
    
    # Test MMS
    print("\n2. Test des moyennes mobiles simples (MMS)...")
    df_sample = add_mms(df_sample, windows=[20, 50, 200], price_col='Close')
    print(f"   [OK] Colonnes ajoutées: MMS_20, MMS_50, MMS_200")
    
    # Test Tendance
    print("\n3. Test de l'analyse de tendance...")
    df_sample = add_tendance(df_sample, mms_short=20, mms_medium=50, mms_long=200, price_col='Close')
    print(f"   [OK] Colonnes ajoutées: Tendance, Tendance_Code")
    tendance_counts = df_sample['Tendance'].value_counts()
    print(f"   Répartition: {dict(tendance_counts)}")
    
    # Test ECT
    print("\n4. Test de l'écart-type...")
    df_sample = add_ecart_type(df_sample, windows=[20, 50, 200], price_col='Close')
    print(f"   [OK] Colonnes ajoutées: ECT_20, ECT_50, ECT_200")
    
    # Test Bollinger
    print("\n5. Test des bandes de Bollinger...")
    df_sample = add_bollinger(df_sample, window=20, price_col='Close', n_std=2.0, add_position=True)
    print(f"   [OK] Colonnes ajoutées: Bollinger_Middle, Bollinger_Upper, Bollinger_Lower, Bollinger_Position")
    
    # Test Fibonacci
    print("\n6. Test des niveaux de Fibonacci...")
    df_sample = add_fibonacci_levels(df_sample, price_col='Close', window=None, add_distance=True)
    print(f"   [OK] Colonnes ajoutées: Fib_0_236, Fib_0_382, Fib_0_5, Fib_0_618, Fib_0_786")

    # Test MACD
    print("\n7. Test du MACD...")
    df_sample = calculate_macd(df_sample)
    print(f"   [OK] Colonnes ajoutées: MACD_Line, Signal_Line, MACD_Hist")

    # Test MME
    print("\n8. Test de la MME...")
    df_sample = calculate_mme(df_sample, window=12)
    df_sample = calculate_mme(df_sample, window=26)
    print(f"   [OK] Colonnes ajoutées: MME_12, MME_26")

    # Test RSI
    print("\n9. Test du RSI...")
    df_sample = calculate_rsi(df_sample, window=14)
    print(f"   [OK] Colonnes ajoutées: RSI_14")

    # Test Volumes
    print("\n10. Test des Volumes...")
    df_sample = analyze_volume(df_sample, window=20)
    print(f"   [OK] Colonnes ajoutées: Volume_SMA_20, Volume_Surge")

    # Test Ichimoku
    print("\n11. Test d'Ichimoku...")
    df_sample = calculate_ichimoku(df_sample)
    print(f"   [OK] Colonnes ajoutées: Tenkan_Sen, Kijun_Sen, Senkou_Span_A, Senkou_Span_B")
    
    # Résumé final
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"DataFrame final: {len(df_sample)} lignes, {len(df_sample.columns)} colonnes")
    
    print("\n[SUCCESS] Tous les calculs sont terminés avec succès !")
    return df_sample


def plot_indicators(df):
    """Crée des visualisations pour tous les indicateurs."""
    print("\n" + "=" * 60)
    print("CRÉATION DES VISUALISATIONS")
    print("=" * 60)
    
    # Configuration de matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    fig = plt.figure(figsize=(20, 25))  # Plus grand pour 10 graphiques
    
    # 1. Prix, MMS et MME
    print("\n[PLOT] Graphique 1/10: Prix, MMS et MME...")
    ax1 = plt.subplot(5, 2, 1)
    ax1.plot(df['DateTime'], df['Close'], label='Close', linewidth=1.5, color='black', alpha=0.6)
    ax1.plot(df['DateTime'], df['MMS_20'], label='MMS 20', linewidth=1, color='blue', alpha=0.7)
    ax1.plot(df['DateTime'], df['MME_12'], label='MME 12', linewidth=1, color='cyan', linestyle='--')
    ax1.plot(df['DateTime'], df['MME_26'], label='MME 26', linewidth=1, color='magenta', linestyle='--')
    ax1.set_title('Prix, Moyennes Mobiles Simples et Exponentielles', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Prix ($)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Tendances
    print("[PLOT] Graphique 2/10: Analyse de tendance...")
    ax2 = plt.subplot(5, 2, 2, sharex=ax1)
    tendance_colors = {'up': 'green', 'down': 'red', 'neutral': 'gray'}
    for tendance, color in tendance_colors.items():
        mask = df['Tendance'] == tendance
        if mask.any():
            ax2.scatter(df.loc[mask, 'DateTime'], df.loc[mask, 'Close'], 
                       label=f'Tendance {tendance}', color=color, alpha=0.6, s=10)
    ax2.set_title('Analyse de Tendance (basée sur MMS)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prix ($)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Écart-types (Volatilité)
    print("[PLOT] Graphique 3/10: Volatilité (écart-type)...")
    ax3 = plt.subplot(5, 2, 3, sharex=ax1)
    ax3.plot(df['DateTime'], df['ECT_20'], label='ECT 20', linewidth=1, color='purple')
    ax3.plot(df['DateTime'], df['ECT_50'], label='ECT 50', linewidth=1, color='orange')
    ax3.set_title('Volatilité (Écart-type glissant)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Écart-type ($)')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Bandes de Bollinger
    print("[PLOT] Graphique 4/10: Bandes de Bollinger...")
    ax4 = plt.subplot(5, 2, 4, sharex=ax1)
    ax4.plot(df['DateTime'], df['Close'], label='Close', linewidth=1, color='black', alpha=0.5)
    ax4.plot(df['DateTime'], df['Bollinger_Upper'], label='Upper', linewidth=1, color='red', linestyle='--')
    ax4.plot(df['DateTime'], df['Bollinger_Lower'], label='Lower', linewidth=1, color='green', linestyle='--')
    ax4.fill_between(df['DateTime'], df['Bollinger_Lower'], df['Bollinger_Upper'], alpha=0.1, color='gray')
    ax4.set_title('Bandes de Bollinger', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Prix ($)')
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Position dans les bandes de Bollinger
    print("[PLOT] Graphique 5/10: Position relative dans Bollinger...")
    ax5 = plt.subplot(5, 2, 5, sharex=ax1)
    ax5.plot(df['DateTime'], df['Bollinger_Position'], linewidth=1, color='purple')
    ax5.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5)
    ax5.axhline(y=0, color='green', linestyle='--', alpha=0.3)
    ax5.axhline(y=1, color='red', linestyle='--', alpha=0.3)
    ax5.set_title('Position relative Bollinger', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Position')
    ax5.grid(True, alpha=0.3)
    
    # 6. Niveaux de Fibonacci
    print("[PLOT] Graphique 6/10: Niveaux de retracement Fibonacci...")
    ax6 = plt.subplot(5, 2, 6, sharex=ax1)
    ax6.plot(df['DateTime'], df['Close'], label='Close', linewidth=1.5, color='black', alpha=0.8)
    fib_levels = ['Fib_0_236', 'Fib_0_382', 'Fib_0_5', 'Fib_0_618', 'Fib_0_786']
    fib_colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#90EE90', '#4169E1']
    for level, color in zip(fib_levels, fib_colors):
        ax6.plot(df['DateTime'], df[level], label=level.replace('_', '.'), 
                linewidth=1, linestyle='--', color=color, alpha=0.7)
    ax6.set_title('Niveaux Fibonacci', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Prix ($)')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)

    # 7. MACD
    print("[PLOT] Graphique 7/10: MACD...")
    ax7 = plt.subplot(5, 2, 7, sharex=ax1)
    ax7.plot(df['DateTime'], df['MACD_Line'], label='MACD', color='blue', linewidth=1.5)
    ax7.plot(df['DateTime'], df['Signal_Line'], label='Signal', color='orange', linewidth=1.5)
    ax7.bar(df['DateTime'], df['MACD_Hist'], label='Hist', color='gray', alpha=0.3)
    ax7.set_title('MACD', fontsize=12, fontweight='bold')
    ax7.legend(loc='best', fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 8. RSI
    print("[PLOT] Graphique 8/10: RSI...")
    ax8 = plt.subplot(5, 2, 8, sharex=ax1)
    ax8.plot(df['DateTime'], df['RSI_14'], label='RSI 14', color='purple', linewidth=1.5)
    ax8.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax8.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax8.fill_between(df['DateTime'], 30, 70, color='gray', alpha=0.1)
    ax8.set_title('RSI (14)', fontsize=12, fontweight='bold')
    ax8.set_ylim(0, 100)
    ax8.legend(loc='best', fontsize=8)
    ax8.grid(True, alpha=0.3)

    # 9. Volumes
    print("[PLOT] Graphique 9/10: Volumes...")
    ax9 = plt.subplot(5, 2, 9, sharex=ax1)
    ax9.bar(df['DateTime'], df['Volume'], label='Volume', color='gray', alpha=0.5)
    ax9.plot(df['DateTime'], df['Volume_SMA_20'], label='SMA 20', color='blue', linewidth=1)
    # Highlight surges
    surge_mask = df['Volume_Surge']
    if surge_mask.any():
        ax9.scatter(df.loc[surge_mask, 'DateTime'], df.loc[surge_mask, 'Volume'], 
                   color='red', label='Surge', zorder=5, s=10)
    ax9.set_title('Volumes', fontsize=12, fontweight='bold')
    ax9.set_ylabel('Volume')
    ax9.legend(loc='best', fontsize=8)
    ax9.grid(True, alpha=0.3)

    # 10. Ichimoku
    print("[PLOT] Graphique 10/10: Ichimoku...")
    ax10 = plt.subplot(5, 2, 10, sharex=ax1)
    ax10.plot(df['DateTime'], df['Close'], label='Close', color='black', alpha=0.5, linewidth=1)
    ax10.plot(df['DateTime'], df['Tenkan_Sen'], label='Tenkan', color='blue', linewidth=1)
    ax10.plot(df['DateTime'], df['Kijun_Sen'], label='Kijun', color='red', linewidth=1)
    # Cloud
    ax10.fill_between(df['DateTime'], df['Senkou_Span_A'], df['Senkou_Span_B'], 
                      color='green', alpha=0.2, label='Cloud')
    ax10.set_title('Ichimoku', fontsize=12, fontweight='bold')
    ax10.set_ylabel('Prix ($)')
    ax10.legend(loc='best', fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # Ajuster l'espacement
    plt.tight_layout()
    
    # Sauvegarder
    output_path = Path(__file__).parent / "indicators_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[SAVE] Graphiques sauvegardés: {output_path}")
    
    # Afficher
    print("[PLOT] Affichage des graphiques...")
    plt.show()


if __name__ == "__main__":
    try:
        df = test_indicators()
        
        # Afficher un aperçu des dernières lignes
        print("\n" + "=" * 60)
        print("APERÇU DES DONNÉES (dernières lignes)")
        print("=" * 60)
        cols_to_show = ['DateTime', 'Close', 'MMS_20', 'Tendance', 'ECT_20', 'Bollinger_Position']
        print(df[cols_to_show].tail(10).to_string(index=False))
        
        # Créer les visualisations
        plot_indicators(df)
        
        print("\n[SUCCESS] Test et visualisation terminés avec succès !")
        
    except Exception as e:
        print(f"\n[X] Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
