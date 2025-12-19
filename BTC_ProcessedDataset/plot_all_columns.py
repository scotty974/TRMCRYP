"""
Script pour visualiser toutes les colonnes du BTC ProcessedDataset
Crée un plot par colonne pour vérifier la qualité des données
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

def plot_all_columns(parquet_file: str = None, sample_size: int = 10000):
    """
    Plot toutes les colonnes d'un fichier parquet en fonction du temps.
    
    Args:
        parquet_file: Chemin vers le fichier parquet (None = utiliser le fusionné)
        sample_size: Nombre de lignes à utiliser (pour accélérer, None = tout)
    """
    current_dir = Path(__file__).parent
    
    # Déterminer quel fichier charger
    if parquet_file is None:
        # Essayer le fichier fusionné d'abord
        merged_file = current_dir / "BTC_processed_merged_2017_2024.parquet"
        if merged_file.exists():
            parquet_file = merged_file
            print(f"📂 Utilisation du fichier fusionné: {merged_file.name}")
        else:
            # Sinon, prendre le premier fichier parquet trouvé
            child_dirs = [d for d in current_dir.iterdir() if d.is_dir()]
            for dir in child_dirs:
                parquet_files = list(dir.glob("*.parquet"))
                if parquet_files:
                    parquet_file = parquet_files[0]
                    print(f"📂 Utilisation de: {parquet_file}")
                    break
    
    if parquet_file is None or not Path(parquet_file).exists():
        print("❌ Aucun fichier parquet trouvé!")
        return
    
    print(f"\n{'=' * 70}")
    print("VISUALISATION DES COLONNES BTC PROCESSED DATASET")
    print(f"{'=' * 70}\n")
    
    # Charger les données
    print(f"📖 Chargement de: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    print(f"   ✓ {len(df):,} lignes, {len(df.columns)} colonnes chargées")
    
    # Prendre un échantillon si nécessaire
    if sample_size and len(df) > sample_size:
        print(f"   → Échantillon de {sample_size:,} lignes (dernières valeurs)")
        df = df.tail(sample_size).copy()
    
    # Identifier la colonne temporelle
    time_col = None
    if 'datetime' in df.columns:
        time_col = 'datetime'
        if df[time_col].dtype != 'datetime64[ns]':
            df[time_col] = pd.to_datetime(df[time_col])
    elif 'Open_Time' in df.columns:
        time_col = 'Open_Time'
        # Convertir en datetime si c'est un timestamp
        df['datetime'] = pd.to_datetime(df[time_col], unit='ms')
        time_col = 'datetime'
    
    if time_col is None:
        print("⚠️  Aucune colonne temporelle trouvée (datetime ou Open_Time)")
        print("    Utilisation de l'index comme axe temporel")
        df['index_time'] = range(len(df))
        time_col = 'index_time'
    
    print(f"   ✓ Colonne temporelle: {time_col}")
    
    # Créer le dossier de sortie
    output_dir = current_dir / "dataPreview"
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Dossier de sortie: {output_dir}")
    
    # Obtenir toutes les colonnes à plotter (exclure la colonne temporelle)
    cols_to_plot = [col for col in df.columns if col != time_col and col != 'index_time']
    
    print(f"\n🎨 Création de {len(cols_to_plot)} plots...")
    print(f"{'=' * 70}\n")
    
    # Style matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plotter chaque colonne
    success_count = 0
    error_count = 0
    
    for i, col in enumerate(cols_to_plot, 1):
        try:
            # Vérifier si la colonne est numérique
            if not np.issubdtype(df[col].dtype, np.number):
                print(f"[{i}/{len(cols_to_plot)}] ⏭️  {col} (non-numérique, ignoré)")
                continue
            
            # Créer le plot
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot principal
            ax.plot(df[time_col], df[col], linewidth=0.8, alpha=0.8, color='blue')
            
            # Titre et labels
            ax.set_title(f'{col} - Évolution temporelle', fontsize=14, fontweight='bold')
            ax.set_xlabel('Temps' if time_col != 'index_time' else 'Index', fontsize=11)
            ax.set_ylabel(col, fontsize=11)
            
            # Grille
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Statistiques en texte
            stats_text = (
                f"Min: {df[col].min():.6g}\n"
                f"Max: {df[col].max():.6g}\n"
                f"Mean: {df[col].mean():.6g}\n"
                f"Std: {df[col].std():.6g}\n"
                f"NaN: {df[col].isna().sum()}"
            )
            
            # Ajouter les stats dans un coin
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9,
                   family='monospace')
            
            # Ajuster la mise en page
            plt.tight_layout()
            
            # Sauvegarder
            output_path = output_dir / f"{col.replace('/', '_').replace(' ', '_')}.png"
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            success_count += 1
            print(f"[{i}/{len(cols_to_plot)}] ✓ {col}")
            
        except Exception as e:
            error_count += 1
            print(f"[{i}/{len(cols_to_plot)}] ❌ {col} - Erreur: {e}")
            plt.close('all')
    
    # Résumé
    print(f"\n{'=' * 70}")
    print("RÉSUMÉ")
    print(f"{'=' * 70}")
    print(f"✓ Plots réussis: {success_count}")
    if error_count > 0:
        print(f"❌ Erreurs: {error_count}")
    print(f"📁 Plots sauvegardés dans: {output_dir}")
    print(f"{'=' * 70}\n")
    
    # Créer un fichier index HTML pour visualiser facilement
    create_html_index(output_dir, cols_to_plot, success_count)


def create_html_index(output_dir: Path, columns: list, success_count: int):
    """Crée un fichier HTML pour visualiser facilement tous les plots"""
    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC ProcessedDataset - Aperçu des colonnes</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .stats {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .card-header {{
            background: #667eea;
            color: white;
            padding: 15px;
            font-weight: bold;
            font-size: 1.1em;
        }}
        .card-body {{
            padding: 0;
        }}
        .card-body img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .search-box {{
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .search-box input {{
            width: 100%;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #667eea;
            border-radius: 5px;
            box-sizing: border-box;
        }}
        .search-box input:focus {{
            outline: none;
            border-color: #764ba2;
        }}
        .no-results {{
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 1.2em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 BTC ProcessedDataset - Aperçu des colonnes</h1>
            <div class="stats">
                {success_count} colonnes visualisées
            </div>
        </div>
        
        <div class="search-box">
            <input type="text" id="searchInput" placeholder="🔍 Rechercher une colonne..." onkeyup="filterColumns()">
        </div>
        
        <div class="grid" id="columnsGrid">
"""
    
    # Ajouter chaque colonne
    for col in columns:
        sanitized_name = col.replace('/', '_').replace(' ', '_')
        img_path = f"{sanitized_name}.png"
        
        html_content += f"""
            <div class="card" data-column="{col.lower()}">
                <div class="card-header">{col}</div>
                <div class="card-body">
                    <img src="{img_path}" alt="{col}" loading="lazy">
                </div>
            </div>
"""
    
    html_content += """
        </div>
        <div class="no-results" id="noResults" style="display: none;">
            Aucune colonne ne correspond à votre recherche.
        </div>
    </div>
    
    <script>
        function filterColumns() {
            const input = document.getElementById('searchInput');
            const filter = input.value.toLowerCase();
            const grid = document.getElementById('columnsGrid');
            const cards = grid.getElementsByClassName('card');
            const noResults = document.getElementById('noResults');
            let visibleCount = 0;
            
            for (let i = 0; i < cards.length; i++) {
                const column = cards[i].getAttribute('data-column');
                if (column.includes(filter)) {
                    cards[i].style.display = '';
                    visibleCount++;
                } else {
                    cards[i].style.display = 'none';
                }
            }
            
            if (visibleCount === 0) {
                grid.style.display = 'none';
                noResults.style.display = 'block';
            } else {
                grid.style.display = 'grid';
                noResults.style.display = 'none';
            }
        }
    </script>
</body>
</html>
"""
    
    # Sauvegarder le fichier HTML
    html_path = output_dir / "index.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"📄 Index HTML créé: {html_path}")
    print(f"   → Ouvrir dans le navigateur pour visualiser tous les plots")


if __name__ == "__main__":
    # Paramètres
    parquet_file = sys.argv[1] if len(sys.argv) > 1 else None
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    
    print("\n" + "=" * 70)
    print("SCRIPT DE VISUALISATION - BTC PROCESSED DATASET")
    print("=" * 70)
    
    if parquet_file:
        print(f"Fichier spécifié: {parquet_file}")
    else:
        print("Aucun fichier spécifié, utilisation du fichier fusionné par défaut")
    
    print(f"Taille d'échantillon: {sample_size:,} lignes")
    
    print("\nOptions:")
    print("  python plot_all_columns.py                          # Utiliser le fichier fusionné")
    print("  python plot_all_columns.py chemin/vers/file.parquet # Fichier spécifique")
    print("  python plot_all_columns.py file.parquet 50000       # Avec échantillon de 50k lignes")
    print()
    
    try:
        plot_all_columns(parquet_file, sample_size)
        print("\n✅ Visualisation terminée avec succès!")
        print("   Ouvrez dataPreview/index.html dans votre navigateur")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
