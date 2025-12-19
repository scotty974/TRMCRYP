# BTC ProcessedDataset

Dataset Bitcoin déjà nettoyé et enrichi avec des features supplémentaires.

## 📁 Structure

```
BTC_ProcessedDataset/
├── 2021/                    # Dossier pour l'année 2021
│   ├── btc_2021_part1.parquet
│   ├── btc_2021_part2.parquet
│   └── ...
├── 2022/                    # Dossier pour l'année 2022
├── 2023/                    # Dossier pour l'année 2023
├── 2024/                    # Dossier pour l'année 2024
├── merge_BTC.py            # Script de fusion
├── test.py                 # Script d'exploration
└── README.md               # Ce fichier
```

Les données sont organisées par année, chaque fichier parquet faisant environ **76 MB**.

## 📊 Colonnes disponibles

Les données contiennent les colonnes OHLCV de base plus des features calculées :

### Colonnes de base
- `datetime` : Date et heure
- `Open_Time` : Timestamp d'ouverture
- `Open`, `High`, `Low`, `Close` : Prix OHLC
- `Volume` : Volume de trading
- `close_time` : Timestamp de clôture
- `Quote_Volume` : Volume en quote
- `Trades` : Nombre de trades
- `Taker_Buy_Base` : Volume acheteur (base)
- `Taker_Buy_Quote` : Volume acheteur (quote)

### Features calculées

#### Rendements
- `ret` : Rendement simple
- `log_ret` : Rendement logarithmique

#### Volatilité réalisée (différentes fenêtres)
- `rv_5`, `rv_ann_5` : Volatilité 5 périodes
- `rv_15`, `rv_ann_15` : Volatilité 15 périodes
- `rv_30`, `rv_ann_30` : Volatilité 30 périodes
- `rv_60`, `rv_ann_60` : Volatilité 60 périodes
- `rv_120`, `rv_ann_120` : Volatilité 120 périodes
- `rv_240`, `rv_ann_240` : Volatilité 240 périodes
- `rv_720`, `rv_ann_720` : Volatilité 720 périodes
- `rv_1440`, `rv_ann_1440` : Volatilité 1440 périodes

#### Moyennes mobiles exponentielles (EMA)
- `ema_20`, `dist_ema_20` : EMA 20 et distance
- `ema_50`, `dist_ema_50` : EMA 50 et distance
- `ema_100`, `dist_ema_100` : EMA 100 et distance
- `ema_200`, `dist_ema_200` : EMA 200 et distance

#### Indicateurs techniques
- `atr_14` : Average True Range (14 périodes)
- `atr_pct_14` : ATR en pourcentage
- `rsi_14` : Relative Strength Index (14 périodes)

#### Mesures de risque (Value at Risk)
- `var_99_60`, `cvar_99_60` : VaR et CVaR 99% sur 60 périodes
- `var_99_240`, `cvar_99_240` : VaR et CVaR 99% sur 240 périodes
- `var_99_1440`, `cvar_99_1440` : VaR et CVaR 99% sur 1440 périodes

## 🚀 Utilisation

### 1. Fusionner tous les fichiers

```bash
# Fusionner toutes les années
python merge_BTC.py

# Fusionner et explorer
python merge_BTC.py --explore
```

Cela créera un fichier `BTC_processed_merged_2021_2024.parquet` contenant toutes les données fusionnées et triées chronologiquement.

### 2. Explorer les données

```bash
python test.py
```

### 3. Charger en Python

```python
import pandas as pd
from pathlib import Path

# Charger le fichier fusionné
df = pd.read_parquet("BTC_processed_merged_2021_2024.parquet")

print(f"Dataset shape: {df.shape}")
print(f"Colonnes: {df.columns.tolist()}")
print(df.head())

# Ou charger un fichier spécifique
df_2021 = pd.read_parquet("2021/btc_2021_part1.parquet")
```

### 4. Charger tous les fichiers d'une année

```python
import pandas as pd
from pathlib import Path

# Charger tous les fichiers de 2021
year_path = Path("2021")
parquet_files = list(year_path.glob("*.parquet"))

dataframes = []
for file in parquet_files:
    df = pd.read_parquet(file)
    dataframes.append(df)

df_2021 = pd.concat(dataframes, ignore_index=True)
df_2021 = df_2021.sort_values('datetime').reset_index(drop=True)

print(f"Données 2021: {len(df_2021):,} lignes")
```

## 📈 Analyse des données

### Exemple : Visualiser le prix et la volatilité

```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_parquet("BTC_processed_merged_2021_2024.parquet")

# Convertir datetime si nécessaire
if 'datetime' not in df.columns and 'Open_Time' in df.columns:
    df['datetime'] = pd.to_datetime(df['Open_Time'], unit='ms')

# Plot prix et volatilité
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

# Prix
ax1.plot(df['datetime'], df['Close'], label='BTC Price', color='blue')
ax1.set_ylabel('Prix ($)')
ax1.set_title('Prix Bitcoin et Volatilité')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Volatilité réalisée annualisée (1 jour)
ax2.plot(df['datetime'], df['rv_ann_1440'], label='Volatilité annualisée (1j)', color='red')
ax2.set_ylabel('Volatilité')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Exemple : Analyser les rendements

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("BTC_processed_merged_2021_2024.parquet")

# Statistiques des rendements
print("Statistiques des rendements logarithmiques:")
print(df['log_ret'].describe())

# Distribution des rendements
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['log_ret'].dropna(), bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Rendement logarithmique')
plt.ylabel('Fréquence')
plt.title('Distribution des rendements')

plt.subplot(1, 2, 2)
plt.plot(df['datetime'], df['log_ret'].cumsum())
plt.xlabel('Date')
plt.ylabel('Rendement cumulé')
plt.title('Rendements cumulés')

plt.tight_layout()
plt.show()
```

## ⚠️ Notes importantes

1. **Intervalles** : Les données sont à intervalle de **1 minute**
2. **Taille** : Le dataset complet (2021-2024) contient plusieurs millions de lignes
3. **Mémoire** : Charger tout le dataset peut nécessiter plusieurs GB de RAM
4. **Features** : Toutes les features sont déjà calculées, pas besoin de les recalculer

## 🔄 Différence avec CryptoDataset

| Caractéristique | CryptoDataset | BTC_ProcessedDataset |
|----------------|---------------|---------------------|
| **État** | Données brutes | Données nettoyées + features |
| **Colonnes** | 12 colonnes de base | 47 colonnes (base + features) |
| **Organisation** | 1 fichier par année | Fichiers fragmentés par année |
| **Taille fichiers** | ~100-500 MB | ~76 MB par fichier |
| **Usage** | Pour calculer vos propres indicateurs | Pour ML/analyse directe |

## 🎯 Cas d'usage

### Pour le Machine Learning
Utilisez `BTC_ProcessedDataset` - les features sont déjà calculées

### Pour l'analyse technique personnalisée
Utilisez `CryptoDataset` et ajoutez vos propres indicateurs avec le module `Finance`

## 📝 Scripts disponibles

- **`merge_BTC.py`** : Fusionne tous les fichiers parquet en un seul
- **`test.py`** : Script simple pour explorer la structure des données
- **`plot_all_columns.py`** : Visualise toutes les colonnes pour vérifier la qualité des données
- **`README.md`** : Cette documentation

### Visualisation des données - `plot_all_columns.py`

Ce script crée un plot pour chaque colonne afin de vérifier visuellement la qualité des données.

```bash
# Utiliser le fichier fusionné (par défaut)
python plot_all_columns.py

# Spécifier un fichier particulier
python plot_all_columns.py 2021/btc_2021_part1.parquet

# Avec un échantillon plus grand (50k lignes au lieu de 10k)
python plot_all_columns.py BTC_processed_merged_2017_2024.parquet 50000
```

**Résultat** :
- Crée un dossier `dataPreview/` avec un plot PNG par colonne
- Génère un fichier `index.html` pour visualiser facilement tous les plots
- Affiche les statistiques de chaque colonne (min, max, mean, std, NaN)

**Pour visualiser** :
1. Exécutez le script
2. Ouvrez `dataPreview/index.html` dans votre navigateur
3. Utilisez la barre de recherche pour trouver une colonne spécifique

## 🆘 Support

Pour toute question ou problème, vérifiez :
1. Que tous les sous-dossiers (2021, 2022, 2023, 2024) existent
2. Que les fichiers parquet sont valides
3. Que vous avez assez de RAM pour charger les données
