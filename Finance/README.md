# Finance - Indicateurs Techniques

Module d'analyse technique pour les données de cryptomonnaies, implémentant les indicateurs assignés à Aymeric.

## Indicateurs Implémentés

### Indicateurs d'Aymeric

#### 1. Moyennes Mobiles Simples (MMS) - `MMS.py`
Calcule les moyennes mobiles simples sur différentes fenêtres temporelles.

**Colonnes créées :** `MMS_20`, `MMS_50`, `MMS_200`

**Utilisation :**
```python
from Finance.MMS import add_mms

df = add_mms(df, windows=[20, 50, 200], price_col="Close")
```

### 2. Analyse de Tendance - `Tendance.py`
Détermine la tendance du marché basée sur les moyennes mobiles.

**Colonnes créées :** `Tendance` (up/down/neutral), `Tendance_Code` (1/-1/0)

**Logique :**
- **Haussière (up)** : MMS_20 > MMS_50 > MMS_200
- **Baissière (down)** : MMS_20 < MMS_50 < MMS_200
- **Neutre (neutral)** : autres cas

**Utilisation :**
```python
from Finance.Tendance import add_tendance

df = add_tendance(df, mms_short=20, mms_medium=50, mms_long=200, price_col="Close")
```

### 3. Écart-type (Volatilité) - `ECT.py`
Mesure la volatilité du marché via l'écart-type glissant.

**Colonnes créées :** `ECT_20`, `ECT_50`, `ECT_200`

**Utilisation :**
```python
from Finance.ECT import add_ecart_type

df = add_ecart_type(df, windows=[20, 50, 200], price_col="Close")
```

### 4. Bandes de Bollinger - `Bollinger.py`
Calcule les bandes de Bollinger pour identifier la volatilité et les niveaux de support/résistance.

**Colonnes créées :** 
- `Bollinger_Middle` : Moyenne mobile (bande médiane)
- `Bollinger_Upper` : Bande supérieure (MMS + 2σ)
- `Bollinger_Lower` : Bande inférieure (MMS - 2σ)
- `Bollinger_Position` : Position relative dans les bandes (0-1)

**Utilisation :**
```python
from Finance.Bollinger import add_bollinger

df = add_bollinger(df, window=20, price_col="Close", n_std=2.0, add_position=True)
```

#### 5. Retracements de Fibonacci - `Fibonacci.py`
Calcule les niveaux de retracement de Fibonacci pour identifier les zones de support/résistance.

**Colonnes créées :** `Fib_0_236`, `Fib_0_382`, `Fib_0_5`, `Fib_0_618`, `Fib_0_786`, `Fib_Nearest_Distance`, `Fib_Nearest_Level`, `Fib_Position`

**Utilisation :**
```python
from Finance.Fibonacci import add_fibonacci_levels

df = add_fibonacci_levels(df, price_col="Close", window=None, add_distance=True)
```

### Indicateurs de Tom

#### 6. Moyenne Mobile Exponentielle (MME) - `MME.py`
Calcule la moyenne mobile exponentielle qui donne plus de poids aux données récentes.

**Colonnes créées :** `MME_{window}`

**Utilisation :**
```python
from Finance.MME import calculate_mme

df = calculate_mme(df, window=20)
```

#### 7. MACD (Moving Average Convergence Divergence) - `MACD.py`
Indicateur de momentum qui montre la relation entre deux moyennes mobiles.

**Colonnes créées :** `MACD_Line`, `Signal_Line`, `MACD_Hist`

**Utilisation :**
```python
from Finance.MACD import calculate_macd

df = calculate_macd(df, fast=12, slow=26, signal=9)
```

#### 8. RSI (Relative Strength Index) - `RSI.py`
Mesure la force relative du momentum des prix (suracheté > 70, survendu < 30).

**Colonnes créées :** `RSI_{window}`

**Utilisation :**
```python
from Finance.RSI import calculate_rsi

df = calculate_rsi(df, window=14)
```

#### 9. Analyse des Volumes - `Volumes.py`
Analyse les volumes de trading et identifie les pics de volume.

**Colonnes créées :** `Volume_SMA_{window}`, `Volume_Surge`

**Utilisation :**
```python
from Finance.Volumes import analyze_volume

df = analyze_volume(df, window=20)
```

#### 10. Nuage d'Ichimoku - `Ichimoku.py`
Système complet d'analyse technique japonais (Tenkan, Kijun, Senkou Span A & B).

**Colonnes créées :** `Tenkan_Sen`, `Kijun_Sen`, `Senkou_Span_A`, `Senkou_Span_B`

**Utilisation :**
```python
from Finance.Ichimoku import calculate_ichimoku

df = calculate_ichimoku(df)
```

## Configuration des Fenêtres Temporelles

⚠️ **Important** : Pour des données à intervalle de **1 minute**, les fenêtres classiques (20/50/200) sont **trop courtes** !

Le module propose des **presets adaptés** selon votre stratégie de trading :

| Preset | Description | Fenêtres MMS/ECT | Bollinger | Usage |
|--------|-------------|------------------|-----------|-------|
| **multi** ⭐ | Multi-timeframe (PAR DÉFAUT) | 1h, 1j, 1sem | 1 jour | Vue complète recommandée |
| **short** | Court terme | 30min, 1h, 4h | 1 heure | Scalping, daytrading |
| **medium** | Moyen terme | 1j, 5j, 10j | 1 jour | Swing trading |
| **long** | Long terme | 1sem, 1mois, 2mois | 1 semaine | Position trading |
| **classic** | Classique 20/50/200 | 20, 50, 200 min | 20 min | ⚠️ NON recommandé pour 1m |

### Liste des presets disponibles

```bash
python Finance/app.py --list-presets
```

## Utilisation

### 1. Script orchestrateur `app.py`

Pour enrichir un fichier parquet avec tous les indicateurs d'Aymeric :

```bash
# Utiliser le preset par défaut (multi-timeframe)
python Finance/app.py CryptoDataset/SOL2021.parquet

# Spécifier un preset
python Finance/app.py CryptoDataset/SOL2021.parquet output.parquet medium

# Court terme pour daytrading
python Finance/app.py CryptoDataset/SOL2021.parquet output.parquet short

# Long terme pour position trading
python Finance/app.py CryptoDataset/SOL2021.parquet output.parquet long
```

### 2. Script de test avec visualisations `test_indicators.py`

Pour tester tous les indicateurs et générer des visualisations :

```bash
# Utiliser le preset par défaut (multi-timeframe)
python Finance/test_indicators.py

# Utiliser un preset spécifique
python Finance/test_indicators.py short
python Finance/test_indicators.py medium
python Finance/test_indicators.py long
```

Ce script :
- Charge automatiquement `CryptoDataset/SOL_merged_2021_2024.parquet` (ou SOL2021.parquet en fallback)
- Applique tous les indicateurs avec le preset choisi
- Adapte la taille de l'échantillon au timeframe (3j pour short, 30j pour medium, 70j pour long)
- Génère 6 graphiques de visualisation avec les bonnes étiquettes
- Sauvegarde les graphiques dans `Finance/indicators_visualization.png`
- Affiche les graphiques à l'écran

### 3. Utilisation modulaire

#### Avec presets (recommandé)

```python
import pandas as pd
from Finance import add_mms, add_tendance, add_ecart_type, add_bollinger, add_fibonacci_levels
from Finance.config import get_preset

# Charger les données
df = pd.read_parquet("CryptoDataset/SOL2021.parquet")

# Charger un preset
config = get_preset('multi')  # ou 'short', 'medium', 'long'

# Ajouter les indicateurs avec les bonnes fenêtres
df = add_mms(df, windows=config['mms_windows'])
df = add_tendance(df, mms_short=config['mms_windows'][0], 
                     mms_medium=config['mms_windows'][1],
                     mms_long=config['mms_windows'][2])
df = add_ecart_type(df, windows=config['ect_windows'])
df = add_bollinger(df, window=config['bollinger_window'])
df = add_fibonacci_levels(df, window=config['fibonacci_window'])

# Sauvegarder
df.to_parquet("output_enhanced.parquet", index=False)
```

#### Utilisation manuelle (fenêtres personnalisées)

```python
import pandas as pd
from Finance import add_mms, add_tendance, add_ecart_type, add_bollinger, add_fibonacci_levels

# Charger les données
df = pd.read_parquet("CryptoDataset/SOL2021.parquet")

# Définir vos propres fenêtres (en minutes pour données 1m)
custom_windows = [120, 480, 2880]  # 2h, 8h, 2j

# Ajouter les indicateurs
df = add_mms(df, windows=custom_windows)
df = add_tendance(df, mms_short=120, mms_medium=480, mms_long=2880)
df = add_ecart_type(df, windows=custom_windows)
df = add_bollinger(df, window=480)
df = add_fibonacci_levels(df)

# Sauvegarder
df.to_parquet("output_enhanced.parquet", index=False)
```

## Structure des fichiers

```
Finance/
├── __init__.py           # Package principal
├── config.py            # Configuration des presets de fenêtres temporelles
├── MMS.py               # Moyennes mobiles simples (Aymeric)
├── Tendance.py          # Analyse de tendance (Aymeric)
├── ECT.py               # Écart-type / volatilité (Aymeric)
├── Bollinger.py         # Bandes de Bollinger (Aymeric)
├── Fibonacci.py         # Retracements de Fibonacci (Aymeric)
├── MME.py               # Moyenne mobile exponentielle (Tom)
├── MACD.py              # MACD (Tom)
├── RSI.py               # RSI (Tom)
├── Volumes.py           # Analyse des volumes (Tom)
├── Ichimoku.py          # Nuage d'Ichimoku (Tom)
├── app.py               # Script orchestrateur avec presets
├── test_indicators.py   # Tests avec visualisations
├── README.md            # Documentation complète
└── GUIDE_DEMARRAGE.md   # Guide de démarrage rapide
```

## Dépendances

- `pandas` : Manipulation de dataframes
- `numpy` : Calculs numériques
- `matplotlib` : Visualisations (pour test_indicators.py)
- `pyarrow` : Lecture/écriture de fichiers parquet

## Paramètres par défaut

- **Preset** : `multi` (multi-timeframe : 1h, 1j, 1sem)
- **Fenêtres Bollinger** : 1 jour, 2 écarts-types
- **Fibonacci** : Calcul global sur toute la série
- **Colonne prix** : `Close`

### Conversion minutes → temps lisible

Pour référence (données 1m) :
- **1 heure** = 60 minutes
- **1 jour** = 1440 minutes  
- **1 semaine** = 10080 minutes
- **1 mois** (~30j) = 43200 minutes
- **2 mois** (~60j) = 86400 minutes

## Notes techniques

- Les calculs utilisent `min_periods=1` pour gérer les premières valeurs
- Les fonctions retournent une copie du dataframe original
- Les erreurs de division par zéro sont gérées automatiquement
- Les valeurs NaN sont traitées de manière appropriée dans les calculs de tendance

## Auteur

Aymeric - Indicateurs techniques pour l'analyse de cryptomonnaies
