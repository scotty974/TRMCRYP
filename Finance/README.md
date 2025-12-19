# Finance - Indicateurs Techniques

Module d'analyse technique pour les données de cryptomonnaies, implémentant les indicateurs assignés à Aymeric.

## Indicateurs Implémentés

### 1. Moyennes Mobiles Simples (MMS) - `MMS.py`
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

### 5. Retracements de Fibonacci - `Fibonacci.py`
Calcule les niveaux de retracement de Fibonacci pour identifier les zones de support/résistance.

**Colonnes créées :** `Fib_0_236`, `Fib_0_382`, `Fib_0_5`, `Fib_0_618`, `Fib_0_786`, `Fib_Nearest_Distance`, `Fib_Nearest_Level`, `Fib_Position`

**Utilisation :**
```python
from Finance.Fibonacci import add_fibonacci_levels

df = add_fibonacci_levels(df, price_col="Close", window=None, add_distance=True)
```

## Utilisation

### 1. Script orchestrateur `app.py`

Pour enrichir un fichier parquet avec tous les indicateurs d'Aymeric :

```bash
python Finance/app.py CryptoDataset/SOL2021.parquet
```

Ou spécifier le fichier de sortie :

```bash
python Finance/app.py CryptoDataset/SOL2021.parquet CryptoDataset/SOL2021_enhanced.parquet
```

### 2. Script de test avec visualisations `test_indicators.py`

Pour tester tous les indicateurs et générer des visualisations :

```bash
python Finance/test_indicators.py
```

Ce script :
- Charge automatiquement `CryptoDataset/SOL2021.parquet`
- Applique tous les indicateurs
- Génère 6 graphiques de visualisation
- Sauvegarde les graphiques dans `Finance/indicators_visualization.png`
- Affiche les graphiques à l'écran

### 3. Utilisation modulaire

Chaque indicateur peut être utilisé individuellement :

```python
import pandas as pd
from Finance import add_mms, add_tendance, add_ecart_type, add_bollinger, add_fibonacci_levels

# Charger les données
df = pd.read_parquet("CryptoDataset/SOL2021.parquet")

# Ajouter les indicateurs un par un
df = add_mms(df, windows=[20, 50, 200])
df = add_tendance(df)
df = add_ecart_type(df, windows=[20, 50, 200])
df = add_bollinger(df, window=20)
df = add_fibonacci_levels(df)

# Sauvegarder
df.to_parquet("output_enhanced.parquet", index=False)
```

## Structure des fichiers

```
Finance/
├── __init__.py           # Package principal
├── MMS.py               # Moyennes mobiles simples
├── Tendance.py          # Analyse de tendance
├── ECT.py               # Écart-type (volatilité)
├── Bollinger.py         # Bandes de Bollinger
├── Fibonacci.py         # Retracements de Fibonacci
├── app.py               # Script orchestrateur
├── test_indicators.py   # Tests avec visualisations
└── README.md            # Cette documentation
```

## Dépendances

- `pandas` : Manipulation de dataframes
- `numpy` : Calculs numériques
- `matplotlib` : Visualisations (pour test_indicators.py)
- `pyarrow` : Lecture/écriture de fichiers parquet

## Paramètres par défaut

- **Fenêtres MMS/ECT** : 20, 50, 200 périodes (configuration classique)
- **Fenêtre Bollinger** : 20 périodes, 2 écarts-types
- **Fibonacci** : Calcul global sur toute la série (peut être fenêtré)
- **Colonne prix** : `Close`

## Notes techniques

- Les calculs utilisent `min_periods=1` pour gérer les premières valeurs
- Les fonctions retournent une copie du dataframe original
- Les erreurs de division par zéro sont gérées automatiquement
- Les valeurs NaN sont traitées de manière appropriée dans les calculs de tendance

## Auteur

Aymeric - Indicateurs techniques pour l'analyse de cryptomonnaies
