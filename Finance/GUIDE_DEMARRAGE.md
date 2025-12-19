# Guide de Démarrage Rapide

## 🎯 Objectif

Ce module calcule des indicateurs techniques financiers adaptés aux données de cryptomonnaies à intervalle de **1 minute**.

## ⚠️ Pourquoi les fenêtres classiques (20/50/200) ne marchent pas ?

Sur des données **1 minute** :
- MMS_20 = moyenne sur **20 minutes** seulement (trop court !)
- MMS_50 = moyenne sur **50 minutes** 
- MMS_200 = moyenne sur **3h20**

Ces fenêtres captent uniquement le **très court terme** et sont sensibles au bruit.

## ✅ Solution : Presets Adaptés

Nous avons créé des **presets** qui ajustent automatiquement les fenêtres selon votre stratégie :

### 🏃 Short (Court terme - Scalping/Daytrading)
```bash
python Finance/app.py CryptoDataset/SOL2021.parquet output.parquet short
```
- **Fenêtres** : 30min, 1h, 4h
- **Usage** : Trading intraday, mouvements rapides
- **Visualisation** : ~3 jours de données

### 📊 Medium (Moyen terme - Swing Trading)
```bash
python Finance/app.py CryptoDataset/SOL2021.parquet output.parquet medium
```
- **Fenêtres** : 1j, 5j, 10j
- **Usage** : Positions de plusieurs jours
- **Visualisation** : ~30 jours de données

### 📈 Long (Long terme - Position Trading)
```bash
python Finance/app.py CryptoDataset/SOL2021.parquet output.parquet long
```
- **Fenêtres** : 1sem, 1mois, 2mois
- **Usage** : Investissement long terme
- **Visualisation** : ~70 jours de données

### 🌟 Multi (PAR DÉFAUT - Recommandé)
```bash
python Finance/app.py CryptoDataset/SOL2021.parquet
```
- **Fenêtres** : 1h, 1j, 1sem
- **Usage** : Vue complète multi-timeframe
- **Visualisation** : ~70 jours de données

## 🚀 Démarrage Rapide

### 1. Lister les presets disponibles
```bash
python Finance/app.py --list-presets
```

### 2. Enrichir vos données
```bash
# Avec preset multi (défaut)
python Finance/app.py ../CryptoDataset/SOL2021.parquet

# Avec preset spécifique
python Finance/app.py ../CryptoDataset/SOL2021.parquet output.parquet medium
```

### 3. Tester avec visualisations
```bash
# Multi-timeframe
python Finance/test_indicators.py

# Court terme
python Finance/test_indicators.py short

# Moyen terme
python Finance/test_indicators.py medium

# Long terme
python Finance/test_indicators.py long
```

## 📊 Indicateurs Calculés

Pour chaque preset, **10 indicateurs** sont ajoutés automatiquement :

### Indicateurs d'Aymeric

#### 1. Moyennes Mobiles Simples (MMS)
- Colonnes : `MMS_X` où X est la fenêtre
- Exemple : `MMS_60`, `MMS_1440`, `MMS_10080` pour preset multi

#### 2. Analyse de Tendance
- Colonne : `Tendance` (up/down/neutral)
- Colonne : `Tendance_Code` (1/-1/0)
- Basé sur la comparaison des 3 MMS

#### 3. Écart-type (Volatilité)
- Colonnes : `ECT_X` où X est la fenêtre
- Mesure la volatilité sur chaque période

#### 4. Bandes de Bollinger
- `Bollinger_Middle` : Moyenne mobile
- `Bollinger_Upper` : Bande supérieure (MMS + 2σ)
- `Bollinger_Lower` : Bande inférieure (MMS - 2σ)
- `Bollinger_Position` : Position relative (0-1)

#### 5. Niveaux de Fibonacci
- `Fib_0_236`, `Fib_0_382`, `Fib_0_5`, `Fib_0_618`, `Fib_0_786`
- `Fib_Nearest_Distance` : Distance au niveau le plus proche
- `Fib_Nearest_Level` : Quel niveau est le plus proche
- `Fib_Position` : Position relative (0-1)

### Indicateurs de Tom

#### 6. Moyenne Mobile Exponentielle (MME)
- Colonne : `MME_X` où X est la fenêtre
- Plus réactive que la MMS

#### 7. MACD (Moving Average Convergence Divergence)
- `MACD_Line` : Différence entre EMA rapide et lente
- `Signal_Line` : EMA de la ligne MACD
- `MACD_Hist` : Histogramme (MACD - Signal)

#### 8. RSI (Relative Strength Index)
- Colonne : `RSI_X` où X est la fenêtre
- Suracheté : > 70
- Survendu : < 30

#### 9. Analyse des Volumes
- `Volume_SMA_X` : Moyenne mobile des volumes
- `Volume_Surge` : Pics de volume détectés

#### 10. Nuage d'Ichimoku
- `Tenkan_Sen` : Ligne de conversion (9 périodes)
- `Kijun_Sen` : Ligne de base (26 périodes)
- `Senkou_Span_A` : Nuage (span A)
- `Senkou_Span_B` : Nuage (span B)

## 💻 Utilisation en Python

```python
from Finance import (add_mms, add_tendance, add_ecart_type, add_bollinger, add_fibonacci_levels,
                     calculate_mme, calculate_macd, calculate_rsi, analyze_volume, calculate_ichimoku)
from Finance.config import get_preset
import pandas as pd

# Charger vos données
df = pd.read_parquet("CryptoDataset/SOL2021.parquet")

# Charger un preset
config = get_preset('multi')  # ou 'short', 'medium', 'long'

# Appliquer les indicateurs d'Aymeric
df = add_mms(df, windows=config['mms_windows'])
df = add_tendance(df, 
                 mms_short=config['mms_windows'][0],
                 mms_medium=config['mms_windows'][1],
                 mms_long=config['mms_windows'][2])
df = add_ecart_type(df, windows=config['ect_windows'])
df = add_bollinger(df, window=config['bollinger_window'])
df = add_fibonacci_levels(df, window=config['fibonacci_window'])

# Appliquer les indicateurs de Tom
df = calculate_mme(df, window=config['mms_windows'][0])
df = calculate_macd(df, fast=12*60, slow=26*60, signal=9*60)  # Adapté pour 1m
df = calculate_rsi(df, window=14*60)  # 14 heures pour 1m
df = analyze_volume(df, window=config['mms_windows'][0])
df = calculate_ichimoku(df)

# Sauvegarder
df.to_parquet("output_enhanced.parquet", index=False)
```

## 🎨 Visualisations

Le script `test_indicators.py` génère **10 graphiques** :

### Indicateurs d'Aymeric
1. **Prix et MMS** : Visualise les moyennes mobiles
2. **Tendances** : Points colorés selon la tendance (vert=haussière, rouge=baissière, gris=neutre)
3. **Volatilité** : Évolution de l'écart-type
4. **Bandes de Bollinger** : Prix dans les bandes
5. **Position Bollinger** : Position relative (0=bas, 1=haut)
6. **Niveaux Fibonacci** : Prix et niveaux de retracement

### Indicateurs de Tom
7. **MACD** : Ligne MACD, Signal et Histogramme
8. **RSI** : Indice de force relative avec zones suracheté/survendu
9. **Volumes** : Volumes avec moyenne mobile et pics détectés
10. **Ichimoku** : Nuage d'Ichimoku avec Tenkan et Kijun

Les graphiques sont sauvegardés dans `Finance/indicators_visualization.png`

## 🔧 Fenêtres Personnalisées

Si vous voulez des fenêtres spécifiques :

```python
from Finance import add_mms

# Fenêtres personnalisées (en minutes)
custom_windows = [120, 720, 4320]  # 2h, 12h, 3j

df = add_mms(df, windows=custom_windows, price_col="Close")
```

## 📝 Conversion Minutes → Temps

Pour référence :
- **1h** = 60 minutes
- **4h** = 240 minutes
- **1j** = 1440 minutes
- **1sem** = 10080 minutes
- **1mois** = 43200 minutes (30 jours)

## ❓ Aide

Pour toute question sur les presets :
```bash
python Finance/app.py --list-presets
```

Pour voir la configuration d'un preset :
```python
from Finance.config import get_preset

config = get_preset('medium')
print(config)
```

## 🎉 C'est tout !

Vos indicateurs sont maintenant **correctement calibrés** pour les données 1 minute. Bon trading ! 🚀
