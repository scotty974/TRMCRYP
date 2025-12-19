# Changelog - Module Finance

## Version 0.2.0 - Intégration complète (18 Déc 2024)

### ✨ Nouvelles fonctionnalités

#### Système de presets pour données 1 minute
- **Ajout de `config.py`** : Module de configuration avec 5 presets adaptés aux données haute fréquence (1m)
  - `short` : Court terme (30min, 1h, 4h) pour scalping/daytrading
  - `medium` : Moyen terme (1j, 5j, 10j) pour swing trading
  - `long` : Long terme (1sem, 1mois, 2mois) pour position trading
  - `multi` : Multi-timeframe (1h, 1j, 1sem) - PAR DÉFAUT recommandé
  - `classic` : Fenêtres classiques 20/50/200 (non recommandé pour 1m)

#### Indicateurs d'Aymeric (complétés)
1. **MMS** (Moyennes Mobiles Simples) - `MMS.py` ✅
2. **Tendance** (Analyse de tendance) - `Tendance.py` ✅
3. **ECT** (Écart-type / Volatilité) - `ECT.py` ✅
4. **Bollinger** (Bandes de Bollinger) - `Bollinger.py` ✅
5. **Fibonacci** (Retracements) - `Fibonacci.py` ✅

#### Indicateurs de Tom (intégrés)
6. **MME** (Moyenne Mobile Exponentielle) - `MME.py` ✅
7. **MACD** (Moving Average Convergence Divergence) - `MACD.py` ✅
8. **RSI** (Relative Strength Index) - `RSI.py` ✅
9. **Volumes** (Analyse des volumes) - `Volumes.py` ✅
10. **Ichimoku** (Nuage d'Ichimoku) - `Ichimoku.py` ✅

### 🔧 Améliorations

#### Script `app.py`
- Support des presets via argument CLI
- Intégration automatique des 10 indicateurs
- Adaptation des fenêtres MACD et RSI selon le preset
- Nouvelle option `--list-presets` pour afficher les presets disponibles

#### Script `test_indicators.py`
- **10 graphiques** au lieu de 6
- Support des presets pour tester différents timeframes
- Visualisations adaptatives selon le preset choisi
- Détection automatique des colonnes créées
- Taille d'échantillon adaptée au timeframe

#### Documentation
- **README.md** : Documentation complète des 10 indicateurs
- **GUIDE_DEMARRAGE.md** : Guide de démarrage rapide en français
- **CHANGELOG.md** : Ce fichier

### 📊 Visualisations

Le script de test génère maintenant 10 graphiques (grille 5x2) :
1. Prix et Moyennes Mobiles Simples
2. Analyse de Tendance (colorée)
3. Volatilité (Écart-type)
4. Bandes de Bollinger
5. Position relative Bollinger
6. Niveaux de retracement Fibonacci
7. MACD (ligne, signal, histogramme)
8. RSI avec zones suracheté/survendu
9. Volumes avec pics détectés
10. Nuage d'Ichimoku

### 🐛 Corrections

- Résolution du problème de merge Git
- Correction des imports manquants dans `test_indicators.py`
- Ajustement des fenêtres pour données 1 minute
- Correction des noms de colonnes dynamiques
- Gestion des colonnes RSI avec fenêtre variable

### 📝 Utilisation

```bash
# Enrichir avec preset multi (défaut)
python Finance/app.py CryptoDataset/SOL2021.parquet

# Enrichir avec preset spécifique
python Finance/app.py CryptoDataset/SOL2021.parquet output.parquet medium

# Tester avec visualisations
python Finance/test_indicators.py multi
python Finance/test_indicators.py short

# Lister les presets disponibles
python Finance/app.py --list-presets
```

### ⚠️ Notes importantes

**Pour des données à intervalle de 1 minute**, les fenêtres classiques (20/50/200) sont **trop courtes** !

- MMS_20 = 20 minutes (beaucoup trop court)
- MMS_200 = 3h20 (capture uniquement très court terme)

**Solution** : Utiliser les presets adaptés qui ajustent automatiquement les fenêtres.

---

## Version 0.1.0 - Première implémentation (17 Déc 2024)

### Indicateurs d'Aymeric (version initiale)
- Implémentation des 5 indicateurs de base
- Fenêtres classiques 20/50/200 (avant adaptation 1m)
- Scripts de test et visualisation de base

### Structure initiale
- Architecture modulaire
- Fonctions `add_*` et `process_parquet_*`
- Tests unitaires basiques
