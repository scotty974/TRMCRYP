"""
Package Finance - Indicateurs techniques pour l'analyse financière.

Ce package contient les modules suivants:
- config: Configuration des presets de fenêtres temporelles (1m data)
- MMS: Moyennes mobiles simples
- Tendance: Analyse de tendance basée sur les moyennes mobiles
- ECT: Écart-type (volatilité)
- Bollinger: Bandes de Bollinger
- Fibonacci: Niveaux de retracement de Fibonacci

Pour des données à intervalle de 1 minute, utilisez les presets :
- 'multi' : Vue complète (1h, 1j, 1sem) [PAR DÉFAUT]
- 'short' : Court terme (30min, 1h, 4h)
- 'medium' : Moyen terme (1j, 5j, 10j)
- 'long' : Long terme (1sem, 1mois, 2mois)
"""

from Finance.MMS import add_mms, process_parquet_mms
from Finance.Tendance import add_tendance, process_parquet_tendance
from Finance.ECT import add_ecart_type, process_parquet_ect
from Finance.Bollinger import add_bollinger, process_parquet_bollinger
from Finance.Fibonacci import add_fibonacci_levels, process_parquet_fibonacci
from Finance.config import get_preset, list_presets, TIMEFRAME_PRESETS, minutes_to_human

__all__ = [
    # Fonctions d'indicateurs
    'add_mms',
    'add_tendance',
    'add_ecart_type',
    'add_bollinger',
    'add_fibonacci_levels',
    # Fonctions de traitement parquet
    'process_parquet_mms',
    'process_parquet_tendance',
    'process_parquet_ect',
    'process_parquet_bollinger',
    'process_parquet_fibonacci',
    # Configuration
    'get_preset',
    'list_presets',
    'TIMEFRAME_PRESETS',
    'minutes_to_human',
]

__version__ = '0.2.0'
