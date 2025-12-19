"""
Package Finance - Indicateurs techniques pour l'analyse financière.

Ce package contient les modules suivants:
- MMS: Moyennes mobiles simples
- Tendance: Analyse de tendance basée sur les moyennes mobiles
- ECT: Écart-type (volatilité)
- Bollinger: Bandes de Bollinger
- Fibonacci: Niveaux de retracement de Fibonacci
"""

from Finance.MMS import add_mms, process_parquet_mms
from Finance.Tendance import add_tendance, process_parquet_tendance
from Finance.ECT import add_ecart_type, process_parquet_ect
from Finance.Bollinger import add_bollinger, process_parquet_bollinger
from Finance.Fibonacci import add_fibonacci_levels, process_parquet_fibonacci

__all__ = [
    'add_mms',
    'add_tendance',
    'add_ecart_type',
    'add_bollinger',
    'add_fibonacci_levels',
    'process_parquet_mms',
    'process_parquet_tendance',
    'process_parquet_ect',
    'process_parquet_bollinger',
    'process_parquet_fibonacci',
]

__version__ = '0.1.0'
