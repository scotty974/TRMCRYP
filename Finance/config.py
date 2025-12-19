"""
Configuration des fenêtres temporelles pour différents timeframes.

Pour des données à intervalle de 1 minute, les fenêtres doivent être ajustées
pour capturer les bonnes tendances selon l'objectif de trading.
"""

# Presets de fenêtres temporelles (en minutes pour données 1m)
TIMEFRAME_PRESETS = {
    # Court terme : Scalping et daytrading
    'short': {
        'mms_windows': [30, 60, 240],          # 30min, 1h, 4h
        'ect_windows': [30, 60, 240],
        'bollinger_window': 60,                 # 1h
        'fibonacci_window': 1440,               # 1 jour
        'description': 'Court terme - Scalping/Daytrading (30min, 1h, 4h)'
    },
    
    # Moyen terme : Swing trading
    'medium': {
        'mms_windows': [1440, 7200, 14400],    # 1j, 5j, 10j
        'ect_windows': [1440, 7200, 14400],
        'bollinger_window': 1440,               # 1 jour
        'fibonacci_window': 10080,              # 1 semaine
        'description': 'Moyen terme - Swing trading (1j, 5j, 10j)'
    },
    
    # Long terme : Position trading
    'long': {
        'mms_windows': [10080, 43200, 86400],  # 1sem, 1mois, 2mois
        'ect_windows': [10080, 43200, 86400],
        'bollinger_window': 10080,              # 1 semaine
        'fibonacci_window': None,               # Global sur toute la série
        'description': 'Long terme - Position trading (1sem, 1mois, 2mois)'
    },
    
    # Multi-timeframe : Vue complète
    'multi': {
        'mms_windows': [60, 1440, 10080],      # 1h, 1j, 1sem
        'ect_windows': [60, 1440, 10080],
        'bollinger_window': 1440,               # 1 jour
        'fibonacci_window': None,               # Global
        'description': 'Multi-timeframe - Vue complète (1h, 1j, 1sem)'
    },
    
    # Classique (ancien comportement - pour comparaison)
    'classic': {
        'mms_windows': [20, 50, 200],          # Fenêtres classiques (trop courtes pour 1m)
        'ect_windows': [20, 50, 200],
        'bollinger_window': 20,
        'fibonacci_window': None,
        'description': 'Classique - Fenêtres traditionnelles 20/50/200 (NON recommandé pour 1m)'
    }
}

# Preset par défaut recommandé
DEFAULT_PRESET = 'multi'


def get_preset(preset_name: str = None) -> dict:
    """
    Récupère un preset de configuration.
    
    Args:
        preset_name: Nom du preset ('short', 'medium', 'long', 'multi', 'classic')
                    Si None, utilise DEFAULT_PRESET
    
    Returns:
        Dictionnaire de configuration
    
    Raises:
        ValueError: Si le preset n'existe pas
    """
    if preset_name is None:
        preset_name = DEFAULT_PRESET
    
    if preset_name not in TIMEFRAME_PRESETS:
        available = ', '.join(TIMEFRAME_PRESETS.keys())
        raise ValueError(f"Preset '{preset_name}' inconnu. Disponibles: {available}")
    
    return TIMEFRAME_PRESETS[preset_name].copy()


def list_presets() -> None:
    """Affiche tous les presets disponibles avec leurs descriptions."""
    print("=" * 70)
    print("PRESETS DE FENÊTRES TEMPORELLES DISPONIBLES")
    print("=" * 70)
    
    for name, config in TIMEFRAME_PRESETS.items():
        is_default = " (PAR DÉFAUT)" if name == DEFAULT_PRESET else ""
        print(f"\n📊 {name.upper()}{is_default}")
        print(f"   {config['description']}")
        print(f"   - MMS/ECT: {config['mms_windows']}")
        print(f"   - Bollinger: {config['bollinger_window']}")
        print(f"   - Fibonacci: {config['fibonacci_window']}")
    
    print("\n" + "=" * 70)


def minutes_to_human(minutes: int) -> str:
    """
    Convertit des minutes en format lisible.
    
    Args:
        minutes: Nombre de minutes
    
    Returns:
        String formaté (ex: "1h", "5j", "2mois")
    """
    if minutes < 60:
        return f"{minutes}min"
    elif minutes < 1440:
        hours = minutes / 60
        return f"{hours:.0f}h" if hours == int(hours) else f"{hours:.1f}h"
    elif minutes < 10080:
        days = minutes / 1440
        return f"{days:.0f}j" if days == int(days) else f"{days:.1f}j"
    elif minutes < 43200:
        weeks = minutes / 10080
        return f"{weeks:.0f}sem" if weeks == int(weeks) else f"{weeks:.1f}sem"
    else:
        months = minutes / 43200
        return f"{months:.0f}mois" if months == int(months) else f"{months:.1f}mois"


if __name__ == "__main__":
    # Afficher tous les presets disponibles
    list_presets()
