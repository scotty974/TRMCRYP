"""
Script de test pour le modèle PricePredictor sur BTC2025
"""

import sys
import gc
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Configuration des chemins
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.insert(0, str(project_root))

# Imports Finance
try:
    from Finance.MMS import add_mms
    from Finance.Tendance import add_tendance
    from Finance.ECT import add_ecart_type
    from Finance.Bollinger import add_bollinger
    from Finance.Fibonacci import add_fibonacci_levels
    from Finance.MME import calculate_mme
    from Finance.MACD import calculate_macd
    from Finance.RSI import calculate_rsi
    from Finance.Volumes import analyze_volume
    from Finance.Ichimoku import calculate_ichimoku
    from Finance.config import get_preset
except ImportError as e:
    print(f"Erreur d'import Finance : {e}")
    print(f"Chemin projet recherché: {project_root}")
    sys.exit(1)

# Imports TensorFlow
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Conv1D, MultiHeadAttention, LayerNormalization as KerasLayerNorm


class PricePredictor(KerasModel):
    """Modèle de prédiction de prix avec CNN, Attention et LSTM"""
    
    def __init__(self, input_shape, forecast_horizon=1, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_config = input_shape
        self.forecast_horizon = forecast_horizon
        
        self.feature_extractor = tf.keras.Sequential([
            Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
            Conv1D(128, 3, activation='relu', padding='same')
        ])
        
        self.temporal_attention = MultiHeadAttention(num_heads=4, key_dim=32)
        self.layernorm = KerasLayerNorm()
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=False)
        
        self.prediction_module = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='gelu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation='gelu'),
            tf.keras.layers.Dense(forecast_horizon)
        ])
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "input_shape": self.input_shape_config,
            "forecast_horizon": self.forecast_horizon,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
    def call(self, inputs):
        features = self.feature_extractor(inputs)
        attn_output = self.temporal_attention(features, features)
        x = self.layernorm(features + attn_output)
        lstm_out = self.lstm(x)
        return self.prediction_module(lstm_out)


class ResidualLSTMBlock(tf.keras.layers.Layer):
    """Bloc résiduel LSTM (utilisé en fallback)"""
    
    def __init__(self, d_model, dropout_rate=0.2):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(d_model, return_sequences=True, activation='tanh')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        residual = x
        x_lstm = self.lstm(x, training=training)
        x_lstm = self.dropout(x_lstm, training=training)
        return self.layer_norm(x_lstm + residual)


def build_price_predictor(input_shape, output_dim=1):
    """Construit un modèle alternatif (utilisé en fallback)"""
    from tensorflow.keras import layers, models, Input
    
    inputs = Input(shape=input_shape)
    x = layers.Normalization(axis=-1, name="normalization")(inputs)
    
    d_model = 64
    x = layers.Dense(d_model, activation="relu", name="projection")(x)
    x = ResidualLSTMBlock(d_model)(x)
    x = ResidualLSTMBlock(d_model)(x)
    x = layers.GlobalAveragePooling1D(name="global_pooling")(x)
    x = layers.Dense(32, activation='relu', name="dense_hidden")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(output_dim, activation='linear', name="output")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="PricePredictor_ResLSTM_ClassStyle")


def load_btc2025():
    """Charge le fichier BTC2025.parquet"""
    btc_path = project_root / "CryptoDataset" / "BTC2025.parquet"
    
    if not btc_path.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {btc_path}")
    
    print(f"Chargement de {btc_path}...")
    df = pd.read_parquet(btc_path)
    
    column_mapping = {
        '0': 'Open Time', '1': 'Open', '2': 'High', '3': 'Low', '4': 'Close',
        '5': 'Volume', '6': 'Close Time', '7': 'Quote Volume', '8': 'Trades',
        '9': 'Taker Buy Base', '10': 'Taker Buy Quote', '11': 'Ignore'
    }
    
    if set(df.columns).intersection(set(column_mapping.keys())):
        df = df.rename(columns=column_mapping)
    
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Open Time' in df.columns:
        df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
        df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✓ Données chargées: {len(df)} lignes")
    return df


def enrich_data(df):
    """Enrichit les données avec tous les indicateurs techniques"""
    print("\nCalcul des indicateurs techniques...")
    
    config = get_preset('multi')
    mms_windows = config['mms_windows']
    ect_windows = config['ect_windows']
    
    df = add_mms(df, windows=mms_windows, price_col='Close')
    df = add_tendance(df, mms_short=mms_windows[0], mms_medium=mms_windows[1], 
                     mms_long=mms_windows[2], price_col='Close')
    df = add_ecart_type(df, windows=ect_windows, price_col='Close')
    df = add_bollinger(df, window=config['bollinger_window'], price_col='Close')
    df = add_fibonacci_levels(df, window=config['fibonacci_window'], add_distance=True)
    df = calculate_mme(df, window=mms_windows[0])
    df = calculate_macd(df)
    df = calculate_rsi(df, window=14)
    df = analyze_volume(df, window=mms_windows[0])
    df = calculate_ichimoku(df)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"✓ Indicateurs calculés: {len(df)} lignes après nettoyage")
    return df


def prepare_features(df):
    """Sélectionne et prépare les features"""
    feature_cols = ['Close', 'Volume', 'Log_Returns']
    
    potential_indicators = [
        'RSI_14', 'MACD_Line', 'Signal_Line', 'MACD_Hist',
        'Tendance_Code', 
        'Bollinger_Upper', 'Bollinger_Lower', 'Bollinger_Middle', 'Bollinger_Position',
        'Fib_0_236', 'Fib_0_382', 'Fib_0_5', 'Fib_0_618', 'Fib_0_786', 'Fib_Position',
        'Tenkan_Sen', 'Kijun_Sen', 'Senkou_Span_A', 'Senkou_Span_B',
        'Volume_Surge'
    ]
    
    for col in df.columns:
        if col.startswith(('MMS_', 'ECT_', 'MME_', 'Volume_SMA_')):
            feature_cols.append(col)
    
    for col in potential_indicators:
        if col in df.columns:
            feature_cols.append(col)
    
    feature_cols = sorted(set(feature_cols))
    print(f"\n✓ {len(feature_cols)} features sélectionnées")
    return feature_cols


def create_sequences(data, targets, sequence_length=60):
    """Crée des séquences pour le modèle LSTM"""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def get_expected_features(model):
    """Obtient le nombre de features attendu par le modèle"""
    try:
        if hasattr(model, 'input_shape_config') and model.input_shape_config:
            return model.input_shape_config[2] if len(model.input_shape_config) > 2 else None
        elif hasattr(model, 'input_shape') and model.input_shape:
            return model.input_shape[2] if len(model.input_shape) > 2 else None
        elif model.layers and hasattr(model.layers[0], 'input_shape') and model.layers[0].input_shape:
            first_layer_shape = model.layers[0].input_shape
            if first_layer_shape and len(first_layer_shape) > 2:
                return first_layer_shape[2]
        return None
    except:
        return None


def test_model():
    """Fonction principale de test"""
    print("=" * 60)
    print("TEST DU MODÈLE PricePredictor SUR BTC2025")
    print("=" * 60)
    
    df = load_btc2025()
    df = enrich_data(df)
    feature_cols = prepare_features(df)
    target_col = 'Log_Returns'
    target_col_idx = feature_cols.index(target_col)
    
    print("\nNormalisation des données...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(df[[target_col]])
    
    print("\nCrée les séquences temporelles...")
    time_stamps = 60
    X, y = create_sequences(scaled_data, scaled_data[:, target_col_idx], time_stamps)
    
    print(f"✓ {len(X)} séquences créées (Shape X: {X.shape}, y: {y.shape})")
    
    print("\nChargement du modèle entraîné...")
    input_shape = (time_stamps, len(feature_cols))
    
    model_path_keras = current_file_path.parent / "best_model.keras"
    model_path_h5 = current_file_path.parent / "best_model.h5"
    model_path_alt = current_file_path.parent / "lstm_financial_model.keras"
    
    model = None
    if model_path_keras.exists():
        print(f"  Chargement de {model_path_keras}...")
        model = tf.keras.models.load_model(
            model_path_keras,
            custom_objects={'PricePredictor': PricePredictor}
        )
        print("✓ Modèle chargé avec succès!")
        
        expected_features = get_expected_features(model)
        if expected_features and expected_features != len(feature_cols):
            raise ValueError(
                f"Le modèle a été entraîné avec {expected_features} features, "
                f"mais les données de test ont {len(feature_cols)} features."
            )
    elif model_path_h5.exists():
        print(f"  Chargement de {model_path_h5}...")
        model = build_price_predictor(input_shape=input_shape, output_dim=1)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.load_weights(model_path_h5)
        print("✓ Poids du modèle chargés avec succès!")
    elif model_path_alt.exists():
        print(f"  Chargement de {model_path_alt}...")
        model = tf.keras.models.load_model(
            model_path_alt,
            custom_objects={'PricePredictor': PricePredictor}
        )
        print("✓ Modèle chargé avec succès!")
        
        expected_features = get_expected_features(model)
        if expected_features and expected_features != len(feature_cols):
            raise ValueError(
                f"Le modèle a été entraîné avec {expected_features} features, "
                f"mais les données de test ont {len(feature_cols)} features."
            )
    else:
        print("⚠ Aucun modèle entraîné trouvé. Création d'un nouveau modèle (non entraîné)...")
        model = build_price_predictor(input_shape=input_shape, output_dim=1)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print("\nRésumé du modèle:")
    model.summary()
    
    print("\nTest des prédictions sur toute la période 2025...")
    batch_size = 256
    print(f"  Prédictions par batches de {batch_size}...")
    
    predictions = model.predict(X, batch_size=batch_size, verbose=1)
    sample_y = y.copy()
    sample_size = len(predictions)
    
    print(f"✓ Prédictions effectuées sur {sample_size} séquences")
    print(f"  Shape: {predictions.shape}, Min: {predictions.min():.6f}, Max: {predictions.max():.6f}")
    
    del X, y
    gc.collect()
    
    print("\nConversion des prédictions en prix Close...")
    pred_log_returns = predictions.flatten()
    true_log_returns = sample_y
    
    pred_log_returns_denorm = target_scaler.inverse_transform(pred_log_returns.reshape(-1, 1)).flatten()
    true_log_returns_denorm = target_scaler.inverse_transform(true_log_returns.reshape(-1, 1)).flatten()
    
    start_idx = time_stamps
    end_idx = start_idx + sample_size
    indices = range(start_idx, end_idx)
    
    print("\nExtraction des prix Close et dates...")
    true_close_prices = df['Close'].iloc[indices].values
    dates = df['Date'].iloc[indices].values if 'Date' in df.columns else None
    
    if dates is None and 'Open Time' in df.columns:
        dates = pd.to_datetime(df['Open Time'].iloc[indices], unit='ms').values
    elif dates is None:
        dates = pd.date_range(start='2025-01-01', periods=sample_size, freq='1H')
    
    print(f"  Période: {dates[0]} à {dates[-1]} ({len(dates)} points)")
    
    pred_close_prices = np.zeros_like(true_close_prices)
    base_price = df['Close'].iloc[start_idx - 1]
    pred_close_prices[0] = base_price * np.exp(pred_log_returns_denorm[0])
    
    for i in range(1, len(pred_close_prices)):
        prev_real_price = df['Close'].iloc[start_idx + i - 1]
        pred_close_prices[i] = prev_real_price * np.exp(pred_log_returns_denorm[i])
    
    del df, scaled_data
    gc.collect()
    
    print("\nCréation du graphique de comparaison...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    ax1.plot(dates, true_close_prices, label='Prix Close Réel', linewidth=2, alpha=0.9, color='#00D9FF')
    ax1.plot(dates, pred_close_prices, label='Prix Close Prédit', linewidth=2, alpha=0.8, color='#FF6B6B', linestyle='--')
    ax1.set_title('Bitcoin (BTC) - Comparaison Prix Close Réel vs Prédit', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Prix Close (USD)', fontsize=12)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.set_facecolor('#F8F9FA')
    ax1.fill_between(dates, true_close_prices, alpha=0.1, color='#00D9FF')
    
    error = true_close_prices - pred_close_prices
    ax2.plot(dates, error, label='Erreur (Réel - Prédit)', linewidth=1.5, alpha=0.7, color='orange')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.fill_between(dates, error, 0, alpha=0.3, color='orange', where=(error >= 0))
    ax2.fill_between(dates, error, 0, alpha=0.3, color='red', where=(error < 0))
    ax2.set_title('Erreur de Prédiction', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Erreur (USD)', fontsize=12)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plot_path = current_file_path.parent / "test_predictions_close.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Graphique sauvegardé: {plot_path}")
    plt.close()
    
    print(f"\nStatistiques sur les prix Close:")
    print(f"  Prix réel - Min: ${true_close_prices.min():.2f}, Max: ${true_close_prices.max():.2f}, Mean: ${true_close_prices.mean():.2f}")
    print(f"  Prix prédit - Min: ${pred_close_prices.min():.2f}, Max: ${pred_close_prices.max():.2f}, Mean: ${pred_close_prices.mean():.2f}")
    print(f"  Erreur moyenne: ${np.mean(np.abs(error)):.2f}")
    print(f"  Erreur relative moyenne: {np.mean(np.abs(error) / true_close_prices) * 100:.2f}%")
    
    print("\n" + "=" * 60)
    print("STATISTIQUES")
    print("=" * 60)
    mse = np.mean((predictions.flatten() - sample_y) ** 2)
    mae = np.mean(np.abs(predictions.flatten() - sample_y))
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    
    print("\n✓ Test terminé avec succès!")
    return model, predictions


if __name__ == "__main__":
    try:
        model, predictions = test_model()
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")
        traceback.print_exc()
        sys.exit(1)