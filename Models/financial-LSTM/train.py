import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Configuration TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Erreur configuration GPU: {e}")

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import timeseries_dataset_from_array

# Configuration des chemins
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parents[2]
sys.path.append(str(project_root))

# Imports Finance
try:
    from Finance.MMS import add_mms
    from Finance.RSI import calculate_rsi
    from Finance.MACD import calculate_macd
    from Finance.Bollinger import add_bollinger
    from Finance.Tendance import add_tendance
    from Finance.ECT import add_ecart_type
    from Finance.Fibonacci import add_fibonacci_levels
    from Finance.MME import calculate_mme
    from Finance.Volumes import analyze_volume
    from Finance.Ichimoku import calculate_ichimoku
    from Finance.config import get_preset
except ImportError as e:
    print(f"Erreur d'import : {e}")
    sys.exit(1)


def safe_print(msg):
    """Print avec flush pour éviter les blocages SSH"""
    print(msg, flush=True)


# Chargement des données
safe_print("Chargement des données...")
file_path = project_root / "CryptoDataset" / "BTC_merged_2021_2024.parquet"

if not file_path.exists():
    safe_print(f"Erreur : Le fichier de données n'existe pas : {file_path}")
    sys.exit(1)

safe_print(f"Lecture du fichier: {file_path}")
df = pd.read_parquet(file_path)
safe_print(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")

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
    df = df.sort_values('Date')

# Feature Engineering
safe_print("Calcul des indicateurs...")
config = get_preset('multi')
mms_windows = config['mms_windows']
ect_windows = config['ect_windows']

df = add_mms(df, windows=mms_windows, price_col='Close')
df = add_tendance(df, mms_short=mms_windows[0], mms_medium=mms_windows[1], mms_long=mms_windows[2], price_col='Close')
df = add_ecart_type(df, windows=ect_windows, price_col='Close')
df = add_bollinger(df, window=config['bollinger_window'], price_col='Close')
df = add_fibonacci_levels(df, window=config['fibonacci_window'])
df = calculate_mme(df, window=mms_windows[0])
df = calculate_macd(df)
df = calculate_rsi(df, window=14)
df = analyze_volume(df, window=mms_windows[0])
df = calculate_ichimoku(df)
safe_print("Indicateurs calculés ✓")

# Calcul de la cible (Log Returns)
df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Sélection des features
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
target_col = 'Log_Returns'
target_col_idx = feature_cols.index(target_col)

# Préparation des datasets
training_to_test_ratio = 0.8
split_val = int(len(df) * training_to_test_ratio)

train_df = df.iloc[:split_val]
test_df = df.iloc[split_val:]

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])

time_stamps = 60
batch_size = 128

safe_print("Création des datasets TensorFlow...")

train_targets = train_scaled[:, target_col_idx]
test_targets = test_scaled[:, target_col_idx]

train_dataset = timeseries_dataset_from_array(
    data=train_scaled,
    targets=train_targets,
    sequence_length=time_stamps,
    batch_size=batch_size,
    shuffle=True,
    seed=42
)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

test_dataset = timeseries_dataset_from_array(
    data=test_scaled,
    targets=test_targets,
    sequence_length=time_stamps,
    batch_size=batch_size,
    shuffle=False
)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

n_train_samples = len(train_scaled) - time_stamps
n_test_samples = len(test_scaled) - time_stamps
safe_print(f"Train samples: ~{n_train_samples}, Test samples: ~{n_test_samples}")

# Architecture du modèle
class PricePredictor(Model):
    """Modèle de prédiction de prix avec CNN, Attention et LSTM"""
    
    def __init__(self, input_shape, forecast_horizon=1, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_config = input_shape
        self.forecast_horizon = forecast_horizon
        
        self.feature_extractor = Sequential([
            Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
            Conv1D(128, 3, activation='relu', padding='same')
        ])
        
        self.temporal_attention = MultiHeadAttention(num_heads=4, key_dim=32)
        self.layernorm = LayerNormalization()
        self.lstm = LSTM(128, return_sequences=False)
        
        self.prediction_module = Sequential([
            Dense(64, activation='gelu'),
            Dropout(0.1),
            Dense(32, activation='gelu'),
            Dense(forecast_horizon)
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

# Initialisation et entraînement
n_features = train_scaled.shape[1]
input_shape = (time_stamps, n_features)

safe_print("Initialisation du modèle PricePredictor...")
model = PricePredictor(input_shape=input_shape, forecast_horizon=1)

safe_print("Compilation du modèle...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    run_eagerly=False
)

model.build(input_shape=(None, time_stamps, n_features))
safe_print("Résumé du modèle:")
model.summary(print_fn=safe_print)

safe_print("Début de l'entraînement...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint(
        str(Path(__file__).parent / 'best_model.keras'), 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
]

history = model.fit(
    train_dataset, 
    epochs=50, 
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

# Sauvegarde
model_save_path = Path(__file__).parent / "lstm_financial_model.keras"
try:
    model.save(model_save_path, save_format='keras')
    safe_print(f"Modèle sauvegardé sous : {model_save_path}")
except Exception as e:
    safe_print(f"Note: Sauvegarde complète impossible ({e}), sauvegarde des poids uniquement.")
    model.save_weights(Path(__file__).parent / "lstm_financial_weights.h5")

# Visualisation
try:
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Courbe d\'apprentissage')
    plt.legend()
    
    plot_save_path = Path(__file__).parent / "training_history.png"
    plt.savefig(plot_save_path)
    plt.close()
    safe_print(f"Graphique sauvegardé sous : {plot_save_path}")
except Exception as e:
    safe_print(f"Erreur lors de l'affichage du graphique: {e}")

# Évaluation directionnelle
safe_print("\nEvaluation de la précision directionnelle...")

y_true_list = []
predictions_list = []

safe_print("Calcul des prédictions...")
for batch_x, batch_y in test_dataset:
    batch_pred = model.predict(batch_x, verbose=0)
    predictions_list.append(batch_pred.flatten())
    y_true_list.append(batch_y.numpy().flatten())

predictions = np.concatenate(predictions_list)
y_true = np.concatenate(y_true_list)

min_len = min(len(predictions), len(y_true))
predictions = predictions[:min_len]
y_true = y_true[:min_len]

pred_sign = np.sign(predictions)
true_sign = np.sign(y_true)
accuracy = np.mean(pred_sign == true_sign)

safe_print(f"Précision Directionnelle (Hausse/Baisse) : {accuracy * 100:.2f}%")
safe_print(f"Nombre d'échantillons évalués: {len(y_true)}")

safe_print("Terminé.")