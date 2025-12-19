import pickle
from models.Transformers import Transformers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

sentiment_mapping = {
    0: "Bearish",      
    1: "Bullish",          
    2: "Neutral", 
}

# Charger la configuration
with open(r"LLMSEM/models/model_config.pkl", "rb") as f:
    config = pickle.load(f)

# Charger le tokenizer
with open(r"LLMSEM/models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Recréer le modèle avec la même architecture
model = Transformers(
    vocab_size=config['VOCAB_SIZE'],
    embed_size=config['EMBED_DIM'],
    num_heads=config['NUM_HEADS'],
    ff_dim=config['FF_DIM']
)
model.build(input_shape=(None, config['MAX_LEN']))

# Compiler le modèle
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Charger les poids
model.load_weights(r"LLMSEM/models/transformers_weights.weights.h5")


test_texts = [
    # Articles crypto simulés

    # Niveau 1 : clair et direct
    "Bitcoin surges past $60,000 as investors rush back into crypto.",
    "Ethereum suffers a minor dip following regulatory news.",
    "Dogecoin rallies as social media hype intensifies.",
    "Litecoin value remains stable amid market calm.",
    "Crypto market shows strong gains across altcoins.",
    "Solana continues its upward trend, hitting new highs.",

    # Niveau 2 : phrases un peu nuancées
    "Investors are cautiously optimistic about Cardano's new update.",
    "The market reacted negatively to news of a hack on a DeFi platform.",
    "Shiba Inu holders feel anxious as volatility spikes.",
    "Traders regret missing the early surge in Avalanche tokens.",
    "Binance Coin price unexpectedly rises despite bearish sentiment.",
    "Polkadot continues steady growth, boosting investor confidence.",

    # Niveau 3 : expressions idiomatiques et sarcasme
    "Oh great, another NFT craze… exactly what the market needed.",
    "Bitcoin is mooning again, apparently gravity doesn't apply here.",
    "My portfolio tanked when the altcoins crashed overnight.",
    "Ethereum gas fees skyrocketed, and shockingly, I'm not surprised.",
    "Well, another rug pull… investors must love surprises!",
    "Nothing beats a bullish DeFi rally, until it collapses tomorrow.",

    # Niveau 4 : phrases ambiguës ou multiples émotions
    "I am thrilled about this Bitcoin surge but wary of a sudden correction.",
    "Ethereum's upgrade excites me, yet the price volatility worries me.",
    "Crypto adoption grows, but regulatory uncertainty looms.",
    "I'm happy with my NFT gains yet concerned about market speculation.",
    "The sudden spike in meme coins made me anxious yet curious.",
    "TradingView shows most top coins under 50- and 200-day SMAs, signaling broad market weakness, though some assets still outperform."
]

for text in test_texts:
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=config['MAX_LEN'])
    prediction = model.predict(padded)
    predicted_index = np.argmax(prediction, axis=-1)[0]
    predicted_sentiment = sentiment_mapping[predicted_index]
    
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {predicted_sentiment}")
    print(f"All Scores: {np.round(prediction[0], 2)}\n")
